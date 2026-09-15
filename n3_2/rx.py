"""N3.2 Header receiver and persistent reception diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np

from .ldpc_codec import StandardLdpc
from .modem import (
    CP, FS, K0, K1, L, N, header_bytes, linear_chirp, parse_header,
    qpsk_llr, read_pcm16_wav, training_symbols,
)
from .sync import SyncError, choose_channel, frame_candidates, synchronize


class DecodeError(RuntimeError):
    def __init__(self, stage, message, metrics):
        self.stage = str(stage)
        self.message = str(message)
        self.metrics = metrics
        super().__init__(f"{self.stage}: {self.message}")


def _diagnostics():
    return {
        "metrics": {"channel_index": None, "chirp_score": None, "training_score": None,
        "sfo": None, "stage": "wav_format", "header_ok": False,
        "verified": False, "strict_stage": None},
        "debug": {}, "raw": b"", "H": np.empty(0, dtype=complex),
        "phase_fit": np.empty((0, 2), dtype=float), "payload": b"",
        "payload_symbols": np.empty((0, K1 - K0 + 1), dtype=np.complex128),
    }


def _samples_at(samples, positions):
    positions = np.asarray(positions)
    if positions.min() < 0 or positions.max() > samples.size - 1:
        raise ValueError("required symbol window is truncated")
    return np.interp(positions.ravel(), np.arange(samples.size), samples).reshape(positions.shape)


def _observations(samples, start, sfo, count):
    offsets = np.arange(count)[:, None] * L + CP + np.arange(N)[None, :]
    symbols = _samples_at(samples, start + (1.0 + sfo) * offsets)
    return np.fft.rfft(symbols, axis=1)[:, K0:K1 + 1]


def _source_debug(state, source):
    if source is None:
        return
    debug = state["debug"]
    debug.update(header_bit_errors=None, header_ber=None)
    if len(state["raw"]) != 249:
        return
    try:
        source = Path(source)
        size = source.stat().st_size
        expected = header_bytes(source.name, size, (size + 248) // 249)
    except (OSError, ValueError) as exc:
        debug["source_error"] = str(exc)
        return
    difference = np.frombuffer(state["raw"], dtype=np.uint8) ^ np.frombuffer(expected, dtype=np.uint8)
    errors = int(np.unpackbits(difference).sum())
    debug.update(header_bit_errors=errors, header_ber=errors / 1992)


def _decode_header_candidate(samples, sync):
    """Demodulate one standard Header candidate without mutating diagnostics."""
    training = training_symbols()[:8, K0:K1 + 1]
    y = _observations(samples, sync.training_start, sync.sfo, 8)
    estimates = y / training
    h = estimates.mean(axis=0)
    power = float(np.sum(np.abs(estimates) ** 2))
    if power <= 0 or not np.all(np.isfinite(h)):
        return None, {}, "training", "training channel estimate has no finite power"
    valid = np.abs(h) > max(float(np.max(np.abs(h))) * 1e-8, 1e-12)
    if not np.any(valid):
        return None, {}, "training", "training channel estimate has no usable carriers"
    residual = y - training * h
    carrier_noise = np.mean(np.abs(residual) ** 2, axis=0) / np.maximum(np.abs(h) ** 2, 1e-12)
    noise_var = max(float(np.median(carrier_noise[valid])), 1e-3)
    reliability = noise_var / np.maximum(carrier_noise, 1e-12)
    llr_scale = 3.0 * np.clip(np.sqrt(reliability), 0.25, 4.0)
    first = sync.training_start + 8 * L * (1.0 + sync.sfo)
    coded = _observations(samples, first, sync.sfo, 1)[0]
    equalized = np.zeros_like(coded)
    np.divide(coded, h, out=equalized, where=valid)
    llr = np.empty(2 * equalized.size, dtype=float)
    llr[0::2] = equalized.imag * llr_scale
    llr[1::2] = equalized.real * llr_scale
    bits, ok = StandardLdpc().decode_llr(np.clip(llr, -30.0, 30.0))
    raw = np.packbits(bits, bitorder="big").tobytes()
    result = {
        "raw": raw,
        "h": h,
        "noise_var": noise_var,
        "llr_scale": llr_scale,
        "front_training": y,
        "phase_fit": np.polyfit(np.arange(K0, K1 + 1), np.unwrap(np.angle(estimates * np.conj(h)), axis=1).T, 1).T,
        "training_score": float(np.sum(np.abs(estimates.sum(axis=0)) ** 2) / (8 * power)),
        "ldpc_ok": bool(ok),
    }
    if not ok:
        return None, result, "ldpc", "Header LDPC parity check failed"
    try:
        header = parse_header(raw)
    except ValueError as exc:
        return None, result, "header_crc" if "CRC" in str(exc) else "header_fields", str(exc)
    if header["payload_symbols"] != (header["size"] + 248) // 249:
        return None, result, "header_fields", "payload count does not match Header file length"
    if not header["name"]:
        return None, result, "header_fields", "Header filename is empty"
    return header, result, None, None


def _receive(path, source, state):
    metrics = state["metrics"]
    stage = "wav_format"
    try:
        _, pcm = read_pcm16_wav(path)
        samples = pcm.astype(float) / 32768.0
        stage = "channel"
        # Header CRC is the final authority.  Do not spend most of a recording
        # committing to one chirp/SFO result before testing any Header.
        strict_stage = "candidate"
        if samples.ndim == 1:
            choice = type("Choice", (), {"index": 0, "samples": samples})()
        else:
            energy = np.mean(samples * samples, axis=0)
            index = int(np.argmax(energy))
            choice = type("Choice", (), {"index": index, "samples": samples[:, index]})()
        metrics.update(verified=False, strict_stage=strict_stage)
        metrics["channel_index"] = choice.index
        samples = choice.samples
        stage = "chirp"
        candidates = []
        # The correlation peak can land anywhere inside the cyclic prefix on a
        # real room channel.  Test a short, symmetric FFT-boundary neighbourhood
        # and let the protected Header choose it; no recording-specific offset
        # is assumed here.
        for base in frame_candidates(samples):
            for offset in (0, -128, 128, -256, 256):
                training_start = base.training_start + offset
                if training_start < 0 or training_start + 9 * L >= samples.size:
                    continue
                candidates.append(type(base)(
                    base.start, base.rear_start, training_start,
                    training_start + int(round(8 * L * (1.0 + base.sfo))),
                    base.sfo, base.score - abs(offset) / (20.0 * CP),
                ))
        attempts, best = [], None
        for sync in candidates:
            header, result, candidate_stage, message = _decode_header_candidate(samples, sync)
            attempts.append({"training_start": sync.training_start, "sfo": sync.sfo,
                             "score": sync.score, "stage": candidate_stage or "header_ok"})
            if result and (best is None or result["training_score"] > best[1]["training_score"]):
                best = (sync, result, candidate_stage, message)
            if header is None:
                continue
            chirp = linear_chirp()
            observed = _samples_at(samples, sync.start + (1 + sync.sfo) * np.arange(chirp.size))
            denominator = np.linalg.norm(observed) * np.linalg.norm(chirp)
            metrics.update(sfo=sync.sfo, sync_score=sync.score, start=sync.start,
                           training_start=sync.training_start, payload_start=sync.payload_start,
                           chirp_score=float(abs(np.vdot(observed, chirp)) / denominator),
                           training_score=result["training_score"], noise_var=result["noise_var"],
                           stage="header_ok", header_ok=True, **header)
            state["debug"].update(ldpc_ok=True, header=header, candidate_attempts=attempts)
            state.update(raw=result["raw"], H=result["h"], h=result["h"], phase_fit=result["phase_fit"],
                         samples=samples, sync=sync, noise_var=result["noise_var"],
                         llr_scale=result["llr_scale"], front_training=result["front_training"])
            return header, metrics
        state["debug"]["candidate_attempts"] = attempts
        if best is not None:
            sync, result, stage, message = best
            state.update(raw=result["raw"], H=result["h"], phase_fit=result["phase_fit"])
            state["debug"]["ldpc_ok"] = result["ldpc_ok"]
            metrics.update(sfo=sync.sfo, sync_score=sync.score, start=sync.start,
                           training_start=sync.training_start, payload_start=sync.payload_start,
                           training_score=result["training_score"], noise_var=result["noise_var"])
            raise ValueError(message)
        stage = "training"
        raise ValueError("no Header candidate had usable training")
    except (SyncError, OSError, ValueError, RuntimeError, wave.Error, EOFError) as exc:
        if isinstance(exc, SyncError):
            stage = exc.stage
        message = exc.message if isinstance(exc, SyncError) else str(exc)
        metrics.update(stage=stage, header_ok=False, error=message)
        raise DecodeError(stage, message, metrics) from exc
    finally:
        state["debug"].update(stage=metrics["stage"], header_ok=metrics["header_ok"])
        _source_debug(state, source)


def decode_header(path: Path, source: Path | None = None) -> tuple[dict, dict]:
    """Decode the first information block; use run_rx to persist diagnostics."""
    return _receive(Path(path), source, _diagnostics())


def _write_diagnostics(out, state):
    out.mkdir(parents=True, exist_ok=True)
    for name, values in (("metrics", state["metrics"]), ("header_debug", state["debug"])):
        (out / f"{name}.json").write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
    (out / "decoded_header.bin").write_bytes(state["raw"])
    if state["payload"]:
        (out / "decoded_payload.bin").write_bytes(state["payload"])
        np.save(out / "payload_symbols.npy", state["payload_symbols"])
    np.save(out / "H.npy", state["H"])
    np.save(out / "training_phase_fit.npy", state["phase_fit"])
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state["metrics"]))
        writer.writeheader()
        writer.writerow(state["metrics"])


def _decode_payload(header, state):
    sync, h = state["sync"], state["h"]
    # Header tells us the exact data-block count, so tail training can now be
    # placed on the OFDM grid without guessing it during synchronization.
    total_blocks = 2 * ((249 + header["size"] + 497) // 498)
    tail_known = training_symbols()[8:, K0:K1 + 1]
    front_known = training_symbols()[:8, K0:K1 + 1]
    joint_known = np.vstack((front_known, tail_known))
    candidates = []
    # Header CRC establishes the frame; use both training edges to refine the
    # clock over a range that remains valid for independent sound devices.
    for candidate_sfo in sync.sfo + np.arange(-50.0, 50.1, 1.0) * 1e-6:
        front = _observations(state["samples"], sync.training_start, candidate_sfo, 8)
        tail_start = sync.training_start + (8 + total_blocks) * L * (1.0 + candidate_sfo)
        tail = _observations(state["samples"], tail_start, candidate_sfo, 8)
        joint = np.vstack((front, tail))
        candidate_h = (joint / joint_known).mean(axis=0)
        error = np.mean(np.abs(joint - joint_known * candidate_h) ** 2, axis=0)
        score = float(np.median(error / np.maximum(np.abs(candidate_h) ** 2, 1e-12)))
        candidates.append((score, float(candidate_sfo), candidate_h, error))
    _, payload_sfo, h, error = min(candidates, key=lambda item: item[0])
    carrier_noise = error / np.maximum(np.abs(h) ** 2, 1e-12)
    base = max(float(np.median(carrier_noise)), 1e-3)
    scale = 3.0 * np.clip(np.sqrt(base / np.maximum(carrier_noise, 1e-12)), 0.25, 4.0)
    start = sync.training_start + 9 * L * (1.0 + payload_sfo)
    observed = _observations(state["samples"], start, payload_sfo, header["payload_symbols"])
    equalized = np.zeros_like(observed)
    valid = np.abs(h) > max(float(np.max(np.abs(h))) * 1e-8, 1e-12)
    np.divide(observed, h, out=equalized, where=valid)
    state["metrics"]["payload_sfo"] = payload_sfo
    codec = StandardLdpc()
    blocks = []
    for index, symbol in enumerate(equalized):
        llr = np.empty(2 * symbol.size, dtype=float)
        llr[0::2] = symbol.imag * scale
        llr[1::2] = symbol.real * scale
        llr = np.clip(llr, -30.0, 30.0)
        bits, ok = codec.decode_llr(llr)
        if not ok:
            raise ValueError(f"payload block {index} LDPC parity check failed")
        blocks.append(np.packbits(bits, bitorder="big").tobytes())
    state["payload_symbols"] = equalized
    state["payload"] = b"".join(blocks)[:header["size"]]


def run_rx(path: Path, out: Path, source: Path | None = None) -> Path:
    """Recover the Header-declared payload and always write diagnostics."""
    state = _diagnostics()
    out = Path(out)
    try:
        header, _ = _receive(Path(path), source, state)
        try:
            _decode_payload(header, state)
        except (ValueError, RuntimeError) as exc:
            state["metrics"].update(stage="ldpc_payload", header_ok=False, error=str(exc))
            raise DecodeError("ldpc_payload", str(exc), state["metrics"]) from exc
    finally:
        _write_diagnostics(out, state)
    recovered = out / Path(header["name"]).name
    recovered.write_bytes(state["payload"])
    return recovered


def main() -> int:
    parser = argparse.ArgumentParser(description="decode an N3.2 standard Header")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    try:
        run_rx(args.input, args.out, args.source)
    except DecodeError as exc:
        print(str(exc))
        return 1
    print(f"File recovered to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
