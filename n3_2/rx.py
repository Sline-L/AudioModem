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
from .sync import SyncError, choose_channel, synchronize


class DecodeError(RuntimeError):
    def __init__(self, stage, message, metrics):
        self.stage = str(stage)
        self.message = str(message)
        self.metrics = metrics
        super().__init__(f"{self.stage}: {self.message}")


def _diagnostics():
    return {
        "metrics": {"channel_index": None, "chirp_score": None, "training_score": None,
                    "sfo": None, "stage": "wav_format", "header_ok": False},
        "debug": {}, "raw": b"", "H": np.empty(0, dtype=complex),
        "phase_fit": np.empty((0, 2), dtype=float),
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


def _receive(path, source, state):
    metrics = state["metrics"]
    stage = "wav_format"
    try:
        _, pcm = read_pcm16_wav(path)
        samples = pcm.astype(float) / 32768.0
        stage = "channel"
        try:
            choice = choose_channel(samples)
        except SyncError:
            # A single channel has no selection ambiguity: expose its sync failure.
            if samples.ndim == 1:
                metrics["channel_index"] = 0
                synchronize(samples)
            raise
        metrics["channel_index"] = choice.index
        samples = choice.samples
        stage = "chirp"
        sync = synchronize(samples)
        metrics.update(sfo=sync.sfo, sync_score=sync.score, start=sync.start,
                       training_start=sync.training_start, payload_start=sync.payload_start)
        chirp = linear_chirp()
        observed = _samples_at(samples, sync.start + (1 + sync.sfo) * np.arange(chirp.size))
        denominator = np.linalg.norm(observed) * np.linalg.norm(chirp)
        metrics["chirp_score"] = float(abs(np.vdot(observed, chirp)) / denominator)

        stage = "training"
        training = training_symbols()[:8, K0:K1 + 1]
        y = _observations(samples, sync.training_start, sync.sfo, 8)
        estimates = y / training
        h = estimates.mean(axis=0)
        state["H"] = h
        power = float(np.sum(np.abs(estimates) ** 2))
        if power <= 0 or not np.all(np.isfinite(h)):
            raise ValueError("training channel estimate has no finite power")
        metrics["training_score"] = float(np.sum(np.abs(estimates.sum(axis=0)) ** 2) / (8 * power))
        phase = np.unwrap(np.angle(estimates * np.conj(h)), axis=1)
        state["phase_fit"] = np.polyfit(np.arange(K0, K1 + 1), phase.T, 1).T
        valid = np.abs(h) > max(float(np.max(np.abs(h))) * 1e-8, 1e-12)
        if not np.any(valid):
            raise ValueError("training channel estimate has no usable carriers")
        residual = (y[:, valid] - training[:, valid] * h[valid]) / h[valid]
        noise_var = max(float(np.mean(np.abs(residual) ** 2)), 1e-3)
        metrics["noise_var"] = noise_var

        stage = "ldpc"
        # Keep fractional timing at the Header boundary rather than rounding it.
        first = sync.training_start + 8 * L * (1.0 + sync.sfo)
        coded = _observations(samples, first, sync.sfo, 1)[0]
        equalized = np.zeros_like(coded)
        np.divide(coded, h, out=equalized, where=valid)
        llr = np.clip(qpsk_llr(equalized, noise_var), -30.0, 30.0)
        bits, ok = StandardLdpc().decode_llr(llr)
        state["raw"] = np.packbits(bits, bitorder="big").tobytes()
        state["debug"]["ldpc_ok"] = bool(ok)
        if not ok:
            raise ValueError("Header LDPC parity check failed")

        stage = "header_fields"
        try:
            header = parse_header(state["raw"])
        except ValueError as exc:
            if "CRC" in str(exc):
                stage = "header_crc"
            raise
        if header["payload_symbols"] != (header["size"] + 248) // 249:
            raise ValueError("payload count does not match Header file length")
        if not header["name"]:
            raise ValueError("Header filename is empty")
        metrics.update(stage="header_ok", header_ok=True, **header)
        state["debug"]["header"] = header
        return header, metrics
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
    np.save(out / "H.npy", state["H"])
    np.save(out / "training_phase_fit.npy", state["phase_fit"])
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state["metrics"]))
        writer.writeheader()
        writer.writerow(state["metrics"])


def run_rx(path: Path, out: Path, source: Path | None = None) -> Path:
    """Write Header diagnostics on success or failure, without recovering payload."""
    state = _diagnostics()
    out = Path(out)
    try:
        _receive(Path(path), source, state)
    finally:
        _write_diagnostics(out, state)
    return out


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
    print(f"Header decoded; diagnostics written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
