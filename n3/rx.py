from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from pathlib import Path

import numpy as np

from n3.ldpc_codec import decode_block, encode_block, llr_from_qpsk
from n3.modem import (
    ACTIVE_BINS,
    CHIRP_SAMPLES,
    CODE_BITS,
    FS,
    INFO_BYTES,
    L,
    SILENCE_SAMPLES,
    TRAINING_SYMBOLS,
    bits_from_bytes,
    coded_symbol_count,
    chirp_wave,
    header_bytes,
    ofdm_rx,
    ofdm_tx,
    parse_header,
    qpsk_to_bits,
    read_wav,
    training_symbols,
)
from n3.sync import (
    equalize,
    estimate_channel,
    frame_candidates,
    phase_correct,
    training_from_recording,
)


LOCAL_PPM_RADIUS = 5.0
LOCAL_PPM_STEP = 0.25
COARSE_PPM_STEP = 1.0
REFINE_PPM_RADIUS = 1.0
BASE_LLR_SCALE = 3.0
MIN_LLR_SCALE = 0.5


def adaptive_llr_scale(training_nrmse: float) -> float:
    """Lower soft-decision confidence when the training residual is large."""
    value = float(training_nrmse)
    if not np.isfinite(value) or value <= 1.0:
        return BASE_LLR_SCALE
    return float(np.clip(BASE_LLR_SCALE / np.sqrt(value), MIN_LLR_SCALE, BASE_LLR_SCALE))


def adaptive_llr_scales(channel: np.ndarray, training_error: np.ndarray) -> np.ndarray:
    """Estimate per-carrier LLR confidence from training residuals and channel gain."""
    h = np.asarray(channel, dtype=complex).ravel()
    error = np.asarray(training_error, dtype=float).ravel()
    if len(h) != len(error):
        raise ValueError("channel and training error must have equal lengths")
    base = adaptive_llr_scale(float(np.nanmedian(error)))
    if not np.isfinite(base) or not np.isfinite(error).any() or float(np.nanmedian(error)) <= 1e-6:
        return np.full(len(h), BASE_LLR_SCALE, dtype=float)
    reliability = np.abs(h) ** 2 / np.maximum(error, 1e-12)
    valid = np.isfinite(reliability) & (reliability > 0)
    if not np.any(valid):
        return np.full(len(h), base, dtype=float)
    reference = float(np.median(reliability[valid]))
    relative = np.ones(len(h), dtype=float)
    relative[valid] = reliability[valid] / max(reference, 1e-12)
    weights = np.clip(np.sqrt(relative), 0.25, 4.0)
    return base * weights


def bit_error_count(received: np.ndarray, truth: np.ndarray) -> tuple[int, int, float | None]:
    """Return errors, compared bits, and BER using the N2 convention."""
    received = np.asarray(received, dtype=np.uint8).ravel()
    truth = np.asarray(truth, dtype=np.uint8).ravel()
    count = min(len(received), len(truth))
    if count == 0:
        return 0, 0, None
    errors = int(np.count_nonzero(received[:count] != truth[:count]))
    return errors, int(count), float(errors / count)


def byte_error_count(received: bytes, truth: bytes) -> tuple[int, int, float | None]:
    """Return byte errors, compared bytes, and byte error rate like N2."""
    count = min(len(received), len(truth))
    if count == 0:
        return 0, 0, None
    errors = sum(a != b for a, b in zip(received[:count], truth[:count]))
    return int(errors), int(count), float(errors / count)


def _truth_payload_coded_bits(source: Path, payload_symbols: int) -> np.ndarray:
    """Rebuild the N3 payload LDPC stream with the same 249-byte block padding."""
    raw = source.read_bytes()
    blocks = [
        raw[offset:offset + INFO_BYTES].ljust(INFO_BYTES, b"\0")
        for offset in range(0, len(raw), INFO_BYTES)
    ]
    if not blocks:
        return np.empty(0, dtype=np.uint8)
    coded = np.concatenate([encode_block(block) for block in blocks])
    return np.asarray(coded[: int(payload_symbols) * CODE_BITS], dtype=np.uint8)


def ber_diagnostics(result: dict | None, source: str | Path | None) -> dict:
    """Calculate N2-style header, raw-coded, decoded, and overall BER metrics."""
    if result is None or source is None:
        return {"ber_available": False, "ber_error": "provide --source <original_file>"}
    source_path = Path(source)
    if not source_path.exists():
        return {"ber_available": False, "ber_error": f"source file not found: {source_path}"}

    truth = source_path.read_bytes()
    header = result["header"]
    truth_header = header_bytes(source_path.name, len(truth), header["payload_symbols"])
    header_errors, header_bits, header_ber = bit_error_count(
        bits_from_bytes(result["header_raw"]), bits_from_bytes(truth_header)
    )

    received_coded_bits = qpsk_to_bits(result["payload_symbols"])
    truth_coded_bits = _truth_payload_coded_bits(source_path, header["payload_symbols"])
    coded_errors, coded_bits, coded_ber = bit_error_count(received_coded_bits, truth_coded_bits)

    payload_bits, truth_bits = bits_from_bytes(result["payload"]), bits_from_bytes(truth)
    payload_errors, payload_bits_count, payload_ber = bit_error_count(payload_bits, truth_bits)
    payload_byte_errors, payload_bytes, payload_byte_rate = byte_error_count(result["payload"], truth)

    overall_errors = header_errors + payload_errors
    overall_bits = header_bits + payload_bits_count
    overall_ber = float(overall_errors / overall_bits) if overall_bits else None
    return {
        "ber_available": True,
        "ber_source": str(source_path),
        "ber_payload_symbols_used": int(header["payload_symbols"]),
        "ber_payload_mod_symbols_used": int(np.asarray(result["payload_symbols"]).size),
        "ber_header_bit_errors": int(header_errors),
        "ber_header_bits": int(header_bits),
        "ber_header_ber": header_ber,
        "ber_ldpc_off_bit_errors": int(coded_errors),
        "ber_ldpc_off_bits": int(coded_bits),
        "ber_ldpc_off": coded_ber,
        "ber_ldpc_off_interleaved_bit_errors": int(coded_errors),
        "ber_ldpc_off_interleaved_bits": int(coded_bits),
        "ber_ldpc_off_interleaved": coded_ber,
        "ber_ldpc_off_deinterleaved_bit_errors": int(coded_errors),
        "ber_ldpc_off_deinterleaved_bits": int(coded_bits),
        "ber_ldpc_off_deinterleaved": coded_ber,
        "ber_ldpc_on_bit_errors": int(payload_errors),
        "ber_ldpc_on_bits": int(payload_bits_count),
        "ber_ldpc_on": payload_ber,
        "ber_payload_raw_coded_bit_errors": int(coded_errors),
        "ber_payload_raw_coded_bits": int(coded_bits),
        "ber_payload_raw_coded_ber": coded_ber,
        "ber_payload_bit_errors": int(payload_errors),
        "ber_payload_bits": int(payload_bits_count),
        "ber_payload_ber": payload_ber,
        "ber_payload_byte_errors": int(payload_byte_errors),
        "ber_payload_bytes": int(payload_bytes),
        "ber_payload_byte_error_rate": payload_byte_rate,
        "ber_overall_useful_bit_errors": int(overall_errors),
        "ber_overall_useful_bits": int(overall_bits),
        "ber_overall_useful_ber": overall_ber,
        "ber_payload_crc32": int(zlib.crc32(result["payload"])),
        "ber_truth_crc32": int(zlib.crc32(truth)),
    }


def decode_data_row(
    row: np.ndarray,
    channel: np.ndarray,
    llr_scale: float | np.ndarray = BASE_LLR_SCALE,
) -> tuple[bytes, int]:
    equalized = equalize(np.asarray(row, dtype=complex).reshape(1, -1), channel).ravel()
    return decode_block(llr_from_qpsk(equalized, scale=llr_scale))


def _training_phase_fit(received: np.ndarray, known: np.ndarray, channel: np.ndarray) -> np.ndarray:
    fits = []
    h = np.where(np.abs(channel) > 1e-12, channel, 1.0)
    for row, reference in zip(received, known):
        residual = row / np.where(np.abs(h * reference) > 1e-12, h * reference, 1.0)
        phase = np.unwrap(np.angle(residual))
        slope, intercept = np.polyfit(ACTIVE_BINS.astype(float), phase, 1)
        error = phase - (slope * ACTIVE_BINS + intercept)
        delta_n = -slope * 8192 / (2.0 * np.pi)
        fits.append([intercept, slope, delta_n, float(np.sqrt(np.mean(error * error)))])
    return np.asarray(fits, dtype=float)


def _data_rows(
    samples: np.ndarray,
    frame_start: float,
    data_symbols: int,
    epsilon: float,
) -> np.ndarray:
    offset = CHIRP_SAMPLES + SILENCE_SAMPLES + TRAINING_SYMBOLS * L
    start = int(round(frame_start + offset))
    block = ofdm_rx(samples[start:start + data_symbols * L])
    starts = start + np.arange(len(block)) * L
    return phase_correct(block, starts, frame_start, epsilon)


def _candidate_raw_blocks(
    samples: np.ndarray,
    frame_start: float,
    training_freq: np.ndarray,
    data_symbols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract FFT blocks once; SFO trials only apply a phase ramp afterward."""
    training_start = int(round(frame_start + CHIRP_SAMPLES + SILENCE_SAMPLES))
    front = ofdm_rx(samples[training_start:training_start + TRAINING_SYMBOLS * L])
    front_starts = training_start + np.arange(len(front)) * L
    known_front = training_freq[:TRAINING_SYMBOLS, ACTIVE_BINS]

    tail_offset = CHIRP_SAMPLES + SILENCE_SAMPLES + TRAINING_SYMBOLS * L + data_symbols * L
    tail_start = int(round(frame_start + tail_offset))
    tail = ofdm_rx(samples[tail_start:tail_start + TRAINING_SYMBOLS * L])
    tail_starts = tail_start + np.arange(len(tail)) * L
    known_tail = training_freq[TRAINING_SYMBOLS:TRAINING_SYMBOLS + len(tail), ACTIVE_BINS]
    if len(tail):
        training_raw = np.vstack((front, tail))
        known_training = np.vstack((known_front, known_tail))
        training_starts = np.r_[front_starts, tail_starts]
    else:
        training_raw = front
        known_training = known_front
        training_starts = front_starts

    data_start = int(round(frame_start + CHIRP_SAMPLES + SILENCE_SAMPLES + TRAINING_SYMBOLS * L))
    data_raw = ofdm_rx(samples[data_start:data_start + data_symbols * L])
    data_starts = data_start + np.arange(len(data_raw)) * L
    return training_raw, known_training, training_starts, data_raw, data_starts


def _attempt(
    samples: np.ndarray,
    candidate: dict,
    training_freq: np.ndarray,
    ppm: float,
    decode_payload: bool = True,
    raw_blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict:
    epsilon = float(ppm) * 1e-6
    frame_start = float(candidate["timing"]["ofdm_frame_start"])
    data_symbols = int(candidate["data_symbols"])
    if raw_blocks is None:
        received_training, known_training, _ = training_from_recording(
            samples, frame_start, training_freq, epsilon, data_symbols=data_symbols
        )
        data = _data_rows(samples, frame_start, data_symbols, epsilon)
    else:
        training_raw, known_training, training_starts, data_raw, data_starts = raw_blocks
        received_training = phase_correct(training_raw, training_starts, frame_start, epsilon)
        data = phase_correct(data_raw, data_starts, frame_start, epsilon)
    channel = estimate_channel(received_training, known_training)
    training_error = np.mean(
        np.abs(received_training - known_training * channel) ** 2,
        axis=0,
    )
    training_nrmse = float(
        np.mean(training_error) / max(float(np.mean(np.abs(channel) ** 2)), 1e-12)
    )
    llr_scales = adaptive_llr_scales(channel, training_error)
    if len(data) != data_symbols:
        raise ValueError("recording ended before all coded data symbols")
    header_raw, header_iterations = decode_data_row(data[0], channel, llr_scales)
    header = parse_header(header_raw)
    if coded_symbol_count(header["file_size"]) != data_symbols:
        raise ValueError(
            f"coded symbol count mismatch: header expects {coded_symbol_count(header['file_size'])}, "
            f"recording has {data_symbols}"
        )
    payload_blocks = []
    iterations = [header_iterations]
    if decode_payload:
        for row in data[1:1 + header["payload_symbols"]]:
            block, count = decode_data_row(row, channel, llr_scales)
            payload_blocks.append(block)
            iterations.append(count)
        if len(payload_blocks) != header["payload_symbols"]:
            raise ValueError("recording ended before all payload symbols")
        payload = b"".join(payload_blocks)[: header["file_size"]]
    else:
        # SFO search only needs a valid header.  Deferring the payload LDPC
        # blocks avoids doing the expensive full decode for every trial.
        payload = b""
    phase_fit = _training_phase_fit(received_training, known_training, channel)
    return {
        "header": header,
        "header_raw": header_raw,
        "payload": payload,
        "channel": channel,
        "training_error": training_error,
        "training_nrmse": training_nrmse,
        "phase_fit": phase_fit,
        "payload_symbols": equalize(data[1:1 + header["payload_symbols"]], channel).ravel(),
        "ldpc_iterations": iterations,
        "llr_scale": float(np.median(llr_scales)),
        "llr_scale_min": float(np.min(llr_scales)),
        "llr_scale_max": float(np.max(llr_scales)),
        "ppm": float(ppm),
        "frame_start": frame_start,
    }


def evaluate_candidate(samples: np.ndarray, candidate: dict, training_freq: np.ndarray) -> dict:
    center = float(candidate.get("coarse_sfo_ppm", 0.0))
    coarse_values = np.arange(
        center - LOCAL_PPM_RADIUS,
        center + LOCAL_PPM_RADIUS + COARSE_PPM_STEP / 2,
        COARSE_PPM_STEP,
    )
    raw_blocks = _candidate_raw_blocks(
        samples,
        float(candidate["timing"]["ofdm_frame_start"]),
        training_freq,
        int(candidate["data_symbols"]),
    )
    attempts = []
    errors = []

    def try_ppm(ppm: float) -> None:
        try:
            result = _attempt(
                samples,
                candidate,
                training_freq,
                float(ppm),
                decode_payload=False,
                raw_blocks=raw_blocks,
            )
            attempts.append(result)
        except (ValueError, IndexError, FloatingPointError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    for ppm in coarse_values:
        try_ppm(float(ppm))
    if not attempts:
        raise ValueError(errors[-1] if errors else "no valid SFO attempt")

    # Refine only around the best coarse point.  If the coarse minimum lands
    # on the search boundary, retain the old dense search as a safety net.
    coarse_best = min(attempts, key=lambda result: (result["training_nrmse"], abs(result["ppm"] - center)))
    at_boundary = abs(float(coarse_best["ppm"]) - center) >= LOCAL_PPM_RADIUS - COARSE_PPM_STEP / 2
    if at_boundary:
        refine_center = center
        refine_radius = LOCAL_PPM_RADIUS
    else:
        refine_center = float(coarse_best["ppm"])
        refine_radius = REFINE_PPM_RADIUS
    refine_values = np.arange(
        refine_center - refine_radius,
        refine_center + refine_radius + LOCAL_PPM_STEP / 2,
        LOCAL_PPM_STEP,
    )
    seen = {round(float(result["ppm"]), 9) for result in attempts}
    for ppm in refine_values:
        key = round(float(ppm), 9)
        if key not in seen:
            try_ppm(float(ppm))
            seen.add(key)

    attempts.sort(key=lambda result: (result["training_nrmse"], abs(result["ppm"] - center)))
    # Only the best training/SFO point gets the complete LDPC payload decode.
    # The header was already validated during the cheap search pass.
    selected = attempts[0]
    chosen = _attempt(
        samples,
        candidate,
        training_freq,
        float(selected["ppm"]),
        decode_payload=True,
        raw_blocks=raw_blocks,
    )
    chosen["candidate"] = candidate
    chosen["attempts"] = [{"ppm": a["ppm"], "training_nrmse": a["training_nrmse"]} for a in attempts]
    return chosen


def _save_summary(path: Path, channel: np.ndarray, training_error: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin", "freq_hz", "abs_h", "training_error"])
        for index, bin_index in enumerate(ACTIVE_BINS):
            writer.writerow(
                [
                    int(bin_index),
                    float(bin_index * FS / 8192),
                    float(abs(channel[index])) if index < len(channel) else float("nan"),
                    float(training_error[index]) if index < len(training_error) else float("nan"),
                ]
            )


def recover_recording(receive: str | Path, out: str | Path, source: str | Path | None = None) -> dict:
    receive = Path(receive)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    header_raw = b""
    payload = b""
    channel = np.zeros(len(ACTIVE_BINS), complex)
    training_error = np.full(len(ACTIVE_BINS), np.nan)
    phase_fit = np.empty((0, 4), float)
    payload_symbols = np.empty(0, complex)
    result = None
    error = ""
    try:
        samples = read_wav(receive)
        training_freq, _ = training_symbols()
        chirp = chirp_wave()
        leading_wave = ofdm_tx(training_freq[:TRAINING_SYMBOLS, ACTIVE_BINS])
        candidates = frame_candidates(samples, chirp, leading_wave)
        if not candidates:
            raise ValueError("no legal front/rear chirp pair found")
        attempts = []
        for candidate in candidates[:8]:
            try:
                attempts.append(evaluate_candidate(samples, candidate, training_freq))
            except ValueError:
                continue
        if not attempts:
            raise ValueError("no candidate produced a valid N3 header")
        result = min(attempts, key=lambda value: (value["training_nrmse"], abs(value["ppm"])))
        header_raw = result["header_raw"]
        payload = result["payload"]
        channel = result["channel"]
        training_error = result["training_error"]
        phase_fit = result["phase_fit"]
        payload_symbols = result["payload_symbols"]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    (output / "decoded_header.bin").write_bytes(header_raw)
    (output / "decoded_payload.bin").write_bytes(payload)
    np.save(output / "H.npy", channel)
    np.save(output / "training_phase_fit.npy", phase_fit)
    np.save(output / "payload_symbols.npy", payload_symbols)
    _save_summary(output / "summary.csv", channel, training_error)

    recovered_name = ""
    file_match = False
    if result is not None:
        recovered_name = result["header"]["name"]
        (output / recovered_name).write_bytes(payload)
        if source is not None and Path(source).exists():
            file_match = payload == Path(source).read_bytes()
    metrics = {
        "standard_profile": "n3_example_tx",
        "input": str(receive),
        "out": str(output),
        "header_ok": bool(result is not None),
        "file_match": bool(file_match),
        "file_crc_ok": None,
        "front_chirp_score": float(result["candidate"]["front_score"]) if result else 0.0,
        "tail_chirp_score": float(result["candidate"]["tail_score"]) if result else 0.0,
        "chirp_front_start": int(result["candidate"]["front_start"]) if result else None,
        "tail_chirp_start": int(result["candidate"]["tail_start"]) if result else None,
        "ofdm_frame_start": float(result["frame_start"]) if result else None,
        "selected_sfo_ppm": float(result["ppm"]) if result else None,
        "coarse_sfo_ppm": float(result["candidate"]["coarse_sfo_ppm"]) if result else None,
        "fft_timing_offset_samples": float(result["candidate"]["timing_offset"]) if result else None,
        "fft_timing_peak_score": float(result["candidate"]["training_score"]) if result else 0.0,
        "fft_timing_confident": bool(result["candidate"]["timing_confident"]) if result else False,
        "data_symbols": int(result["candidate"]["data_symbols"]) if result else None,
        "mean_abs_h": float(np.mean(np.abs(channel))) if channel.size else None,
        "median_training_error": float(np.nanmedian(training_error)) if np.isfinite(training_error).any() else None,
        "llr_scale": float(result["llr_scale"]) if result else None,
        "llr_scale_min": float(result["llr_scale_min"]) if result else None,
        "llr_scale_max": float(result["llr_scale_max"]) if result else None,
        "ldpc_decode_iterations": result["ldpc_iterations"] if result else [],
        "payload_bit_errors": None,
        "payload_ber": None,
        "payload_byte_errors": None,
        "payload_byte_error_rate": None,
        "recovered_name": recovered_name,
        "recovered_bytes": int(len(payload)),
        "error": error,
    }
    metrics.update(ber_diagnostics(result, source))
    if metrics.get("ber_available"):
        metrics["payload_bit_errors"] = metrics["ber_payload_bit_errors"]
        metrics["payload_ber"] = metrics["ber_payload_ber"]
        metrics["payload_byte_errors"] = metrics["ber_payload_byte_errors"]
        metrics["payload_byte_error_rate"] = metrics["ber_payload_byte_error_rate"]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="recover Standardization/example_tx.py-compatible N3 WAVs")
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    options = args()
    results = []
    many = len(options.input) > 1
    for receive in options.input:
        destination = options.out / receive.stem if many else options.out
        metrics = recover_recording(receive, destination, options.source)
        results.append(metrics)
        print(
            f"{receive}: header_ok={metrics['header_ok']} "
            f"file_match={metrics['file_match']} recovered={metrics['recovered_name'] or '-'}"
        )
        if metrics.get("ber_available"):
            print(
                f"ber_header={metrics['ber_header_ber']:.6f} "
                f"ber_payload={metrics['ber_payload_ber']:.6f} "
                f"ber_overall={metrics['ber_overall_useful_ber']:.6f}"
            )
            print(
                f"ber_ldpc_off={metrics['ber_ldpc_off']:.6f} "
                f"ber_ldpc_on={metrics['ber_ldpc_on']:.6f}"
            )
        else:
            print(f"ber_unavailable={metrics.get('ber_error', 'provide --source <original_file>')}")
    failed = [result for result in results if not result["header_ok"] or (options.source and not result["file_match"])]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
