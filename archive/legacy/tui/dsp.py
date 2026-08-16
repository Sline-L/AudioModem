from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import wave
import zlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "run" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import legacy_step8 as legacy
from .codec import (
    HEADER_SIZE,
    encoded_bit_length,
    encoded_frames,
    fec_decode,
    fec_decode_values,
    map_data,
    demap_llr,
    parse_data_block,
    parse_header,
)
from .profiles import ModemProfile


I16 = 32768.0


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def configure_legacy(profile: ModemProfile) -> np.ndarray:
    bins = np.asarray(profile.active_bins, dtype=int)
    legacy.FS = profile.fs
    legacy.N = profile.fft_size
    legacy.CP = profile.cp_samples
    legacy.L = profile.symbol_len
    legacy.SEGMENTS = tuple(tuple(item) for item in profile.active_ranges)
    legacy.ACTIVE_BINS = bins
    legacy.PILOT_SPACING = profile.pilot_spacing
    legacy.SEGMENT_INDICES = [
        np.flatnonzero((bins >= start) & (bins <= end)) for start, end in profile.active_ranges
    ]
    legacy.ofdm_tx.__defaults__ = (bins,)
    legacy.ofdm_rx.__defaults__ = (bins,)
    legacy.pilot_indices.__defaults__ = (profile.pilot_spacing,)
    legacy.data_indices.__defaults__ = (profile.pilot_spacing,)
    return bins


def read_wav(path: Path, profile: ModemProfile) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != profile.fs:
            raise ValueError(f"expected mono 16-bit {profile.fs} Hz WAV")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / I16


def write_wav(path: Path, samples: np.ndarray, profile: ModemProfile) -> float:
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        raise ValueError("empty signal")
    gain = 0.95 * (I16 - 1) / I16 / peak
    pcm = np.clip(samples * gain * I16, -I16 + 1, I16 - 1).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(profile.fs)
        wav.writeframes(pcm.tobytes())
    return gain


def ofdm_tx(symbols: np.ndarray, profile: ModemProfile) -> np.ndarray:
    bins = np.asarray(profile.active_bins)
    freq = np.zeros((len(symbols), profile.fft_size), complex)
    freq[:, bins] = symbols
    freq[:, profile.fft_size - bins] = np.conj(symbols)
    time = np.fft.ifft(freq, axis=1).real
    return np.c_[time[:, -profile.cp_samples :], time].ravel()


def ofdm_rx(samples: np.ndarray, profile: ModemProfile) -> np.ndarray:
    rows = len(samples) // profile.symbol_len
    if rows == 0:
        return np.empty((0, len(profile.active_bins)), complex)
    time = samples[: rows * profile.symbol_len].reshape(rows, profile.symbol_len)[:, profile.cp_samples :]
    return np.fft.fft(time, axis=1)[:, profile.active_bins]


def random_qpsk(rows: int, profile: ModemProfile, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (rows, len(profile.active_bins), 2), dtype=np.uint8)
    return (np.where(bits[:, :, 1], -1, 1) + 1j * np.where(bits[:, :, 0], -1, 1)) / np.sqrt(2)


def segment_indices(profile: ModemProfile) -> list[np.ndarray]:
    bins = np.asarray(profile.active_bins)
    return [np.flatnonzero((bins >= a) & (bins <= b)) for a, b in profile.active_ranges]


def pilot_indices(symbol: int, profile: ModemProfile) -> np.ndarray:
    offset = symbol % profile.pilot_spacing
    return np.concatenate(
        [part[np.arange(len(part)) % profile.pilot_spacing == offset] for part in segment_indices(profile)]
    )


def data_indices(symbol: int, profile: ModemProfile) -> np.ndarray:
    mask = np.ones(len(profile.active_bins), dtype=bool)
    mask[pilot_indices(symbol, profile)] = False
    return np.flatnonzero(mask)


def pilot_values(symbol: int, indices: np.ndarray, profile: ModemProfile) -> np.ndarray:
    rng = np.random.default_rng(profile.pilot_seed + symbol)
    bits = rng.integers(0, 2, (len(profile.active_bins), 2), dtype=np.uint8)
    values = (np.where(bits[:, 1], -1, 1) + 1j * np.where(bits[:, 0], -1, 1)) / np.sqrt(2)
    return values[indices]


def payload_symbols(bits: np.ndarray, profile: ModemProfile) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rows: list[np.ndarray] = []
    used: list[int] = []
    pos = 0
    logical = 0
    bits_per_value = 1 if profile.modulation == "bpsk" else 2
    while pos < len(bits):
        pidx = pilot_indices(logical, profile)
        didx = data_indices(logical, profile)
        take = min(len(didx) * bits_per_value, len(bits) - pos)
        value_count = (take + bits_per_value - 1) // bits_per_value
        row = np.zeros(len(profile.active_bins), complex)
        row[pidx] = pilot_values(logical, pidx, profile)
        row[didx[:value_count]] = map_data(bits[pos : pos + take], profile.modulation)
        rows.append(row)
        used.append(take)
        pos += take
        logical += 1
    return np.asarray(rows), np.asarray(used, dtype=int)


def frame_payload(payload: np.ndarray, profile: ModemProfile) -> tuple[np.ndarray, np.ndarray]:
    parts: list[np.ndarray] = []
    starts: list[int] = []
    logical = physical = anchor = 0
    while logical < len(payload):
        take = min(profile.timing_anchor_interval, len(payload) - logical)
        parts.append(payload[logical : logical + take])
        logical += take
        physical += take
        if logical < len(payload):
            starts.append(physical)
            parts.append(random_qpsk(profile.timing_anchor_symbols, profile, profile.timing_anchor_seed + anchor))
            physical += profile.timing_anchor_symbols
            anchor += 1
    return np.vstack(parts), np.asarray(starts, dtype=int)


def encode_file(source: Path, out: Path, profile: ModemProfile) -> dict:
    profile.validate()
    configure_legacy(profile)
    source = Path(source).resolve()
    out = Path(out).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    emit("progress", stage="framing", progress=0.05)
    frames, labels, _ = encoded_frames(source, profile)
    coded = np.concatenate(frames)
    logical, used = payload_symbols(coded, profile)
    physical, anchor_starts = frame_payload(logical, profile)
    sync = random_qpsk(profile.sync_symbols, profile, profile.sync_seed)
    preambles = [
        random_qpsk(profile.preamble_symbols, profile, profile.preamble_seed + i)
        for i in range(profile.preamble_repeats)
    ]
    start_anchor = random_qpsk(
        profile.payload_start_anchor_symbols, profile, profile.payload_start_anchor_seed
    ) if profile.payload_start_anchor_symbols else np.empty((0, len(profile.active_bins)), complex)
    emit("progress", stage="ofdm", progress=0.45)
    noise = np.zeros(round(profile.noise_seconds * profile.fs))
    tail = np.zeros(round(profile.tail_seconds * profile.fs))
    raw = np.r_[
        noise,
        ofdm_tx(sync, profile),
        ofdm_tx(np.vstack(preambles), profile),
        ofdm_tx(start_anchor, profile),
        ofdm_tx(physical, profile),
        tail,
    ]
    gain = write_wav(out, raw, profile)
    np.save(out.with_suffix(".sync.npy"), sync)
    np.save(out.with_suffix(".preamble.npy"), np.vstack(preambles))
    np.save(out.with_suffix(".start_anchor.npy"), start_anchor)
    np.save(out.with_suffix(".anchor_starts.npy"), anchor_starts)
    meta = {
        "protocol": profile.protocol,
        "profile": profile.to_dict(),
        "profile_id": profile.profile_id,
        "input": str(source),
        "out": str(out),
        "input_bytes": source.stat().st_size,
        "frame_labels": labels,
        "frame_coded_bits": [len(frame) for frame in frames],
        "payload_coded_bits": len(coded),
        "payload_data_slots": int(used.sum()),
        "logical_payload_symbols": len(logical),
        "physical_payload_symbols": len(physical),
        "timing_anchor_count": len(anchor_starts),
        "total_samples": len(raw),
        "seconds": len(raw) / profile.fs,
        "gain": gain,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    emit("result", status="success", wav=str(out), meta=str(out.with_suffix(".meta.json")), seconds=meta["seconds"])
    return meta


def _anchor_channel(rows: np.ndarray, known: np.ndarray, current: np.ndarray) -> np.ndarray:
    estimates = rows / known
    aligned = []
    for estimate in estimates:
        weight = np.maximum(np.abs(current) * np.abs(estimate), 1e-12)
        phase = np.angle(np.sum(weight * np.conj(estimate) * current))
        aligned.append(estimate * np.exp(1j * phase))
    return np.mean(aligned, axis=0)


def payload_llrs(y: np.ndarray, h_initial: np.ndarray, training_var: np.ndarray,
                 silence_var: np.ndarray, profile: ModemProfile) -> dict:
    h = np.asarray(h_initial, complex).copy()
    bins = np.asarray(profile.active_bins)
    base_var = np.maximum(training_var, silence_var)
    positive = base_var[base_var > 0]
    floor = float(np.median(positive)) if len(positive) else 1e-8
    base_var = np.maximum(base_var, floor * 0.1)
    llrs: list[np.ndarray] = []
    llr_bins: list[np.ndarray] = []
    h_track: list[np.ndarray] = []
    anchor_track: list[np.ndarray] = []
    cpes: list[float] = []
    slopes: list[float] = []
    raw_slopes: list[float] = []
    residuals: list[float] = []
    history: list[float] = []
    physical = logical = anchor = 0
    while physical < len(y):
        count = min(profile.timing_anchor_interval, len(y) - physical)
        for row in y[physical : physical + count]:
            pidx = pilot_indices(logical, profile)
            didx = data_indices(logical, profile)
            pilots = pilot_values(logical, pidx, profile)
            observed_h = row[pidx] / pilots
            cpe, raw_slope, center = legacy._phase_measurement(observed_h, h, pidx)
            raw_slope = float(np.clip(raw_slope, -profile.slope_clip, profile.slope_clip))
            history.append(raw_slope)
            slope = float(np.median(history[-profile.slope_window :])) if profile.phase_slope == "slow" else 0.0
            ramp = np.exp(1j * (cpe + slope * (bins - center)))
            effective = h * ramp
            residual = np.abs(row[pidx] - effective[pidx] * pilots) ** 2
            current_noise = max(float(np.median(residual)), floor * 0.1)
            if profile.phase_slope == "slow":
                magnitude = (1.0 - profile.channel_alpha) * np.abs(h[pidx]) + profile.channel_alpha * np.abs(observed_h)
                h[pidx] = magnitude * np.exp(1j * np.angle(h[pidx]))
            else:
                observed_base = observed_h / ramp[pidx]
                h[pidx] = (1.0 - profile.channel_alpha) * h[pidx] + profile.channel_alpha * observed_base
            numerator = np.conj(effective[didx]) * row[didx]
            variance = np.maximum(base_var[didx], current_noise)
            decoded = np.clip(demap_llr(numerator, variance, profile.modulation), -24, 24)
            llrs.append(decoded)
            repeats = 1 if profile.modulation == "bpsk" else 2
            llr_bins.append(np.repeat(bins[didx], repeats))
            h_track.append(effective.copy())
            cpes.append(cpe)
            slopes.append(slope)
            raw_slopes.append(raw_slope)
            residuals.append(float(np.median(residual)))
            logical += 1
        physical += count
        if count < profile.timing_anchor_interval or physical + profile.timing_anchor_symbols > len(y):
            break
        known = random_qpsk(profile.timing_anchor_symbols, profile, profile.timing_anchor_seed + anchor)
        anchor_h = _anchor_channel(y[physical : physical + profile.timing_anchor_symbols], known, h)
        if profile.anchor_h_alpha:
            h = (1 - profile.anchor_h_alpha) * h + profile.anchor_h_alpha * anchor_h
        anchor_track.append(anchor_h)
        history.clear()
        physical += profile.timing_anchor_symbols
        anchor += 1
    return {
        "llr": np.concatenate(llrs) if llrs else np.empty(0),
        "bins": np.concatenate(llr_bins) if llr_bins else np.empty(0, int),
        "H": np.asarray(h_track), "anchor_H": np.asarray(anchor_track),
        "cpe": np.asarray(cpes), "phase_slope": np.asarray(slopes),
        "phase_slope_raw": np.asarray(raw_slopes), "pilot_residual": np.asarray(residuals),
        "logical_symbols": logical, "physical_symbols": physical, "anchors_consumed": anchor,
    }


def _save_plots(out: Path, profile: ModemProfile, h: np.ndarray, training_var: np.ndarray,
                silence_var: np.ndarray, clock: np.ndarray, intercept: float, scale: float,
                track: dict) -> None:
    freq = np.asarray(profile.active_bins) * profile.fs / profile.fft_size
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(freq, np.abs(h), marker=".")
    axes[0].set_ylabel("|H|")
    axes[1].semilogy(freq, np.maximum(training_var, 1e-15), label="training")
    axes[1].semilogy(freq, np.maximum(silence_var, 1e-15), label="silence")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "channel_and_noise.png", dpi=140); plt.close(fig)
    if len(clock):
        nominal, observed = clock[:, 0], clock[:, 1]
        used = clock[:, 4].astype(bool)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].scatter(nominal / profile.fs, observed - nominal, c=clock[:, 2])
        axes[0].plot(nominal / profile.fs, intercept + (scale - 1) * nominal, color="black")
        axes[1].scatter(nominal[used] / profile.fs, clock[used, 3], c=clock[used, 2])
        axes[1].set_xlabel("Nominal time (s)")
        axes[0].set_ylabel("Observed - nominal")
        axes[1].set_ylabel("Fit residual")
        fig.tight_layout(); fig.savefig(out / "clock_fit.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(np.unwrap(track["cpe"])); axes[0].set_ylabel("CPE")
    axes[1].plot(track["phase_slope_raw"], alpha=.4); axes[1].plot(track["phase_slope"])
    axes[1].set_ylabel("Slope")
    axes[2].semilogy(np.maximum(track["pilot_residual"], 1e-15)); axes[2].set_ylabel("Pilot residual")
    axes[2].set_xlabel("Logical symbol")
    fig.tight_layout(); fig.savefig(out / "phase_tracking.png", dpi=140); plt.close(fig)


def decode_file(receive: Path, out: Path, profile: ModemProfile, source: Path | None = None) -> dict:
    profile.validate()
    configure_legacy(profile)
    receive, out = Path(receive).resolve(), Path(out).resolve()
    source = Path(source).resolve() if source else None
    rx = read_wav(receive, profile)
    sync = random_qpsk(profile.sync_symbols, profile, profile.sync_seed)
    preambles = [random_qpsk(profile.preamble_symbols, profile, profile.preamble_seed + i)
                 for i in range(profile.preamble_repeats)]
    sync_template = ofdm_tx(sync[: profile.sync_correlation_symbols], profile)
    coarse, score = legacy.find_sync(rx, sync_template)
    emit("progress", stage="sync", progress=.1, sync_score=score)
    training = np.vstack([sync, *preambles])
    intercept, scale, clock_table, clock_status, initial = legacy.detect_clock_anchors(
        rx, training, coarse, profile.training_anchor_symbols, profile.training_anchor_step,
        profile.timing_anchor_interval, profile.timing_anchor_symbols, profile.timing_anchor_seed,
        profile.payload_start_anchor_symbols, profile.payload_start_anchor_seed,
        profile.clock_search, profile.anchor_min_score,
    )
    corrected = legacy.correct_clock(rx, intercept, scale)
    noise_end = max(0, round(intercept))
    noise_start = max(0, noise_end - round(profile.noise_seconds * profile.fs))
    silence_var = legacy.noise_variance(rx[noise_start:noise_end])
    sync_end = profile.sync_symbols * profile.symbol_len
    preamble_count = profile.preamble_symbols * profile.preamble_repeats
    preamble_end = sync_end + preamble_count * profile.symbol_len
    sync_y = ofdm_rx(corrected[:sync_end], profile)[:profile.sync_symbols]
    preamble_y = ofdm_rx(corrected[sync_end:preamble_end], profile)[:preamble_count]
    y_blocks, x_blocks = [sync_y], [sync]
    pos = 0
    for block in preambles:
        y_blocks.append(preamble_y[pos : pos + len(block)])
        x_blocks.append(block); pos += len(block)
    if any(len(y) != len(x) for y, x in zip(y_blocks, x_blocks)):
        raise ValueError("recording ends inside preamble")
    h, h_blocks, training_var = legacy.estimate_training(y_blocks, x_blocks)
    start_score = None
    start_status = "disabled"
    start_position = preamble_end
    start_h_track = np.empty((0, len(profile.active_bins)), complex)
    if profile.payload_start_anchor_symbols:
        known = random_qpsk(profile.payload_start_anchor_symbols, profile, profile.payload_start_anchor_seed)
        position, start_score = legacy.correlate_near(
            corrected, ofdm_tx(known, profile), preamble_end, profile.payload_search
        )
        start_position = round(position) if start_score >= profile.anchor_min_score else preamble_end
        start_status = "detected" if start_score >= profile.anchor_min_score else "structural_fallback"
        rows = ofdm_rx(corrected[start_position : start_position + len(known) * profile.symbol_len], profile)
        start_h = _anchor_channel(rows, known, h)
        h = (1 - profile.payload_start_anchor_h_alpha) * h + profile.payload_start_anchor_h_alpha * start_h
        start_h_track = start_h[None, :]
    payload_start = start_position + profile.payload_start_anchor_symbols * profile.symbol_len
    track = payload_llrs(ofdm_rx(corrected[payload_start:], profile), h, training_var, silence_var, profile)
    llr = track["llr"]
    emit("progress", stage="fec", progress=.72, llr_count=len(llr))
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("H.npy", h), ("H_training_blocks.npy", h_blocks),
        ("H_payload_start_anchor.npy", start_h_track), ("H_anchor_track.npy", track["anchor_H"]),
        ("training_noise.npy", training_var), ("silence_noise.npy", silence_var),
        ("rx_llr.npy", llr), ("pilot_H_track.npy", track["H"]),
        ("cpe.npy", track["cpe"]), ("phase_slope.npy", track["phase_slope"]),
        ("phase_slope_raw.npy", track["phase_slope_raw"]), ("clock_anchors.npy", clock_table),
    ):
        np.save(out / name, value)
    header_len = encoded_bit_length(HEADER_SIZE, profile.fec)
    need_header = profile.header_repeats * header_len
    header = None
    header_raw = b""
    chunks: list[bytes | None] = []
    rows: list[dict] = []
    error = ""
    consumed = 0
    post_errors = post_total = 0
    source_bytes = source.read_bytes() if source and source.exists() else None
    try:
        if len(llr) < need_header:
            raise ValueError("recording ends inside repeated header")
        combined = np.zeros(header_len)
        for repeat in range(profile.header_repeats):
            part = llr[repeat * header_len : (repeat + 1) * header_len]
            combined += legacy.deinterleave(part, profile.fec_seed + repeat)
        header_raw = fec_decode_values(combined, HEADER_SIZE, profile.fec)
        header = parse_header(header_raw, profile)
        consumed = need_header
        raw_block_bytes = header["block_size"] + 8
        block_bits = encoded_bit_length(raw_block_bytes, profile.fec)
        for index in range(header["block_count"]):
            part = llr[consumed : consumed + block_bits]
            expected_size = min(header["block_size"], header["file_size"] - index * header["block_size"])
            try:
                raw = fec_decode(part, raw_block_bytes, profile.fec, profile.fec_seed + 1000 + index)
                chunk = parse_data_block(raw, index, header["block_size"])
                chunks.append(chunk)
                rows.append({"block": index, "crc_ok": True, "bytes": len(chunk), "error": ""})
                if source_bytes is not None:
                    truth = source_bytes[index * header["block_size"] : index * header["block_size"] + expected_size]
                    xor = np.frombuffer(chunk[:expected_size], np.uint8) ^ np.frombuffer(truth, np.uint8)
                    post_errors += int(np.unpackbits(xor).sum()); post_total += expected_size * 8
            except Exception as exc:
                chunks.append(None)
                rows.append({"block": index, "crc_ok": False, "bytes": expected_size, "error": str(exc)})
            consumed += block_bits
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    (out / "decoded_header.bin").write_bytes(header_raw)
    with (out / "blocks.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["block", "crc_ok", "bytes", "error"])
        writer.writeheader(); writer.writerows(rows)
    file_crc_ok = file_match = False
    recovered = ""
    if header:
        body = b"".join(
            chunk if chunk is not None else b"\0" * min(header["block_size"], header["file_size"] - i * header["block_size"])
            for i, chunk in enumerate(chunks)
        )[:header["file_size"]]
        file_crc_ok = all(chunk is not None for chunk in chunks) and zlib.crc32(body) == header["file_crc32"]
        recovered = header["name"] if file_crc_ok else header["name"] + ".partial"
        (out / recovered).write_bytes(body)
        file_match = bool(source_bytes is not None and file_crc_ok and body == source_bytes)
    coded_errors = coded_total = 0
    coded_ber = None
    if source and source.exists():
        truth_frames, _, _ = encoded_frames(source, profile)
        truth = np.concatenate(truth_frames)
        count = min(len(llr), len(truth))
        coded_errors = int(np.count_nonzero((llr[:count] < 0).astype(np.uint8) != truth[:count])) + max(0, len(truth) - count)
        coded_total = len(truth)
        coded_ber = coded_errors / coded_total if coded_total else 0.0
    _save_plots(out, profile, h, training_var, silence_var, clock_table, intercept, scale, track)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["bin", "freq_hz", "abs_h", "training_noise", "silence_noise"])
        for i, bin_index in enumerate(profile.active_bins):
            writer.writerow([bin_index, bin_index * profile.fs / profile.fft_size, abs(h[i]), training_var[i], silence_var[i]])
    accepted = clock_table[:, 4].astype(bool) if len(clock_table) else np.zeros(0, bool)
    residual = clock_table[accepted, 3] if len(clock_table) else np.empty(0)
    metrics = {
        "input": str(receive), "out": str(out), "protocol": profile.protocol,
        "profile": profile.to_dict(), "profile_id": profile.profile_id,
        "sync_score": score, "clock_status": clock_status,
        "initial_clock_error_ppm": (initial["scale"] - 1) * 1e6,
        "clock_error_ppm": (scale - 1) * 1e6,
        "clock_anchor_candidates": len(clock_table), "clock_anchor_used": int(accepted.sum()),
        "clock_fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))) if len(residual) else None,
        "payload_start_anchor_score": start_score, "payload_start_anchor_status": start_status,
        "coded_bit_errors": coded_errors, "coded_bit_total": coded_total,
        "coded_bit_error_rate": coded_ber,
        "post_fec_bit_errors": post_errors, "post_fec_bit_total": post_total,
        "post_fec_bit_error_rate": post_errors / post_total if post_total else None,
        "header_ok": header is not None, "header": header,
        "blocks_ok": sum(bool(row["crc_ok"]) for row in rows), "blocks_total": len(rows),
        "file_crc_ok": file_crc_ok, "file_match": file_match, "recovered_name": recovered,
        "mean_abs_h": float(np.mean(np.abs(h))),
        "median_training_noise": float(np.median(training_var)),
        "median_silence_noise": float(np.median(silence_var)),
        "median_abs_llr": float(np.median(np.abs(llr[:consumed]))) if consumed else None,
        "median_pilot_residual": float(np.median(track["pilot_residual"])) if len(track["pilot_residual"]) else None,
        "phase_slope_mode": profile.phase_slope, "error": error,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    status = "exact" if file_match else "crc_ok" if file_crc_ok else "partial" if header else "failed"
    emit("result", status=status, metrics=str(out / "metrics.json"), file_match=file_match,
         blocks_ok=metrics["blocks_ok"], blocks_total=metrics["blocks_total"])
    return metrics
