from pathlib import Path
import argparse
import csv
import json
import zlib

import numpy as np
from scipy import signal

from modem_n1 import (
    ACTIVE_BINS,
    CHIRP_GUARD_SAMPLES,
    CHIRP_SAMPLES,
    CP,
    FS,
    HEADER_BITS,
    HEADER_MOD,
    HEADER_SIZE,
    HEADER_SYMBOLS,
    HEADER_COPIES,
    HEADER_COPY_SYMBOLS,
    HEADER_PERM_SEEDS,
    INTER_FRAME_GAP_SAMPLES,
    L,
    N,
    TAIL_TRAINING_SYMBOLS,
    TRAINING_SYMBOLS,
    bits_from_bytes,
    bits_from_mod,
    bytes_from_mod,
    bytes_from_bits,
    chirp_wave,
    decode_payload,
    equalize,
    estimate_channel,
    find_chirp,
    frame_sample_counts,
    ofdm_rx,
    ofdm_tx,
    parse_header,
    phase_correct,
    profile_meta,
    read_wav,
    training_phase_fit,
    training_symbols,
    header_bytes,
    header_copy_bytes,
)


CHIRP_MAX_PEAKS = 96
PAIR_SHORTLIST = 8
FFT_TIMING_PATH_RATIO = 0.5
FFT_TIMING_MARGIN = max(8, CP // 64)
GLOBAL_PPM_LIMIT = 80.0
GLOBAL_PPM_STEP = 5.0
LOCAL_PPM_RADIUS = 5.0
LOCAL_PPM_STEP = 0.25


def args():
    p = argparse.ArgumentParser(description="recover an N1 chirp/training/header/payload/tail-chirp recording")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--training-seed", type=int, default=3026)
    p.add_argument(
        "--tail-training",
        action="store_true",
        help="decode frames that include 6 training OFDM symbols after payload before the gap",
    )
    p.add_argument("--tail-search-seconds", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("runs/n1"))
    return p.parse_args()


def validate(a):
    if a.tail_search_seconds < 0:
        raise SystemExit("--tail-search-seconds must be >= 0")


def output_dir(base, receive, many):
    if not many:
        return base
    suffix = receive.stem.rsplit("_", 1)[-1]
    return base / (suffix if suffix.isdigit() else receive.stem)


def save_summary(path, h, training_error):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "freq_hz", "abs_h", "training_error"])
        for index, bin_index in enumerate(ACTIVE_BINS):
            writer.writerow(
                [
                    int(bin_index),
                    float(bin_index * FS / N),
                    float(abs(h[index])),
                    float(training_error[index]),
                ]
            )


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def symbol_block(rx, front_start, offset, rows):
    start = int(round(front_start + offset))
    return ofdm_rx(rx[start : start + rows * L])[:rows], start


def block_starts(front_start, offset, rows):
    start = float(front_start + offset)
    return start + np.arange(rows) * L


def training_blocks(rx, front_start, training, epsilon, tail_training=False, payload_rows=None):
    reference = float(front_start)
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    header_offset = training_offset + TRAINING_SYMBOLS * L

    front_y, _ = symbol_block(rx, front_start, training_offset, TRAINING_SYMBOLS)
    front_y = phase_correct(
        front_y,
        block_starts(front_start, training_offset, len(front_y)),
        reference,
        epsilon,
    )

    blocks = [front_y]
    known = [training[: len(front_y)]]
    if tail_training and payload_rows is not None:
        tail_offset = header_offset + HEADER_SYMBOLS * L + int(payload_rows) * L
        tail_y, _ = symbol_block(rx, front_start, tail_offset, TAIL_TRAINING_SYMBOLS)
        if len(tail_y):
            tail_y = phase_correct(
                tail_y,
                block_starts(front_start, tail_offset, len(tail_y)),
                reference,
                epsilon,
            )
            blocks.append(tail_y)
            known.append(training[: len(tail_y)])
    return np.vstack(blocks), np.vstack(known)


def decode_header_pass(rx, front_start, training, epsilon=0.0, tail_training=False, payload_rows=None):
    reference = float(front_start)
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    header_offset = training_offset + TRAINING_SYMBOLS * L

    train_y, known_training = training_blocks(
        rx,
        front_start,
        training,
        epsilon,
        tail_training=tail_training,
        payload_rows=payload_rows,
    )
    h = estimate_channel(train_y, known_training)
    training_error = np.mean(np.abs(train_y - known_training[: len(train_y)] * h) ** 2, axis=0)
    training_nrmse = float(
        np.mean(training_error) / max(float(np.mean(np.abs(h) ** 2)), 1e-12)
    )
    phase_fit = training_phase_fit(train_y, known_training, h)

    header_y, header_start = symbol_block(rx, front_start, header_offset, HEADER_SYMBOLS)
    header_y = phase_correct(
        header_y,
        block_starts(front_start, header_offset, len(header_y)),
        reference,
        epsilon,
    )
    header_eq = equalize(header_y, h)
    header_scale = max(float(np.median(np.abs(header_eq.real))), 1e-12)
    header_normalized = header_eq / header_scale
    header_soft_error = float(
        np.mean(header_normalized.imag ** 2 + (np.abs(header_normalized.real) - 1.0) ** 2)
    )
    raw_copies, bit_copies, copy_crc_ok = header_copy_bytes(header_eq)
    votes = np.sum(bit_copies, axis=0)
    voted_bits = (votes >= 2).astype(np.uint8)
    header_raw = bytes_from_bits(voted_bits)[:HEADER_SIZE]
    diagnostics = {
        "copy_crc_ok": copy_crc_ok,
        "bit_disagreements": int(np.count_nonzero((votes != 0) & (votes != HEADER_COPIES))),
        "training_nrmse": training_nrmse,
        "channel_training_symbols_used": int(len(train_y)),
        "tail_training_used_for_h": bool(tail_training and payload_rows is not None),
        "header_soft_error": header_soft_error,
        "raw_copies": raw_copies,
    }
    try:
        header = parse_header(header_raw)
    except Exception as exc:
        exc.header_raw = header_raw
        exc.header_diag = diagnostics
        exc.h = h
        exc.training_error = training_error
        exc.phase_fit = phase_fit
        raise
    return header, header_raw, h, training_error, phase_fit, header_start, diagnostics


def find_tail(rx, chirp, front_start, d_tx, search_seconds):
    expected = float(front_start + d_tx)
    radius = int(round(search_seconds * FS))
    start = max(0, int(round(expected)) - radius)
    end = min(len(rx), int(round(expected)) + radius + len(chirp))
    try:
        return find_chirp(rx, chirp, start, end)
    except ValueError:
        return find_chirp(rx, chirp, int(round(front_start + CHIRP_SAMPLES)), None)


def normalized_correlation(rx, template):
    if len(rx) < len(template):
        return np.empty(0, dtype=float)
    corr = signal.correlate(rx, template, mode="valid", method="fft")
    energy = signal.fftconvolve(rx * rx, np.ones(len(template)), mode="valid")
    denom = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(template)
    return np.divide(np.abs(corr), denom, out=np.zeros_like(denom), where=denom > 0)


def chirp_candidates(rx, chirp, max_peaks=CHIRP_MAX_PEAKS, min_spacing=None):
    if min_spacing is None:
        min_spacing = max(64, CP // 8)
    score = normalized_correlation(rx, chirp)
    if not score.size:
        return []
    peaks, _ = signal.find_peaks(score, distance=min_spacing)
    edge = []
    if len(score) == 1 or score[0] >= score[1]:
        edge.append(0)
    if len(score) > 1 and score[-1] >= score[-2]:
        edge.append(len(score) - 1)
    peaks = np.unique(np.r_[peaks, edge]).astype(int)
    if not peaks.size:
        peaks = np.array([int(np.argmax(score))])
    order = peaks[np.argsort(score[peaks])[::-1][:max_peaks]]
    return sorted(
        [(int(index), float(np.clip(score[index], 0.0, 1.0))) for index in order],
        key=lambda item: item[0],
    )


def refine_fft_timing(training_score, chirp_front_start):
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    expected = int(round(chirp_front_start + training_offset))
    start = max(0, expected - CP)
    stop = min(len(training_score), expected + CP + 1)
    local = training_score[start:stop]
    fallback = {
        "ofdm_frame_start": float(chirp_front_start),
        "training_start": float(expected),
        "timing_offset": 0.0,
        "peak_start": None,
        "peak_score": 0.0,
        "confident": False,
        "significant_path_count": 0,
    }
    if not local.size:
        return fallback

    peak_local = int(np.argmax(local))
    peak_start = start + peak_local
    peak_score = float(local[peak_local])
    median = float(np.median(local))
    mad = float(np.median(np.abs(local - median)))
    confidence_floor = median + 6.0 * max(mad, 1e-12)
    path_floor = max(FFT_TIMING_PATH_RATIO * peak_score, confidence_floor)
    paths, _ = signal.find_peaks(
        local,
        height=path_floor,
        distance=max(8, CP // 128),
    )
    if not paths.size:
        paths = np.array([peak_local])
    chosen_training_start = float(start + int(paths[0]) - FFT_TIMING_MARGIN)
    ofdm_frame_start = chosen_training_start - training_offset
    return {
        "ofdm_frame_start": float(ofdm_frame_start),
        "training_start": float(chosen_training_start),
        "timing_offset": float(ofdm_frame_start - chirp_front_start),
        "peak_start": float(peak_start),
        "peak_score": peak_score,
        "confident": bool(peak_score >= confidence_floor),
        "significant_path_count": int(len(paths)),
    }


def find_chirp_pairs(rx, chirp, training, tail_training=False):
    training_wave = ofdm_tx(training)
    training_score = normalized_correlation(rx, training_wave)
    chirp_score = normalized_correlation(rx, chirp)
    min_spacing = max(64, CP // 8)
    all_chirp_peaks, _ = signal.find_peaks(chirp_score, distance=min_spacing)
    edge_peaks = []
    if chirp_score.size == 1 or chirp_score[0] >= chirp_score[1]:
        edge_peaks.append(0)
    if chirp_score.size > 1 and chirp_score[-1] >= chirp_score[-2]:
        edge_peaks.append(len(chirp_score) - 1)
    if edge_peaks:
        all_chirp_peaks = np.unique(np.r_[all_chirp_peaks, edge_peaks]).astype(int)
    if not all_chirp_peaks.size and chirp_score.size:
        all_chirp_peaks = np.array([int(np.argmax(chirp_score))])
    top_chirp_peaks = all_chirp_peaks[
        np.argsort(chirp_score[all_chirp_peaks])[::-1][:CHIRP_MAX_PEAKS]
    ]

    training_peaks, _ = signal.find_peaks(training_score, distance=max(L, len(training_wave) // 2))
    if not training_peaks.size and training_score.size:
        training_peaks = np.array([int(np.argmax(training_score))])
    top_training_peaks = training_peaks[
        np.argsort(training_score[training_peaks])[::-1][:8]
    ]
    anchored_fronts = set()
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    for training_peak in top_training_peaks:
        expected_front = int(training_peak - training_offset)
        nearby = all_chirp_peaks[
            (all_chirp_peaks >= expected_front - CP)
            & (all_chirp_peaks <= expected_front + CP)
        ]
        nearby = nearby[np.argsort(chirp_score[nearby])[::-1][:8]]
        anchored_fronts.update(int(index) for index in nearby)

    candidate_indices = list(
        set(int(index) for index in top_chirp_peaks) | anchored_fronts
    )
    candidate_indices.sort(
        key=lambda index: (
            index not in anchored_fronts,
            -float(chirp_score[index]),
            index,
        )
    )
    candidate_indices = sorted(candidate_indices[:CHIRP_MAX_PEAKS])
    candidates = [
        (index, float(np.clip(chirp_score[index], 0.0, 1.0)))
        for index in candidate_indices
    ]
    fixed_part = fixed_part_samples(tail_training=tail_training)
    timing_cache = {}
    pairs = []
    seen_pairs = set()

    def grid_tail_indices(front_start):
        available = len(rx) - front_start - fixed_part - CHIRP_SAMPLES
        max_payload = max(0, int(np.floor(available / L)))
        indices = set()
        for payload_symbols in range(max_payload + 1):
            d_tx = float(frame_sample_counts(payload_symbols, tail_training=tail_training)["D_tx"])
            expected = int(round(front_start + d_tx))
            radius = max(1, min(500, int(np.floor(d_tx * 300e-6))))
            start = max(1, expected - radius)
            stop = min(len(chirp_score) - 1, expected + radius + 1)
            if stop <= start:
                continue
            local = chirp_score[start:stop]
            peaks, _ = signal.find_peaks(local, distance=8)
            if not peaks.size:
                peaks = np.array([int(np.argmax(local))])
            strongest = peaks[np.argsort(local[peaks])[::-1][:2]]
            indices.update(int(start + index) for index in strongest)
        return np.asarray(sorted(indices), dtype=int)

    for front_start, front_score in candidates:
        tail_indices = np.unique(
            np.r_[top_chirp_peaks, grid_tail_indices(front_start)]
        )
        for tail_start in tail_indices:
            tail_start = int(tail_start)
            if tail_start <= front_start or (front_start, tail_start) in seen_pairs:
                continue
            seen_pairs.add((front_start, tail_start))
            tail_score = float(np.clip(chirp_score[tail_start], 0.0, 1.0))
            d_rx = float(tail_start - front_start)
            payload_symbols = int(round((d_rx - fixed_part) / L))
            if payload_symbols < 0:
                continue
            counts = frame_sample_counts(payload_symbols, tail_training=tail_training)
            d_tx = float(counts["D_tx"])
            grid_error = abs(d_rx - d_tx)
            epsilon = (d_rx - d_tx) / d_tx if d_tx else 0.0
            if grid_error > 500:
                continue
            if abs(epsilon * 1e6) > 300:
                continue
            if front_start + d_tx + CHIRP_SAMPLES > len(rx):
                continue
            if front_start not in timing_cache:
                timing_cache[front_start] = refine_fft_timing(training_score, front_start)
            candidate = {
                "front_start": float(front_start),
                "front_score": float(front_score),
                "tail_start": float(tail_start),
                "tail_score": float(tail_score),
                "payload_symbols": int(payload_symbols),
                "counts": counts,
                "D_rx": float(d_rx),
                "D_tx": float(d_tx),
                "epsilon": float(epsilon),
                "grid_error": float(grid_error),
                "timing": timing_cache[front_start],
            }
            pairs.append(candidate)
    pairs.sort(
        key=lambda item: (
            not item["timing"]["confident"],
            -item["timing"]["peak_score"],
            -(item["front_score"] + item["tail_score"]),
            item["grid_error"],
            item["front_start"],
        )
    )
    return pairs, candidates, training_score


def find_chirp_pair(rx, chirp, training=None, tail_training=False):
    if training is not None:
        pairs, candidates, _ = find_chirp_pairs(rx, chirp, training, tail_training=tail_training)
        if pairs:
            result = dict(pairs[0])
            result["candidates"] = candidates
            return result
    else:
        candidates = chirp_candidates(rx, chirp)
    front_start, front_score = find_chirp(rx, chirp)
    return {
        "front_start": float(front_start),
        "front_score": float(front_score),
        "tail_start": None,
        "tail_score": 0.0,
        "payload_symbols": None,
        "counts": None,
        "D_rx": None,
        "D_tx": None,
        "epsilon": 0.0,
        "grid_error": None,
        "candidates": candidates,
        "timing": {
            "ofdm_frame_start": float(front_start),
            "training_start": float(front_start + CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES),
            "timing_offset": 0.0,
            "peak_start": None,
            "peak_score": 0.0,
            "confident": False,
            "significant_path_count": 0,
        },
    }


def evaluate_header_epsilon(rx, front_start, training, ppm, tail_training=False, payload_rows=None):
    epsilon = float(ppm) * 1e-6
    try:
        result = decode_header_pass(
            rx,
            front_start,
            training,
            epsilon,
            tail_training=tail_training,
            payload_rows=payload_rows,
        )
        header, header_raw, h, training_error, phase_fit, header_start, header_diag = result
        return {
            "ppm": float(ppm),
            "epsilon": epsilon,
            "header_ok": True,
            "header": header,
            "header_raw": header_raw,
            "h": h,
            "training_error": training_error,
            "phase_fit": phase_fit,
            "header_start": header_start,
            "header_diag": header_diag,
            "relaxed_header": header,
        }
    except Exception as exc:
        header_diag = getattr(exc, "header_diag", {})
        header_raw = getattr(exc, "header_raw", b"")
        relaxed_header = None
        if len(header_raw) >= HEADER_SIZE:
            first = header_raw[: HEADER_SIZE - 4]
            repaired = first + zlib.crc32(first).to_bytes(4, "big")
            try:
                relaxed_header = parse_header(repaired)
            except Exception:
                pass
        return {
            "ppm": float(ppm),
            "epsilon": epsilon,
            "header_ok": False,
            "header": None,
            "header_raw": header_raw,
            "h": getattr(exc, "h", np.zeros(len(ACTIVE_BINS), complex)),
            "training_error": getattr(exc, "training_error", np.full(len(ACTIVE_BINS), np.nan)),
            "phase_fit": getattr(exc, "phase_fit", np.empty((0, 4))),
            "header_start": None,
            "header_diag": {
                "copy_crc_ok": header_diag.get("copy_crc_ok", []),
                "bit_disagreements": header_diag.get("bit_disagreements"),
                "training_nrmse": header_diag.get("training_nrmse", float("inf")),
                "channel_training_symbols_used": header_diag.get("channel_training_symbols_used", 0),
                "tail_training_used_for_h": header_diag.get("tail_training_used_for_h", False),
                "header_soft_error": header_diag.get("header_soft_error", float("inf")),
                "raw_copies": header_diag.get("raw_copies", []),
            },
            "relaxed_header": relaxed_header,
            "error": f"{type(exc).__name__}: {exc}",
        }


def metric_ranks(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.inf)
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    unique, inverse = np.unique(values, return_inverse=True)
    if len(unique) == 1:
        return np.zeros(len(values), dtype=float)
    return inverse.astype(float) / (len(unique) - 1)


def score_epsilon_attempts(attempts, center_ppm):
    disagreements = [
        attempt["header_diag"].get("bit_disagreements")
        if attempt["header_diag"].get("bit_disagreements") is not None
        else HEADER_COPIES * HEADER_BITS
        for attempt in attempts
    ]
    training_rank = metric_ranks(
        [attempt["header_diag"].get("training_nrmse", float("inf")) for attempt in attempts]
    )
    soft_rank = metric_ranks(
        [attempt["header_diag"].get("header_soft_error", float("inf")) for attempt in attempts]
    )
    disagreement_rank = metric_ranks(disagreements)
    for index, attempt in enumerate(attempts):
        attempt["joint_score"] = float(
            0.5 * training_rank[index] + 0.3 * soft_rank[index] + 0.2 * disagreement_rank[index]
        )
        attempt["copy_crc_count"] = int(
            sum(bool(value) for value in attempt["header_diag"].get("copy_crc_ok", []))
        )
    return min(
        attempts,
        key=lambda attempt: (
            not attempt["header_ok"],
            -attempt["copy_crc_count"],
            attempt["joint_score"],
            abs(attempt["ppm"] - center_ppm),
        ),
    )


def ppm_attempt_metrics(attempts):
    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    return [
        {
            "ppm": float(attempt["ppm"]),
            "joint_score": float(attempt["joint_score"]),
            "header_ok": bool(attempt["header_ok"]),
            "copy_crc_ok": [
                bool(value) for value in attempt["header_diag"].get("copy_crc_ok", [])
            ],
            "bit_disagreements": attempt["header_diag"].get("bit_disagreements"),
            "training_nrmse": finite_or_none(
                attempt["header_diag"].get("training_nrmse", float("inf"))
            ),
            "channel_training_symbols_used": int(
                attempt["header_diag"].get("channel_training_symbols_used", 0)
            ),
            "tail_training_used_for_h": bool(
                attempt["header_diag"].get("tail_training_used_for_h", False)
            ),
            "header_soft_error": finite_or_none(
                attempt["header_diag"].get("header_soft_error", float("inf"))
            ),
        }
        for attempt in attempts
    ]


def search_header_epsilon(
    rx,
    front_start,
    training,
    ppm_values,
    center_ppm,
    tail_training=False,
    payload_rows=None,
):
    attempts = []
    for ppm in ppm_values:
        attempts.append(
            evaluate_header_epsilon(
                rx,
                front_start,
                training,
                float(ppm),
                tail_training=tail_training,
                payload_rows=payload_rows,
            )
        )
    best = score_epsilon_attempts(attempts, center_ppm)
    return best, ppm_attempt_metrics(attempts)


def select_epsilon(rx, front_start, training, coarse_ppm, tail_training=False, payload_rows=None):
    global_scores = []
    local_steps = int(round(LOCAL_PPM_RADIUS / LOCAL_PPM_STEP))
    if abs(coarse_ppm) <= GLOBAL_PPM_LIMIT:
        center = float(coarse_ppm)
        source = "chirp_local_search"
        initial_values = center + np.arange(
            -local_steps, local_steps + 1, dtype=float
        ) * LOCAL_PPM_STEP
        initial, initial_scores = search_header_epsilon(
            rx,
            front_start,
            training,
            initial_values,
            center,
            tail_training=tail_training,
            payload_rows=payload_rows,
        )
        if initial["header_ok"]:
            return initial, {
                "source": source,
                "coarse_ppm": float(coarse_ppm),
                "center_ppm": center,
                "min_ppm": float(initial_values[0]),
                "max_ppm": float(initial_values[-1]),
                "step_ppm": LOCAL_PPM_STEP,
                "global_scores": global_scores,
                "local_scores": initial_scores,
            }
        source = "chirp_local_then_global"
    else:
        source = "global_then_local"

    global_values = np.arange(
        -GLOBAL_PPM_LIMIT,
        GLOBAL_PPM_LIMIT + GLOBAL_PPM_STEP / 2.0,
        GLOBAL_PPM_STEP,
    )
    coarse, global_scores = search_header_epsilon(
        rx,
        front_start,
        training,
        global_values,
        0.0,
        tail_training=tail_training,
        payload_rows=payload_rows,
    )
    center = float(coarse["ppm"])

    local_values = center + np.arange(
        -local_steps, local_steps + 1, dtype=float
    ) * LOCAL_PPM_STEP
    local_values = local_values[
        (local_values >= -GLOBAL_PPM_LIMIT) & (local_values <= GLOBAL_PPM_LIMIT)
    ]
    best, local_scores = search_header_epsilon(
        rx,
        front_start,
        training,
        local_values,
        center,
        tail_training=tail_training,
        payload_rows=payload_rows,
    )
    return best, {
        "source": source,
        "coarse_ppm": float(coarse_ppm),
        "center_ppm": center,
        "min_ppm": float(local_values[0]),
        "max_ppm": float(local_values[-1]),
        "step_ppm": LOCAL_PPM_STEP,
        "global_scores": global_scores,
        "local_scores": local_scores,
    }


def refine_tail_from_epsilon(rx, chirp, front_start, d_tx, epsilon, radius_samples=20):
    if d_tx is None:
        return None, 0.0, None
    expected = float(front_start + d_tx * (1.0 + epsilon))
    start = max(0, int(round(expected - radius_samples)))
    end = min(len(rx), int(round(expected + radius_samples + len(chirp))))
    try:
        tail_start, tail_score = find_chirp(rx, chirp, start, end)
    except ValueError:
        return None, 0.0, None
    refined_epsilon = ((tail_start - front_start) - d_tx) / d_tx if d_tx else None
    return tail_start, tail_score, refined_epsilon


def fixed_part_samples(tail_training=False):
    return (
        CHIRP_SAMPLES
        + CHIRP_GUARD_SAMPLES
        + TRAINING_SYMBOLS * L
        + HEADER_SYMBOLS * L
        + (TAIL_TRAINING_SYMBOLS * L if tail_training else 0)
        + INTER_FRAME_GAP_SAMPLES
    )


def estimate_payload_from_tail(rx, chirp, front_start, tail_training=False):
    fixed_part = fixed_part_samples(tail_training=tail_training)
    search_start = int(round(front_start + fixed_part - L // 2))
    tail_start, tail_score = find_chirp(rx, chirp, search_start, None)
    d_rx = float(tail_start - front_start)
    payload_symbols = max(0, int(round((d_rx - fixed_part) / L)))
    counts = frame_sample_counts(payload_symbols, tail_training=tail_training)
    d_tx = float(counts["D_tx"])
    epsilon = (d_rx - d_tx) / d_tx if d_tx else 0.0
    return payload_symbols, counts, tail_start, tail_score, d_rx, d_tx, epsilon


def hard_bits(symbols, mod):
    return bits_from_mod(symbols, mod)


def bit_error_count(received, truth):
    count = min(len(received), len(truth))
    if count == 0:
        return 0, 0, None
    errors = int(np.count_nonzero(received[:count] != truth[:count]))
    return errors, int(count), float(errors / count)


def byte_error_count(received, truth):
    count = min(len(received), len(truth))
    if count == 0:
        return 0, 0, None
    received_bytes = np.frombuffer(received[:count], dtype=np.uint8)
    truth_bytes = np.frombuffer(truth[:count], dtype=np.uint8)
    errors = int(np.count_nonzero(received_bytes != truth_bytes))
    return errors, int(count), float(errors / count)


def truth_header_from_source(source, payload_rows, mod="qpsk"):
    source_bytes = source.read_bytes()
    payload_mod_symbols = int(np.ceil(len(source_bytes) * 8 / {
        "bpsk": 1,
        "qpsk": 2,
        "qam16": 4,
    }[mod]))
    return header_bytes(source, mod, payload_rows, payload_mod_symbols)


def ber_diagnostics(rx, front_start, training, epsilon, source, payload_rows, header, h=None, tail_training=False):
    if source is None or not source.exists() or payload_rows is None:
        return {}

    source_bytes = source.read_bytes()
    mod = header["mod"] if header else "qpsk"
    payload_rows = int(payload_rows)
    payload_mod_symbols = int(np.ceil(len(source_bytes) * 8 / {"bpsk": 1, "qpsk": 2, "qam16": 4}[mod]))

    reference = float(front_start)
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    header_offset = training_offset + TRAINING_SYMBOLS * L
    payload_offset = header_offset + HEADER_SYMBOLS * L

    train_y, known_training = training_blocks(
        rx,
        front_start,
        training,
        epsilon,
        tail_training=tail_training,
        payload_rows=payload_rows,
    )
    if h is not None and h.size and np.any(np.abs(h) > 1e-12):
        diag_h = h
    else:
        diag_h = estimate_channel(train_y, known_training)

    header_y, _ = symbol_block(rx, front_start, header_offset, HEADER_SYMBOLS)
    header_y = phase_correct(
        header_y,
        block_starts(front_start, header_offset, len(header_y)),
        reference,
        epsilon,
    )
    header_eq = equalize(header_y, diag_h)
    _, header_bit_copies, header_copy_crc_ok = header_copy_bytes(header_eq)
    votes = np.sum(header_bit_copies, axis=0)
    voted_header_bits = (votes >= 2).astype(np.uint8)
    truth_header_bits = bits_from_bytes(truth_header_from_source(source, payload_rows, mod))[:HEADER_BITS]
    header_vote_errors, header_vote_bits, header_vote_ber = bit_error_count(voted_header_bits, truth_header_bits)
    header_copy_errors = [bit_error_count(bits, truth_header_bits)[0] for bits in header_bit_copies]
    header_copy_bers = [errors / HEADER_BITS for errors in header_copy_errors]

    payload_y, _ = symbol_block(rx, front_start, payload_offset, payload_rows)
    payload_y = phase_correct(
        payload_y,
        block_starts(front_start, payload_offset, len(payload_y)),
        reference,
        epsilon,
    )
    payload_eq = equalize(payload_y, diag_h).ravel()[:payload_mod_symbols]
    payload_bits = hard_bits(payload_eq, mod)[: len(source_bytes) * 8]
    truth_payload_bits = bits_from_bytes(source_bytes)
    payload_errors, payload_bits_count, payload_ber = bit_error_count(payload_bits, truth_payload_bits)
    decoded_payload = bytes_from_bits(payload_bits)[: len(source_bytes)]
    payload_byte_errors, payload_bytes_count, payload_byte_error_rate = byte_error_count(decoded_payload, source_bytes)

    overall_errors = header_vote_errors + payload_errors
    overall_bits = header_vote_bits + payload_bits_count
    return {
        "ber_available": True,
        "ber_source": str(source),
        "ber_mod_assumed": mod,
        "ber_payload_symbols_used": int(payload_rows),
        "ber_payload_mod_symbols_used": int(payload_mod_symbols),
        "ber_header_copy_crc_ok": [bool(value) for value in header_copy_crc_ok],
        "ber_header_copy_bit_errors": [int(value) for value in header_copy_errors],
        "ber_header_copy_ber": [float(value) for value in header_copy_bers],
        "ber_header_vote_bit_errors": int(header_vote_errors),
        "ber_header_vote_bits": int(header_vote_bits),
        "ber_header_vote_ber": header_vote_ber,
        "ber_header_bit_disagreements": int(np.count_nonzero((votes != 0) & (votes != HEADER_COPIES))),
        "ber_payload_bit_errors": int(payload_errors),
        "ber_payload_bits": int(payload_bits_count),
        "ber_payload_ber": payload_ber,
        "ber_payload_byte_errors": int(payload_byte_errors),
        "ber_payload_bytes": int(payload_bytes_count),
        "ber_payload_byte_error_rate": payload_byte_error_rate,
        "ber_overall_useful_bit_errors": int(overall_errors),
        "ber_overall_useful_bits": int(overall_bits),
        "ber_overall_useful_ber": float(overall_errors / overall_bits) if overall_bits else None,
        "ber_payload_crc32": int(zlib.crc32(decoded_payload)),
        "ber_truth_crc32": int(zlib.crc32(source_bytes)),
    }


def evaluate_sync_pair(rx, pair, training, tail_training=False):
    timing = pair["timing"]
    selected, search = select_epsilon(
        rx,
        timing["ofdm_frame_start"],
        training,
        pair["epsilon"] * 1e6,
        tail_training=tail_training,
        payload_rows=pair["payload_symbols"],
    )
    header_hint = selected["header"] or selected.get("relaxed_header")
    header_match = bool(
        header_hint
        and header_hint["payload_symbols"] == pair["payload_symbols"]
    )
    selected_at_boundary = bool(
        np.isclose(selected["ppm"], search["min_ppm"])
        or np.isclose(selected["ppm"], search["max_ppm"])
    )
    return {
        "pair": pair,
        "selected": selected,
        "search": search,
        "header_payload_match": header_match,
        "selected_at_boundary": selected_at_boundary,
    }


def sync_pair_rank(result):
    pair = result["pair"]
    selected = result["selected"]
    return (
        not selected["header_ok"],
        not result["header_payload_match"],
        -selected["copy_crc_count"],
        result["selected_at_boundary"],
        selected["joint_score"],
        -pair["timing"]["peak_score"],
        -(pair["front_score"] + pair["tail_score"]),
        pair["grid_error"],
    )


def sync_pair_summary(result):
    pair = result["pair"]
    selected = result["selected"]
    return {
        "front_start": float(pair["front_start"]),
        "tail_start": float(pair["tail_start"]),
        "payload_symbols": int(pair["payload_symbols"]),
        "front_score": float(pair["front_score"]),
        "tail_score": float(pair["tail_score"]),
        "grid_error_samples": float(pair["grid_error"]),
        "coarse_sfo_ppm": float(pair["epsilon"] * 1e6),
        "ofdm_frame_start": float(pair["timing"]["ofdm_frame_start"]),
        "fft_timing_peak_score": float(pair["timing"]["peak_score"]),
        "fft_timing_confident": bool(pair["timing"]["confident"]),
        "selected_sfo_ppm": float(selected["ppm"]),
        "header_ok": bool(selected["header_ok"]),
        "header_payload_match": bool(result["header_payload_match"]),
        "selected_at_local_boundary": bool(result["selected_at_boundary"]),
        "copy_crc_count": int(selected["copy_crc_count"]),
        "joint_score": float(selected["joint_score"]),
    }


def run_one(receive, out, a, training, chirp):
    rx = read_wav(receive)
    header = None
    header_raw = b""
    h = np.zeros(len(ACTIVE_BINS), complex)
    training_error = np.full(len(ACTIVE_BINS), np.nan)
    phase_fit = np.empty((0, 4))
    chirp_front_start = 0.0
    front_start = 0.0
    front_score = 0.0
    tail_start = None
    tail_score = 0.0
    d_tx = None
    d_rx = None
    chirp_epsilon = 0.0
    epsilon = 0.0
    payload_symbols_guess = None
    payload_symbols_from_tail = None
    payload_symbols_header_match = None
    sync_mode = "sync_fallback"
    tail_search_mode = sync_mode
    sync_grid_error = None
    sync_candidates = []
    sync_pairs_evaluated = 0
    sync_pair_shortlist = []
    epsilon_search_scores = []
    global_epsilon_search_scores = []
    selected_sfo_source = "local_search"
    local_ppm_center = 0.0
    local_ppm_min = 0.0
    local_ppm_max = 0.0
    local_ppm_step = LOCAL_PPM_STEP
    refined_tail_start = None
    refined_tail_score = 0.0
    refined_chirp_epsilon = None
    timing = {
        "ofdm_frame_start": 0.0,
        "training_start": float(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES),
        "timing_offset": 0.0,
        "peak_start": None,
        "peak_score": 0.0,
        "confident": False,
        "significant_path_count": 0,
    }
    fixed_part = fixed_part_samples(tail_training=a.tail_training)
    header_ok = False
    header_diag = {
        "copy_crc_ok": [],
        "bit_disagreements": None,
        "raw_copies": [],
    }
    error = ""

    try:
        pairs, sync_candidates, training_score = find_chirp_pairs(
            rx, chirp, training, tail_training=a.tail_training
        )
        sync_pairs_evaluated = len(pairs)
        evaluated = [
            evaluate_sync_pair(rx, pair, training, tail_training=a.tail_training)
            for pair in pairs[:PAIR_SHORTLIST]
        ]
        sync_pair_shortlist = [sync_pair_summary(result) for result in evaluated]

        if evaluated:
            chosen = min(evaluated, key=sync_pair_rank)
            header_hint = chosen["selected"]["header"] or chosen["selected"].get(
                "relaxed_header"
            )
            if header_hint:
                chosen_timing = chosen["pair"]["timing"]["ofdm_frame_start"]
                matching_pairs = [
                    pair
                    for pair in pairs
                    if abs(pair["timing"]["ofdm_frame_start"] - chosen_timing) <= 0.5
                    and pair["payload_symbols"] == header_hint["payload_symbols"]
                ]
                if matching_pairs:
                    original_pair = chosen["pair"]
                    redirected = min(
                        matching_pairs,
                        key=lambda pair: (
                            -(pair["front_score"] + pair["tail_score"]),
                            pair["grid_error"],
                        ),
                    )
                    chosen = dict(chosen)
                    chosen["pair"] = redirected
                    chosen["header_payload_match"] = True
                    if redirected is not original_pair:
                        redirected_selected, redirected_search = select_epsilon(
                            rx,
                            redirected["timing"]["ofdm_frame_start"],
                            training,
                            redirected["epsilon"] * 1e6,
                            tail_training=a.tail_training,
                            payload_rows=redirected["payload_symbols"],
                        )
                        chosen["selected"] = redirected_selected
                        chosen["search"] = redirected_search
            sync = chosen["pair"]
            selected = chosen["selected"]
            search = chosen["search"]
            sync_mode = "pair_training_joint"
            tail_search_mode = sync_mode
        else:
            sync = find_chirp_pair(rx, chirp, tail_training=a.tail_training)
            timing = refine_fft_timing(training_score, sync["front_start"])
            (
                payload_symbols_guess,
                counts,
                tail_start,
                tail_score,
                d_rx,
                d_tx,
                chirp_epsilon,
            ) = estimate_payload_from_tail(
                rx, chirp, sync["front_start"], tail_training=a.tail_training
            )
            sync.update(
                {
                    "tail_start": tail_start,
                    "tail_score": tail_score,
                    "payload_symbols": payload_symbols_guess,
                    "counts": counts,
                    "D_rx": d_rx,
                    "D_tx": d_tx,
                    "epsilon": chirp_epsilon,
                    "grid_error": abs(d_rx - d_tx),
                    "timing": timing,
                }
            )
            selected, search = select_epsilon(
                rx,
                timing["ofdm_frame_start"],
                training,
                chirp_epsilon * 1e6,
                tail_training=a.tail_training,
                payload_rows=payload_symbols_guess,
            )
            payload_symbols_from_tail = payload_symbols_guess
            tail_search_mode = "sync_fallback"

        chirp_front_start = float(sync["front_start"])
        front_start = float(sync["timing"]["ofdm_frame_start"])
        timing = sync["timing"]
        front_score = float(sync["front_score"])
        tail_start = sync["tail_start"]
        tail_score = float(sync["tail_score"])
        payload_symbols_guess = sync["payload_symbols"]
        payload_symbols_from_tail = payload_symbols_guess
        d_tx = sync["D_tx"]
        d_rx = sync["D_rx"]
        chirp_epsilon = float(sync["epsilon"])
        sync_grid_error = sync["grid_error"]

        epsilon = float(selected["epsilon"])
        selected_sfo_source = search["source"]
        epsilon_search_scores = search["local_scores"]
        global_epsilon_search_scores = search["global_scores"]
        local_ppm_center = search["center_ppm"]
        local_ppm_min = search["min_ppm"]
        local_ppm_max = search["max_ppm"]
        local_ppm_step = search["step_ppm"]
        header = selected["header"]
        header_raw = selected["header_raw"]
        h = selected["h"]
        training_error = selected["training_error"]
        phase_fit = selected["phase_fit"]
        header_diag = selected["header_diag"]
        header_ok = bool(selected["header_ok"])
        error = "" if header_ok else selected.get("error", "ValueError: header CRC mismatch")
        if header_ok:
            payload_symbols_header_match = payload_symbols_guess == header["payload_symbols"]
            counts = frame_sample_counts(header["payload_symbols"], tail_training=a.tail_training)
            d_tx = float(counts["D_tx"])
        refined_tail_start, refined_tail_score, refined_chirp_epsilon = refine_tail_from_epsilon(
            rx, chirp, chirp_front_start, d_tx, epsilon
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    out.mkdir(parents=True, exist_ok=True)
    (out / "decoded_header.bin").write_bytes(header_raw)
    np.save(out / "H.npy", h)
    np.save(out / "training_phase_fit.npy", phase_fit)
    save_summary(out / "summary.csv", h, training_error)

    recovered_name = ""
    recovered_bytes = b""
    file_crc_ok = False
    file_match = False
    payload_count = 0
    payload_eq = np.empty(0, complex)

    if header_ok and not error:
        try:
            payload_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + (TRAINING_SYMBOLS + HEADER_SYMBOLS) * L
            payload_y, _ = symbol_block(rx, front_start, payload_offset, header["payload_symbols"])
            payload_y = phase_correct(
                payload_y,
                block_starts(front_start, payload_offset, len(payload_y)),
                float(front_start),
                epsilon,
            )
            payload_eq = equalize(payload_y, h).ravel()[: header["payload_mod_symbols"]]
            recovered_bytes = decode_payload(payload_eq, header)
            recovered_name = header["name"]
            (out / recovered_name).write_bytes(recovered_bytes)
            file_crc_ok = True
            if a.source and a.source.exists():
                file_match = recovered_bytes == a.source.read_bytes()
            payload_count = int(len(payload_y))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw = bytes_from_mod(payload_eq, header["mod"]) if header else b""
            (out / "decoded_payload.bin").write_bytes(raw)
    else:
        (out / "decoded_payload.bin").write_bytes(b"")

    ber_payload_rows = header["payload_symbols"] if header else payload_symbols_guess
    ber_metrics = {}
    try:
        ber_metrics = ber_diagnostics(
            rx,
            front_start,
            training,
            epsilon,
            a.source,
            ber_payload_rows,
            header,
            h,
            tail_training=a.tail_training,
        )
    except Exception as exc:
        ber_metrics = {
            "ber_available": False,
            "ber_error": f"{type(exc).__name__}: {exc}",
        }

    np.save(out / "payload_symbols.npy", payload_eq)
    metrics = {
        "input": str(receive),
        "out": str(out),
        "front_chirp_start": float(chirp_front_start),
        "chirp_front_start": float(chirp_front_start),
        "ofdm_frame_start": float(front_start),
        "front_chirp_score": float(front_score),
        "tail_chirp_start": float(tail_start) if tail_start is not None else None,
        "tail_chirp_score": float(tail_score),
        "D_tx": d_tx,
        "D_rx": d_rx,
        "chirp_sfo_epsilon": float(chirp_epsilon),
        "chirp_sfo_ppm": float(chirp_epsilon * 1e6),
        "sfo_epsilon": float(epsilon),
        "sfo_ppm": float(epsilon * 1e6),
        "selected_sfo_epsilon": float(epsilon),
        "selected_sfo_ppm": float(epsilon * 1e6),
        "selected_sfo_source": selected_sfo_source,
        "coarse_sfo_ppm": float(chirp_epsilon * 1e6),
        "local_ppm_center": float(local_ppm_center),
        "local_ppm_min": float(local_ppm_min),
        "local_ppm_max": float(local_ppm_max),
        "local_ppm_step": float(local_ppm_step),
        "local_ppm_scores": epsilon_search_scores,
        "global_ppm_scores": global_epsilon_search_scores,
        "epsilon_search_ppm": float(epsilon * 1e6),
        "epsilon_search_scores": epsilon_search_scores,
        "refined_tail_chirp_start": float(refined_tail_start) if refined_tail_start is not None else None,
        "refined_tail_chirp_score": float(refined_tail_score),
        "refined_chirp_sfo_epsilon": float(refined_chirp_epsilon) if refined_chirp_epsilon is not None else None,
        "refined_chirp_sfo_ppm": float(refined_chirp_epsilon * 1e6) if refined_chirp_epsilon is not None else None,
        "fixed_part_samples": int(fixed_part),
        "payload_symbols_guess": int(payload_symbols_guess) if payload_symbols_guess is not None else None,
        "payload_symbols_from_tail": int(payload_symbols_from_tail) if payload_symbols_from_tail is not None else None,
        "payload_symbols_header_match": payload_symbols_header_match,
        "tail_training_enabled": bool(a.tail_training),
        "tail_training_symbols": int(TAIL_TRAINING_SYMBOLS if a.tail_training else 0),
        "channel_training_symbols_used": int(
            header_diag.get("channel_training_symbols_used", 0)
        ),
        "tail_training_used_for_h": bool(
            header_diag.get("tail_training_used_for_h", False)
        ),
        "tail_search_mode": tail_search_mode,
        "sync_mode": sync_mode,
        "sync_grid_error_samples": float(sync_grid_error) if sync_grid_error is not None else None,
        "sync_front_score": float(front_score),
        "sync_tail_score": float(tail_score),
        "sync_pair_candidates_evaluated": int(sync_pairs_evaluated),
        "sync_pair_shortlist": sync_pair_shortlist,
        "sync_candidates": [
            {"start": int(start), "score": float(score)}
            for start, score in sync_candidates
        ],
        "fft_training_start": float(timing["training_start"]),
        "fft_timing_offset_samples": float(timing["timing_offset"]),
        "fft_timing_peak_start": timing["peak_start"],
        "fft_timing_peak_score": float(timing["peak_score"]),
        "fft_timing_confident": bool(timing["confident"]),
        "fft_significant_path_count": int(timing["significant_path_count"]),
        "training_start": int(round(timing["training_start"])),
        "payload_ofdm_symbols_received": int(payload_count),
        "header_ok": bool(header_ok),
        "header": header,
        "header_copies": int(HEADER_COPIES),
        "header_copy_symbols": int(HEADER_COPY_SYMBOLS),
        "header_size": int(HEADER_SIZE),
        "header_copy_crc_ok": [bool(value) for value in header_diag["copy_crc_ok"]],
        "header_bit_disagreements": header_diag["bit_disagreements"],
        "header_vote_used": True,
        "header_permutation_used": True,
        "header_permutation_seeds": [int(seed) for seed in HEADER_PERM_SEEDS],
        "file_crc_ok": bool(file_crc_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "recovered_bytes": int(len(recovered_bytes)),
        "mean_abs_h": float(np.mean(np.abs(h))) if h.size else None,
        "median_training_error": finite_median(training_error),
        "error": error,
    }
    metrics.update(ber_metrics)
    metrics.update(profile_meta(tail_training=a.tail_training))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"{receive}: front={front_score:.6f} tail={tail_score:.6f} "
        f"sfo_ppm={epsilon * 1e6:+.4f} header_ok={metrics['header_ok']} "
        f"file_crc_ok={file_crc_ok} file_match={file_match}"
    )
    if metrics.get("ber_available"):
        print(
            f"ber_header_vote={metrics['ber_header_vote_ber']:.6f} "
            f"ber_payload={metrics['ber_payload_ber']:.6f} "
            f"ber_overall={metrics['ber_overall_useful_ber']:.6f}"
        )
    if error:
        print(f"decode_error={error}")
    return metrics


def main():
    a = args()
    validate(a)
    training = training_symbols(a.training_seed)
    chirp = chirp_wave()
    many = len(a.input) > 1
    results = []
    for receive in a.input:
        results.append(run_one(receive, output_dir(a.out, receive, many), a, training, chirp))
    if many:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = [
            "input",
            "front_chirp_score",
            "tail_chirp_score",
            "sfo_ppm",
            "header_ok",
            "file_crc_ok",
            "file_match",
            "error",
        ]
        with (a.out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {a.out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
