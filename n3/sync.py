from __future__ import annotations

import math

import numpy as np
from scipy import signal

from n3.modem import (
    ACTIVE_BINS,
    CHIRP_SAMPLES,
    CP,
    L,
    N,
    SILENCE_SAMPLES,
    TRAINING_SYMBOLS,
    ofdm_rx,
)


def normalized_correlation(samples: np.ndarray, template: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float).ravel()
    reference = np.asarray(template, dtype=float).ravel()
    if len(values) < len(reference):
        return np.empty(0, dtype=float)
    numerator = signal.fftconvolve(values, reference[::-1], mode="valid")
    energy = signal.fftconvolve(values * values, np.ones(len(reference)), mode="valid")
    denominator = np.sqrt(np.maximum(energy, 1e-18) * np.sum(reference * reference))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def chirp_candidates(
    samples: np.ndarray,
    chirp: np.ndarray,
    max_peaks: int = 96,
    min_spacing: int | None = None,
) -> list[tuple[int, float]]:
    score = normalized_correlation(samples, chirp)
    if score.size == 0:
        return []
    distance = int(min_spacing or max(1, len(chirp) // 2))
    peaks, _ = signal.find_peaks(score, distance=distance)
    edges = []
    if score[0] >= (score[1] if len(score) > 1 else score[0]):
        edges.append(0)
    if len(score) > 1 and score[-1] >= score[-2]:
        edges.append(len(score) - 1)
    peaks = np.unique(np.r_[peaks, edges]).astype(int)
    order = np.argsort(score[peaks])[::-1][:max_peaks]
    return [(int(peaks[index]), float(np.clip(score[peaks[index]], -1.0, 1.0))) for index in order]


def refine_training_start(
    samples: np.ndarray,
    expected_training_start: int,
    training_wave: np.ndarray,
    radius: int = CP,
) -> dict:
    score = normalized_correlation(samples, training_wave)
    start = max(0, int(expected_training_start) - int(radius))
    stop = min(len(score), int(expected_training_start) + int(radius) + 1)
    if stop <= start:
        raise ValueError("recording does not contain the training search window")
    local = score[start:stop]
    peak = start + int(np.argmax(local))
    return {
        "training_start": int(peak),
        "training_score": float(score[peak]),
        "ofdm_frame_start": float(peak - CHIRP_SAMPLES - SILENCE_SAMPLES),
        "timing_offset": float(peak - expected_training_start),
        "confident": bool(score[peak] >= 0.5),
    }


def frame_candidates(
    samples: np.ndarray,
    chirp: np.ndarray,
    training_wave: np.ndarray,
    max_peaks: int = 96,
) -> list[dict]:
    values = np.asarray(samples, dtype=float).ravel()
    peaks = chirp_candidates(values, chirp, max_peaks=max_peaks)
    if len(peaks) < 2:
        return []
    training_offset = CHIRP_SAMPLES + SILENCE_SAMPLES
    fixed_distance = CHIRP_SAMPLES + 2 * SILENCE_SAMPLES + (2 * TRAINING_SYMBOLS) * L
    candidates = []
    for front_start, front_score in peaks:
        for tail_start, tail_score in peaks:
            if tail_start <= front_start:
                continue
            distance = tail_start - front_start
            if distance < fixed_distance:
                continue
            data_float = (distance - fixed_distance) / L
            data_symbols = int(round(data_float / 2.0) * 2)
            if data_symbols < 2:
                continue
            expected_distance = fixed_distance + data_symbols * L
            grid_error = abs(distance - expected_distance)
            if grid_error > CP:
                continue
            expected_training = int(round(front_start + training_offset))
            try:
                timing = refine_training_start(values, expected_training, training_wave)
            except ValueError:
                continue
            d_tx = float(expected_distance)
            epsilon = (float(distance) - d_tx) / d_tx
            candidates.append(
                {
                    "front_start": int(front_start),
                    "tail_start": int(tail_start),
                    "front_score": float(front_score),
                    "tail_score": float(tail_score),
                    "data_symbols": data_symbols,
                    "distance_samples": int(distance),
                    "grid_error": float(grid_error),
                    "coarse_sfo_ppm": float(epsilon * 1e6),
                    "epsilon": float(epsilon),
                    "timing": timing,
                    "training_start": int(timing["training_start"]),
                    "training_score": float(timing["training_score"]),
                    "timing_offset": float(timing["timing_offset"]),
                    "timing_confident": bool(timing["confident"]),
                }
            )
    candidates.sort(key=lambda item: (-item["training_score"], -item["front_score"] - item["tail_score"], item["grid_error"]))
    return candidates


def phase_correct(
    symbols: np.ndarray,
    symbol_starts: np.ndarray,
    reference_start: float,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(symbols, dtype=complex)
    starts = np.asarray(symbol_starts, dtype=float).ravel()
    if len(values) != len(starts):
        raise ValueError("one start is required for each OFDM symbol")
    drift = float(epsilon) * (starts - float(reference_start))
    ramp = np.exp(2j * np.pi * drift[:, None] * ACTIVE_BINS[None, :] / N)
    return values * ramp


def estimate_channel(received: np.ndarray, known: np.ndarray) -> np.ndarray:
    rx = np.asarray(received, dtype=complex)
    ref = np.asarray(known, dtype=complex)
    rows = min(len(rx), len(ref))
    if rows == 0:
        raise ValueError("no training symbols available for channel estimate")
    denominator = np.sum(np.abs(ref[:rows]) ** 2, axis=0)
    numerator = np.sum(np.conj(ref[:rows]) * rx[:rows], axis=0)
    return np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-12)


def equalize(received: np.ndarray, channel: np.ndarray) -> np.ndarray:
    values = np.asarray(received, dtype=complex)
    h = np.asarray(channel, dtype=complex)
    return values / np.where(np.abs(h) > 1e-12, h, 1.0)


def training_from_recording(
    samples: np.ndarray,
    front_start: float,
    training_freq: np.ndarray,
    epsilon: float = 0.0,
    data_symbols: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    start = int(round(front_start + CHIRP_SAMPLES + SILENCE_SAMPLES))
    front = ofdm_rx(values[start:start + TRAINING_SYMBOLS * L])
    starts = start + np.arange(len(front)) * L
    front = phase_correct(front, starts, front_start, epsilon)
    known_front = training_freq[:TRAINING_SYMBOLS, ACTIVE_BINS]
    if data_symbols is None:
        return front, known_front, starts
    tail_offset = CHIRP_SAMPLES + SILENCE_SAMPLES + TRAINING_SYMBOLS * L + data_symbols * L
    tail_start = int(round(front_start + tail_offset))
    tail = ofdm_rx(values[tail_start:tail_start + TRAINING_SYMBOLS * L])
    tail_starts = tail_start + np.arange(len(tail)) * L
    tail = phase_correct(tail, tail_starts, front_start, epsilon)
    known_tail = training_freq[TRAINING_SYMBOLS:TRAINING_SYMBOLS + len(tail), ACTIVE_BINS]
    if len(tail):
        return np.vstack((front, tail)), np.vstack((known_front, known_tail)), np.r_[starts, tail_starts]
    return front, known_front, starts
