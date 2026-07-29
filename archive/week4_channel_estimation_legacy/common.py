#!/usr/bin/env python3
"""Shared helpers for OFDM/QPSK channel estimation."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


FFT_SIZE = 1024
CP_LEN = 32
SYMBOL_LEN = FFT_SIZE + CP_LEN
SAMPLE_RATE = 48000
INT16_LIMIT = 32767
INT16_SCALE = 32768.0
QPSK_LEVEL = 1.0


def data_bins(bin_start: int, bin_end: int) -> np.ndarray:
    if bin_start < 1 or bin_end > FFT_SIZE // 2 - 1 or bin_start > bin_end:
        raise ValueError("bin range must satisfy 1 <= bin_start <= bin_end <= 511")
    return np.arange(bin_start, bin_end + 1)


def training_symbols(symbol_count: int, bins_count: int, seed: int) -> np.ndarray:
    if symbol_count <= 0:
        raise ValueError("symbols must be positive")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(symbol_count, bins_count, 2), dtype=np.uint8)
    imag = np.where(bits[:, :, 0] == 1, -QPSK_LEVEL, QPSK_LEVEL)
    real = np.where(bits[:, :, 1] == 1, -QPSK_LEVEL, QPSK_LEVEL)
    return real + 1j * imag


def ofdm_modulate(active_symbols: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if active_symbols.ndim != 2 or active_symbols.shape[1] != bins.size:
        raise ValueError("active symbol shape does not match selected bins")
    fft_frames = np.zeros((active_symbols.shape[0], FFT_SIZE), dtype=np.complex128)
    fft_frames[:, bins] = active_symbols
    fft_frames[:, FFT_SIZE - bins] = np.conj(active_symbols)
    time_symbols = np.fft.ifft(fft_frames, axis=1).real
    with_cp = np.concatenate((time_symbols[:, -CP_LEN:], time_symbols), axis=1)
    return with_cp.reshape(-1)


def write_wav(samples: np.ndarray, output: Path) -> None:
    scale = wav_output_scale(samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples * scale * INT16_SCALE, -INT16_LIMIT, INT16_LIMIT)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.astype("<i2").tobytes())


def wav_output_scale(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        raise ValueError("empty signal")
    return 0.95 * INT16_LIMIT / INT16_SCALE / peak


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
            raise ValueError(f"unsupported WAV format: {path}")
        if wav_file.getnchannels() != 1:
            raise ValueError(f"expected mono WAV: {path}")
        if wav_file.getframerate() != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz WAV: {path}")
        raw = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / INT16_SCALE


def make_probe_samples(bin_start: int, bin_end: int, symbol_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = data_bins(bin_start, bin_end)
    active_symbols = training_symbols(symbol_count, bins.size, seed)
    samples = ofdm_modulate(active_symbols, bins)
    return samples, active_symbols, bins
