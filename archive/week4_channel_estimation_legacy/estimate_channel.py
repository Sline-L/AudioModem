#!/usr/bin/env python3
"""Estimate OFDM channel bins from a recorded probe WAV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import signal

from common import CP_LEN, FFT_SIZE, SAMPLE_RATE, SYMBOL_LEN, make_probe_samples, read_wav, wav_output_scale


def f01_parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Estimate frequency-domain channel bins from a recorded probe.")
    parser.add_argument("recorded", type=Path)
    parser.add_argument("--bin-start", type=int, default=8)
    parser.add_argument("--bin-end", type=int, default=420)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-prefix", type=Path, default=here / "estimated_channel")
    return parser.parse_args()


def t01_find_probe_start(recorded: np.ndarray, probe: np.ndarray) -> tuple[int, float]:
    if recorded.size < probe.size:
        raise ValueError("recorded WAV is shorter than the expected probe")
    correlation = signal.correlate(recorded, probe, mode="valid", method="fft")
    start = int(np.argmax(np.abs(correlation)))
    peak = float(np.abs(correlation[start]))
    energy = float(np.linalg.norm(recorded[start : start + probe.size]) * np.linalg.norm(probe))
    score = peak / energy if energy > 0.0 else 0.0
    return start, score


def t02_estimate_h(recorded_probe: np.ndarray, known_symbols: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, float]:
    symbols = recorded_probe[: known_symbols.shape[0] * SYMBOL_LEN].reshape(known_symbols.shape[0], SYMBOL_LEN)
    no_cp = symbols[:, CP_LEN:]
    received_bins = np.fft.fft(no_cp, n=FFT_SIZE, axis=1)[:, bins]
    per_symbol_h = received_bins / known_symbols
    h = np.mean(per_symbol_h, axis=0)
    variance = float(np.mean(np.abs(per_symbol_h - h) ** 2))
    return h, variance


def t03_write_csv(path: Path, bins: np.ndarray, h: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "frequency_hz", "real", "imag", "abs", "phase"])
        for bin_index, value in zip(bins, h):
            frequency = bin_index * SAMPLE_RATE / FFT_SIZE
            writer.writerow(
                [
                    int(bin_index),
                    f"{frequency:.6f}",
                    f"{value.real:.12g}",
                    f"{value.imag:.12g}",
                    f"{abs(value):.12g}",
                    f"{np.angle(value):.12g}",
                ]
            )


def f02_run() -> None:
    args = f01_parse_args()
    try:
        probe, known_symbols, bins = make_probe_samples(args.bin_start, args.bin_end, args.symbols, args.seed)
        known_symbols = known_symbols * wav_output_scale(probe)
        recorded = read_wav(args.recorded)
        start, score = t01_find_probe_start(recorded, probe)
        recorded_probe = recorded[start : start + probe.size]
        h, variance = t02_estimate_h(recorded_probe, known_symbols, bins)

        npy_path = args.out_prefix.with_suffix(".npy")
        csv_path = args.out_prefix.with_suffix(".csv")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, h)
        t03_write_csv(csv_path, bins, h)

        magnitudes = np.abs(h)
        print(f"sync_start={start}")
        print(f"sync_score={score:.6f}")
        print(f"min_abs_h={float(np.min(magnitudes)):.6g}")
        print(f"mean_abs_h={float(np.mean(magnitudes)):.6g}")
        print(f"max_abs_h={float(np.max(magnitudes)):.6g}")
        print(f"mean_symbol_h_variance={variance:.6g}")
        print(f"wrote={npy_path}")
        print(f"wrote={csv_path}")
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    f02_run()
