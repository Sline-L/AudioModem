#!/usr/bin/env python3
"""Generate a known OFDM/QPSK probe WAV for channel estimation."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import SAMPLE_RATE, make_probe_samples, write_wav


def f01_parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate an OFDM/QPSK channel-estimation probe WAV.")
    parser.add_argument("--bin-start", type=int, default=8)
    parser.add_argument("--bin-end", type=int, default=420)
    parser.add_argument("--symbols", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=here / "probe.wav")
    return parser.parse_args()


def f02_run() -> None:
    args = f01_parse_args()
    try:
        samples, _active_symbols, bins = make_probe_samples(args.bin_start, args.bin_end, args.symbols, args.seed)
        write_wav(samples, args.output)
        seconds = samples.size / SAMPLE_RATE
        print(
            f"Wrote {args.output} with {args.symbols} symbol(s), "
            f"bins {bins[0]}..{bins[-1]}, {seconds:.3f} s"
        )
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    f02_run()
