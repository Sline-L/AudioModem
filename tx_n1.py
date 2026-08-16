from pathlib import Path
import argparse
import json

import numpy as np

from modem_n1 import (
    ACTIVE_BINS,
    FS,
    L,
    MODS,
    bits_per_symbol,
    file_symbols,
    ofdm_tx,
    profile_meta,
    random_qpsk,
    wav_gain,
    write_wav,
)


def args():
    p = argparse.ArgumentParser(description="make a n1 sync + preamble + data WAV")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/source/file16_test.txt"))
    p.add_argument("--noise-seconds", type=float, default=0.5)
    p.add_argument("--sync-symbols", type=int, default=16)
    p.add_argument("--sync-seed", type=int, default=2026)
    p.add_argument("--preamble-symbols", type=int, default=64)
    p.add_argument("--preamble-seed", type=int, default=3026)
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--tail-seconds", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("data/n1/n1.wav"))
    return p.parse_args()


def validate(a):
    if not a.input.exists():
        raise SystemExit(f"input file does not exist: {a.input}")
    for name in ("sync_symbols", "preamble_symbols"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if a.noise_seconds < 0 or a.tail_seconds < 0:
        raise SystemExit("silence durations must be non-negative")


def main():
    a = args()
    validate(a)

    sync = random_qpsk(a.sync_symbols, seed=a.sync_seed)
    preamble = random_qpsk(a.preamble_symbols, seed=a.preamble_seed)
    payload, payload_data_symbols = file_symbols(a.input, a.mod)

    noise_samples = int(round(a.noise_seconds * FS))
    tail_samples = int(round(a.tail_seconds * FS))
    sync_wave = ofdm_tx(sync)
    preamble_wave = ofdm_tx(preamble)
    payload_wave = ofdm_tx(payload)
    raw = np.r_[np.zeros(noise_samples), sync_wave, preamble_wave, payload_wave, np.zeros(tail_samples)]
    gain = wav_gain(raw)
    write_wav(a.out, raw)

    np.save(a.out.with_suffix(".sync.npy"), sync)
    np.save(a.out.with_suffix(".preamble.npy"), preamble)
    meta = {
        "input": str(a.input),
        "out": str(a.out),
        "input_bytes": int(a.input.stat().st_size),
        "noise_seconds": float(a.noise_seconds),
        "noise_samples": int(noise_samples),
        "sync_symbols": int(a.sync_symbols),
        "sync_seed": int(a.sync_seed),
        "preamble_symbols": int(a.preamble_symbols),
        "preamble_seed": int(a.preamble_seed),
        "mod": a.mod,
        "bits_per_symbol": int(bits_per_symbol(a.mod)),
        "payload_data_symbols": int(payload_data_symbols),
        "payload_ofdm_symbols": int(len(payload)),
        "tail_seconds": float(a.tail_seconds),
        "tail_samples": int(tail_samples),
        "gain": float(gain),
        "sync_start_sample": int(noise_samples),
        "preamble_start_sample": int(noise_samples + len(sync_wave)),
        "payload_start_sample": int(noise_samples + len(sync_wave) + len(preamble_wave)),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(profile_meta())
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"wrote {a.out} seconds={len(raw) / FS:.3f} mod={a.mod} "
        f"active_bins={len(ACTIVE_BINS)} symbol_len={L}"
    )
    print(
        f"structure=silence {a.noise_seconds:.3f}s + sync {len(sync)} + "
        f"preamble {len(preamble)} + data {len(payload)} + tail {a.tail_seconds:.3f}s"
    )
    print(f"band={meta['bin_start_hz']:.1f}-{meta['bin_end_hz']:.1f}Hz bins={meta['bin_start']}-{meta['bin_end']}")
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
