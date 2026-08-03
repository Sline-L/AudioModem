from pathlib import Path
import argparse
import json

import numpy as np

from step7_modem import (
    ACTIVE_BINS,
    BLOCK_SIZE,
    FEC_SEED,
    FS,
    HEADER_REPEATS,
    L,
    PILOT_SEED,
    coded_frames,
    ofdm_tx,
    payload_symbols,
    profile_meta,
    training_symbols,
    wav_gain,
    write_wav,
)


def args():
    p = argparse.ArgumentParser(description="make a Step7 adaptive-pilot, soft-FEC file WAV")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/step6_newexp/file.jpeg"))
    p.add_argument("--noise-seconds", type=float, default=0.5)
    p.add_argument("--sync-symbols", type=int, default=64)
    p.add_argument("--sync-seed", type=int, default=7026)
    p.add_argument("--preamble-symbols", type=int, default=128)
    p.add_argument("--preamble-repeats", type=int, default=2)
    p.add_argument("--preamble-seed", type=int, default=8026)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument("--header-repeats", type=int, default=HEADER_REPEATS)
    p.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    p.add_argument("--fec-seed", type=int, default=FEC_SEED)
    p.add_argument("--tail-seconds", type=float, default=0.25)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/step7_adaptive_fec/file_combo_step7_adaptive_fec.wav"),
    )
    return p.parse_args()


def validate(a):
    if not a.input.exists():
        raise SystemExit(f"input file does not exist: {a.input}")
    for name in ("sync_symbols", "preamble_symbols", "preamble_repeats", "block_size", "header_repeats"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if a.block_size > 65535:
        raise SystemExit("--block-size must be <= 65535")
    if a.noise_seconds < 0 or a.tail_seconds < 0:
        raise SystemExit("silence durations must be non-negative")


def main():
    a = args()
    validate(a)

    sync = training_symbols(a.sync_symbols, a.sync_seed)
    preamble_blocks = [
        training_symbols(a.preamble_symbols, a.preamble_seed + i) for i in range(a.preamble_repeats)
    ]
    frames, labels = coded_frames(a.input, a.block_size, a.header_repeats, a.fec_seed)
    coded = np.concatenate(frames)
    payload, used = payload_symbols(coded, a.pilot_seed)

    noise_samples = int(round(a.noise_seconds * FS))
    tail_samples = int(round(a.tail_seconds * FS))
    sync_wave = ofdm_tx(sync)
    preamble_wave = ofdm_tx(np.vstack(preamble_blocks))
    payload_wave = ofdm_tx(payload)
    raw = np.r_[np.zeros(noise_samples), sync_wave, preamble_wave, payload_wave, np.zeros(tail_samples)]
    gain = wav_gain(raw)
    write_wav(a.out, raw)

    np.save(a.out.with_suffix(".sync.npy"), sync)
    np.save(a.out.with_suffix(".preamble.npy"), np.vstack(preamble_blocks))
    frame_lengths = [int(len(frame)) for frame in frames]
    meta = {
        "input": str(a.input),
        "out": str(a.out),
        "input_bytes": int(a.input.stat().st_size),
        "noise_seconds": float(a.noise_seconds),
        "noise_samples": noise_samples,
        "sync_symbols": int(a.sync_symbols),
        "sync_seed": int(a.sync_seed),
        "preamble_symbols_per_repeat": int(a.preamble_symbols),
        "preamble_repeats": int(a.preamble_repeats),
        "preamble_seed": int(a.preamble_seed),
        "payload_symbols": int(len(payload)),
        "payload_coded_bits": int(len(coded)),
        "payload_data_slots": int(used.sum()),
        "frame_labels": labels,
        "frame_coded_bits": frame_lengths,
        "block_size": int(a.block_size),
        "header_repeats": int(a.header_repeats),
        "pilot_seed": int(a.pilot_seed),
        "fec_seed": int(a.fec_seed),
        "gain": float(gain),
        "payload_start_sample": int(noise_samples + len(sync_wave) + len(preamble_wave)),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(profile_meta())
    meta["block_size"] = int(a.block_size)
    meta["header_repeats"] = int(a.header_repeats)
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"wrote {a.out} seconds={len(raw) / FS:.3f} N=512 CP=256 "
        f"active_bins={len(ACTIVE_BINS)} payload_symbols={len(payload)}"
    )
    print(
        f"structure=noise {a.noise_seconds:.3f}s + sync {len(sync)} + "
        f"preamble {a.preamble_symbols}x{a.preamble_repeats} + coded payload x1"
    )
    print(f"coded_frames={len(frames)} coded_bits={len(coded)} block_size={a.block_size}")
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
