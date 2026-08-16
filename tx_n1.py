from pathlib import Path
import argparse
import json

import numpy as np

from modem_n1 import (
    ACTIVE_BINS,
    CHIRP_GUARD_SAMPLES,
    CHIRP_SAMPLES,
    FS,
    HEADER_SYMBOLS,
    INTER_FRAME_GAP_SAMPLES,
    L,
    MODS,
    TRAINING_SYMBOLS,
    bits_per_symbol,
    chirp_wave,
    frame_sample_counts,
    header_symbols,
    ofdm_tx,
    payload_symbols,
    profile_meta,
    training_symbols,
    wav_gain,
    write_wav,
)


def args():
    p = argparse.ArgumentParser(description="make an N1 chirp/training/header/payload/tail-chirp WAV")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/source/file16_test.txt"))
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--training-seed", type=int, default=3026)
    p.add_argument("--out", type=Path, default=Path("data/n1/n1.wav"))
    return p.parse_args()


def validate(a):
    if not a.input.exists():
        raise SystemExit(f"input file does not exist: {a.input}")


def main():
    a = args()
    validate(a)

    chirp = chirp_wave()
    guard = np.zeros(CHIRP_GUARD_SAMPLES)
    gap = np.zeros(INTER_FRAME_GAP_SAMPLES)
    training = training_symbols(a.training_seed)
    payload, payload_mod_symbols = payload_symbols(a.input, a.mod)
    header = header_symbols(a.input, a.mod, len(payload), payload_mod_symbols)

    training_wave = ofdm_tx(training)
    header_wave = ofdm_tx(header)
    payload_wave = ofdm_tx(payload)
    raw = np.r_[chirp, guard, training_wave, header_wave, payload_wave, gap, chirp]
    gain = wav_gain(raw)
    write_wav(a.out, raw)

    counts = frame_sample_counts(len(payload))
    np.save(a.out.with_suffix(".training.npy"), training)
    np.save(a.out.with_suffix(".header.npy"), header)
    meta = {
        "input": str(a.input),
        "out": str(a.out),
        "input_bytes": int(a.input.stat().st_size),
        "mod": a.mod,
        "bits_per_symbol": int(bits_per_symbol(a.mod)),
        "payload_mod_symbols": int(payload_mod_symbols),
        "payload_symbols": int(len(payload)),
        "gain": float(gain),
        "front_chirp_start_sample": 0,
        "guard_start_sample": int(CHIRP_SAMPLES),
        "training_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES),
        "header_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + TRAINING_SYMBOLS * L),
        "payload_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + (TRAINING_SYMBOLS + HEADER_SYMBOLS) * L),
        "gap_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + (TRAINING_SYMBOLS + HEADER_SYMBOLS + len(payload)) * L),
        "tail_chirp_start_sample": int(counts["tail_chirp_start"]),
        "D_tx": int(counts["D_tx"]),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(profile_meta())
    meta.update(counts)
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"wrote {a.out} seconds={len(raw) / FS:.3f} mod={a.mod} "
        f"active_bins={len(ACTIVE_BINS)} symbol_len={L}"
    )
    print(
        f"structure=chirp {CHIRP_SAMPLES} + guard {CHIRP_GUARD_SAMPLES} + "
        f"training {TRAINING_SYMBOLS} + header {HEADER_SYMBOLS} + "
        f"payload {len(payload)} + gap {INTER_FRAME_GAP_SAMPLES} + chirp {CHIRP_SAMPLES}"
    )
    print(f"D_tx={counts['D_tx']} tail_chirp_start={counts['tail_chirp_start']}")
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
