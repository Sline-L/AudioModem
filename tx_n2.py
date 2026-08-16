from pathlib import Path
import argparse
import json

import numpy as np

from modem_n2 import (
    ACTIVE_BINS,
    CHIRP_GUARD_SAMPLES,
    CHIRP_SAMPLES,
    FS,
    HEADER_SYMBOLS,
    INTER_FRAME_GAP_SAMPLES,
    L,
    MODS,
    TAIL_TRAINING_SYMBOLS,
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
    p = argparse.ArgumentParser(description="make an N2 chirp/training/header/payload/tail-chirp WAV")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/source/file16_test.txt"))
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--training-seed", type=int, default=3026)
    p.add_argument(
        "--tail-training",
        action="store_true",
        help="append the same known training block after payload before the gap",
    )
    p.add_argument("--ldpc", action="store_true", help="LDPC-encode payload bits before modulation")
    p.add_argument("--out", type=Path, default=Path("data/n2/n2.wav"))
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
    payload, payload_mod_symbols, ldpc_payload_meta = payload_symbols(
        a.input,
        a.mod,
        ldpc_enabled=a.ldpc,
    )
    header = header_symbols(
        a.input,
        a.mod,
        len(payload),
        payload_mod_symbols,
        ldpc_enabled=a.ldpc,
    )

    training_wave = ofdm_tx(training)
    header_wave = ofdm_tx(header)
    payload_wave = ofdm_tx(payload)
    tail_training_wave = training_wave if a.tail_training else np.empty(0, dtype=float)
    raw = np.r_[chirp, guard, training_wave, header_wave, payload_wave, tail_training_wave, gap, chirp]
    gain = wav_gain(raw)
    write_wav(a.out, raw)

    counts = frame_sample_counts(len(payload), tail_training=a.tail_training)
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
        "ldpc_enabled": bool(a.ldpc),
        "gain": float(gain),
        "front_chirp_start_sample": 0,
        "guard_start_sample": int(CHIRP_SAMPLES),
        "training_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES),
        "header_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + TRAINING_SYMBOLS * L),
        "payload_start_sample": int(CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + (TRAINING_SYMBOLS + HEADER_SYMBOLS) * L),
        "tail_training_enabled": bool(a.tail_training),
        "tail_training_symbols": int(TAIL_TRAINING_SYMBOLS if a.tail_training else 0),
        "tail_training_start_sample": counts["tail_training_start"],
        "gap_start_sample": int(counts["gap_start"]),
        "tail_chirp_start_sample": int(counts["tail_chirp_start"]),
        "D_tx": int(counts["D_tx"]),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(ldpc_payload_meta)
    meta.update(profile_meta(tail_training=a.tail_training, ldpc_enabled=a.ldpc))
    meta.update(counts)
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"wrote {a.out} seconds={len(raw) / FS:.3f} mod={a.mod} ldpc={a.ldpc} "
        f"active_bins={len(ACTIVE_BINS)} symbol_len={L}"
    )
    print(
        f"structure=chirp {CHIRP_SAMPLES} + guard {CHIRP_GUARD_SAMPLES} + "
        f"training {TRAINING_SYMBOLS} + header {HEADER_SYMBOLS} + "
        f"payload {len(payload)} + tail_training {TAIL_TRAINING_SYMBOLS if a.tail_training else 0} + "
        f"gap {INTER_FRAME_GAP_SAMPLES} + chirp {CHIRP_SAMPLES}"
    )
    print(f"D_tx={counts['D_tx']} tail_chirp_start={counts['tail_chirp_start']}")
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
