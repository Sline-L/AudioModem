from pathlib import Path
import argparse
import json

import numpy as np

from step8_modem import (
    ACTIVE_BINS,
    BLOCK_SIZE,
    FEC_SEED,
    FS,
    HEADER_REPEATS,
    L,
    PILOT_SEED,
    PAYLOAD_START_ANCHOR_SEED,
    PAYLOAD_START_ANCHOR_SYMBOLS,
    coded_frames,
    frame_payload,
    ofdm_tx,
    payload_symbols,
    payload_start_anchor_symbols,
    profile_meta,
    TIMING_ANCHOR_INTERVAL,
    TIMING_ANCHOR_SEED,
    TIMING_ANCHOR_SYMBOLS,
    timing_anchor_symbols,
    training_symbols,
    wav_gain,
    write_wav,
)


def args():
    p = argparse.ArgumentParser(description="make a Step8 periodic-clock-anchor WAV")
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/step8_clock_anchor/observatory_64_uncompressed.tiff"),
    )
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
    p.add_argument("--timing-anchor-interval", type=int, default=TIMING_ANCHOR_INTERVAL)
    p.add_argument("--timing-anchor-symbols", type=int, default=TIMING_ANCHOR_SYMBOLS)
    p.add_argument("--timing-anchor-seed", type=int, default=TIMING_ANCHOR_SEED)
    p.add_argument("--payload-start-anchor-symbols", type=int, default=PAYLOAD_START_ANCHOR_SYMBOLS)
    p.add_argument("--payload-start-anchor-seed", type=int, default=PAYLOAD_START_ANCHOR_SEED)
    p.add_argument("--tail-seconds", type=float, default=0.25)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/step8_clock_anchor/observatory_64_uncompressed_step8_start_anchor.wav"),
    )
    return p.parse_args()


def validate(a):
    if not a.input.exists():
        raise SystemExit(f"input file does not exist: {a.input}")
    positive = (
        "sync_symbols",
        "preamble_symbols",
        "preamble_repeats",
        "block_size",
        "header_repeats",
        "timing_anchor_interval",
        "timing_anchor_symbols",
    )
    for name in positive:
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if a.payload_start_anchor_symbols < 0:
        raise SystemExit("--payload-start-anchor-symbols must be >= 0")
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
    logical_payload, used = payload_symbols(coded, a.pilot_seed)
    physical_payload, anchor_starts = frame_payload(
        logical_payload,
        a.timing_anchor_interval,
        a.timing_anchor_symbols,
        a.timing_anchor_seed,
    )

    noise_samples = int(round(a.noise_seconds * FS))
    tail_samples = int(round(a.tail_seconds * FS))
    sync_wave = ofdm_tx(sync)
    preamble_wave = ofdm_tx(np.vstack(preamble_blocks))
    start_anchor = payload_start_anchor_symbols(
        a.payload_start_anchor_symbols, a.payload_start_anchor_seed
    )
    start_anchor_wave = ofdm_tx(start_anchor)
    payload_wave = ofdm_tx(physical_payload)
    raw = np.r_[
        np.zeros(noise_samples), sync_wave, preamble_wave, start_anchor_wave,
        payload_wave, np.zeros(tail_samples)
    ]
    gain = wav_gain(raw)
    write_wav(a.out, raw)

    np.save(a.out.with_suffix(".sync.npy"), sync)
    np.save(a.out.with_suffix(".preamble.npy"), np.vstack(preamble_blocks))
    np.save(a.out.with_suffix(".start_anchor.npy"), start_anchor)
    np.save(a.out.with_suffix(".anchor_starts.npy"), anchor_starts)
    if len(anchor_starts):
        all_anchors = np.asarray(
            [
                timing_anchor_symbols(i, a.timing_anchor_symbols, a.timing_anchor_seed)
                for i in range(len(anchor_starts))
            ]
        )
    else:
        all_anchors = np.empty((0, a.timing_anchor_symbols, len(ACTIVE_BINS)), complex)
    np.save(a.out.with_suffix(".anchors.npy"), all_anchors)
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
        "logical_payload_symbols": int(len(logical_payload)),
        "physical_payload_symbols": int(len(physical_payload)),
        "payload_coded_bits": int(len(coded)),
        "payload_data_slots": int(used.sum()),
        "payload_repeats": 1,
        "frame_labels": labels,
        "frame_coded_bits": [int(len(frame)) for frame in frames],
        "block_size": int(a.block_size),
        "header_repeats": int(a.header_repeats),
        "pilot_seed": int(a.pilot_seed),
        "fec_seed": int(a.fec_seed),
        "timing_anchor_interval": int(a.timing_anchor_interval),
        "timing_anchor_symbols": int(a.timing_anchor_symbols),
        "timing_anchor_seed": int(a.timing_anchor_seed),
        "payload_start_anchor_symbols": int(a.payload_start_anchor_symbols),
        "payload_start_anchor_seed": int(a.payload_start_anchor_seed),
        "timing_anchor_count": int(len(anchor_starts)),
        "timing_anchor_overhead_symbols": int(len(anchor_starts) * a.timing_anchor_symbols),
        "timing_anchor_overhead_fraction": float(
            len(anchor_starts) * a.timing_anchor_symbols / max(1, len(physical_payload))
        ),
        "gain": float(gain),
        "payload_start_anchor_sample": int(noise_samples + len(sync_wave) + len(preamble_wave)),
        "payload_start_sample": int(
            noise_samples + len(sync_wave) + len(preamble_wave) + len(start_anchor_wave)
        ),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(profile_meta())
    meta.update(
        {
            "timing_anchor_interval": int(a.timing_anchor_interval),
            "timing_anchor_symbols": int(a.timing_anchor_symbols),
            "timing_anchor_seed": int(a.timing_anchor_seed),
            "payload_start_anchor_symbols": int(a.payload_start_anchor_symbols),
            "payload_start_anchor_seed": int(a.payload_start_anchor_seed),
            "block_size": int(a.block_size),
            "header_repeats": int(a.header_repeats),
        }
    )
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"wrote {a.out} seconds={len(raw) / FS:.3f} active_bins={len(ACTIVE_BINS)} "
        f"logical_payload={len(logical_payload)} physical_payload={len(physical_payload)}"
    )
    print(
        f"structure=noise {a.noise_seconds:.3f}s + sync {len(sync)} + "
        f"preamble {a.preamble_symbols}x{a.preamble_repeats} + "
        f"start_anchor {a.payload_start_anchor_symbols} + "
        f"payload x1 with anchor {a.timing_anchor_symbols}/{a.timing_anchor_interval}"
    )
    print(f"timing_anchors={len(anchor_starts)} overhead_seconds={len(anchor_starts) * a.timing_anchor_symbols * L / FS:.3f}")
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
