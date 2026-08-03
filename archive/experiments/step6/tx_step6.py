from pathlib import Path
import argparse
import json

import numpy as np

from step6_modem import (
    ACTIVE_BINS,
    FS,
    L,
    MODS,
    bits_per_symbol,
    file_symbols_with_pilots,
    ofdm_tx,
    profile_meta,
    training_blocks,
    wav_gain,
    write_wav,
)


def args():
    p = argparse.ArgumentParser(description="make the Step 6 N=512, CP=256 overlap-band test WAV")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/step6_newexp/file.jpeg"))
    p.add_argument("--mod", choices=MODS, default="bpsk")
    p.add_argument("--probe-symbols", type=int, default=256)
    p.add_argument("--probe-repeats", type=int, default=3)
    p.add_argument("--probe-seed", type=int, default=2026)
    p.add_argument("--sync-symbols", type=int, default=128)
    p.add_argument("--sync-repeats", type=int, default=3)
    p.add_argument("--sync-seed", type=int, default=3026)
    p.add_argument("--pilot-seed", type=int, default=2027)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/step6_newexp/file_combo_overlap_n512_cp256.wav"),
    )
    return p.parse_args()


def validate(a):
    for name in ("probe_symbols", "probe_repeats", "sync_symbols", "sync_repeats"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if not a.input.exists():
        raise SystemExit(f"input file does not exist: {a.input}")


def main():
    a = args()
    validate(a)
    probe_blocks = training_blocks(ACTIVE_BINS, a.probe_symbols, a.probe_repeats, a.probe_seed)
    sync_blocks = training_blocks(ACTIVE_BINS, a.sync_symbols, a.sync_repeats, a.sync_seed)
    payload, pilots = file_symbols_with_pilots(a.input, a.mod, a.pilot_seed)

    probe_symbols = np.vstack(probe_blocks)
    sync_symbols = np.vstack(sync_blocks)
    probe_wave = ofdm_tx(probe_symbols)
    sync_wave = ofdm_tx(sync_symbols)
    payload_wave = ofdm_tx(payload)
    raw = np.r_[probe_wave, sync_wave, payload_wave]
    gain = wav_gain(raw)
    write_wav(a.out, raw)

    np.save(a.out.with_suffix(".probe.npy"), probe_symbols * gain)
    np.save(a.out.with_suffix(".preamble.npy"), sync_symbols * gain)
    meta = {
        "input": str(a.input),
        "out": str(a.out),
        "fs": FS,
        "mod": a.mod,
        "bits_per_symbol": bits_per_symbol(a.mod),
        "probe_symbols_per_repeat": int(a.probe_symbols),
        "probe_repeats": int(a.probe_repeats),
        "probe_symbols_total": int(len(probe_symbols)),
        "probe_seed": int(a.probe_seed),
        "sync_symbols_per_repeat": int(a.sync_symbols),
        "sync_repeats": int(a.sync_repeats),
        "sync_symbols_total": int(len(sync_symbols)),
        "sync_seed": int(a.sync_seed),
        "payload_repeats": 1,
        "payload_symbols": int(len(payload)),
        "pilot_seed": int(a.pilot_seed),
        "gain": float(gain),
        "probe_samples": int(len(probe_wave)),
        "sync_samples": int(len(sync_wave)),
        "payload_samples": int(len(payload_wave)),
        "file_preamble_start": int(len(probe_wave)),
        "payload_start": int(len(probe_wave) + len(sync_wave)),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(profile_meta())
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"wrote {a.out} seconds={len(raw) / FS:.3f} N=512 CP=256 "
        f"active_bins={ACTIVE_BINS[0]}-{ACTIVE_BINS[-1]}"
    )
    print(
        f"probe={a.probe_symbols}x{a.probe_repeats} "
        f"file_preamble={a.sync_symbols}x{a.sync_repeats} "
        f"payload={len(payload)}x1 symbol_len={L}"
    )
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
