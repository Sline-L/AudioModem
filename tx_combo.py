from pathlib import Path
import argparse
import json

import numpy as np

from audiomodem import (
    FS,
    L,
    MODS,
    bins,
    bits_per_symbol,
    file_symbols,
    ofdm_tx,
    preamble_wave,
    probe_symbols,
    wav_gain,
    write_wav,
)
from fband import fband_profile, file_symbols_with_comb_pilots, profile_meta


def args():
    p = argparse.ArgumentParser(description="make one-shot probe + file OFDM wav")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/step2_file/exp2.txt"))
    p.add_argument("--bins", nargs=2, type=int, default=(8, 150), metavar=("START", "END"))
    p.add_argument("--fband-profile", choices=["conservative", "trimmed"], default=None)
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--probe-kind", choices=["ones", "chirp", "step", "bandstep", "random"], default="ones")
    p.add_argument("--probe-symbols", type=int, default=256)
    p.add_argument("--probe-seed", type=int, default=2026)
    p.add_argument("--sync-symbols", type=int, default=32)
    p.add_argument("--sync-seed", type=int, default=2026)
    p.add_argument("--pilot-interval", type=int, default=0)
    p.add_argument("--pilot-len", type=int, default=1)
    p.add_argument("--pilot-kind", choices=["ones", "chirp", "step", "bandstep", "random"], default="random")
    p.add_argument("--pilot-seed", type=int, default=2027)
    p.add_argument("--payload-repeats", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("data/step2_file/exp2_combo_8_150.wav"))
    return p.parse_args()


def pilot_count(data_symbols, interval, pilot_len):
    return (data_symbols // interval) * pilot_len if interval > 0 else 0


def frame_payload(data, pilots, interval, pilot_len):
    if interval <= 0:
        return data
    parts = []
    pilot_i = 0
    for i in range(0, len(data), interval):
        chunk = data[i : i + interval]
        parts.append(chunk)
        if len(chunk) == interval:
            parts.append(pilots[pilot_i : pilot_i + pilot_len])
            pilot_i += pilot_len
    return np.vstack(parts) if parts else data


def main():
    a = args()
    if a.pilot_interval < 0:
        raise SystemExit("--pilot-interval must be >= 0")
    if a.pilot_len < 1:
        raise SystemExit("--pilot-len must be >= 1")
    if a.payload_repeats < 1:
        raise SystemExit("--payload-repeats must be >= 1")
    profile = fband_profile(a.fband_profile)
    k = profile["k"] if profile else bins(*a.bins)
    probe = probe_symbols(a.probe_kind, k, a.probe_symbols, a.probe_seed)
    if profile:
        if a.pilot_interval > 0:
            raise SystemExit("--fband-profile uses per-symbol comb pilots; leave --pilot-interval at 0")
        data_payload, comb_pilots = file_symbols_with_comb_pilots(
            a.input,
            k,
            profile["data_idx"],
            profile["pilot_idx"],
            a.mod,
            a.pilot_seed,
            a.pilot_kind,
        )
        n_pilots = 0
        pilots = np.empty((0, len(k)), complex)
        payload_once = data_payload
    else:
        data_payload = file_symbols(a.input, k, a.mod)
        comb_pilots = np.empty((0, 0), complex)
        n_pilots = pilot_count(len(data_payload), a.pilot_interval, a.pilot_len)
        pilots = probe_symbols(a.pilot_kind, k, n_pilots, a.pilot_seed)
        payload_once = frame_payload(data_payload, pilots, a.pilot_interval, a.pilot_len)
    payload = np.vstack([payload_once] * a.payload_repeats)
    probe_wave = ofdm_tx(probe, k)
    sync_wave = preamble_wave(k, a.sync_symbols, a.sync_seed)
    payload_wave = ofdm_tx(payload, k)
    raw = np.r_[probe_wave, sync_wave, payload_wave]
    gain = wav_gain(raw)

    write_wav(a.out, raw)
    np.save(a.out.with_suffix(".probe.npy"), probe * gain)
    meta = {
        "input": str(a.input),
        "out": str(a.out),
        "fs": FS,
        "symbol_len": L,
        "bins": [int(k[0]), int(k[-1])],
        "bin_list": [int(x) for x in k],
        "mod": a.mod,
        "bits_per_symbol": bits_per_symbol(a.mod),
        "probe_kind": a.probe_kind,
        "probe_symbols": int(a.probe_symbols),
        "probe_seed": int(a.probe_seed),
        "sync_symbols": int(a.sync_symbols),
        "sync_seed": int(a.sync_seed),
        "data_symbols": int(len(data_payload)),
        "comb_pilot_symbols": int(len(comb_pilots)),
        "pilot_symbols": int(n_pilots),
        "pilot_interval": int(a.pilot_interval),
        "pilot_len": int(a.pilot_len),
        "pilot_kind": a.pilot_kind,
        "pilot_seed": int(a.pilot_seed),
        "payload_repeats": int(a.payload_repeats),
        "payload_symbols_once": int(len(payload_once)),
        "payload_symbols": int(len(payload)),
        "gain": float(gain),
        "probe_samples": int(len(probe_wave)),
        "sync_samples": int(len(sync_wave)),
        "payload_samples": int(len(payload_wave)),
        "file_preamble_start": int(len(probe_wave)),
        "payload_start": int(len(probe_wave) + len(sync_wave)),
        "total_samples": int(len(raw)),
        "seconds": float(len(raw) / FS),
    }
    meta.update(profile_meta(profile))
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    label = a.fband_profile or f"{k[0]}-{k[-1]}"
    print(f"wrote {a.out} mod={a.mod} bins={label} active_bins={len(k)} seconds={len(raw) / FS:.3f}")
    print(
        f"probe_symbols={len(probe)} sync_symbols={a.sync_symbols} "
        f"data_symbols={len(data_payload)} pilot_symbols={n_pilots} comb_pilot_symbols={len(comb_pilots)} "
        f"payload_symbols_once={len(payload_once)} payload_repeats={a.payload_repeats} "
        f"payload_symbols={len(payload)}"
    )
    print(f"payload_start={meta['payload_start']} symbol_len={L}")
    print(f"wrote {a.out.with_suffix('.probe.npy')}")
    print(f"wrote {a.out.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
