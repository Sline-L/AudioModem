from pathlib import Path
import argparse

import numpy as np

from audiomodem import L, MODS, bins, bytes_from_mod, find_sync, ofdm_rx, preamble_wave, read_wav, unpack


def args():
    p = argparse.ArgumentParser(description="recover file from OFDM/QPSK wav")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/tx/file.wav"))
    p.add_argument("--bins", nargs=2, type=int, default=(8, 420), metavar=("START", "END"))
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--sync-symbols", type=int, default=32)
    p.add_argument("--sync-seed", type=int, default=2026)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--h", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/recovered"))
    return p.parse_args()


def main():
    a = args()
    k = bins(*a.bins)
    h = np.load(a.h) if a.h else np.ones(len(k), complex)
    if h.shape != (len(k),):
        raise SystemExit(f"bad H shape: expected {(len(k),)}, got {h.shape}")
    x = read_wav(a.input)
    if a.no_sync:
        payload_start = 0
    else:
        sync = preamble_wave(k, a.sync_symbols, a.sync_seed)
        start, score = find_sync(x, sync)
        payload_start = start + len(sync)
        print(f"sync_start={start}")
        print(f"sync_score={score:.6f}")
        print(f"payload_start={payload_start}")
    z = ofdm_rx(x[payload_start:], k) / h
    name, body = unpack(bytes_from_mod(z, a.mod))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / name).write_bytes(body)
    np.save(a.out / "rx_symbols.npy", z)
    print(f"wrote {a.out / name} mod={a.mod} bytes={len(body)} symbol_len={L}")


if __name__ == "__main__":
    main()
