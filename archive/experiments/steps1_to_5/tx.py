from pathlib import Path
import argparse

import numpy as np

from audiomodem import L, MODS, bits_per_symbol, bins, file_symbols, ofdm_tx, preamble_wave, write_wav


def args():
    p = argparse.ArgumentParser(description="make OFDM/QPSK tx wav")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/source/file16_test.txt"))
    p.add_argument("--bins", nargs=2, type=int, default=(8, 420), metavar=("START", "END"))
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--sync-symbols", type=int, default=32)
    p.add_argument("--sync-seed", type=int, default=2026)
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--h", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/tx/file.wav"))
    return p.parse_args()


def main():
    a = args()
    k = bins(*a.bins)
    s = file_symbols(a.input, k, a.mod)
    h = np.load(a.h) if a.h else np.ones(len(k), complex)
    if h.shape != (len(k),):
        raise SystemExit(f"bad H shape: expected {(len(k),)}, got {h.shape}")
    payload = ofdm_tx(s * h, k)
    sync = np.array([], dtype=float) if a.no_sync else preamble_wave(k, a.sync_symbols, a.sync_seed)
    write_wav(a.out, np.r_[sync, payload])
    print(f"wrote {a.out} mod={a.mod} bits_per_symbol={bits_per_symbol(a.mod)} symbols={len(s)} bins={k[0]}-{k[-1]}")
    if not a.no_sync:
        print(f"sync_symbols={a.sync_symbols} sync_samples={len(sync)} payload_start={len(sync)} symbol_len={L}")


if __name__ == "__main__":
    main()
