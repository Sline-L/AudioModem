from pathlib import Path
import argparse

import numpy as np

from audiomodem import MODS, bins, bytes_from_mod, ofdm_rx, read_wav, unpack


def args():
    p = argparse.ArgumentParser(description="recover file from OFDM/QPSK wav")
    p.add_argument("input", nargs="?", type=Path, default=Path("data/tx/file.wav"))
    p.add_argument("--bins", nargs=2, type=int, default=(8, 420), metavar=("START", "END"))
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--h", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/recovered"))
    return p.parse_args()


def main():
    a = args()
    k = bins(*a.bins)
    h = np.load(a.h) if a.h else np.ones(len(k), complex)
    if h.shape != (len(k),):
        raise SystemExit(f"bad H shape: expected {(len(k),)}, got {h.shape}")
    z = ofdm_rx(read_wav(a.input), k) / h
    name, body = unpack(bytes_from_mod(z, a.mod))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / name).write_bytes(body)
    np.save(a.out / "rx_symbols.npy", z)
    print(f"wrote {a.out / name} mod={a.mod} bytes={len(body)}")


if __name__ == "__main__":
    main()
