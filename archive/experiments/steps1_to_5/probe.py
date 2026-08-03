from pathlib import Path
import argparse

import numpy as np

from audiomodem import FS, L, bins, ofdm_tx, probe_symbols, write_wav


def args():
    p = argparse.ArgumentParser(description="make channel probe wav")
    p.add_argument("--kind", choices=["ones", "chirp", "step", "bandstep", "singlebin", "random"], default="ones")
    p.add_argument("--bins", nargs=2, type=int, default=(8, 420), metavar=("START", "END"))
    p.add_argument("--symbols", type=int, default=256)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--bandstep-parts", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("data/tx/probe.wav"))
    return p.parse_args()


def main():
    a = args()
    k = bins(*a.bins)
    if a.kind == "singlebin" and a.symbols < len(k):
        raise SystemExit("--kind singlebin requires --symbols >= number of bins")
    s = probe_symbols(a.kind, k, a.symbols, a.seed, a.bandstep_parts)
    write_wav(a.out, ofdm_tx(s, k))
    np.save(a.out.with_suffix(".symbols.npy"), s)
    print(f"wrote {a.out} kind={a.kind} symbols={len(s)} bins={k[0]}-{k[-1]} seconds={len(s) * L / FS:.3f}")


if __name__ == "__main__":
    main()
