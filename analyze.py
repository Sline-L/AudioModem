from pathlib import Path
import argparse
import csv
import os

import numpy as np
from scipy import signal

from audiomodem import FS, bins, ofdm_rx, ofdm_tx, probe_symbols, read_wav, wav_gain

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


def args():
    p = argparse.ArgumentParser(description="estimate H and plot Y/H from received probe")
    p.add_argument("receive", type=Path)
    p.add_argument("--kind", choices=["ones", "chirp", "step", "bandstep", "random"], default="ones")
    p.add_argument("--bins", nargs=2, type=int, default=(8, 420), metavar=("START", "END"))
    p.add_argument("--symbols", type=int, default=256)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", type=Path, default=Path("runs/probe"))
    return p.parse_args()


def sync(rx, tx):
    if len(rx) < len(tx):
        raise ValueError("receive wav is shorter than probe")
    c = signal.correlate(rx, tx, mode="valid", method="fft")
    i = int(np.argmax(np.abs(c)))
    e = np.linalg.norm(rx[i : i + len(tx)]) * np.linalg.norm(tx)
    return i, float(abs(c[i]) / e) if e else 0.0


def save_csv(path, k, h, y, yt):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "freq_hz", "abs_y", "abs_y_theory", "real_h", "imag_h", "abs_h", "phase_h"])
        for n, yy, tt, hh in zip(k, y, yt, h):
            w.writerow([int(n), n * FS / 1024, abs(yy), abs(tt), hh.real, hh.imag, abs(hh), np.angle(hh)])


def plot(path, k, a, b, title, ylabel):
    plt.figure(figsize=(9, 4))
    plt.plot(k, a, label="received")
    if b is not None:
        plt.plot(k, b, label="theory", alpha=0.75)
    plt.title(title)
    plt.xlabel("bin")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    a = args()
    k = bins(*a.bins)
    x = probe_symbols(a.kind, k, a.symbols, a.seed)
    tx_raw = ofdm_tx(x, k)
    g = wav_gain(tx_raw)
    tx = tx_raw * g
    x = x * g
    rx = read_wav(a.receive)
    start, score = sync(rx, tx)
    y = ofdm_rx(rx[start : start + len(tx)], k)
    mask = np.abs(x) > 0
    h_each = np.divide(y, x, out=np.full_like(y, np.nan), where=mask)
    seen = np.any(mask, axis=0)
    h = np.full(len(k), np.nan + 1j * np.nan)
    h[seen] = np.nanmean(h_each[:, seen], axis=0)
    yt = np.sqrt(np.nanmean(np.abs(x) ** 2, axis=0))
    ym = np.sqrt(np.nanmean(np.abs(y) ** 2, axis=0))

    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "Y.npy", y)
    np.save(a.out / "Y_theory.npy", x)
    np.save(a.out / "H.npy", h)
    save_csv(a.out / "summary.csv", k, h, ym, yt)
    plot(a.out / "Y_spectrum.png", k, np.abs(ym), np.abs(yt), "Y spectrum", "magnitude")
    plot(a.out / "H.png", k, np.abs(h), None, "H magnitude", "|H|")

    m = np.abs(h[np.isfinite(h)])
    print(f"sync_start={start}")
    print(f"sync_score={score:.6f}")
    print(f"mean_abs_h={float(np.mean(m)):.6g}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
