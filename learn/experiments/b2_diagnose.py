"""打印 _lock_frame 的中间量，并保存 Chirp / 训练相关曲线。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _boot  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from capture import coarse_body_start, find_two_chirps
from params import derived
from phy import read_wav_mono, train_freq_time
from receiver import _lock_frame
from scipy.signal import fftconvolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    wav = Path(args.wav)
    out = Path(args.out_dir) if args.out_dir else _boot.ROOT / "run" / "n2" / f"study_b2_{wav.stem}"
    out.mkdir(parents=True, exist_ok=True)

    _fs, rx = read_wav_mono(wav)
    d = derived()
    fr = _lock_frame(rx, d)
    first, second, metric, cap = find_two_chirps(rx, d)
    lines = [
        f"wav={wav}",
        f"samples={rx.size}",
        f"capture={json.dumps({k: cap[k] for k in ('ok','snr1','snr2','peak1','peak2')}, default=float)}",
    ]
    if fr is None:
        lines.append("lock_frame=None")
    else:
        lines.append(f"coarse={coarse_body_start(fr['first'], d)}")
        lines.append(f"lock={fr['lock']}")
        lines.append(f"end={fr['end']}")
        nraw = fr["end"]["n_sym_raw"]
        lines.append(f"distance_to_integer={abs(nraw - round(nraw))}")
    text = "\n".join(lines)
    (out / "lock.txt").write_text(text, encoding="utf-8")
    print(text)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(metric)
    if first is not None:
        ax.axvline(first, color="C1", label="peak1")
        ax.axvline(second, color="C2", label="peak2")
    ax.legend()
    ax.set_title("chirp matched metric")
    fig.tight_layout()
    fig.savefig(out / "chirp_metric.png", dpi=120)
    plt.close(fig)

    _, train_t = train_freq_time(d)
    tmpl = np.concatenate(train_t[:2])
    train_metric = fftconvolve(rx, tmpl[::-1], mode="valid") ** 2
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(train_metric)
    if fr is not None:
        ax.axvline(fr["lock"]["start"], color="C1", label="lock.start")
        ax.legend()
    ax.set_title("front-2 training metric")
    fig.tight_layout()
    fig.savefig(out / "train_metric.png", dpi=120)
    plt.close(fig)

    print("期望: 成功录音 lock.snr>=4 且 n_sym_raw 离整数 < 0.08；align 失败时 lock_frame=None")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
