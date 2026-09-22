"""用 r1 的起点和钟偏走一个符号：H、星座、LLR、与 cute.jpg 对字节。"""

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

from align import extract_active_sfo
from channel import h_and_llr_scale, lerp_h
from decode_path import LdpcBank
from params import derived
from phy import bits_to_bytes, read_wav_mono, train_freq_time
from receiver import _equalize_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=str(_boot.ROOT / "data" / "r1.wav"))
    parser.add_argument("--metrics", default=str(_boot.ROOT / "run" / "n2" / "r1" / "metrics.json"))
    parser.add_argument("--out-dir", default=str(_boot.ROOT / "run" / "n2" / "study_c1"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    _fs, rx = read_wav_mono(args.wav)
    d = derived()
    start = float(metrics["sync_start"])
    sfo = float(metrics["clock_error_ppm"]) * 1e-6
    k = int(metrics["K"])
    n_pre = d["preamble_count"]
    body = extract_active_sfo(rx, start, n_pre + k + n_pre, sfo, d)
    freq, _ = train_freq_time(d)
    known = freq[:, d["start_index"] : d["end_index"]]
    h_front, scale_front, _nv, _v = h_and_llr_scale(body[:n_pre], known[:n_pre])
    h_tail, scale_tail, _, _ = h_and_llr_scale(body[n_pre + k :], known[n_pre:])
    mean_h = float(np.mean(np.abs(h_front)))

    bins = np.arange(d["start_index"], d["end_index"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(bins, np.abs(h_front), label="|H_front|")
    ax.plot(bins, np.abs(h_tail), label="|H_tail|", alpha=0.8)
    ax.legend()
    ax.set_xlabel("bin")
    fig.tight_layout()
    fig.savefig(out / "h_front_tail.png", dpi=120)
    plt.close(fig)

    bank = LdpcBank(d)
    payload = body[n_pre + 1 :]  # 载荷符号，不含已经译过的帧头符号
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    pick = [0, min(20, payload.shape[0] - 1), payload.shape[0] - 1]
    sym0 = {}
    for col, idx in enumerate(pick):
        alpha = (idx + 1) / float(max(k - 1, 1))
        h = lerp_h(h_front, h_tail, alpha)
        scale = (1.0 - alpha) * scale_front + alpha * scale_tail
        y = payload[idx]
        z_raw = y / (h + 1e-12)
        info, it, llr, z, _theta = _equalize_once(payload[idx], h, scale, bank)
        for row, (cloud, title) in enumerate(
            ((y, "Y"), (z_raw, "Z=Y/H"), (z, f"after CPE it={it}"))
        ):
            axes[row, col].plot(cloud.real, cloud.imag, ".", ms=1)
            axes[row, col].set_title(f"sym {idx} {title}")
            axes[row, col].set_aspect("equal", adjustable="datalim")
        if idx == 0:
            raw = bits_to_bytes(info)
            src = (_boot.ROOT / "source" / "cute.jpg").read_bytes()
            ncmp = min(249, len(src))
            sym0 = {
                "it": it,
                "med": float(np.median(np.abs(llr))),
                "match": raw[:ncmp] == src[:ncmp],
                "llr": llr,
            }
    fig.tight_layout()
    fig.savefig(out / "constellations.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(sym0["llr"], bins=60)
    ax.set_title(f"LLR of payload symbol 0, median |LLR|={sym0['med']:.3f}")
    fig.tight_layout()
    fig.savefig(out / "llr_hist.png", dpi=120)
    plt.close(fig)

    cpe_path = _boot.ROOT / "run" / "n2" / "r1" / "cpe.npy"
    if cpe_path.exists():
        cpe = np.load(cpe_path)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(cpe)
        ax.set_xlabel("payload symbol")
        ax.set_ylabel("CPE rad")
        fig.tight_layout()
        fig.savefig(out / "cpe.png", dpi=120)
        plt.close(fig)

    lines = [
        f"mean |H_front|={mean_h:.4f}  metrics mean_abs_h={metrics['mean_abs_h']:.4f}",
        f"payload symbol 0 iters={sym0['it']} median|LLR|={sym0['med']:.4f}",
        f"first 249 file bytes match cute.jpg = {sym0['match']}",
        "clock_error_ppm 是搜头阶段的钟偏；metrics.json 没有单独的 payload_ppm 字段",
        "mean_abs_h 来自联合 H，这里打印的是前训练 H，两者接近但不要求相等",
    ]
    ok = bool(sym0.get("match")) and int(sym0["it"]) < 200
    text = "\n".join(lines)
    (out / "walk.txt").write_text(text + f"\n{'OK' if ok else 'FAIL'}\n", encoding="utf-8")
    print(text)
    print("期望: 载荷第 0 块与 cute.jpg 前 249 字节一致，it<200")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
