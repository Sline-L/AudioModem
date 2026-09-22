"""r1 时间轴：Chirp 峰、粗起点、训练锁、K、钟偏累积、sfo=0 的尾残差。"""

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

from align import extract_active_sfo, find_end_preamble, matched_start
from capture import coarse_body_start, find_two_chirps, parabolic_peak
from channel import h_and_llr_scale
from params import derived
from phy import read_wav_mono, train_freq_time


def _tail_resid(y, x, h):
    err = np.mean(np.abs(y - x * h) ** 2, axis=0)
    return float(np.median(err / np.maximum(np.abs(h) ** 2, 1e-12)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default=str(_boot.ROOT / "data" / "r1.wav"))
    parser.add_argument("--metrics", default=str(_boot.ROOT / "run" / "n2" / "r1" / "metrics.json"))
    parser.add_argument("--out-dir", default=str(_boot.ROOT / "run" / "n2" / "study_b1"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    _fs, rx = read_wav_mono(args.wav)
    d = derived()
    first, second, metric, cap = find_two_chirps(rx, d)
    coarse = coarse_body_start(first, d)
    lock = matched_start(rx, coarse, d)
    end = find_end_preamble(rx, lock["start"], d)
    sfo = float(metrics["clock_error_ppm"]) * 1e-6
    n_body = 8 + int(end["K"]) + 8
    accum = abs(sfo) * n_body * d["symbol_len"]

    freq, _ = train_freq_time(d)
    bins_known = freq[:, d["start_index"] : d["end_index"]]
    body_ok = extract_active_sfo(rx, lock["start"], n_body, sfo, d)
    body_0 = extract_active_sfo(rx, lock["start"], n_body, 0.0, d)
    h, _scale, _nv, _valid = h_and_llr_scale(body_ok[:8], bins_known[:8])
    resid_front = _tail_resid(body_ok[:8], bins_known[:8], h)
    resid_tail = _tail_resid(body_ok[-8:], bins_known[8:16], h)
    h0, _, _, _ = h_and_llr_scale(body_0[:8], bins_known[:8])
    resid_tail_0 = _tail_resid(body_0[-8:], bins_known[8:16], h0)

    lines = [
        f"peak1={first:.3f}  metrics={metrics['capture']['peak1']:.3f}",
        f"peak2={second:.3f}",
        f"coarse=peak1+144000+24000={coarse:.3f}",
        f"lock.start={lock['start']:.3f}  snr={lock['snr']:.1f}",
        f"coarse-lock samples={coarse - lock['start']:.3f}",
        f"end_offset={end['end_offset']:.3f}  n_sym_raw={end['n_sym_raw']:.6f}  K={end['K']}",
        f"metrics K={metrics['end']['K']} n_sym_raw={metrics['end']['n_sym_raw']:.6f}",
        f"|sfo|={abs(sfo):.3e} over {n_body} symbols accumulates {accum:.2f} samples",
        f"front residual (true sfo)={resid_front:.4e}",
        f"tail residual (true sfo)={resid_tail:.4e}",
        f"tail residual (sfo=0)={resid_tail_0:.4e}",
    ]
    ok = (
        abs(first - metrics["capture"]["peak1"]) < 0.2
        and abs(lock["start"] - metrics["lock"]["start"]) < 0.2
        and int(end["K"]) == int(metrics["end"]["K"])
        and abs(end["n_sym_raw"] - metrics["end"]["n_sym_raw"]) < 1e-3
        and resid_tail_0 > resid_tail
    )
    text = "\n".join(lines)
    (out / "timeline.txt").write_text(text + f"\n{'OK' if ok else 'FAIL'}\n", encoding="utf-8")
    print(text)

    i1 = int(round(first))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(metric)
    ax[0].axvline(first, color="C1")
    ax[0].axvline(second, color="C2")
    ax[0].set_title("chirp metric")
    sl = metric[i1 - 40 : i1 + 40]
    ax[1].plot(np.arange(i1 - 40, i1 - 40 + sl.size), sl)
    peak_i = int(np.argmax(metric))
    frac, _y = parabolic_peak(metric, peak_i)
    ax[1].axvline(frac, color="C1", label=f"parabolic {frac:.3f}")
    ax[1].legend()
    ax[1].set_title("peak1 zoom")
    fig.tight_layout()
    fig.savefig(out / "chirp_peaks.png", dpi=120)
    plt.close(fig)

    print("期望: peak/lock/K 与 run/n2/r1/metrics.json 一致，sfo=0 时尾训练残差更大")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
