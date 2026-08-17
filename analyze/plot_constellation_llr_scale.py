from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-audiomodem")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

import modem_n2
from modem_n2 import llr_from_symbols, parse_header


DEFAULT_RUN = Path("runs/n2_3_ldpc_z27_rate/r2_3_10dm_1")
DEFAULT_SWEEP = Path("runs/n2_6_llr_sweep/rate2_3_1")
DEFAULT_SCALES = (2 ** -0.5, 1.0, 2.2, 3.4)


def args():
    p = argparse.ArgumentParser(
        description="plot N2 payload constellation colored by LLR confidence"
    )
    p.add_argument("--run", type=Path, default=DEFAULT_RUN, help="decode run directory")
    p.add_argument(
        "--sweep",
        type=Path,
        default=DEFAULT_SWEEP,
        help="optional LLR sweep directory containing llr_XpY/metrics.json",
    )
    p.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="LLR scales to plot",
    )
    p.add_argument("--mod", choices=modem_n2.MODS, default=None, help="override modulation")
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument(
        "--axis-limit",
        type=float,
        default=None,
        help="fixed I/Q axis limit; default uses a robust payload percentile",
    )
    return p.parse_args()


def scale_name(scale):
    return f"{scale:.1f}".replace(".", "p")


def load_metrics(run, sweep, scale):
    paths = []
    if sweep is not None:
        paths.append(sweep / f"llr_{scale_name(scale)}" / "metrics.json")
    paths.append(run / "metrics.json")
    for path in paths:
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if float(metrics.get("ldpc_llr_scale", scale)) == float(scale):
                return metrics
    return {}


def infer_mod(run, override):
    if override is not None:
        return override
    header_path = run / "decoded_header.bin"
    if header_path.exists():
        return parse_header(header_path.read_bytes())["mod"]
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        mod = json.loads(metrics_path.read_text(encoding="utf-8")).get("mod")
        if mod:
            return mod
    raise SystemExit("cannot infer modulation; pass --mod")


def symbol_confidence(symbols, mod, scale):
    old_scale = modem_n2.LDPC_LLR_SCALE
    modem_n2.LDPC_LLR_SCALE = float(scale)
    try:
        llr = llr_from_symbols(symbols, mod)
    finally:
        modem_n2.LDPC_LLR_SCALE = old_scale
    return np.mean(np.abs(llr.reshape(-1, modem_n2.bits_per_symbol(mod))), axis=1)


def ideal_points(mod):
    if mod == "bpsk":
        return np.array([1.0, -1.0], dtype=complex)
    if mod == "qpsk":
        return np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)
    labels = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    return modem_n2.mod_symbols_from_bits(labels.ravel(), "qam16")


def metric_text(metrics):
    ber = metrics.get("ber_payload_ber")
    mean_it = metrics.get("ldpc_decode_iterations_mean")
    ber_text = f"BER={ber * 100:.3f}%" if isinstance(ber, (int, float)) else "BER=n/a"
    it_text = f"mean it={mean_it:.2f}" if isinstance(mean_it, (int, float)) else "mean it=n/a"
    return ber_text, it_text


def main():
    a = args()
    run = a.run
    out = a.out if a.out is not None else run / "constellation_llr_scale"
    out.mkdir(parents=True, exist_ok=True)

    symbols_path = run / "payload_symbols.npy"
    if not symbols_path.exists():
        raise SystemExit(f"missing payload symbols: {symbols_path}")
    symbols = np.load(symbols_path).ravel()
    mod = infer_mod(run, a.mod)
    scales = [float(scale) for scale in a.scale]

    confidence = {scale: symbol_confidence(symbols, mod, scale) for scale in scales}
    vmax = float(np.nanpercentile(np.concatenate(list(confidence.values())), 99))
    vmax = max(vmax, 1e-6)
    limit = a.axis_limit
    if limit is None:
        limit = float(np.nanpercentile(np.abs(np.r_[symbols.real, symbols.imag]), 99.5))
        limit = max(1.8, min(limit, 8.0))

    ideal = ideal_points(mod)
    for scale in scales:
        metrics = load_metrics(run, a.sweep, scale)
        ber_text, it_text = metric_text(metrics)
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
        points = ax.scatter(
            symbols.real,
            symbols.imag,
            c=confidence[scale],
            s=7,
            alpha=0.68,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            ideal.real,
            ideal.imag,
            marker="x",
            s=90,
            c="crimson",
            linewidths=2.0,
            label=f"ideal {mod.upper()}",
        )
        ax.axhline(0, color="0.55", linewidth=0.8)
        ax.axvline(0, color="0.55", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.grid(True, color="0.88", linewidth=0.7)
        ax.set_xlabel("In-phase / Real")
        ax.set_ylabel("Quadrature / Imag")
        ax.set_title(
            f"{run.name} {mod.upper()} constellation, LLR scale={scale:g}\n"
            f"color = mean |bit LLR|, {ber_text}, {it_text}"
        )
        ax.legend(loc="upper right", frameon=True)
        colorbar = fig.colorbar(points, ax=ax, pad=0.02)
        colorbar.set_label("confidence: mean |LLR| per symbol")
        fig.tight_layout()
        path = out / f"constellation_llr_scale_{scale_name(scale)}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
