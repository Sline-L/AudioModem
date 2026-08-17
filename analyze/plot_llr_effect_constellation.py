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
from modem_n2 import (
    bits_from_bytes,
    bits_from_mod,
    bits_per_symbol,
    interleave_bits,
    ldpc_encode_bits,
    mod_symbols_from_bits,
    parse_header,
)


DEFAULT_RUN = Path("runs/n2_3_ldpc_z27_rate/r2_3_10dm_1")
DEFAULT_SCALES = (2 ** -0.5, 1.0, 2.2, 3.4)


PASTEL = {
    "00": "#8fb9a8",
    "01": "#f0c987",
    "11": "#b9a7d9",
    "10": "#8fb3d9",
    "error": "#d97b73",
}


def args():
    p = argparse.ArgumentParser(
        description="plot QPSK constellation in LLR coordinates and highlight hard-decision errors"
    )
    p.add_argument("--run", type=Path, default=DEFAULT_RUN, help="decode run directory")
    p.add_argument("--source", type=Path, default=None, help="original payload file")
    p.add_argument("--scale", type=float, nargs="+", default=list(DEFAULT_SCALES))
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument("--axis-limit", type=float, default=None)
    return p.parse_args()


def scale_name(scale):
    return f"{scale:.1f}".replace(".", "p")


def load_metrics(run):
    path = run / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def configure_ldpc(metrics):
    for name, key in [
        ("LDPC_STANDARD", "ldpc_standard"),
        ("LDPC_RATE", "ldpc_rate"),
        ("LDPC_Z", "ldpc_z"),
        ("LDPC_PTYPE", "ldpc_ptype"),
        ("LDPC_INTERLEAVER_ENABLED", "ldpc_interleaver_enabled"),
        ("LDPC_INTERLEAVER_SEED", "ldpc_interleaver_seed"),
    ]:
        if key in metrics and metrics[key] is not None:
            setattr(modem_n2, name, metrics[key])
    modem_n2._LDPC_CODE = None


def source_path(run, metrics, override):
    if override is not None:
        return override
    if metrics.get("ber_source"):
        return Path(metrics["ber_source"])
    candidate = run / parse_header((run / "decoded_header.bin").read_bytes())["name"]
    if candidate.exists():
        return candidate
    raise SystemExit("cannot find original source; pass --source")


def truth_interleaved_coded_bits(source, header):
    payload_bits = bits_from_bytes(source.read_bytes())
    coded, _ = ldpc_encode_bits(payload_bits)
    if header["ldpc_enabled"]:
        coded = interleave_bits(coded)
    return coded


def qpsk_symbol_labels(bits, symbol_count):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    need = symbol_count * 2
    if bits.size < need:
        bits = np.r_[bits, np.zeros(need - bits.size, dtype=np.uint8)]
    pairs = bits[:need].reshape(symbol_count, 2)
    return np.array([f"{b0}{b1}" for b0, b1 in pairs])


def hard_symbol_errors(symbols, truth_bits):
    hard = bits_from_mod(symbols, "qpsk")[: len(truth_bits)]
    truth = truth_bits[: len(hard)]
    pair_count = len(hard) // 2
    return np.any(
        hard[: pair_count * 2].reshape(pair_count, 2)
        != truth[: pair_count * 2].reshape(pair_count, 2),
        axis=1,
    )


def main():
    a = args()
    run = a.run
    metrics = load_metrics(run)
    configure_ldpc(metrics)
    header = parse_header((run / "decoded_header.bin").read_bytes())
    if header["mod"] != "qpsk":
        raise SystemExit(f"this plot is QPSK-specific; run uses {header['mod']}")

    source = source_path(run, metrics, a.source)
    symbols = np.load(run / "payload_symbols.npy").ravel()[: header["payload_mod_symbols"]]
    truth_bits = truth_interleaved_coded_bits(source, header)[: header["payload_mod_symbols"] * 2]
    labels = qpsk_symbol_labels(truth_bits, len(symbols))
    errors = hard_symbol_errors(symbols, truth_bits)
    scales = [float(scale) for scale in a.scale]

    largest_scale = max(scales)
    limit = a.axis_limit
    if limit is None:
        base_limit = float(np.nanpercentile(np.abs(np.r_[symbols.real, symbols.imag]), 99.5))
        limit = max(1.8, min(base_limit, 8.0)) * largest_scale * np.sqrt(2.0)

    ideal = mod_symbols_from_bits(np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8), "qpsk")
    error_count = int(np.count_nonzero(errors))
    total = int(len(errors))
    out_dir = a.out if a.out is not None else run / "constellation_llr_scale"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale in scales:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
        factor = scale * np.sqrt(2.0)
        x = factor * symbols.real
        y = factor * symbols.imag
        for label, color in PASTEL.items():
            if label == "error":
                continue
            mask = (labels == label) & ~errors
            ax.scatter(x[mask], y[mask], s=4, alpha=0.44, color=color, linewidths=0, label=label)
        ax.scatter(
            x[errors],
            y[errors],
            s=8,
            alpha=0.74,
            color=PASTEL["error"],
            linewidths=0,
            label="hard error",
        )
        ax.scatter(
            factor * ideal.real,
            factor * ideal.imag,
            marker="x",
            s=85,
            color="#5a5a5a",
            linewidths=1.8,
            label="ideal",
        )
        ax.axhline(0, color="0.65", linewidth=0.8)
        ax.axvline(0, color="0.65", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.7)
        ax.set_title(
            f"{run.name}: QPSK LLR-space constellation, scale={scale:g}\n"
            f"I/Q * scale * sqrt(2), {error_count}/{total} hard-decision symbol errors"
        )
        ax.set_xlabel("real-bit LLR coordinate")
        ax.set_ylabel("imag-bit LLR coordinate")
        ax.legend(loc="upper right", frameon=True, markerscale=2.0)
        fig.tight_layout()
        path = out_dir / f"llr_effect_qpsk_errors_scale_{scale_name(scale)}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
