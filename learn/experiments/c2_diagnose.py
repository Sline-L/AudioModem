"""读一份 run 目录：未收敛码字、每符号中位 |LLR|、公共相位。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _boot  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else _boot.ROOT / "run" / "n2" / f"study_c2_{run.name}"
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((run / "ldpc_symbols.csv").open(encoding="utf-8")))
    iters = np.array([int(r["iters"]) for r in rows])
    med = []
    for r in rows:
        med.append(float(r["median_abs_llr"]) if r["median_abs_llr"] else np.nan)
    med = np.array(med)
    bad = np.flatnonzero(iters >= 200)
    cpe_path = run / "cpe.npy"
    cpe = np.load(cpe_path) if cpe_path.exists() else None

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(iters, lw=0.8)
    ax[0].axhline(200, color="C3", lw=0.6)
    ax[0].set_ylabel("iters")
    ax[1].plot(med, lw=0.8)
    ax[1].set_ylabel("median |LLR|")
    if cpe is not None and cpe.size:
        ax[2].plot(cpe, lw=0.8)
    ax[2].set_ylabel("CPE rad")
    ax[2].set_xlabel("payload symbol")
    fig.tight_layout()
    fig.savefig(out / "ldpc_llr_cpe.png", dpi=120)
    plt.close(fig)

    metrics = {}
    mj = run / "metrics.json"
    if mj.exists():
        metrics = json.loads(mj.read_text(encoding="utf-8"))
    lines = [
        f"run={run}",
        f"stage={metrics.get('stage')} blocks={metrics.get('blocks_ok')}/{metrics.get('blocks_total')}",
        f"unconverged count={bad.size} index={bad.tolist()[:40]}",
    ]
    text = "\n".join(lines)
    (out / "diagnose.txt").write_text(text + "\nOK\n", encoding="utf-8")
    print(text)
    print("期望: 未收敛下标与 ldpc_symbols.csv 里 converged=False 的行一致")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
