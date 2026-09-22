"""随机信息块：编码、扰码、加噪声、解扰译码。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _boot  # noqa: E402

import numpy as np

from decode_path import LdpcBank
from phy import lfsr_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(_boot.ROOT / "run" / "n2" / "study_a2"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bank = LdpcBank()
    coder = bank.coder
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, coder.K, dtype=np.uint8)
    coded = np.asarray(coder.encode(u), dtype=np.uint8)
    scram = lfsr_sequence(coded.size)
    air = coded ^ scram

    lines = [
        f"coder.K={coder.K} coder.N={coder.N} z={coder.z}",
        "sigma  it  info_ber  note",
    ]
    ok = coder.K == 1992 and coder.N == 3984
    noiseless_it = None
    for sigma in (0.05, 0.4, 0.8, 1.5, 3.0):
        signal = 1.0 - 2.0 * air.astype(np.float64)
        llr = signal + rng.normal(0.0, sigma, signal.size)
        info, _full, it = bank.decode_codeword(llr)
        ber = float(np.mean(info != u))
        note = "应 0 误码且 it<200" if sigma <= 0.05 else "噪声变大后允许不收敛"
        lines.append(f"{sigma:<5}  {it:<3}  {ber:.4f}    {note}")
        if sigma == 0.05:
            noiseless_it = it
            ok = ok and ber == 0.0 and it < 200
    ok = ok and noiseless_it is not None and noiseless_it < 200

    text = "\n".join(lines)
    (out / "a2_report.txt").write_text(text + f"\n{'OK' if ok else 'FAIL'}\n", encoding="utf-8")
    print(text)
    print("期望: K=1992 N=3984，sigma=0.05 时 info_ber=0 且 it<200")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
