"""对照发射端：Chirp、训练、扰码、PH 头。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _boot  # noqa: E402

import numpy as np

from params import derived
from phy import lfsr_sequence, linear_chirp, parse_ph_header, train_freq_time

LDPC_ROOT = _boot.LDPC_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(_boot.ROOT / "run" / "n2" / "study_a1"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(LDPC_ROOT)
    try:
        from example_tx import Ofdm_tx

        tx = Ofdm_tx()
    finally:
        os.chdir(cwd)

    d = derived()
    lines = []

    chirp_err = float(np.max(np.abs(linear_chirp(d) - tx.log_chirp_gen())))
    lines.append(f"chirp max abs diff = {chirp_err:.3e}")
    ok = chirp_err < 1e-9

    lfsr_ok = np.array_equal(lfsr_sequence(64), tx.sequence(64))
    lines.append(f"lfsr[64] equal = {bool(lfsr_ok)}")
    ok = ok and bool(lfsr_ok)

    freq, time_cp = train_freq_time(d)
    tf, tt = tx.train_symbol_gen(
        tx.start_index, tx.start_index + 4 * tx.chunk_size, N=tx.N, symbol_count=16
    )
    train_err = float(np.max(np.abs(freq - tf)))
    time_err = float(np.max(np.abs(time_cp - tt)))
    lines.append(f"train freq max abs diff = {train_err:.3e}")
    lines.append(f"train time max abs diff = {time_err:.3e}")
    lines.append(f"train freq shape = {freq.shape}, time shape = {time_cp.shape}")
    ok = ok and train_err < 1e-8 and time_err < 1e-8

    import random

    random.seed(80)
    npy = np.random.default_rng(80)
    py_seq = [random.randint(0, 3) for _ in range(8)]
    np_seq = [int(npy.integers(0, 4)) for _ in range(8)]
    lines.append(f"CPython random.seed(80) first 8 = {py_seq}")
    lines.append(f"NumPy default_rng(80) first 8 = {np_seq}")
    lines.append(f"RNG sequences differ = {py_seq != np_seq}")
    ok = ok and py_seq != np_seq

    name = "study_name_20_chars!"
    assert len(name.encode("utf-8")) == 20
    header = bytearray(tx.header_gen(name, 1000, 5, d["info_bytes"]))
    lines.append(f"header len = {len(header)}")
    lines.append(f"hex[0:40] = {header[:40].hex()}")
    parsed = parse_ph_header(bytes(header))
    lines.append(f"parse filename={parsed['filename']!r} size={parsed['file_size']} decl={parsed['declared_symbols']}")
    ok = ok and parsed["filename"] == name and parsed["file_size"] == 1000

    name_len = len(name.encode("utf-8"))
    crc_at = 12 + name_len
    lines.append(f"filename bytes at [11:{11+name_len}]")
    lines.append(f"0x00 at index {11+name_len} value={header[11+name_len]}")
    lines.append(f"CRC 4 bytes at [{crc_at}:{crc_at+4}] = {header[crc_at:crc_at+4].hex()}")

    broken = bytearray(header)
    broken[11] ^= 0x01
    lines.append(f"flip filename byte -> parse is None: {parse_ph_header(bytes(broken)) is None}")
    ok = ok and parse_ph_header(bytes(broken)) is None

    outside = bytearray(header)
    pad_at = crc_at + 4
    outside[pad_at] ^= 0xFF
    parsed_out = parse_ph_header(bytes(outside))
    lines.append(f"flip byte after CRC -> still parses: {parsed_out is not None}")
    ok = ok and parsed_out is not None

    text = "\n".join(lines)
    (out / "a1_report.txt").write_text(text + f"\n{'OK' if ok else 'FAIL'}\n", encoding="utf-8")
    print(text)
    print("期望: chirp/train/lfsr 差值为 0，NumPy RNG 与 CPython 不同，改文件名内字节解析失败，改 CRC 之后的补零仍能解析")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
