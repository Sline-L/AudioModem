# -*- coding: utf-8 -*-
"""功能自测：TX→WAV→RX 环回、文件头校验、轻度损伤。"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
N1 = Path(__file__).resolve().parent
sys.path.insert(0, str(N1))
sys.path.insert(0, str(ROOT / "standard"))

# 注入仓库课程 LDPC（LDPC/new_ldpc），供 standard/example_tx 使用
import course_ldpc

course_ldpc.install_as_ldpc_module()

from example_tx import Ofdm_tx  # noqa: E402
from ofdm_rx import Ofdm_rx  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _write_payload(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run_ideal_loopback(workdir: Path) -> dict:
    """理想环回：字节级一致。"""
    payload = (
        b"OFDM-n1-selftest:"
        + bytes(range(64))
        + "中文文件名载荷测试".encode("utf-8")
        + np.random.bytes(200)
    )
    src = workdir / "payload.bin"
    _write_payload(src, payload)
    wav_path = workdir / "tx_ideal.wav"
    out_dir = workdir / "rx_ideal"

    tx = Ofdm_tx(tx_mode="save", send_file=str(src))
    # TX 会写 send.tif 到 cwd；切换到 workdir
    cwd = Path.cwd()
    try:
        import os

        os.chdir(workdir)
        tx.tx(saved_name=str(wav_path))
    finally:
        os.chdir(cwd)

    rx = Ofdm_rx()
    recovered = rx.rx(str(wav_path), out_dir=str(out_dir))
    got = Path(recovered).read_bytes()
    header = rx.debug.get("header", {})
    ok = got == payload
    return {
        "name": "ideal",
        "pass": ok,
        "detail": f"sha_tx={_sha(payload)} sha_rx={_sha(got)} name={header.get('filename')} size={header.get('file_size')} K={rx.debug.get('K')}",
        "stage": rx.last_stage,
        "header_crc": header.get("crc_ok", False),
        "symbol_count_field": header.get("symbol_count"),
        "true_K": rx.debug.get("K"),
    }


def run_delay_impairment(workdir: Path, delay: int = 1234) -> dict:
    """前缀时延损伤。"""
    payload = b"delay-test-" + np.random.bytes(120)
    src = workdir / "payload_delay.bin"
    _write_payload(src, payload)
    wav_path = workdir / "tx_delay.wav"
    out_dir = workdir / "rx_delay"

    tx = Ofdm_tx(tx_mode="save", send_file=str(src))
    import os

    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        tx.tx(saved_name=str(wav_path))
    finally:
        os.chdir(cwd)

    fs, data = wavfile.read(str(wav_path))
    delayed = np.concatenate([np.zeros(delay, dtype=data.dtype), data])
    delayed_path = workdir / "tx_delay_shifted.wav"
    wavfile.write(str(delayed_path), fs, delayed)

    rx = Ofdm_rx()
    try:
        recovered = rx.rx(str(delayed_path), out_dir=str(out_dir))
        got = Path(recovered).read_bytes()
        ok = got == payload
        return {
            "name": "delay",
            "pass": ok,
            "detail": f"delay={delay} match={ok} stage={rx.last_stage}",
            "stage": rx.last_stage,
        }
    except Exception as e:
        return {
            "name": "delay",
            "pass": False,
            "detail": f"delay={delay} err={e}",
            "stage": rx.last_stage,
        }


def run_awgn_impairment(workdir: Path, snr_db: float = 25.0) -> dict:
    """加性高斯噪声。"""
    payload = b"awgn-test-" + np.random.bytes(120)
    src = workdir / "payload_awgn.bin"
    _write_payload(src, payload)
    wav_path = workdir / "tx_awgn.wav"
    out_dir = workdir / "rx_awgn"

    tx = Ofdm_tx(tx_mode="save", send_file=str(src))
    import os

    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        tx.tx(saved_name=str(wav_path))
    finally:
        os.chdir(cwd)

    fs, data = wavfile.read(str(wav_path))
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x = x / 32768.0
    p = np.mean(x ** 2) + 1e-12
    sigma = np.sqrt(p / (10 ** (snr_db / 10)))
    y = x + sigma * np.random.randn(len(x))
    y_i16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    noisy_path = workdir / "tx_awgn_noisy.wav"
    wavfile.write(str(noisy_path), fs, y_i16)

    rx = Ofdm_rx()
    try:
        recovered = rx.rx(str(noisy_path), out_dir=str(out_dir))
        got = Path(recovered).read_bytes()
        ok = got == payload
        return {
            "name": "awgn",
            "pass": ok,
            "detail": f"snr_db={snr_db} match={ok} stage={rx.last_stage}",
            "stage": rx.last_stage,
        }
    except Exception as e:
        return {
            "name": "awgn",
            "pass": False,
            "detail": f"snr_db={snr_db} err={e}",
            "stage": getattr(rx, "last_stage", "unknown"),
        }


def main():
    np.random.seed(0)
    workdir = Path(tempfile.mkdtemp(prefix="ofdm_n1_"))
    print(f"工作目录: {workdir}")
    results = []
    results.append(run_ideal_loopback(workdir))
    results.append(run_delay_impairment(workdir))
    results.append(run_awgn_impairment(workdir, snr_db=25.0))

    all_pass = True
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print(f"[{status}] {r['name']}: {r['detail']}")

    # 头字段 symbol_count 与真实 K 的说明性检查（理想用例）
    ideal = results[0]
    if ideal.get("symbol_count_field") is not None and ideal.get("true_K") is not None:
        print(
            f"[INFO] header.symbol_count={ideal['symbol_count_field']} true_K={ideal['true_K']} "
            f"(二者可不一致；RX 不以 symbol_count 为界)"
        )

    if not all_pass:
        print("功能校验失败")
        raise SystemExit(1)
    print("功能校验全部通过")


if __name__ == "__main__":
    main()
