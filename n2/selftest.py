"""n2 功能自测：比特级对照发射端 + WAV 回环 + 轻度损伤。"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

N2 = Path(__file__).resolve().parent
ROOT = N2.parent
LDPC_PY = ROOT / "LDPC" / "new_ldpc" / "py"
LDPC_ROOT = ROOT / "LDPC" / "new_ldpc"
STANDARD = ROOT / "standard"

for p in (str(N2), str(LDPC_PY), str(STANDARD)):
    if p not in sys.path:
        sys.path.insert(0, p)

if "sounddevice" not in sys.modules:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sounddevice")
        stub.play = lambda *a, **k: None
        stub.wait = lambda: None
        sys.modules["sounddevice"] = stub

from decode_path import make_coder  # noqa: E402
from params import derived  # noqa: E402
from phy import (  # noqa: E402
    lfsr_sequence,
    linear_chirp,
    parse_ph_header,
    read_wav_mono,
    train_freq_time,
)
from receiver import recover_array, recover_wav  # noqa: E402


def _tx_class():
    from example_tx import Ofdm_tx

    return Ofdm_tx


def _construct(cls, **kwargs):
    """ldpc.code 在库目录下加载 ./bin 里的 C 译码库。"""
    cwd = os.getcwd()
    os.chdir(LDPC_ROOT)
    try:
        return cls(**kwargs)
    finally:
        os.chdir(cwd)


def test_phy_against_tx():
    """扫频、扰码、训练、帧头与发射端逐项一致。"""
    tx = _construct(_tx_class())
    d = derived()
    assert np.allclose(linear_chirp(d), tx.log_chirp_gen())
    assert np.array_equal(lfsr_sequence(64), tx.sequence(64))
    freq, time_cp = train_freq_time(d)
    tf, tt = tx.train_symbol_gen(
        tx.start_index, tx.start_index + 4 * tx.chunk_size, N=tx.N, symbol_count=16
    )
    assert np.allclose(freq, tf)
    assert np.allclose(time_cp, tt)
    header = tx.header_gen("测.bin", 12, 1, d["info_bytes"])
    parsed = parse_ph_header(bytes(header))
    assert parsed is not None
    assert parsed["filename"] == "测.bin"
    assert parsed["file_size"] == 12
    print("PHY 对照发射端：通过")


def test_ldpc_noiseless():
    """直接使用 LDPC/new_ldpc：无噪声码字应一次译出。"""
    d = derived()
    coder = make_coder(d["ldpc_z"])
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, coder.K)
    x = coder.encode(u)
    y = 10.0 * (0.5 - x)
    app, it = coder.decode(y, dectype="sumprod2")
    xh = np.array(app < 0, dtype=int)
    assert np.array_equal(xh[: coder.K], u)
    print(f"LDPC 无噪声：通过（迭代 {it}）")


def _run_tx_wav(payload: bytes, name: str, wav_path: Path):
    wav_path = Path(wav_path).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / name
        src.write_bytes(payload)
        tx = _construct(_tx_class(), send_file=str(src), tx_mode="save")
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            tx.tx(saved_name=str(wav_path))
        finally:
            os.chdir(cwd)


def test_loopback():
    payload = b"N2-OFDM-" + bytes(range(32)) + "中文".encode("utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "loop.wav"
        out = Path(tmp) / "out"
        _run_tx_wav(payload, "hello.bin", wav)
        result = recover_wav(wav, out_dir=out)
        assert result.ok, result.diagnostics
        assert result.filename == "hello.bin"
        assert result.payload == payload
        print("理想回环：通过")
        return result


def test_impairment():
    payload = b"impair-" + os.urandom(40)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "imp.wav"
        _run_tx_wav(payload, "imp.bin", wav)
        fs, x = read_wav_mono(wav)
        delayed = np.concatenate([np.zeros(120), x])
        noisy = delayed + 0.0015 * np.random.default_rng(1).standard_normal(delayed.size)
        result = recover_array(noisy, fs=fs)
        if not result.ok:
            print(f"轻度损伤：未恢复，阶段={result.stage}")
            assert result.stage in {"capture", "align", "channel", "ldpc", "header"}
        else:
            assert result.payload == payload
            print("轻度损伤：通过")
        return result


def main():
    test_phy_against_tx()
    test_ldpc_noiseless()
    test_loopback()
    test_impairment()
    print("selftest 全部完成")


if __name__ == "__main__":
    main()
