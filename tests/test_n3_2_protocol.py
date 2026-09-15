import hashlib

import numpy as np

from n3_2.modem import (
    CP,
    FS,
    L,
    M,
    N,
    header_bytes,
    linear_chirp,
    ofdm_time,
    parse_header,
    qpsk_map,
    qpsk_llr,
    read_pcm16_wav,
    scramble_bits,
    training_symbols,
    write_pcm16_wav,
)


def test_reference_golden_vectors():
    raw = header_bytes("x.bin", 5, 1)
    assert hashlib.sha256(raw).hexdigest() == (
        "e158f3dff4d6c8b8494c3e9cc48ebd03cdbcb4350d45e9dbb991cc6ce7793a4e"
    )
    assert raw[:18].hex() == "5048000000000500000001782e62696e00d0"
    assert "".join(map(str, scramble_bits(np.zeros(32, np.uint8)))) == "10110100100110111011101101011001"
    t = training_symbols()
    assert t.shape == (16, 8192)
    assert hashlib.sha256(t.tobytes()).hexdigest() == "1eaf58db08942730d41a049d5f77b1ba9e0e672d225773b69530538887f36404"


def test_qpsk_cp_and_chirp():
    assert np.array_equal(qpsk_map(np.array([0, 0, 0, 1], np.uint8)), np.array([1+1j, -1+1j]))
    x = ofdm_time(np.ones((1, M), np.complex128))
    assert x.shape == (1, L)
    assert np.allclose(x[0, :CP], x[0, -CP:])
    assert linear_chirp().shape == (3 * FS,)


def test_header_round_trip_and_qpsk_llr():
    raw = header_bytes("测试.bin", 123, 17)
    parsed = parse_header(raw)
    assert parsed == {"name": "测试.bin", "size": 123, "payload_symbols": 17}
    assert np.array_equal(qpsk_llr(np.array([1+1j, -1-1j]), 2.0), np.array([1., 1., -1., -1.]))
    assert np.array_equal(qpsk_llr(np.array([-1+1j]), 1.0), np.array([2., -2.]))


def test_wav_round_trip(tmp_path):
    path = tmp_path / "x.wav"
    samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    write_pcm16_wav(path, samples)
    rate, got = read_pcm16_wav(path)
    assert rate == FS
    assert np.array_equal(got, samples)
