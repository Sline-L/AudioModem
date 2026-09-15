import json

from n3_2.modem import FS, L, read_pcm16_wav
from n3_2.tx import run_tx


def test_tx_frame_layout_and_sidecars(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"abc")

    out = run_tx(src, tmp_path / "a.wav")

    fs, samples = read_pcm16_wav(out)
    assert fs == FS
    assert len(samples) == 2 * 3 * FS + FS + (8 + 2 + 8) * L
    assert out.with_suffix(".training.npy").exists()
    assert out.with_suffix(".header.bin").read_bytes()[:2] == b"PH"
    assert json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["payload_symbols"] == 1
