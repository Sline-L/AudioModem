import importlib
import json
import wave
import zlib
from pathlib import Path

import numpy as np
import pytest

from n3_2.ldpc_codec import StandardLdpc
from n3_2.modem import FS, L, ofdm_time, qpsk_map, read_pcm16_wav, scramble_bits, write_pcm16_wav
from n3_2.tx import run_tx


def load_receiver():
    assert importlib.util.find_spec("n3_2.rx") is not None, "Header receiver is missing"
    return importlib.import_module("n3_2.rx")


@pytest.fixture
def standard(tmp_path):
    source = tmp_path / "sample.bin"
    source.write_bytes(b"abc" * 100)
    return source, run_tx(source, tmp_path / "standard.wav")


def diagnostics(out, stage):
    expected = {"metrics.json", "header_debug.json", "decoded_header.bin", "H.npy",
                "training_phase_fit.npy", "summary.csv"}
    assert expected <= {p.name for p in out.iterdir()}
    metrics = json.loads((out / "metrics.json").read_text())
    debug = json.loads((out / "header_debug.json").read_text())
    assert metrics["stage"] == stage
    assert metrics["header_ok"] == (stage == "header_ok")
    assert {"channel_index", "chirp_score", "training_score", "sfo"} <= metrics.keys()
    assert "header_ber" not in metrics
    assert (out / "summary.csv").read_text().count("\n") == 2
    return metrics, debug


def test_standard_header_and_success_artifacts(standard, tmp_path):
    receiver = load_receiver()
    source, wav = standard
    header, metrics = receiver.decode_header(wav)
    assert header == {"name": "sample.bin", "size": 300, "payload_symbols": 2}
    assert metrics["header_ok"] is True
    out = tmp_path / "received"
    recovered = receiver.run_rx(wav, out, source)
    assert recovered.read_bytes() == source.read_bytes()
    metrics, debug = diagnostics(out, "header_ok")
    assert debug["header_bit_errors"] == 0
    assert debug["header_ber"] == 0.0
    assert metrics["channel_index"] == 0
    assert np.load(out / "H.npy").shape == (1992,)
    assert np.load(out / "training_phase_fit.npy").shape == (8, 2)
    assert (out / "decoded_header.bin").read_bytes() == wav.with_suffix(".header.bin").read_bytes()
    assert (out / "decoded_payload.bin").read_bytes() == source.read_bytes()
    assert np.load(out / "payload_symbols.npy").shape == (2, 1992)


def test_nonstandard_rate_preserves_wav_format_stage(tmp_path):
    receiver = load_receiver()
    wav = tmp_path / "wrong_rate.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(100))
    with pytest.raises(receiver.DecodeError) as failure:
        receiver.decode_header(wav)
    assert failure.value.stage == "wav_format"
    out = tmp_path / "failed"
    with pytest.raises(receiver.DecodeError):
        receiver.run_rx(wav, out)
    _, debug = diagnostics(out, "wav_format")
    assert (out / "decoded_header.bin").read_bytes() == b""
    assert "header_ber" not in debug


@pytest.mark.parametrize("damage,stage", [("noise", "ldpc"), ("crc", "header_crc"),
                                         ("count", "header_fields"), ("utf8", "header_fields")])
def test_corrupt_header_has_specific_stage_and_diagnostics(standard, tmp_path, damage, stage):
    receiver = load_receiver()
    _, wav = standard
    _, samples = read_pcm16_wav(wav)
    first = 3 * FS + FS // 2 + 8 * L
    raw = bytearray(wav.with_suffix(".header.bin").read_bytes())
    if damage == "noise":
        symbol = np.random.default_rng(9).normal(0, 0.015, L)
    else:
        if damage == "crc":
            raw[11] ^= 1
        else:
            if damage == "count":
                raw[7:11] = (1).to_bytes(4, "big")
            else:
                raw[11] = 255
            end = raw.index(0, 11)
            raw[end + 1:end + 5] = zlib.crc32(raw[:end]).to_bytes(4, "big")
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
        symbol = ofdm_time(qpsk_map(scramble_bits(StandardLdpc().encode(bits))))
    samples[first:first + L] = np.rint(symbol * 32767).astype(np.int16)
    write_pcm16_wav(wav, samples)
    out = tmp_path / "failed"
    with pytest.raises(receiver.DecodeError) as failure:
        receiver.run_rx(wav, out)
    assert failure.value.stage == stage
    diagnostics(out, stage)
    assert len((out / "decoded_header.bin").read_bytes()) == 249


def test_mono_without_chirps_retains_chirp_stage(tmp_path):
    receiver = load_receiver()
    wav = tmp_path / "silence.wav"
    write_pcm16_wav(wav, np.zeros(FS, dtype=np.int16))
    with pytest.raises(receiver.DecodeError) as failure:
        receiver.run_rx(wav, tmp_path / "failed")
    assert failure.value.stage == "chirp"
    diagnostics(tmp_path / "failed", "chirp")


def test_stereo_selects_signal_channel(standard, tmp_path):
    receiver = load_receiver()
    _, wav = standard
    _, samples = read_pcm16_wav(wav)
    write_pcm16_wav(wav, np.column_stack((np.zeros_like(samples), samples)))
    out = tmp_path / "received"
    receiver.run_rx(wav, out)
    metrics, debug = diagnostics(out, "header_ok")
    assert metrics["channel_index"] == 1
    assert "header_ber" not in debug


def test_low_confidence_chirp_still_recovers_candidate_payload(standard, tmp_path):
    receiver = load_receiver()
    source, wav = standard
    _, samples = read_pcm16_wav(wav)
    # Keep the acquisition prefix but destroy the rest of the final marker.
    rear = len(samples) - 3 * FS
    samples[rear + FS // 10:] = 0
    write_pcm16_wav(wav, samples)
    out = tmp_path / "candidate"
    recovered = receiver.run_rx(wav, out, source)
    assert recovered.read_bytes() == source.read_bytes()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["verified"] is False
    assert metrics["strict_stage"] == "candidate"


def test_recorded_r1_recovers_with_independent_robust_receiver(tmp_path):
    receiver = load_receiver()
    wav = Path("data/n3_2/r1.wav")
    source = Path("data/source/cute.jpg")
    assert wav.exists() and source.exists()
    recovered = receiver.run_rx(wav, tmp_path / "r1", source)
    assert recovered.read_bytes() == source.read_bytes()


def test_recorded_r5_recovers_by_header_selected_candidate(tmp_path):
    receiver = load_receiver()
    wav = Path("data/n3_2/r5.wav")
    recovered = receiver.run_rx(wav, tmp_path / "r5")
    assert recovered.name == "duck_image_34k.tiff"
    assert recovered.stat().st_size == 34685


def test_recorded_r4_persists_all_candidate_diagnostics(tmp_path):
    receiver = load_receiver()
    out = tmp_path / "r4"
    with pytest.raises(receiver.DecodeError) as failure:
        receiver.run_rx(Path("data/n3_2/r4.wav"), out)
    assert failure.value.stage == "ldpc"
    debug = json.loads((out / "header_debug.json").read_text())
    assert len((out / "decoded_header.bin").read_bytes()) == 249
    assert len(debug["candidate_attempts"]) >= 4
