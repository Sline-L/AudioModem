import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from n3.rx import adaptive_llr_scale, adaptive_llr_scales, recover_recording
from n3.tx import write_transmission


class ReceiverTests(unittest.TestCase):
    def make_file(self, directory: str, name: str, data: bytes) -> Path:
        path = Path(directory) / name
        path.write_bytes(data)
        return path

    def test_adaptive_llr_scale_preserves_clean_and_reduces_noisy_confidence(self):
        self.assertEqual(adaptive_llr_scale(0.0), 3.0)
        self.assertLess(adaptive_llr_scale(2.0), 3.0)
        self.assertGreaterEqual(adaptive_llr_scale(100.0), 0.5)
        scales = adaptive_llr_scales(np.ones(4), np.ones(4) * 2.0)
        self.assertEqual(scales.shape, (4,))
        self.assertTrue(np.all(np.isfinite(scales)))

    def test_clean_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_file(directory, "message.bin", bytes(range(251)))
            wav = root / "tx.wav"
            write_transmission(source, wav)
            metrics = recover_recording(wav, root / "rx", source)
            self.assertTrue(metrics["header_ok"], metrics["error"])
            self.assertTrue(metrics["file_match"], metrics)
            self.assertEqual((root / "rx" / "message.bin").read_bytes(), source.read_bytes())
            self.assertEqual(metrics["standard_profile"], "n3_example_tx")
            self.assertTrue(metrics["ber_available"])
            self.assertEqual(metrics["ber_header_bit_errors"], 0)
            self.assertEqual(metrics["ber_header_ber"], 0.0)
            self.assertEqual(metrics["ber_ldpc_off"], 0.0)
            self.assertEqual(metrics["ber_ldpc_on"], 0.0)
            self.assertEqual(metrics["ber_overall_useful_ber"], 0.0)
            saved = json.loads((root / "rx" / "metrics.json").read_text())
            self.assertEqual(saved["payload_bit_errors"], 0)

    def test_gain_and_delay_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_file(directory, "message.bin", b"hello")
            wav = root / "tx.wav"
            write_transmission(source, wav)
            with wave.open(str(wav), "rb") as handle:
                frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(float) / 32768.0
                rate = handle.getframerate()
            impaired = np.r_[np.zeros(1379), frames * 0.23]
            impaired_path = root / "impaired.wav"
            with wave.open(str(impaired_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(np.clip(impaired * 32767, -32768, 32767).astype("<i2").tobytes())
            metrics = recover_recording(impaired_path, root / "rx", source)
            self.assertTrue(metrics["header_ok"], metrics["error"])
            self.assertTrue(metrics["file_match"], metrics)

    def test_light_seeded_noise_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_file(directory, "message.bin", b"hello")
            wav = root / "tx.wav"
            write_transmission(source, wav)
            with wave.open(str(wav), "rb") as handle:
                frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(float) / 32768.0
                rate = handle.getframerate()
            rng = np.random.default_rng(20260913)
            noisy = frames + rng.normal(0.0, 0.01 * np.sqrt(np.mean(frames * frames)), len(frames))
            noisy_path = root / "noisy.wav"
            with wave.open(str(noisy_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(np.clip(noisy * 32767, -32768, 32767).astype("<i2").tobytes())
            metrics = recover_recording(noisy_path, root / "rx", source)
            self.assertTrue(metrics["header_ok"], metrics)
            self.assertTrue(metrics["file_match"], metrics)

    def test_truncated_recording_does_not_publish_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_file(directory, "message.bin", b"hello")
            wav = root / "tx.wav"
            write_transmission(source, wav)
            with wave.open(str(wav), "rb") as handle:
                frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
                rate = handle.getframerate()
            truncated = root / "truncated.wav"
            with wave.open(str(truncated), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(rate)
                handle.writeframes(frames[:-10240].tobytes())
            out = root / "rx"
            metrics = recover_recording(truncated, out, source)
            self.assertFalse(metrics["header_ok"])
            self.assertFalse((out / "message.bin").exists())

    def test_invalid_wav_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "bad.wav"
            with wave.open(str(invalid), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(struct.pack("<hhhh", 0, 0, 0, 0))
            metrics = recover_recording(invalid, root / "rx")
            self.assertFalse(metrics["header_ok"])
            self.assertIn("expected mono 16-bit", metrics["error"])
            self.assertTrue((root / "rx" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
