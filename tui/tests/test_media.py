from pathlib import Path
from io import BytesIO
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from tui.media import parse_wpctl_status, record, wav_seconds


class MediaTests(unittest.TestCase):
    def test_wpctl_parser(self):
        sample = """
Audio
 ├─ Sinks:
 │  * 45. Built-in Audio Analog Stereo [vol: 0.60]
 ├─ Sources:
 │    52. USB Microphone Mono [vol: 1.00]
"""
        devices = parse_wpctl_status(sample)
        self.assertEqual(devices["sinks"][0]["id"], "45")
        self.assertEqual(devices["sources"][0]["name"], "USB Microphone Mono")

    def test_wav_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(48000)
                wav.writeframes(b"\0\0" * 24000)
            self.assertEqual(wav_seconds(path), 0.5)

    def test_simulated_capture_metrics_and_wav(self):
        class FakeProcess:
            def __init__(self, raw):
                self.stdout = BytesIO(raw); self.stderr = BytesIO(); self.returncode = 0
            def wait(self, timeout=None): return 0
            def poll(self): return self.returncode
            def terminate(self): self.returncode = -15

        samples = np.array([0, 16384, -32768, 8192] * 120, dtype="<i2")
        fake = FakeProcess(samples.tobytes())
        with tempfile.TemporaryDirectory() as folder, patch("tui.media.subprocess.Popen", return_value=fake):
            path = Path(folder) / "recording.wav"
            result = record(path, len(samples) / 48000)
            self.assertAlmostEqual(result["peak"], 1.0)
            self.assertEqual(result["clipping_samples"], 120)
            self.assertAlmostEqual(wav_seconds(path), len(samples) / 48000)
