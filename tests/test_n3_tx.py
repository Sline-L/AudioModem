import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from n3.modem import (
    ACTIVE_BINS,
    CHIRP_SAMPLES,
    FS,
    INFO_BYTES,
    L,
    SILENCE_SAMPLES,
    chirp_wave,
    training_symbols,
)
from n3.tx import build_frame, information_blocks, write_transmission


class TransmitterTests(unittest.TestCase):
    def make_file(self, directory: str, name: str, data: bytes) -> Path:
        path = Path(directory) / name
        path.write_bytes(data)
        return path

    def test_information_blocks_match_reference_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, "x.bin", b"hello")
            blocks, meta = information_blocks(path)
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0], meta["header"])
            self.assertEqual(blocks[1][:5], b"hello")
            self.assertEqual(blocks[1][5:], bytes(244))
            self.assertEqual(meta["payload_information_symbols"], 1)

    def test_250_byte_payload_has_reference_filler_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, "x.bin", bytes(range(250)))
            blocks, _ = information_blocks(path)
            self.assertEqual(len(blocks), 4)
            self.assertEqual(blocks[-1], bytes(249))

    def test_frame_layout_and_standard_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_file(directory, "x.bin", b"hello")
            frame, meta, training = build_frame(path)
            chirp = chirp_wave()
            self.assertEqual(meta["sample_rate"], FS)
            self.assertEqual(meta["active_bin_start"], 400)
            self.assertEqual(meta["active_bin_end"], 2391)
            self.assertEqual(meta["active_subcarriers"], len(ACTIVE_BINS))
            self.assertEqual(meta["training_symbols"], 8)
            self.assertEqual(meta["trailing_training_symbols"], 8)
            self.assertEqual(meta["coded_symbols"], 2)
            self.assertEqual(len(frame), meta["total_samples"])
            np.testing.assert_array_equal(frame[:CHIRP_SAMPLES], chirp)
            self.assertEqual(meta["training_start_sample"], CHIRP_SAMPLES + SILENCE_SAMPLES)
            self.assertEqual(meta["header_start_sample"], CHIRP_SAMPLES + SILENCE_SAMPLES + 8 * L)
            self.assertEqual(meta["tail_chirp_start_sample"], len(frame) - CHIRP_SAMPLES)
            self.assertEqual(training.shape, (16, 8192))
            self.assertEqual(meta["total_samples"] % 1, 0)

    def test_write_transmission_sidecars_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_file(directory, "x.bin", b"hello")
            output = Path(directory) / "tx.wav"
            meta = write_transmission(source, output)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".training.npy").exists())
            self.assertTrue(output.with_suffix(".header.bin").exists())
            self.assertTrue(output.with_suffix(".meta.json").exists())
            self.assertEqual(json.loads(output.with_suffix(".meta.json").read_text())["profile"], "n3_example_tx")
            self.assertEqual(meta["input_bytes"], 5)
            result = subprocess.run(
                [sys.executable, "-m", "n3.tx", str(source), "--out", str(Path(directory) / "cli.wav")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            unsupported = subprocess.run(
                [sys.executable, "-m", "n3.tx", str(source), "--mod", "qpsk", "--out", str(Path(directory) / "bad.wav")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(unsupported.returncode, 0)


if __name__ == "__main__":
    unittest.main()
