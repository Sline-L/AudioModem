from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from tui.dsp import decode_file, encode_file
from tui.profiles import default_profile


class LoopbackTests(unittest.TestCase):
    def test_qpsk_rs_conv_loopback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.bin"
            source.write_bytes(bytes(range(251)))
            profile = replace(
                default_profile(), modulation="qpsk", fec="rs_conv",
                block_size=223, sync_symbols=32, preamble_symbols=32,
                preamble_repeats=2, training_anchor_step=16,
            )
            wav = root / "tx.wav"
            encode_file(source, wav, profile)
            metrics = decode_file(wav, root / "decode", profile, source)
            self.assertTrue(metrics["file_match"])
            self.assertEqual(metrics["coded_bit_error_rate"], 0.0)
