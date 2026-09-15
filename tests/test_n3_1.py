import json
import tempfile
import unittest
from pathlib import Path

from n3_1.ldpc_codec import get_code
from n3_1.modem import ACTIVE_BINS, CP, N
from n3_1.rx import recover_recording
from n3_1.sync import frame_candidates
from n3_1.tx import write_transmission


class N31CompatibilityTests(unittest.TestCase):
    def test_n3_1_exposes_the_fixed_standard_phy(self):
        self.assertEqual((N, CP), (8192, 2048))
        self.assertEqual((ACTIVE_BINS[0], ACTIVE_BINS[-1]), (400, 2391))
        self.assertEqual((get_code().standard, get_code().rate, get_code().z), ("802.16", "1/2", 166))
        self.assertTrue(callable(frame_candidates))

    def test_n3_1_round_trip_writes_header_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "message.bin"
            source.write_bytes(b"n3_1 header diagnostics")
            wave = root / "tx.wav"

            meta = write_transmission(source, wave)
            metrics = recover_recording(wave, root / "rx", source)

            self.assertEqual(meta["profile"], "n3_1_example_tx")
            self.assertTrue(metrics["header_ok"], metrics["error"])
            self.assertTrue(metrics["file_match"], metrics)
            debug = json.loads((root / "rx" / "header_debug.json").read_text(encoding="utf-8"))
            self.assertTrue(debug["header_available"])
            self.assertTrue(debug["header_parse_ok"])
            self.assertEqual(debug["expected_header_bit_errors"], 0)
            self.assertEqual(debug["expected_header_bit_error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
