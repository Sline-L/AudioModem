import tempfile
import unittest
from pathlib import Path

import numpy as np

from n3.ber_test import inject_payload_bit_errors
from n3.tx import build_frame


class BerInjectionTests(unittest.TestCase):
    def test_injects_exact_payload_bit_count_and_preserves_frame_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "message.bin"
            source.write_bytes(b"hello")
            frame, meta, _ = build_frame(source)
            modified, injection = inject_payload_bit_errors(frame, meta["coded_symbols"], 0.10, seed=7)
            self.assertEqual(modified.shape, frame.shape)
            self.assertEqual(injection["injected_bit_errors"], round(0.10 * injection["payload_coded_bits"]))
            self.assertFalse(np.array_equal(modified, frame))
            self.assertEqual(injection["header_untouched"], True)
            self.assertEqual(injection["training_untouched"], True)


if __name__ == "__main__":
    unittest.main()
