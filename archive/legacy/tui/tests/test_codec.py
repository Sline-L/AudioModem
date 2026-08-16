from dataclasses import replace
import unittest

import numpy as np

from tui.codec import encoded_bit_length, fec_decode, fec_encode
from tui.profiles import default_profile


class CodecTests(unittest.TestCase):
    def test_all_fec_modes_roundtrip(self):
        raw = bytes(range(256)) + b"audio-modem"
        for mode in ("none", "conv", "rs", "rs_conv"):
            with self.subTest(mode=mode):
                encoded = fec_encode(raw, mode, 1234)
                llr = np.where(encoded == 0, 8.0, -8.0)
                self.assertEqual(len(encoded), encoded_bit_length(len(raw), mode))
                self.assertEqual(fec_decode(llr, len(raw), mode, 1234), raw)

    def test_rs_corrects_byte_errors(self):
        raw = bytes(range(200))
        encoded = fec_encode(raw, "rs", 88)
        order = np.random.default_rng(88).permutation(len(encoded))
        deinterleaved = np.empty_like(encoded); deinterleaved[order] = encoded
        for byte in range(12):
            begin = byte * 8
            deinterleaved[begin : begin + 8] ^= 1
        damaged = deinterleaved[order]
        llr = np.where(damaged == 0, 8.0, -8.0)
        self.assertEqual(fec_decode(llr, len(raw), "rs", 88), raw)
