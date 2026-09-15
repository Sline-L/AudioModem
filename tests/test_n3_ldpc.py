import unittest

import numpy as np

from n3.ldpc_codec import decode_block, encode_block, get_code, llr_from_qpsk
from n3.modem import qpsk_from_bits, scrambler_sequence


class LdpcTests(unittest.TestCase):
    def test_parameters_and_zero_syndrome(self):
        code = get_code()
        self.assertEqual((code.standard, code.rate, code.z), ("802.16", "1/2", 166))
        self.assertEqual((code.K, code.N), (1992, 3984))
        info = bytes(range(249))
        scrambled = encode_block(info)
        coded = scrambled ^ scrambler_sequence(3984)
        syndrome = np.mod(coded @ code.pcmat().T, 2)
        self.assertFalse(np.any(syndrome))

    def test_noiseless_round_trip(self):
        info = bytes(range(249))
        scrambled = encode_block(info)
        llr = np.where(scrambled == 0, 12.0, -12.0)
        decoded, iterations = decode_block(llr)
        self.assertEqual(decoded, info)
        self.assertGreaterEqual(iterations, 0)

    def test_qpsk_llr_order(self):
        symbols = qpsk_from_bits(np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8))
        llr = llr_from_qpsk(symbols, scale=1.0)
        np.testing.assert_allclose(llr, np.array([1, 1, 1, -1, -1, 1, -1, -1], float))

    def test_qpsk_llr_accepts_per_carrier_scales(self):
        symbols = qpsk_from_bits(np.array([0, 0, 0, 1], dtype=np.uint8))
        llr = llr_from_qpsk(symbols, scale=np.array([2.0, 3.0]))
        np.testing.assert_allclose(llr, np.array([2, 2, 3, -3], float))


if __name__ == "__main__":
    unittest.main()
