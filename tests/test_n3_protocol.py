import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from n3.modem import (
    ACTIVE_BINS,
    CHIRP_SAMPLES,
    CODE_BYTES,
    CP,
    INFO_BYTES,
    L,
    N,
    SILENCE_SAMPLES,
    coded_symbol_count,
    chirp_wave,
    header_bytes,
    ofdm_rx,
    ofdm_tx,
    parse_header,
    payload_symbol_count,
    qpsk_from_bits,
    qpsk_to_bits,
    read_wav,
    scrambler_sequence,
    training_symbols,
    write_wav,
)


class HeaderTests(unittest.TestCase):
    def test_header_matches_reference_hash(self):
        raw = header_bytes("x.bin", 5, 1)
        self.assertEqual(len(raw), INFO_BYTES)
        self.assertEqual(raw[:18].hex(), "5048000000000500000001782e62696e00d0")
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "e158f3dff4d6c8b8494c3e9cc48ebd03cdbcb4350d45e9dbb991cc6ce7793a4e",
        )
        parsed = parse_header(raw)
        self.assertEqual(parsed["name"], "x.bin")
        self.assertEqual(parsed["file_size"], 5)
        self.assertEqual(parsed["payload_symbols"], 1)

    def test_header_rejects_bad_crc(self):
        raw = bytearray(header_bytes("x.bin", 5, 1))
        raw[3] ^= 1
        with self.assertRaisesRegex(ValueError, "header CRC mismatch"):
            parse_header(bytes(raw))


class DeterministicTests(unittest.TestCase):
    def test_scrambler_matches_reference(self):
        seq = scrambler_sequence(3984)
        self.assertEqual("".join(map(str, seq[:32])), "10110100100110111011101101011001")
        self.assertEqual(
            hashlib.sha256(seq.tobytes()).hexdigest(),
            "5d40678d55efd4b5253b42d3ef59dc72dff9ab0bad52fd4890fdbfd21ffa77e8",
        )

    def test_training_matches_reference(self):
        freq, time_cp = training_symbols()
        self.assertEqual(freq.shape, (16, N))
        self.assertEqual(time_cp.shape, (16, L))
        self.assertEqual(
            hashlib.sha256(freq.tobytes()).hexdigest(),
            "1eaf58db08942730d41a049d5f77b1ba9e0e672d225773b69530538887f36404",
        )
        self.assertEqual(
            hashlib.sha256(time_cp.tobytes()).hexdigest(),
            "d27491ce1f274ec383246d89ab74217f46574dabe0c8e35695199866f2567755",
        )

    def test_qpsk_mapping_is_big_endian_and_invertible(self):
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols = qpsk_from_bits(bits)
        np.testing.assert_array_equal(symbols, np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]))
        np.testing.assert_array_equal(qpsk_to_bits(symbols), bits)

    def test_ofdm_round_trip_and_cp(self):
        active = np.zeros((1, len(ACTIVE_BINS)), complex)
        active[0, :4] = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]
        waveform = ofdm_tx(active)
        self.assertEqual(waveform.shape, (L,))
        freq = ofdm_rx(waveform)
        np.testing.assert_allclose(freq, active, atol=1e-9)
        full = np.zeros(N, complex)
        full[ACTIVE_BINS] = active[0]
        full[N - ACTIVE_BINS] = np.conj(active[0])
        time = np.fft.ifft(full).real
        np.testing.assert_allclose(waveform[:CP], time[-CP:])

    def test_chirp_and_counts(self):
        chirp = chirp_wave()
        self.assertEqual(len(chirp), CHIRP_SAMPLES)
        self.assertEqual(SILENCE_SAMPLES, 24000)
        np.testing.assert_array_equal(chirp, chirp_wave())
        for size, payload, coded in ((0, 0, 2), (1, 1, 2), (249, 1, 2), (250, 2, 4)):
            self.assertEqual(payload_symbol_count(size), payload)
            self.assertEqual(coded_symbol_count(size), coded)

    def test_wav_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            samples = np.linspace(-0.7, 0.7, 97)
            write_wav(path, samples)
            recovered = read_wav(path)
            np.testing.assert_allclose(recovered, samples, atol=2 / 32767)


if __name__ == "__main__":
    unittest.main()
