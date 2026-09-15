import tempfile
import unittest
from pathlib import Path

import numpy as np

from n3.modem import ACTIVE_BINS, CHIRP_SAMPLES, SILENCE_SAMPLES, chirp_wave, ofdm_tx, training_symbols
from n3.sync import (
    equalize,
    estimate_channel,
    frame_candidates,
    normalized_correlation,
    phase_correct,
    refine_training_start,
)
from n3.tx import build_frame


class SyncTests(unittest.TestCase):
    def test_channel_estimate_recovers_known_response(self):
        training_freq, _ = training_symbols()
        known = training_freq[:8, ACTIVE_BINS]
        h = np.linspace(0.6, 1.4, len(ACTIVE_BINS)) * np.exp(1j * 0.2)
        received = known * h
        np.testing.assert_allclose(estimate_channel(received, known), h, atol=1e-12)
        np.testing.assert_allclose(equalize(received, h), known, atol=1e-12)

    def test_phase_correction_zero_epsilon_is_identity(self):
        values = np.ones((3, len(ACTIVE_BINS)), complex)
        starts = np.array([10.0, 10250.0, 20490.0])
        np.testing.assert_allclose(phase_correct(values, starts, 10.0, 0.0), values)

    def test_training_refinement_finds_expected_start(self):
        training_freq, _ = training_symbols()
        training_wave = ofdm_tx(training_freq[:8, ACTIVE_BINS])
        samples = np.r_[np.zeros(123), training_wave, np.zeros(100)]
        result = refine_training_start(samples, 123, training_wave)
        self.assertEqual(result["training_start"], 123)
        self.assertGreater(result["training_score"], 0.99)

    def test_delay_and_gain_produce_a_valid_frame_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "x.bin"
            source.write_bytes(b"hello")
            frame, _, _ = build_frame(source)
            delayed = np.r_[np.zeros(1379), frame * 0.23]
            candidates = frame_candidates(delayed, chirp_wave(), ofdm_tx(training_symbols()[0][:8, ACTIVE_BINS]))
            self.assertTrue(candidates)
            best = candidates[0]
            self.assertLessEqual(abs(best["front_start"] - 1379), 2)
            self.assertEqual(best["data_symbols"], 2)
            self.assertLess(abs(best["coarse_sfo_ppm"]), 0.1)

    def test_correlation_returns_one_for_an_exact_template(self):
        chirp = chirp_wave()
        samples = np.r_[np.zeros(41), chirp, np.zeros(17)]
        score = normalized_correlation(samples, chirp)
        self.assertEqual(int(np.argmax(score)), 41)
        self.assertGreater(score[41], 0.999999)


if __name__ == "__main__":
    unittest.main()
