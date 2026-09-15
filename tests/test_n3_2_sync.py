import numpy as np
import pytest
from scipy.signal import resample_poly

from n3_2.modem import FS, K0, K1, M, linear_chirp, ofdm_time, training_symbols
from n3_2.sync import SyncError, choose_channel, synchronize


def known_standard_frame():
    training = training_symbols()
    front_training = ofdm_time(training[:8, K0:K1 + 1]).reshape(-1)
    rear_training = ofdm_time(training[8:, K0:K1 + 1]).reshape(-1)
    payload = ofdm_time(np.ones((1, M), dtype=np.complex128)).reshape(-1)
    silence = np.zeros(FS // 2)
    chirp = linear_chirp()
    return np.concatenate(
        (chirp, silence, front_training, payload, rear_training, silence, chirp)
    )


def test_stereo_selects_cleaner_channel_and_finds_training():
    clean = known_standard_frame()
    noisy = np.random.default_rng(4).normal(0, 0.3, len(clean))
    choice = choose_channel(np.column_stack([noisy, clean]))
    result = synchronize(choice.samples)
    assert choice.index == 1
    assert result.training_start == 3 * FS + FS // 2
    assert result.score > 0.9


def test_sync_reports_specific_chirp_failure():
    with pytest.raises(SyncError, match="chirp"):
        synchronize(np.zeros(10 * FS))


def test_sync_estimates_bounded_sample_frequency_offset():
    stretched = resample_poly(known_standard_frame(), 5001, 5000)
    result = synchronize(stretched)
    assert result.sfo == pytest.approx(0.0002, abs=3e-5)
