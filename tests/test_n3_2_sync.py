import numpy as np
import pytest
from scipy.signal import resample_poly

from n3_2.modem import FS, K0, K1, L, M, linear_chirp, ofdm_time, training_symbols
from n3_2.sync import SyncError, choose_channel, synchronize


def known_standard_frame(payload_symbols=1):
    training = training_symbols()
    front_training = ofdm_time(training[:8, K0:K1 + 1]).reshape(-1)
    rear_training = ofdm_time(training[8:, K0:K1 + 1]).reshape(-1)
    payload_symbol = ofdm_time(np.ones((1, M), dtype=np.complex128)).reshape(-1)
    payload = np.tile(payload_symbol, payload_symbols)
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


def test_long_frame_sfo_does_not_depend_on_unknown_payload_count():
    stretched = resample_poly(known_standard_frame(payload_symbols=200), 251, 250)
    result = synchronize(stretched)
    assert result.sfo == pytest.approx(0.004, abs=3e-5)


def test_sync_rejects_rear_chirp_with_only_acquisition_prefix():
    truncated = known_standard_frame()
    rear_start = len(truncated) - 3 * FS
    truncated[rear_start + FS // 10:] = 0.0
    with pytest.raises(SyncError) as caught:
        synchronize(truncated)
    assert caught.value.stage == "chirp"


def test_sync_rejects_rear_chirp_off_the_payload_symbol_grid():
    frame = known_standard_frame()
    rear_start = len(frame) - 3 * FS
    incompatible = np.insert(frame, rear_start, np.zeros(L // 2))
    with pytest.raises(SyncError) as caught:
        synchronize(incompatible)
    assert caught.value.stage == "chirp"


def test_sync_accepts_minimum_interval_with_fixed_rear_training():
    result = synchronize(known_standard_frame(payload_symbols=0))
    assert result.score > 0.9


def test_sync_accepts_negative_sfo_at_minimum_frame_interval():
    compressed = resample_poly(known_standard_frame(payload_symbols=0), 39999, 40000)
    result = synchronize(compressed)
    assert result.sfo == pytest.approx(-1 / 40000, abs=3e-6)


@pytest.mark.parametrize("advance", [1, 1000])
def test_sync_rejects_complete_rear_chirp_before_minimum_interval(advance):
    frame = known_standard_frame(payload_symbols=0)
    rear_start = len(frame) - 3 * FS
    shifted = np.delete(frame, slice(rear_start - advance, rear_start))
    with pytest.raises(SyncError) as caught:
        synchronize(shifted)
    assert caught.value.stage == "chirp"
