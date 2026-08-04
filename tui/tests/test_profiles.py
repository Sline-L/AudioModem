from dataclasses import replace
import unittest

from tui.profiles import default_profile, parse_ranges


class ProfileTests(unittest.TestCase):
    def test_parse_ranges_and_frequency_mapping(self):
        profile = replace(default_profile(), active_ranges=parse_ranges("64-120, 158-178"))
        profile.validate()
        self.assertEqual(profile.active_bins[0], 64)
        self.assertEqual(profile.frequency_ranges_hz[0], [6000.0, 11250.0])

    def test_receiver_tuning_does_not_change_air_profile_id(self):
        profile = default_profile()
        tuned = replace(profile, phase_slope="slow", anchor_h_alpha=0.0, clock_search=64)
        self.assertEqual(profile.profile_id, tuned.profile_id)

    def test_invalid_profiles_are_rejected(self):
        cases = (
            replace(default_profile(), fft_size=500),
            replace(default_profile(), cp_samples=513),
            replace(default_profile(), active_ranges=[[0, 12]]),
            replace(default_profile(), active_ranges=[[64, 120], [100, 130]]),
        )
        for profile in cases:
            with self.subTest(profile=profile):
                with self.assertRaises(ValueError):
                    profile.validate()
