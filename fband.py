import numpy as np

from audiomodem import FS, mod_symbols, pack, probe_symbols


FBAND_PROFILES = {
    "conservative": {
        "ranges": [(128, 160), (164, 240), (315, 400)],
        "pilot_bins": [
            128,
            136,
            144,
            152,
            160,
            164,
            172,
            180,
            188,
            196,
            204,
            212,
            220,
            228,
            236,
            240,
            315,
            323,
            331,
            339,
            347,
            355,
            363,
            371,
            379,
            387,
            395,
            400,
        ],
    },
    "trimmed": {
        "ranges": [(128, 160), (164, 169), (172, 240), (315, 357)],
        "pilot_bins": [
            128,
            136,
            144,
            152,
            160,
            164,
            169,
            172,
            180,
            188,
            196,
            204,
            212,
            220,
            228,
            236,
            240,
            315,
            323,
            331,
            339,
            347,
            355,
            357,
        ],
    },
}


def fband_profile(name):
    if name is None:
        return None
    if name not in FBAND_PROFILES:
        known = ", ".join(sorted(FBAND_PROFILES))
        raise ValueError(f"--fband-profile must be one of: {known}")
    profile = FBAND_PROFILES[name]
    k = np.asarray(
        [kk for a, b in profile["ranges"] for kk in range(a, b + 1)],
        dtype=int,
    )
    pilots = np.asarray(profile["pilot_bins"], dtype=int)
    if not np.all(np.isin(pilots, k)):
        raise ValueError(f"pilot bins for profile {name} must be inside active ranges")
    pilot_idx = np.flatnonzero(np.isin(k, pilots))
    data_idx = np.flatnonzero(~np.isin(k, pilots))
    return {
        "name": name,
        "ranges": profile["ranges"],
        "k": k,
        "pilot_bins": pilots,
        "pilot_idx": pilot_idx,
        "data_bins": k[data_idx],
        "data_idx": data_idx,
    }


def profile_meta(profile):
    if profile is None:
        return {
            "fband_profile": None,
            "active_ranges": None,
            "active_bins": None,
            "data_bins": None,
            "comb_pilot_bins": None,
        }
    return {
        "fband_profile": profile["name"],
        "active_ranges": [[int(a), int(b)] for a, b in profile["ranges"]],
        "active_bins": [int(x) for x in profile["k"]],
        "data_bins": [int(x) for x in profile["data_bins"]],
        "comb_pilot_bins": [int(x) for x in profile["pilot_bins"]],
        "data_bin_count": int(len(profile["data_idx"])),
        "comb_pilot_bin_count": int(len(profile["pilot_idx"])),
    }


def comb_pilot_symbols(n, pilot_bins, seed=2027, kind="random"):
    if n == 0 or len(pilot_bins) == 0:
        return np.zeros((n, len(pilot_bins)), complex)
    return probe_symbols(kind, np.asarray(pilot_bins), n, seed)


def file_symbols_with_comb_pilots(path, k, data_idx, pilot_idx, mod="bpsk", pilot_seed=2027, pilot_kind="random"):
    z = mod_symbols(pack(path), mod)
    n = int(np.ceil(len(z) / len(data_idx)))
    payload = np.zeros((n, len(k)), complex)
    padded = np.zeros(n * len(data_idx), complex)
    padded[: len(z)] = z
    payload[:, data_idx] = padded.reshape(n, len(data_idx))
    pilots = comb_pilot_symbols(n, k[pilot_idx], pilot_seed, pilot_kind)
    payload[:, pilot_idx] = pilots
    return payload, pilots


def interp_h_from_comb_pilots(k, h_pilot, pilot_idx, data_idx, ranges):
    h_data = np.empty(len(data_idx), complex)
    for a, b in ranges:
        in_seg = (k >= a) & (k <= b)
        seg_p = pilot_idx[in_seg[pilot_idx]]
        seg_d = data_idx[in_seg[data_idx]]
        if len(seg_p) == 0 or len(seg_d) == 0:
            continue
        kp = k[seg_p].astype(float)
        kd = k[seg_d].astype(float)
        hp = h_pilot[seg_p]
        mag = np.interp(kd, kp, np.abs(hp))
        phase = np.interp(kd, kp, np.unwrap(np.angle(hp)))
        h_data[np.searchsorted(data_idx, seg_d)] = mag * np.exp(1j * phase)
    return h_data


def freq_hz(k):
    return np.asarray(k) * FS / 1024
