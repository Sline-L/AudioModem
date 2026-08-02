from pathlib import Path

import numpy as np

from audiomodem import (
    FS,
    MODS,
    bits_per_symbol,
    bytes_from_mod,
    find_sync,
    mod_symbols,
    pack,
    probe_symbols,
    read_wav,
    unpack,
    wav_gain,
    write_wav,
)

PROFILE = "step4_trimmed_n1024"

N = 1024
CP = 256
L = N + CP

# Step 4 trimmed profile in its original N=1024 bins.
ACTIVE_BINS = np.r_[
    np.arange(128, 161, dtype=int),
    np.arange(164, 170, dtype=int),
    np.arange(172, 241, dtype=int),
    np.arange(315, 358, dtype=int),
]
PILOT_BINS = np.asarray(
    [
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
    dtype=int,
)
PILOT_IDX = np.flatnonzero(np.isin(ACTIVE_BINS, PILOT_BINS))
DATA_IDX = np.flatnonzero(~np.isin(ACTIVE_BINS, PILOT_BINS))
DATA_BINS = ACTIVE_BINS[DATA_IDX]


def ofdm_tx(symbols, k=ACTIVE_BINS):
    f = np.zeros((len(symbols), N), complex)
    f[:, k] = symbols
    f[:, N - k] = np.conj(symbols)
    x = np.fft.ifft(f, axis=1).real
    return np.c_[x[:, -CP:], x].ravel()


def ofdm_rx(samples, k=ACTIVE_BINS):
    n = len(samples) // L
    y = samples[: n * L].reshape(n, L)[:, CP:]
    return np.fft.fft(y, axis=1)[:, k]


def random_qpsk_symbols(k, n, seed):
    return probe_symbols("random", np.asarray(k), n, seed)


def training_blocks(k, symbols_per_repeat, repeats, seed):
    return [random_qpsk_symbols(k, symbols_per_repeat, seed + i) for i in range(repeats)]


def file_symbols_with_pilots(path, mod="bpsk", pilot_seed=2027):
    z = mod_symbols(pack(path), mod)
    n = int(np.ceil(len(z) / len(DATA_IDX)))
    payload = np.zeros((n, len(ACTIVE_BINS)), complex)
    padded = np.zeros(n * len(DATA_IDX), complex)
    padded[: len(z)] = z
    payload[:, DATA_IDX] = padded.reshape(n, len(DATA_IDX))
    pilots = random_qpsk_symbols(PILOT_BINS, n, pilot_seed)
    payload[:, PILOT_IDX] = pilots
    return payload, pilots


def estimate_h(y, x):
    n = min(len(y), len(x))
    if n == 0:
        return np.full(x.shape[1], np.nan + 1j * np.nan)
    return np.mean(y[:n] / x[:n], axis=0)


def phase_aligned_mean(h_blocks, reference=None):
    hs = np.asarray(h_blocks, dtype=complex)
    if len(hs) == 0:
        raise ValueError("at least one H estimate is required")
    ref = hs[-1] if reference is None else np.asarray(reference)
    aligned = []
    for h in hs:
        valid = np.isfinite(h) & np.isfinite(ref)
        phase = np.angle(np.vdot(h[valid], ref[valid])) if np.any(valid) else 0.0
        aligned.append(h * np.exp(1j * phase))
    return np.mean(aligned, axis=0), np.asarray(aligned)


def interp_h_from_pilots(h_pilot):
    kp = PILOT_BINS.astype(float)
    kd = DATA_BINS.astype(float)
    h = np.empty(len(ACTIVE_BINS), complex)
    h[PILOT_IDX] = h_pilot
    mag = np.interp(kd, kp, np.abs(h_pilot))
    phase = np.interp(kd, kp, np.unwrap(np.angle(h_pilot)))
    h[DATA_IDX] = mag * np.exp(1j * phase)
    return h


def decode_payload(y, pilots, h_initial, smooth=0.25):
    n = min(len(y), len(pilots))
    z = np.empty((n, len(DATA_IDX)), complex)
    hs = np.empty((n, len(ACTIVE_BINS)), complex)
    residuals = np.empty(n, float)
    h = np.asarray(h_initial, dtype=complex).copy()
    for i in range(n):
        h_instant = interp_h_from_pilots(y[i, PILOT_IDX] / pilots[i])
        if i == 0 or smooth >= 1:
            h = h_instant
        elif smooth > 0:
            h = (1.0 - smooth) * h + smooth * h_instant
        else:
            h = h_instant
        hs[i] = h
        z[i] = y[i, DATA_IDX] / h[DATA_IDX]
        residuals[i] = np.mean(np.abs(y[i, PILOT_IDX] / h[PILOT_IDX] - pilots[i]) ** 2)
    return z, hs, residuals


def profile_meta():
    return {
        "profile": PROFILE,
        "fft_size": N,
        "cp_samples": CP,
        "symbol_len": L,
        "symbol_seconds": L / FS,
        "active_ranges": [[128, 160], [164, 169], [172, 240], [315, 357]],
        "active_bins": [int(x) for x in ACTIVE_BINS],
        "data_bins": [int(x) for x in DATA_BINS],
        "pilot_bins": [int(x) for x in PILOT_BINS],
        "active_frequencies_hz": [float(ACTIVE_BINS[0] * FS / N), float(ACTIVE_BINS[-1] * FS / N)],
        "data_bin_count": int(len(DATA_IDX)),
        "pilot_bin_count": int(len(PILOT_IDX)),
    }


__all__ = [
    "ACTIVE_BINS",
    "CP",
    "DATA_BINS",
    "DATA_IDX",
    "FS",
    "L",
    "MODS",
    "N",
    "PILOT_BINS",
    "PILOT_IDX",
    "Path",
    "bits_per_symbol",
    "bytes_from_mod",
    "decode_payload",
    "estimate_h",
    "file_symbols_with_pilots",
    "find_sync",
    "ofdm_rx",
    "ofdm_tx",
    "phase_aligned_mean",
    "profile_meta",
    "read_wav",
    "training_blocks",
    "unpack",
    "wav_gain",
    "write_wav",
]
