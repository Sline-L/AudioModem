from pathlib import Path
import struct
import zlib

import numpy as np

from audiomodem import FS, find_sync, read_wav, wav_gain, write_wav


N = 512
CP = 256
L = N + CP

PROFILE = "adaptive_fec_n512_cp256"
SEGMENTS = ((64, 120), (158, 178))
ACTIVE_BINS = np.concatenate([np.arange(a, b + 1, dtype=int) for a, b in SEGMENTS])

PILOT_SPACING = 4
PILOT_SEED = 2027
FEC_SEED = 7027
BLOCK_SIZE = 512
HEADER_SIZE = 128
HEADER_REPEATS = 3
MAGIC = b"AM7F"
VERSION = 1

CONSTRAINT = 7
GENERATORS = (0o171, 0o133)
NSTATES = 1 << (CONSTRAINT - 1)


def ofdm_tx(symbols, k=ACTIVE_BINS):
    f = np.zeros((len(symbols), N), complex)
    f[:, k] = symbols
    f[:, N - k] = np.conj(symbols)
    x = np.fft.ifft(f, axis=1).real
    return np.c_[x[:, -CP:], x].ravel()


def ofdm_rx(samples, k=ACTIVE_BINS):
    n = len(samples) // L
    if n == 0:
        return np.empty((0, len(k)), complex)
    x = samples[: n * L].reshape(n, L)[:, CP:]
    return np.fft.fft(x, axis=1)[:, k]


def random_qpsk(rows, bins, seed):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, (rows, len(bins), 2), dtype=np.uint8)
    return (np.where(b[:, :, 1], -1, 1) + 1j * np.where(b[:, :, 0], -1, 1)) / np.sqrt(2)


def training_symbols(rows, seed):
    return random_qpsk(rows, ACTIVE_BINS, seed)


def bits_from_bytes(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bytes_from_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits[: bits.size // 8 * 8], bitorder="big").tobytes()


def parity(value):
    return value.bit_count() & 1


def conv_encode(bits):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    src = np.r_[bits, np.zeros(CONSTRAINT - 1, dtype=np.uint8)]
    out = np.empty(src.size * len(GENERATORS), dtype=np.uint8)
    state = 0
    for i, bit in enumerate(src):
        reg = (state << 1) | int(bit)
        out[2 * i] = parity(reg & GENERATORS[0])
        out[2 * i + 1] = parity(reg & GENERATORS[1])
        state = reg & (NSTATES - 1)
    return out


def encoded_length(raw_bytes):
    return 2 * (raw_bytes * 8 + CONSTRAINT - 1)


def _incoming_trellis():
    prev = np.empty((NSTATES, 2), dtype=np.int16)
    signs = np.empty((NSTATES, 2, 2), dtype=float)
    for state in range(NSTATES):
        bit = state & 1
        for branch, old in enumerate((state >> 1, (state >> 1) | (NSTATES >> 1))):
            reg = (old << 1) | bit
            coded = [parity(reg & g) for g in GENERATORS]
            prev[state, branch] = old
            signs[state, branch] = 1.0 - 2.0 * np.asarray(coded)
    return prev, signs


INCOMING_PREV, INCOMING_SIGNS = _incoming_trellis()


def viterbi_decode(llr, raw_bits):
    llr = np.asarray(llr, dtype=float).ravel()
    steps = raw_bits + CONSTRAINT - 1
    if llr.size < 2 * steps:
        raise ValueError(f"need {2 * steps} coded LLRs, got {llr.size}")
    pairs = llr[: 2 * steps].reshape(steps, 2)
    metrics = np.full(NSTATES, -np.inf)
    metrics[0] = 0.0
    choices = np.empty((steps, NSTATES), dtype=np.uint8)
    for t, pair in enumerate(pairs):
        branch = metrics[INCOMING_PREV] + np.sum(INCOMING_SIGNS * pair, axis=2)
        choice = np.argmax(branch, axis=1)
        metrics = branch[np.arange(NSTATES), choice]
        choices[t] = choice
        finite = np.isfinite(metrics)
        if np.any(finite):
            metrics[finite] -= np.max(metrics[finite])
    state = 0
    decoded = np.empty(steps, dtype=np.uint8)
    for t in range(steps - 1, -1, -1):
        decoded[t] = state & 1
        state = int(INCOMING_PREV[state, choices[t, state]])
    return decoded[:raw_bits]


def interleave(values, seed):
    values = np.asarray(values)
    order = np.random.default_rng(seed).permutation(values.size)
    return values[order], order


def deinterleave(values, seed):
    values = np.asarray(values)
    order = np.random.default_rng(seed).permutation(values.size)
    out = np.empty_like(values)
    out[order] = values
    return out


def header_bytes(path, block_size=BLOCK_SIZE):
    path = Path(path)
    body = path.read_bytes()
    name = path.name.encode("utf-8")
    if len(name) > 100:
        raise ValueError("UTF-8 filename must be at most 100 bytes")
    blocks = (len(body) + block_size - 1) // block_size
    first = struct.pack(
        ">4sBBHQHHI100s",
        MAGIC,
        VERSION,
        len(name),
        0,
        len(body),
        block_size,
        blocks,
        zlib.crc32(body),
        name.ljust(100, b"\0"),
    )
    return first + struct.pack(">I", zlib.crc32(first))


def parse_header(raw):
    if len(raw) != HEADER_SIZE:
        raise ValueError("invalid Step7 header length")
    first, stored_crc = raw[:-4], struct.unpack(">I", raw[-4:])[0]
    if zlib.crc32(first) != stored_crc:
        raise ValueError("header CRC mismatch")
    magic, version, name_len, _, size, block_size, blocks, file_crc, name_raw = struct.unpack(
        ">4sBBHQHHI100s", first
    )
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported Step7 header")
    if name_len > len(name_raw) or block_size < 1:
        raise ValueError("invalid Step7 header fields")
    name = Path(name_raw[:name_len].decode("utf-8")).name
    expected_blocks = (size + block_size - 1) // block_size
    if blocks != expected_blocks:
        raise ValueError("inconsistent Step7 block count")
    return {
        "name": name,
        "file_size": int(size),
        "block_size": int(block_size),
        "block_count": int(blocks),
        "file_crc32": int(file_crc),
    }


def data_block_bytes(index, chunk, block_size=BLOCK_SIZE):
    if len(chunk) > block_size:
        raise ValueError("chunk exceeds block size")
    prefix = struct.pack(">HH", index, len(chunk))
    crc = zlib.crc32(prefix + chunk)
    return prefix + chunk.ljust(block_size, b"\0") + struct.pack(">I", crc)


def parse_data_block(raw, expected_index, block_size):
    if len(raw) != block_size + 8:
        raise ValueError("invalid data block length")
    index, size = struct.unpack(">HH", raw[:4])
    if index != expected_index or size > block_size:
        raise ValueError("invalid data block header")
    chunk = raw[4 : 4 + size]
    stored_crc = struct.unpack(">I", raw[-4:])[0]
    if zlib.crc32(raw[:4] + chunk) != stored_crc:
        raise ValueError("data block CRC mismatch")
    return chunk


def coded_frames(path, block_size=BLOCK_SIZE, header_repeats=HEADER_REPEATS, fec_seed=FEC_SEED):
    path = Path(path)
    header_coded = conv_encode(bits_from_bytes(header_bytes(path, block_size)))
    frames = []
    labels = []
    for repeat in range(header_repeats):
        frame, _ = interleave(header_coded, fec_seed + repeat)
        frames.append(frame)
        labels.append(f"header{repeat + 1}")
    body = path.read_bytes()
    for index, start in enumerate(range(0, len(body), block_size)):
        raw = data_block_bytes(index, body[start : start + block_size], block_size)
        coded = conv_encode(bits_from_bytes(raw))
        frame, _ = interleave(coded, fec_seed + 1000 + index)
        frames.append(frame)
        labels.append(f"block{index}")
    return frames, labels


def _segment_indices():
    return [np.flatnonzero((ACTIVE_BINS >= a) & (ACTIVE_BINS <= b)) for a, b in SEGMENTS]


SEGMENT_INDICES = _segment_indices()


def pilot_indices(symbol_index, spacing=PILOT_SPACING):
    offset = symbol_index % spacing
    parts = [idx[np.arange(len(idx)) % spacing == offset] for idx in SEGMENT_INDICES]
    return np.concatenate(parts)


def data_indices(symbol_index, spacing=PILOT_SPACING):
    mask = np.ones(len(ACTIVE_BINS), dtype=bool)
    mask[pilot_indices(symbol_index, spacing)] = False
    return np.flatnonzero(mask)


def pilot_values(symbol_index, indices, seed=PILOT_SEED):
    rng = np.random.default_rng(seed + symbol_index)
    b = rng.integers(0, 2, (len(ACTIVE_BINS), 2), dtype=np.uint8)
    values = (np.where(b[:, 1], -1, 1) + 1j * np.where(b[:, 0], -1, 1)) / np.sqrt(2)
    return values[indices]


def payload_symbols(coded_bits, pilot_seed=PILOT_SEED, spacing=PILOT_SPACING):
    bits = np.asarray(coded_bits, dtype=np.uint8).ravel()
    symbols = []
    used = []
    pos = 0
    symbol_index = 0
    while pos < len(bits):
        pidx = pilot_indices(symbol_index, spacing)
        didx = data_indices(symbol_index, spacing)
        take = min(len(didx), len(bits) - pos)
        s = np.zeros(len(ACTIVE_BINS), complex)
        s[pidx] = pilot_values(symbol_index, pidx, pilot_seed)
        s[didx[:take]] = 1.0 - 2.0 * bits[pos : pos + take]
        symbols.append(s)
        used.append(take)
        pos += take
        symbol_index += 1
    return np.asarray(symbols), np.asarray(used, dtype=int)


def phase_aligned_mean(h_blocks):
    hs = np.asarray(h_blocks, dtype=complex)
    ref = hs[-1]
    aligned = []
    for h in hs:
        weight = np.abs(h) * np.abs(ref)
        phase = np.angle(np.sum(weight * np.conj(h) * ref))
        aligned.append(h * np.exp(1j * phase))
    return np.mean(aligned, axis=0), np.asarray(aligned)


def estimate_training(y_blocks, x_blocks):
    hs = []
    variances = []
    for y, x in zip(y_blocks, x_blocks):
        n = min(len(y), len(x))
        h = np.mean(y[:n] / x[:n], axis=0)
        hs.append(h)
        variances.append(np.mean(np.abs(y[:n] - x[:n] * h) ** 2, axis=0))
    h, aligned = phase_aligned_mean(hs)
    return h, np.asarray(aligned), np.median(np.asarray(variances), axis=0)


def noise_variance(samples):
    y = ofdm_rx(samples)
    if len(y) == 0:
        return np.zeros(len(ACTIVE_BINS))
    return np.median(np.abs(y) ** 2, axis=0)


def _phase_ramp(observed_h, current_h, pidx):
    ratio = observed_h / np.where(np.abs(current_h[pidx]) > 1e-12, current_h[pidx], 1e-12)
    phase = np.unwrap(np.angle(ratio))
    x = ACTIVE_BINS[pidx].astype(float)
    center = float(np.average(x, weights=np.maximum(np.abs(current_h[pidx]) ** 2, 1e-12)))
    xc = x - center
    weights = np.maximum(np.abs(current_h[pidx]) ** 2, np.max(np.abs(current_h[pidx]) ** 2) * 1e-3)
    design = np.c_[np.ones(len(xc)), xc]
    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * phase)
    try:
        cpe, slope = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        cpe, slope = float(np.angle(np.sum(ratio))), 0.0
    slope = float(np.clip(slope, -0.2, 0.2))
    return float(cpe), slope, center


def payload_llrs(y, h_initial, training_var, silence_var, pilot_seed=PILOT_SEED, spacing=PILOT_SPACING, alpha=0.35):
    h = np.asarray(h_initial, dtype=complex).copy()
    base_var = np.maximum(np.asarray(training_var), np.asarray(silence_var))
    positive = base_var[base_var > 0]
    floor = float(np.median(positive)) if positive.size else 1e-8
    base_var = np.maximum(base_var, floor * 0.1)
    llrs = []
    symbol_bins = []
    h_track = []
    cpes = []
    slopes = []
    residuals = []
    for symbol_index, row in enumerate(y):
        pidx = pilot_indices(symbol_index, spacing)
        didx = data_indices(symbol_index, spacing)
        pilots = pilot_values(symbol_index, pidx, pilot_seed)
        observed_h = row[pidx] / pilots
        cpe, slope, center = _phase_ramp(observed_h, h, pidx)
        h *= np.exp(1j * (cpe + slope * (ACTIVE_BINS - center)))
        residual = np.abs(row[pidx] - h[pidx] * pilots) ** 2
        current_noise = max(float(np.median(residual)), floor * 0.1)
        h[pidx] = (1.0 - alpha) * h[pidx] + alpha * observed_h
        variance = np.maximum(base_var[didx], current_noise)
        llr = 2.0 * np.real(np.conj(h[didx]) * row[didx]) / variance
        llrs.append(np.clip(llr, -24.0, 24.0))
        symbol_bins.append(ACTIVE_BINS[didx])
        h_track.append(h.copy())
        cpes.append(cpe)
        slopes.append(slope)
        residuals.append(float(np.median(residual)))
    return {
        "llr": np.concatenate(llrs) if llrs else np.empty(0),
        "bins": np.concatenate(symbol_bins) if symbol_bins else np.empty(0, dtype=int),
        "H": np.asarray(h_track),
        "cpe": np.asarray(cpes),
        "phase_slope": np.asarray(slopes),
        "pilot_residual": np.asarray(residuals),
    }


def profile_meta():
    return {
        "profile": PROFILE,
        "fs": FS,
        "fft_size": N,
        "cp_samples": CP,
        "symbol_len": L,
        "symbol_seconds": L / FS,
        "active_ranges": [list(x) for x in SEGMENTS],
        "active_bins": [int(x) for x in ACTIVE_BINS],
        "active_bin_count": int(len(ACTIVE_BINS)),
        "active_frequency_ranges_hz": [[a * FS / N, b * FS / N] for a, b in SEGMENTS],
        "pilot_spacing": PILOT_SPACING,
        "pilot_pattern": "rotating_comb",
        "mod": "bpsk",
        "fec": "convolutional_k7_r1_2_171_133_soft_viterbi",
        "block_size": BLOCK_SIZE,
        "header_size": HEADER_SIZE,
        "header_repeats": HEADER_REPEATS,
    }
