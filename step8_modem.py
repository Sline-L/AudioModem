from pathlib import Path
import struct
import wave
import zlib

import numpy as np
from scipy import signal, stats


FS = 48000
N = 512
CP = 256
L = N + CP
I16 = 32768.0

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


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != FS:
            raise ValueError(f"expected mono 16-bit {FS} Hz wav: {path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / I16


def wav_gain(samples):
    peak = np.max(np.abs(samples))
    if peak == 0:
        raise ValueError("empty signal")
    return 0.95 * (I16 - 1) / I16 / peak


def write_wav(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples * wav_gain(samples) * I16, -I16 + 1, I16 - 1).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(FS)
        wav.writeframes(pcm.tobytes())


def find_sync(rx, template):
    if len(rx) < len(template):
        raise ValueError("receive wav is shorter than preamble")
    corr = signal.correlate(rx, template, mode="valid", method="fft")
    peak = int(np.argmax(np.abs(corr)))
    energy = np.linalg.norm(rx[peak : peak + len(template)]) * np.linalg.norm(template)
    return peak, float(abs(corr[peak]) / energy) if energy else 0.0


def ofdm_tx(symbols, bins=ACTIVE_BINS):
    freq = np.zeros((len(symbols), N), complex)
    freq[:, bins] = symbols
    freq[:, N - bins] = np.conj(symbols)
    time = np.fft.ifft(freq, axis=1).real
    return np.c_[time[:, -CP:], time].ravel()


def ofdm_rx(samples, bins=ACTIVE_BINS):
    rows = len(samples) // L
    if rows == 0:
        return np.empty((0, len(bins)), complex)
    time = samples[: rows * L].reshape(rows, L)[:, CP:]
    return np.fft.fft(time, axis=1)[:, bins]


def random_qpsk(rows, bins, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (rows, len(bins), 2), dtype=np.uint8)
    return (np.where(bits[:, :, 1], -1, 1) + 1j * np.where(bits[:, :, 0], -1, 1)) / np.sqrt(2)


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
    previous = np.empty((NSTATES, 2), dtype=np.int16)
    signs = np.empty((NSTATES, 2, 2), dtype=float)
    for state in range(NSTATES):
        bit = state & 1
        for branch, old in enumerate((state >> 1, (state >> 1) | (NSTATES >> 1))):
            reg = (old << 1) | bit
            coded = [parity(reg & generator) for generator in GENERATORS]
            previous[state, branch] = old
            signs[state, branch] = 1.0 - 2.0 * np.asarray(coded)
    return previous, signs


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
        raise ValueError("invalid Step8 header length")
    first, stored_crc = raw[:-4], struct.unpack(">I", raw[-4:])[0]
    if zlib.crc32(first) != stored_crc:
        raise ValueError("header CRC mismatch")
    magic, version, name_len, _, size, block_size, blocks, file_crc, name_raw = struct.unpack(
        ">4sBBHQHHI100s", first
    )
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported Step8 header")
    if name_len > len(name_raw) or block_size < 1:
        raise ValueError("invalid Step8 header fields")
    name = Path(name_raw[:name_len].decode("utf-8")).name
    expected_blocks = (size + block_size - 1) // block_size
    if blocks != expected_blocks:
        raise ValueError("inconsistent Step8 block count")
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
    parts = [indices[np.arange(len(indices)) % spacing == offset] for indices in SEGMENT_INDICES]
    return np.concatenate(parts)


def data_indices(symbol_index, spacing=PILOT_SPACING):
    mask = np.ones(len(ACTIVE_BINS), dtype=bool)
    mask[pilot_indices(symbol_index, spacing)] = False
    return np.flatnonzero(mask)


def pilot_values(symbol_index, indices, seed=PILOT_SEED):
    rng = np.random.default_rng(seed + symbol_index)
    bits = rng.integers(0, 2, (len(ACTIVE_BINS), 2), dtype=np.uint8)
    values = (np.where(bits[:, 1], -1, 1) + 1j * np.where(bits[:, 0], -1, 1)) / np.sqrt(2)
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
        row = np.zeros(len(ACTIVE_BINS), complex)
        row[pidx] = pilot_values(symbol_index, pidx, pilot_seed)
        row[didx[:take]] = 1.0 - 2.0 * bits[pos : pos + take]
        symbols.append(row)
        used.append(take)
        pos += take
        symbol_index += 1
    return np.asarray(symbols), np.asarray(used, dtype=int)


def phase_aligned_mean(h_blocks):
    channels = np.asarray(h_blocks, dtype=complex)
    reference = channels[-1]
    aligned = []
    for channel in channels:
        weight = np.abs(channel) * np.abs(reference)
        phase = np.angle(np.sum(weight * np.conj(channel) * reference))
        aligned.append(channel * np.exp(1j * phase))
    return np.mean(aligned, axis=0), np.asarray(aligned)


def estimate_training(y_blocks, x_blocks):
    channels = []
    variances = []
    for received, known in zip(y_blocks, x_blocks):
        rows = min(len(received), len(known))
        channel = np.mean(received[:rows] / known[:rows], axis=0)
        channels.append(channel)
        variances.append(np.mean(np.abs(received[:rows] - known[:rows] * channel) ** 2, axis=0))
    channel, aligned = phase_aligned_mean(channels)
    return channel, np.asarray(aligned), np.median(np.asarray(variances), axis=0)


def noise_variance(samples):
    freq = ofdm_rx(samples)
    if len(freq) == 0:
        return np.zeros(len(ACTIVE_BINS))
    return np.median(np.abs(freq) ** 2, axis=0)


PROFILE = "clock_anchor_n512_cp256"
TIMING_ANCHOR_INTERVAL = 128
TIMING_ANCHOR_SYMBOLS = 8
TIMING_ANCHOR_SEED = 9028


def timing_anchor_symbols(index, rows=TIMING_ANCHOR_SYMBOLS, seed=TIMING_ANCHOR_SEED):
    return random_qpsk(rows, ACTIVE_BINS, seed + index)


def frame_payload(payload, interval=TIMING_ANCHOR_INTERVAL, anchor_rows=TIMING_ANCHOR_SYMBOLS, seed=TIMING_ANCHOR_SEED):
    payload = np.asarray(payload, dtype=complex)
    parts = []
    anchor_starts = []
    logical = 0
    physical = 0
    anchor_index = 0
    while logical < len(payload):
        take = min(interval, len(payload) - logical)
        parts.append(payload[logical : logical + take])
        logical += take
        physical += take
        if logical < len(payload):
            anchor_starts.append(physical)
            parts.append(timing_anchor_symbols(anchor_index, anchor_rows, seed))
            physical += anchor_rows
            anchor_index += 1
    framed = np.vstack(parts) if parts else np.empty((0, len(ACTIVE_BINS)), complex)
    return framed, np.asarray(anchor_starts, dtype=int)


def payload_anchor_nominals(training_symbols, interval, anchor_rows, count):
    group = np.arange(count, dtype=float)
    physical = interval + group * (interval + anchor_rows)
    return (training_symbols + physical) * L


def correlate_near(rx, template, expected, radius):
    center = int(round(expected))
    lo = max(0, center - radius)
    hi = min(len(rx) - len(template), center + radius)
    if hi < lo:
        raise ValueError("recording is too short for timing anchor")
    section = rx[lo : hi + len(template)]
    corr = signal.correlate(section, template, mode="valid", method="fft")
    energy = signal.fftconvolve(section * section, np.ones(len(template)), mode="valid")
    denom = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(template)
    score = np.divide(np.abs(corr), denom, out=np.zeros_like(denom), where=denom > 0)
    peak = int(np.argmax(score))
    delta = 0.0
    if 0 < peak < len(score) - 1:
        left, middle, right = score[peak - 1 : peak + 2]
        curvature = left - 2.0 * middle + right
        if abs(curvature) > 1e-12:
            delta = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(lo + peak + delta), float(np.clip(score[peak], 0.0, 1.0))


def _weighted_line(nominal, observed, weights):
    design = np.c_[np.ones(len(nominal)), nominal]
    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * observed)
    return np.linalg.solve(lhs, rhs)


def robust_clock_fit(nominal, observed, scores, iterations=5):
    nominal = np.asarray(nominal, dtype=float)
    observed = np.asarray(observed, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(nominal) < 3:
        raise ValueError("not enough anchors for clock fit")
    slope, intercept, _, _ = stats.theilslopes(observed, nominal)
    weights = np.maximum(scores, 1e-3) ** 2
    for _ in range(iterations):
        residual = observed - (intercept + slope * nominal)
        center = np.median(residual)
        sigma = max(1.4826 * np.median(np.abs(residual - center)), 0.25)
        distance = np.abs(residual - center)
        huber = np.minimum(1.0, 2.5 * sigma / np.maximum(distance, 1e-12))
        intercept, slope = _weighted_line(nominal, observed, weights * huber)
    residual = observed - (intercept + slope * nominal)
    center = np.median(residual)
    sigma = max(1.4826 * np.median(np.abs(residual - center)), 0.25)
    used = np.abs(residual - center) <= max(3.0, 4.0 * sigma)
    if np.count_nonzero(used) >= 3:
        intercept, slope = _weighted_line(nominal[used], observed[used], weights[used])
        residual = observed - (intercept + slope * nominal)
    if not 0.995 <= slope <= 1.005:
        raise ValueError(f"implausible sample-clock scale: {slope:.8f}")
    return float(intercept), float(slope), residual, used


def detect_clock_anchors(
    rx,
    training,
    coarse_start,
    training_anchor_rows,
    training_anchor_step,
    interval,
    anchor_rows,
    anchor_seed,
    radius,
    min_score,
):
    nominal = []
    observed = []
    scores = []
    kinds = []
    indices = []
    for symbol in range(0, len(training) - training_anchor_rows + 1, training_anchor_step):
        template = ofdm_tx(training[symbol : symbol + training_anchor_rows])
        start, score = correlate_near(rx, template, coarse_start + symbol * L, radius)
        nominal.append(float(symbol * L))
        observed.append(start)
        scores.append(score)
        kinds.append(0.0)
        indices.append(float(symbol))

    train_mask = np.asarray(scores) >= min_score
    train_i, train_scale, _, train_used = robust_clock_fit(
        np.asarray(nominal)[train_mask], np.asarray(observed)[train_mask], np.asarray(scores)[train_mask]
    )
    template_samples = anchor_rows * L
    anchor_index = 0
    attempted_payload = []
    while True:
        anchor_nominal = float((len(training) + interval + anchor_index * (interval + anchor_rows)) * L)
        expected = train_i + train_scale * anchor_nominal
        if expected - radius >= len(rx) - template_samples:
            break
        template = ofdm_tx(timing_anchor_symbols(anchor_index, anchor_rows, anchor_seed))
        start, score = correlate_near(rx, template, expected, radius)
        nominal.append(anchor_nominal)
        observed.append(start)
        scores.append(score)
        kinds.append(1.0)
        indices.append(float(anchor_index))
        attempted_payload.append(anchor_nominal)
        anchor_index += 1

    nominal = np.asarray(nominal)
    observed = np.asarray(observed)
    scores = np.asarray(scores)
    kinds = np.asarray(kinds)
    valid = scores >= min_score
    full_i, full_scale, full_residual, full_used = robust_clock_fit(
        nominal[valid], observed[valid], scores[valid]
    )
    accepted = np.zeros(len(nominal), dtype=bool)
    accepted[np.flatnonzero(valid)] = full_used
    payload_used = accepted & (kinds == 1)
    attempted_span = np.ptp(attempted_payload) if len(attempted_payload) > 1 else 0.0
    used_span = np.ptp(nominal[payload_used]) if np.count_nonzero(payload_used) > 1 else 0.0
    full_ok = np.count_nonzero(payload_used) >= 4 and (attempted_span == 0.0 or used_span >= 0.5 * attempted_span)
    if full_ok:
        intercept, scale = full_i, full_scale
        status = "full"
        residual = observed - (intercept + scale * nominal)
    else:
        intercept, scale = train_i, train_scale
        status = "training_fallback"
        accepted = np.zeros(len(nominal), dtype=bool)
        accepted[np.flatnonzero((kinds == 0) & valid)] = train_used
        residual = observed - (intercept + scale * nominal)
    table = np.c_[nominal, observed, scores, residual, accepted.astype(float), kinds, indices]
    initial = {"intercept": train_i, "scale": train_scale}
    return intercept, scale, table, status, initial


def correct_clock(rx, intercept, scale):
    length = int(np.floor((len(rx) - intercept - 1) / scale))
    if length <= 0:
        raise ValueError("recording ends before synchronized signal")
    source = intercept + np.arange(length) * scale
    return np.interp(source, np.arange(len(rx)), rx)


def _phase_measurement(observed_h, current_h, pidx):
    safe = np.where(np.abs(current_h[pidx]) > 1e-12, current_h[pidx], 1e-12)
    ratio = observed_h / safe
    x = ACTIVE_BINS[pidx].astype(float)
    power = np.maximum(np.abs(current_h[pidx]) ** 2, 1e-12)
    center = float(np.average(x, weights=power))
    splits = np.r_[0, np.flatnonzero(np.diff(x) > 1) + 1, len(x)]
    segment_slopes = []
    segment_weights = []
    for begin, end in zip(splits[:-1], splits[1:]):
        sx = x[begin:end]
        sw = power[begin:end]
        if len(sx) < 2:
            continue
        phase = np.unwrap(np.angle(ratio[begin:end]))
        local_center = float(np.average(sx, weights=sw))
        xc = sx - local_center
        information = float(np.sum(sw * xc * xc))
        if information > 1e-12:
            segment_slopes.append(float(np.sum(sw * xc * phase) / information))
            segment_weights.append(information)
    if segment_slopes:
        slope = float(np.average(segment_slopes, weights=segment_weights))
    else:
        slope = 0.0
    cpe = float(np.angle(np.sum(power * ratio * np.exp(-1j * slope * (x - center)))))
    return float(cpe), float(slope), center


def _anchor_channel(rows, known, current_h):
    estimates = rows / known
    aligned = []
    for estimate in estimates:
        weight = np.maximum(np.abs(current_h) * np.abs(estimate), 1e-12)
        phase = np.angle(np.sum(weight * np.conj(estimate) * current_h))
        aligned.append(estimate * np.exp(1j * phase))
    return np.mean(aligned, axis=0)


def payload_llrs_anchored(
    y,
    h_initial,
    training_var,
    silence_var,
    pilot_seed=PILOT_SEED,
    spacing=PILOT_SPACING,
    channel_alpha=0.35,
    interval=TIMING_ANCHOR_INTERVAL,
    anchor_rows=TIMING_ANCHOR_SYMBOLS,
    anchor_seed=TIMING_ANCHOR_SEED,
    anchor_h_alpha=0.5,
    phase_slope="off",
    slope_window=64,
    slope_clip=0.05,
):
    h = np.asarray(h_initial, dtype=complex).copy()
    base_var = np.maximum(np.asarray(training_var), np.asarray(silence_var))
    positive = base_var[base_var > 0]
    floor = float(np.median(positive)) if positive.size else 1e-8
    base_var = np.maximum(base_var, floor * 0.1)
    llrs = []
    symbol_bins = []
    h_track = []
    anchor_h_track = []
    cpes = []
    slopes = []
    raw_slopes = []
    slope_history = []
    residuals = []
    physical = 0
    logical = 0
    anchor_index = 0
    while physical < len(y):
        data_count = min(interval, len(y) - physical)
        for row in y[physical : physical + data_count]:
            pidx = pilot_indices(logical, spacing)
            didx = data_indices(logical, spacing)
            pilots = pilot_values(logical, pidx, pilot_seed)
            observed_h = row[pidx] / pilots
            cpe, raw_slope, center = _phase_measurement(observed_h, h, pidx)
            raw_slope = float(np.clip(raw_slope, -slope_clip, slope_clip))
            raw_slopes.append(raw_slope)
            slope_history.append(raw_slope)
            if phase_slope == "slow":
                slope = float(np.median(slope_history[-slope_window:]))
            else:
                slope = 0.0
            ramp = np.exp(1j * (cpe + slope * (ACTIVE_BINS - center)))
            h_effective = h * ramp
            residual = np.abs(row[pidx] - h_effective[pidx] * pilots) ** 2
            current_noise = max(float(np.median(residual)), floor * 0.1)
            if phase_slope == "slow":
                observed_magnitude = np.abs(observed_h)
                magnitude = (1.0 - channel_alpha) * np.abs(h[pidx]) + channel_alpha * observed_magnitude
                h[pidx] = magnitude * np.exp(1j * np.angle(h[pidx]))
            else:
                observed_base = observed_h / ramp[pidx]
                h[pidx] = (1.0 - channel_alpha) * h[pidx] + channel_alpha * observed_base
            variance = np.maximum(base_var[didx], current_noise)
            llr = 2.0 * np.real(np.conj(h_effective[didx]) * row[didx]) / variance
            llrs.append(np.clip(llr, -24.0, 24.0))
            symbol_bins.append(ACTIVE_BINS[didx])
            h_track.append(h_effective.copy())
            cpes.append(cpe)
            slopes.append(slope)
            residuals.append(float(np.median(residual)))
            logical += 1
        physical += data_count
        if data_count < interval or physical + anchor_rows > len(y):
            break
        known = timing_anchor_symbols(anchor_index, anchor_rows, anchor_seed)
        anchor_h = _anchor_channel(y[physical : physical + anchor_rows], known, h)
        if anchor_h_alpha > 0:
            h = (1.0 - anchor_h_alpha) * h + anchor_h_alpha * anchor_h
        anchor_h_track.append(anchor_h.copy())
        slope_history.clear()
        physical += anchor_rows
        anchor_index += 1
    return {
        "llr": np.concatenate(llrs) if llrs else np.empty(0),
        "bins": np.concatenate(symbol_bins) if symbol_bins else np.empty(0, dtype=int),
        "H": np.asarray(h_track),
        "anchor_H": np.asarray(anchor_h_track),
        "cpe": np.asarray(cpes),
        "phase_slope": np.asarray(slopes),
        "phase_slope_raw": np.asarray(raw_slopes),
        "pilot_residual": np.asarray(residuals),
        "logical_symbols": logical,
        "physical_symbols": physical,
        "anchors_consumed": anchor_index,
    }


def profile_meta():
    return {
        "profile": PROFILE,
        "fs": FS,
        "fft_size": N,
        "cp_samples": CP,
        "symbol_len": L,
        "symbol_seconds": L / FS,
        "active_ranges": [list(segment) for segment in SEGMENTS],
        "active_bins": [int(bin_index) for bin_index in ACTIVE_BINS],
        "active_bin_count": int(len(ACTIVE_BINS)),
        "active_frequency_ranges_hz": [[a * FS / N, b * FS / N] for a, b in SEGMENTS],
        "pilot_spacing": PILOT_SPACING,
        "pilot_pattern": "rotating_comb",
        "mod": "bpsk",
        "fec": "convolutional_k7_r1_2_171_133_soft_viterbi",
        "block_size": BLOCK_SIZE,
        "header_size": HEADER_SIZE,
        "header_repeats": HEADER_REPEATS,
        "timing_anchor_pattern": "periodic_full_active_band_qpsk",
        "timing_anchor_interval": TIMING_ANCHOR_INTERVAL,
        "timing_anchor_symbols": TIMING_ANCHOR_SYMBOLS,
        "timing_anchor_seed": TIMING_ANCHOR_SEED,
    }
