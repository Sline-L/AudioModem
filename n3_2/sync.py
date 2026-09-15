from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .modem import CP, FS, K0, K1, L, N, linear_chirp, training_symbols


_FRONT_TRAINING = training_symbols()[:8, K0:K1 + 1]
_CHIRP_THRESHOLD = 0.35
_CHIRP_MATCH_SAMPLES = FS // 10
_TRAINING_THRESHOLD = 0.55
_MAX_SFO = 0.005
_SFO_STEP = 0.00025
_TIMING_RADIUS = CP
_TIMING_STEP = 32
_MIN_CHIRP_INTERVAL = 4 * FS + 16 * L


@dataclass(frozen=True)
class ChannelChoice:
    index: int
    score: float
    samples: np.ndarray


@dataclass(frozen=True)
class SyncResult:
    start: int
    training_start: int
    payload_start: int
    sfo: float
    score: float


class SyncError(RuntimeError):
    def __init__(self, stage, message):
        self.stage = str(stage)
        self.message = str(message)
        super().__init__(f"{self.stage}: {self.message}")


def _mono_samples(samples):
    x = np.asarray(samples)
    if x.ndim != 1 or np.iscomplexobj(x):
        raise SyncError("channel", "synchronization requires real mono samples")
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise SyncError("channel", "samples must be nonempty and finite")
    return x.astype(np.float64, copy=False)


def _normalized_correlation(samples, template):
    if samples.size < template.size:
        return np.empty(0, dtype=float)
    reference = np.asarray(template, dtype=float)
    reference = reference - np.mean(reference)
    reference_energy = float(np.dot(reference, reference))
    correlation = fftconvolve(samples, reference[::-1], mode="valid")
    sums = np.concatenate(([0.0], np.cumsum(samples, dtype=float)))
    squares = np.concatenate(([0.0], np.cumsum(samples * samples, dtype=float)))
    count = reference.size
    window_sum = sums[count:] - sums[:-count]
    window_energy = squares[count:] - squares[:-count] - window_sum * window_sum / count
    denominator = np.sqrt(np.maximum(window_energy, 0.0) * reference_energy)
    scores = np.zeros(correlation.shape, dtype=float)
    np.divide(np.abs(correlation), denominator, out=scores, where=denominator > 0)
    return np.clip(scores, 0.0, 1.0)


def _chirp_candidates(samples):
    chirp = linear_chirp()
    reference = chirp[:_CHIRP_MATCH_SAMPLES]
    scores = _normalized_correlation(samples, reference)
    candidates = []
    work = scores.copy()
    exclusion = chirp.size // 2
    for _ in range(12):
        if work.size == 0:
            break
        index = int(np.argmax(work))
        score = float(work[index])
        if score < _CHIRP_THRESHOLD:
            break
        candidates.append((index, score))
        lo = max(0, index - exclusion)
        hi = min(work.size, index + exclusion + 1)
        work[lo:hi] = 0.0
    return sorted(candidates)


def _compatible_pair(samples):
    candidates = _chirp_candidates(samples)
    if not candidates:
        raise SyncError("chirp", "no chirp candidate passed normalized correlation")
    pairs = []
    for front_index, front_score in candidates:
        for rear_index, rear_score in candidates:
            if rear_index <= front_index:
                continue
            interval = rear_index - front_index
            payload_symbols = max(0, int(round((interval - _MIN_CHIRP_INTERVAL) / L)))
            nominal = _MIN_CHIRP_INTERVAL + payload_symbols * L
            interval_sfo = interval / nominal - 1.0
            if abs(interval_sfo) <= _MAX_SFO:
                score = float(np.sqrt(front_score * rear_score))
                pairs.append((score, front_index, rear_index, interval_sfo))
    if not pairs:
        raise SyncError("chirp", "no compatible front/rear chirp pair")
    return max(pairs, key=lambda item: item[0])


def _interpolate(samples, positions):
    flat = np.asarray(positions, dtype=float).reshape(-1)
    if flat[0] < 0 or flat[-1] > samples.size - 1:
        return None
    values = np.interp(flat, np.arange(samples.size), samples)
    return values.reshape(np.shape(positions))


def _training_observations(samples, start, sfo):
    scale = 1.0 + sfo
    if sfo == 0.0 and int(start) == start:
        first = int(start)
        last = first + 8 * L
        if first < 0 or last > samples.size:
            return None
        symbols = samples[first:last].reshape(8, L)[:, CP:]
    else:
        symbol_offsets = np.arange(8, dtype=float)[:, None] * L
        useful_offsets = CP + np.arange(N, dtype=float)[None, :]
        positions = float(start) + scale * (symbol_offsets + useful_offsets)
        symbols = _interpolate(samples, positions)
        if symbols is None:
            return None
    return np.fft.rfft(symbols, n=N, axis=1)[:, K0:K1 + 1]


def _frequency_score(observations):
    if observations is None:
        return 0.0, None
    products = observations * np.conj(_FRONT_TRAINING)
    power = float(np.sum(np.abs(products) ** 2))
    if power <= 0.0 or not np.isfinite(power):
        return 0.0, products
    coherent = float(np.sum(np.abs(np.sum(products, axis=0)) ** 2))
    score = coherent / (products.shape[0] * power)
    return float(np.clip(score, 0.0, 1.0)), products


def _cp_score(samples, start, sfo):
    scale = 1.0 + sfo
    symbol_offsets = np.arange(8, dtype=float)[:, None] * L
    prefix_offsets = np.arange(CP, dtype=float)[None, :]
    prefix = _interpolate(samples, float(start) + scale * (symbol_offsets + prefix_offsets))
    tail = _interpolate(samples, float(start) + scale * (symbol_offsets + N + prefix_offsets))
    if prefix is None or tail is None:
        return 0.0
    numerator = abs(float(np.vdot(prefix, tail).real))
    denominator = float(np.linalg.norm(prefix) * np.linalg.norm(tail))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _timing_score(samples, start, sfo):
    frequency, _ = _frequency_score(_training_observations(samples, start, sfo))
    cyclic_prefix = _cp_score(samples, start, sfo)
    return 0.75 * frequency + 0.25 * cyclic_prefix


def _training_start(samples, expected, sfo):
    lo = max(0, int(expected) - _TIMING_RADIUS)
    hi = min(samples.size - 8 * L, int(expected) + _TIMING_RADIUS)
    if hi < lo:
        raise SyncError("training", "front training window is truncated")
    coarse = list(range(lo, hi + 1, _TIMING_STEP))
    if int(expected) not in coarse and lo <= int(expected) <= hi:
        coarse.append(int(expected))
    coarse_scores = [(_timing_score(samples, index, sfo), index) for index in coarse]
    _, coarse_start = max(coarse_scores)
    fine_lo = max(lo, coarse_start - _TIMING_STEP)
    fine_hi = min(hi, coarse_start + _TIMING_STEP)
    fine_scores = [(_timing_score(samples, index, sfo), index) for index in range(fine_lo, fine_hi + 1)]
    score, start = max(fine_scores)
    if score < _TRAINING_THRESHOLD:
        raise SyncError("training", f"frequency-domain training score {score:.3f} is too low")
    return start, score


def _phase_fit_sfo(products, coarse_sfo):
    cross = np.sum(products[1:] * np.conj(products[:-1]), axis=0)
    weights = np.abs(cross)
    if not np.any(weights > 0) or not np.all(np.isfinite(cross)):
        raise SyncError("sfo", "training phase fit has no finite weight")
    phase = np.unwrap(np.angle(cross))
    bins = np.arange(K0, K1 + 1, dtype=float)
    design = np.column_stack((bins, np.ones_like(bins)))
    root_weight = np.sqrt(weights / np.max(weights))
    fitted, _, _, _ = np.linalg.lstsq(design * root_weight[:, None], phase * root_weight, rcond=None)
    slope = float(fitted[0])
    residual = slope * N / (2.0 * np.pi * L)
    refined = (coarse_sfo - residual) / (1.0 + residual)
    if not np.isfinite(refined) or abs(refined) > _MAX_SFO:
        raise SyncError("sfo", "training phase fit is outside the supported range")
    return float(refined)


def _estimate_sfo(samples, training_start, interval_sfo):
    centers = np.arange(-_MAX_SFO, _MAX_SFO + _SFO_STEP / 2.0, _SFO_STEP)
    centers = np.unique(np.append(centers, np.clip(interval_sfo, -_MAX_SFO, _MAX_SFO)))
    scored = []
    for candidate in centers:
        score, products = _frequency_score(
            _training_observations(samples, training_start, float(candidate))
        )
        scored.append((score, float(candidate), products))
    coarse_score, coarse_sfo, products = max(scored, key=lambda item: item[0])
    if products is None or coarse_score < _TRAINING_THRESHOLD:
        raise SyncError("sfo", f"coarse SFO score {coarse_score:.3f} is too low")
    refined_sfo = _phase_fit_sfo(products, coarse_sfo)
    refined_score, _ = _frequency_score(
        _training_observations(samples, training_start, refined_sfo)
    )
    if refined_score < _TRAINING_THRESHOLD:
        raise SyncError("sfo", f"fitted SFO score {refined_score:.3f} is too low")
    return refined_sfo, refined_score


def synchronize(samples):
    x = _mono_samples(samples)
    chirp_score, front_start, _, interval_sfo = _compatible_pair(x)
    expected_training = front_start + int(round((3 * FS + FS // 2) * (1.0 + interval_sfo)))
    training_start, timing_score = _training_start(x, expected_training, interval_sfo)
    sfo, sfo_score = _estimate_sfo(x, training_start, interval_sfo)
    payload_start = training_start + int(round(8 * L * (1.0 + sfo)))
    score = min(chirp_score, timing_score, sfo_score)
    return SyncResult(front_start, training_start, payload_start, sfo, float(score))


def choose_channel(samples):
    data = np.asarray(samples)
    if np.iscomplexobj(data) or data.ndim not in (1, 2):
        raise SyncError("channel", "samples must be real mono or stereo")
    if data.ndim == 2 and data.shape[1] not in (1, 2):
        raise SyncError("channel", "samples must have one or two channels")
    channels = [data] if data.ndim == 1 else [data[:, index] for index in range(data.shape[1])]
    valid = []
    failures = []
    for index, channel in enumerate(channels):
        try:
            result = synchronize(channel)
        except SyncError as exc:
            failures.append(f"channel {index} {exc.stage}")
        else:
            valid.append((result.score, index, channel))
    if not valid:
        detail = ", ".join(failures) if failures else "no channels"
        raise SyncError("channel", f"no channel contains a valid standard frame ({detail})")
    score, index, channel = max(valid, key=lambda item: item[0])
    return ChannelChoice(index, float(score), np.asarray(channel))
