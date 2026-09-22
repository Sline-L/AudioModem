"""已知前导的二维 STO–CFO 对准，不以 CP 自相关为主定时。"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from capture import parabolic_peak
from params import derived
from phy import active_bins, train_freq_time


def _cfo_grid(center_hz, span_hz=24.0, n=9):
    return np.linspace(center_hz - span_hz, center_hz + span_hz, n)


def matched_start(rx, coarse, p=None, search_syms=2.5):
    """
    用前两个已知训练符号做匹配滤波，抛物线亚采样锁定起点。
    返回整数起点、分数 STO、相关能量。
    """
    d = derived(p)
    _, train_t = train_freq_time(d)
    tmpl = np.concatenate(train_t[:2])
    slen = d["symbol_len"]
    w = int(search_syms * slen)
    i0 = max(0, int(round(coarse)) - w)
    i1 = min(len(rx), int(round(coarse)) + w + len(tmpl))
    window = np.asarray(rx[i0:i1], dtype=np.float64)
    if window.size < len(tmpl) + 4:
        return None
    metric = fftconvolve(window, tmpl[::-1], mode="valid") ** 2
    ip = int(np.argmax(metric))
    frac_rel, peak = parabolic_peak(metric, ip)
    start = i0 + frac_rel
    noise = float(np.median(metric)) + 1e-18
    return {
        "start": start,
        "peak_energy": peak,
        "snr": peak / noise,
        "integer": int(np.round(start)),
        "frac": float(start - np.round(start)),
    }


def cfo_search(rx, start, p=None):
    """
    在已锁定的整数起点附近，对残余频偏做小网格搜索。
    代价：已知前导与接收的复相关能量。
    """
    d = derived(p)
    fs = float(d["sample_rate"])
    _, train_t = train_freq_time(d)
    tmpl = np.concatenate(train_t[:2])
    L = len(tmpl)
    i0 = int(np.round(start))
    if i0 < 0 or i0 + L > len(rx):
        return 0.0, 0.0
    sl = np.asarray(rx[i0 : i0 + L], dtype=np.float64)
    n = np.arange(L, dtype=np.float64)
    best_f = 0.0
    best_e = -1.0
    for f in _cfo_grid(0.0):
        rot = np.exp(-2j * np.pi * f * n / fs)
        e = abs(np.vdot(tmpl, sl * rot))
        if e > best_e:
            best_e = e
            best_f = float(f)
    # 在最佳点附近再细搜一档
    for f in _cfo_grid(best_f, span_hz=3.0, n=7):
        rot = np.exp(-2j * np.pi * f * n / fs)
        e = abs(np.vdot(tmpl, sl * rot))
        if e > best_e:
            best_e = e
            best_f = float(f)
    return best_f, float(best_e)


def find_end_preamble(rx, start, p=None):
    """用后 8 个训练符号匹配，估计数据符号数 K。"""
    d = derived(p)
    _, train_t = train_freq_time(d)
    tmpl = np.concatenate(train_t[d["preamble_count"] :])
    slen = d["symbol_len"]
    i0 = int(np.round(start))
    body = np.asarray(rx[i0:], dtype=np.float64)
    if body.size < len(tmpl) + slen:
        return None
    metric = fftconvolve(body, tmpl[::-1], mode="valid") ** 2
    ip = int(np.argmax(metric))
    pos, energy = parabolic_peak(metric, ip)
    n_sym = pos / slen
    k = int(round(n_sym)) - d["preamble_count"]  # 峰在后训练起点：符号数 = 8+K
    noise = float(np.median(metric)) + 1e-18
    return {
        "K": k,
        "end_offset": pos,
        "snr": energy / noise,
        "n_sym_raw": n_sym,
    }


def sto_phase(n_fft, frac):
    """分数 STO 的频域线性相位补偿：X[k] *= exp(-j 2π k Δ / N)。"""
    k = np.arange(n_fft, dtype=np.float64)
    return np.exp(-2j * np.pi * k * float(frac) / n_fft)


def extract_active_phase(rx, start, n_sym, sfo, p=None, symbol_offset=0):
    """
    等间隔切窗，再用频域线性相位补偿采样钟。
    第 g 个符号的等效时延（样点）为
    τ = frac(start) + sfo * (g * symbol_len + CP)，
    频谱乘 exp(-j 2π k τ / N)。不改 WAV 采样率，也不在时域插值拉伸取样点。
    symbol_offset 是本段第一个符号在整帧里的序号，尾训练要传 8+K。
    """
    d = derived(p)
    n = d["N"]
    cp = d["CP"]
    slen = d["symbol_len"]
    bins = active_bins(d)
    n_sym = int(n_sym)
    if n_sym < 1:
        raise ValueError("符号数无效")
    rx = np.asarray(rx, dtype=np.float64)
    i_base = int(np.round(float(start)))
    frac = float(start) - i_base
    sfo = float(sfo)
    k = bins.astype(np.float64)
    specs = np.empty((n_sym, bins.size), dtype=np.complex128)
    for m in range(n_sym):
        g = int(symbol_offset) + m
        i0 = i_base + g * slen + cp
        if i0 < 0 or i0 + n > len(rx):
            raise ValueError("符号越界，对准失败")
        tau = frac + sfo * (g * slen + cp)
        spec = np.fft.rfft(rx[i0 : i0 + n], n=n)
        specs[m] = spec[bins] * np.exp(-2j * np.pi * k * tau / n)
    return specs


def extract_active_sfo(rx, start, n_sym, sfo, p=None):
    """
    n3_2 同款：在 start+(1+sfo)*(m*(N+CP)+CP+n) 上线性插值再 rFFT。
    钟偏留在取样网格里，不先整段 resample。
    返回 (n_sym, n_active) 正频率数据子载波。
    """
    d = derived(p)
    n = d["N"]
    cp = d["CP"]
    slen = d["symbol_len"]
    bins = active_bins(d)
    n_sym = int(n_sym)
    if n_sym < 1:
        raise ValueError("符号数无效")
    offsets = (
        np.arange(n_sym, dtype=np.float64)[:, None] * slen
        + cp
        + np.arange(n, dtype=np.float64)[None, :]
    )
    pos = float(start) + (1.0 + float(sfo)) * offsets
    if pos.min() < 0 or pos.max() > len(rx) - 1:
        raise ValueError("符号越界，对准失败")
    time = np.interp(
        pos.ravel(),
        np.arange(len(rx), dtype=np.float64),
        np.asarray(rx, dtype=np.float64),
    ).reshape(n_sym, n)
    spec = np.fft.rfft(time, n=n, axis=1)
    return spec[:, bins]


def training_peaks(rx, p=None, n_peaks=4):
    """
    全段搜索前两个训练符号的相关峰（n3_2 frame_candidates）。
    不依赖 Chirp 对，尾扫频损坏或立体声很弱时仍能给出定时候选。
    """
    d = derived(p)
    _, train_t = train_freq_time(d)
    tmpl = np.concatenate(train_t[:2])
    slen = d["symbol_len"]
    metric = fftconvolve(np.asarray(rx, dtype=np.float64), tmpl[::-1], mode="valid") ** 2
    if metric.size < 8:
        return []
    work = metric.copy()
    noise = float(np.median(metric)) + 1e-18
    half = slen // 2
    out = []
    for _ in range(int(n_peaks)):
        ip = int(np.argmax(work))
        pos, energy = parabolic_peak(metric, ip)
        snr = float(energy / noise)
        if out and snr < 3.0:
            break
        out.append({"start": float(pos), "snr": snr, "energy": float(energy)})
        lo = max(0, ip - half)
        hi = min(work.size, ip + half)
        work[lo:hi] = 0.0
    return out


def best_sfo_near(rx, start, known_front, p=None, span_ppm=200.0, step_ppm=25.0):
    """在训练起点上用 8 个前导残差选钟偏，避免被错误的双 Chirp 间距带偏。"""
    d = derived(p)
    best = None
    for ppm in np.arange(-float(span_ppm), float(span_ppm) + 0.5, float(step_ppm)):
        sfo = ppm * 1e-6
        try:
            y = extract_active_phase(rx, start, 8, sfo, d)
        except ValueError:
            continue
        h = np.mean(y / (known_front + 1e-12), axis=0)
        err = np.mean(np.abs(y - known_front * h) ** 2, axis=0)
        score = float(np.median(err / np.maximum(np.abs(h) ** 2, 1e-12)))
        if not np.isfinite(score):
            continue
        item = (score, float(sfo))
        if best is None or score < best[0]:
            best = item
    return best


def extract_ffts(rx, start, n_sym, cfo_hz, p=None):
    """
    从对准后的体里取出 n_sym 个 OFDM 符号的 FFT。
    去 CP；CFO 时域补偿后保持复谱；分数 STO 只补偿一次。
    Δ>0 表示窗相对信号偏晚。
    """
    d = derived(p)
    n = d["N"]
    cp = d["CP"]
    slen = d["symbol_len"]
    fs = float(d["sample_rate"])
    i_start = int(np.round(start))
    frac = float(start - i_start)
    rot_sto = sto_phase(n, frac)
    specs = np.zeros((n_sym, n), dtype=np.complex128)
    for m in range(n_sym):
        i0 = i_start + m * slen + cp
        if i0 + n > len(rx):
            raise ValueError("符号越界，对准失败")
        n_abs = i0 + np.arange(n, dtype=np.float64)
        sl = np.asarray(rx[i0 : i0 + n], dtype=np.float64)
        sl_c = sl * np.exp(-2j * np.pi * cfo_hz * n_abs / fs)
        spec = np.fft.fft(sl_c)
        specs[m] = spec * rot_sto
    return specs


def apply_symbol_delays(specs, delays, n_fft):
    """给每个 OFDM 符号补残余时延（采样点），等价于频域线性相位。"""
    specs = np.asarray(specs, dtype=np.complex128)
    k = np.arange(int(n_fft), dtype=np.float64)
    out = np.empty_like(specs)
    for m, delay in enumerate(delays):
        out[m] = specs[m] * np.exp(-2j * np.pi * k * float(delay) / n_fft)
    return out


def delay_from_h(href, hmeas, bins, n_fft):
    """由 H_end/H_start 的相位斜率估计额外时延（采样点）。只用低频连续段。"""
    href = np.asarray(href, dtype=np.complex128)
    hmeas = np.asarray(hmeas, dtype=np.complex128)
    b = np.asarray(bins, dtype=np.float64)
    order = np.argsort(b)
    b = b[order]
    href = href[order]
    hmeas = hmeas[order]
    ang = np.unwrap(np.angle(hmeas / (href + 1e-12)))
    w = np.maximum(np.abs(href) * np.abs(hmeas), 1e-12)
    n = max(32, min(len(b), len(b) // 3))
    coef = np.polyfit(b[:n], ang[:n], 1, w=w[:n])
    delay = float(-coef[0] * n_fft / (2.0 * np.pi))
    return delay
