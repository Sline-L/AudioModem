# -*- coding: utf-8 -*-
"""同步模块：chirp 相关（亚采样）、SFO、精定时、CFO 网格、尾训练定 K。"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import fftconvolve, resample_poly


def load_wav_mono(path: str, expected_fs: int | None = None) -> tuple[np.ndarray, int]:
    """读取 WAV 为单声道浮点 [-1,1]；立体声取左声道（与常见录音一致）。"""
    from scipy.io import wavfile

    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        audio = data.astype(np.float64) / 32768.0
    else:
        audio = data.astype(np.float64)
        peak = np.max(np.abs(audio)) + 1e-12
        if peak > 1.5:
            audio = audio / peak
    if expected_fs is not None and fs != expected_fs:
        g = gcd(expected_fs, fs)
        audio = resample_poly(audio, expected_fs // g, fs // g)
        fs = expected_fs
    return audio, fs


def gen_linear_chirp(sample_rate: int, start_freq: float, end_freq: float,
                     duration: float, amp: float = 0.8) -> np.ndarray:
    """与 TX 相同的线性 chirp。"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    B = end_freq - start_freq
    phi = 2 * np.pi * start_freq * t + t * t * np.pi * B / duration
    return amp * np.cos(phi)


def parabolic_peak(y: np.ndarray, i: int) -> tuple[float, float]:
    """三点抛物线插值，返回 (亚采样峰位置, 峰高)。"""
    if i <= 0 or i >= len(y) - 1:
        return float(i), float(y[i])
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2 * b + c
    if abs(denom) < 1e-18:
        return float(i), b
    delta = 0.5 * (a - c) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    peak = b - 0.25 * (a - c) * delta
    return float(i) + delta, float(peak)


def find_chirp_peaks(audio: np.ndarray, chirp: np.ndarray,
                     min_separation: int) -> tuple[float, float, np.ndarray]:
    """首尾 chirp 相关峰（能量平方 + 亚采样）。返回浮点起点。"""
    metric = fftconvolve(audio, chirp[::-1], mode="full") ** 2
    L = len(chirp)
    # full 相关中：音频起点对应 lag=L-1 → start=0
    # start = lag - (L - 1)
    def lag_to_start(lag: float) -> float:
        return float(lag) - (L - 1)

    # 合法 start∈[0, len-L]
    starts = np.arange(len(metric)) - (L - 1)
    valid = (starts >= 0) & (starts + L <= len(audio))
    m = np.where(valid, metric, 0.0)

    i1 = int(np.argmax(m))
    f1, _ = parabolic_peak(m, i1)
    first = lag_to_start(f1)

    mask = m.copy()
    lo = max(0, i1 - min_separation)
    hi = min(len(mask), i1 + min_separation)
    mask[lo:hi] = 0.0
    i2 = int(np.argmax(mask))
    f2, _ = parabolic_peak(mask, i2)
    second = lag_to_start(f2)

    if second < first:
        first, second = second, first
    return first, second, metric


def estimate_sfo(first_chirp: float, second_chirp: float, nominal_gap: float) -> float:
    """返回 scale≈fs_rx/fs_tx = measured/nominal。"""
    measured = second_chirp - first_chirp
    if measured <= 0 or nominal_gap <= 0:
        return 1.0
    return float(measured) / float(nominal_gap)


def resample_audio(audio: np.ndarray, rate_ratio: float,
                   ppm_skip: float = 2.0) -> np.ndarray:
    """按采样钟 scale 校正。

    与 AudioModem / n2 一致：``corrected[n] = rx[n * scale]`` 线性插值，
    可表示十几 ppm。``resample_poly(10000)`` 会把 ~11 ppm 四舍五入成 0。
    """
    audio = np.asarray(audio, dtype=np.float64)
    scale = float(rate_ratio)
    if not np.isfinite(scale) or scale <= 0.5 or scale >= 1.5:
        return audio
    if abs(scale - 1.0) * 1e6 < float(ppm_skip):
        return audio
    n_out = int(np.floor((len(audio) - 1) / scale))
    if n_out < 16:
        return audio
    src = np.arange(n_out, dtype=np.float64) * scale
    return np.interp(src, np.arange(len(audio), dtype=np.float64), audio)


def coarse_body_start(first_chirp: float, chirp_len: int, silence_samples: int) -> float:
    return float(first_chirp) + chirp_len + silence_samples


def fine_timing_with_preamble(audio: np.ndarray, search_center: float,
                              preamble_time: np.ndarray,
                              search_radius: int) -> tuple[float, float]:
    """前导匹配 + 亚采样，返回 (浮点起点, 相关能量)。"""
    pref = preamble_time.ravel().astype(np.float64)
    lo = max(0, int(round(search_center)) - search_radius)
    hi = min(len(audio), int(round(search_center)) + search_radius + len(pref))
    segment = np.asarray(audio[lo:hi], dtype=np.float64)
    if len(segment) < len(pref) + 4:
        return float(search_center), 0.0
    metric = fftconvolve(segment, pref[::-1], mode="valid") ** 2
    ip = int(np.argmax(metric))
    frac_rel, peak = parabolic_peak(metric, ip)
    return float(lo + frac_rel), float(peak)


def find_end_preamble(audio: np.ndarray, body_start: float,
                      end_preamble_time: np.ndarray, sym_len: int,
                      preamble_count: int) -> int:
    """用尾训练匹配估计数据符号数 K。"""
    tmpl = end_preamble_time.ravel().astype(np.float64)
    i0 = int(round(body_start))
    body = np.asarray(audio[i0:], dtype=np.float64)
    if body.size < len(tmpl) + sym_len:
        return 0
    metric = fftconvolve(body, tmpl[::-1], mode="valid") ** 2
    ip = int(np.argmax(metric))
    pos, _ = parabolic_peak(metric, ip)
    k = int(round(pos / sym_len)) - preamble_count
    return max(0, k)


def cfo_search_hz(audio: np.ndarray, start: float, preamble_time: np.ndarray,
                  sample_rate: float, span_hz: float = 24.0) -> float:
    """已知前导复相关网格搜索残余频偏（Hz）。"""
    tmpl = preamble_time.ravel().astype(np.float64)
    L = len(tmpl)
    i0 = int(round(start))
    if i0 < 0 or i0 + L > len(audio):
        return 0.0
    sl = np.asarray(audio[i0 : i0 + L], dtype=np.float64)
    n = np.arange(L, dtype=np.float64)

    def score(f_hz: float) -> float:
        rot = np.exp(-2j * np.pi * f_hz * n / sample_rate)
        return abs(np.vdot(tmpl, sl * rot))

    best_f, best_e = 0.0, -1.0
    for f in np.linspace(-span_hz, span_hz, 9):
        e = score(float(f))
        if e > best_e:
            best_e, best_f = e, float(f)
    for f in np.linspace(best_f - 3.0, best_f + 3.0, 7):
        e = score(float(f))
        if e > best_e:
            best_e, best_f = e, float(f)
    return best_f


def sto_phase(n_fft: int, frac: float) -> np.ndarray:
    """分数 STO 频域相位：Δ>0 表示窗偏晚。"""
    k = np.arange(n_fft, dtype=np.float64)
    return np.exp(-2j * np.pi * k * float(frac) / n_fft)


def extract_symbol_ffts(audio: np.ndarray, start: float, n_sym: int,
                        n_fft: int, cp: int, cfo_hz: float,
                        sample_rate: float) -> np.ndarray:
    """去 CP、时域复数 CFO、分数 STO 相位，返回 (n_sym, N) 复谱。"""
    sym_len = n_fft + cp
    i_start = int(round(start))
    frac = float(start - i_start)
    rot_sto = sto_phase(n_fft, frac)
    specs = np.zeros((n_sym, n_fft), dtype=np.complex128)
    for m in range(n_sym):
        i0 = i_start + m * sym_len + cp
        if i0 + n_fft > len(audio):
            raise ValueError("符号越界，对准失败")
        n_abs = i0 + np.arange(n_fft, dtype=np.float64)
        sl = np.asarray(audio[i0 : i0 + n_fft], dtype=np.float64)
        sl_c = sl * np.exp(-2j * np.pi * cfo_hz * n_abs / sample_rate)
        specs[m] = np.fft.fft(sl_c) * rot_sto
    return specs
