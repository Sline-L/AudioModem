"""去斜拍频捕获：检测线性扫频、亚采样峰值、粗估采样钟偏。"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from params import derived
from phy import chirp_phase_samples, linear_chirp


def parabolic_peak(mag, index):
    """三点抛物线插值，得到亚采样峰值位置。"""
    i = int(index)
    if i <= 0 or i >= mag.size - 1:
        return float(i), float(mag[i])
    a, b, c = float(mag[i - 1]), float(mag[i]), float(mag[i + 1])
    den = a - 2.0 * b + c
    if abs(den) < 1e-18:
        return float(i), b
    delta = 0.5 * (a - c) / den
    delta = float(np.clip(delta, -1.0, 1.0))
    y = b - 0.25 * (a - c) * delta
    return i + delta, y


def matched_chirp_metric(rx, template):
    """线性扫频匹配滤波（脉冲压缩）。"""
    rx = np.asarray(rx, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    corr = fftconvolve(rx, template[::-1], mode="valid")
    return corr * corr


def find_two_chirps(rx, p=None):
    """
    找出首尾两个 Chirp 起点（亚采样）。
    返回 (first, second, metric, diagnostics)
    """
    d = derived(p)
    tmpl = linear_chirp(d)
    metric = matched_chirp_metric(rx, tmpl)
    if metric.size < 8:
        return None, None, metric, {"ok": False, "reason": "信号短于扫频模板"}
    i1 = int(np.argmax(metric))
    p1, y1 = parabolic_peak(metric, i1)
    masked = metric.copy()
    half = d["chirp_len"] // 2
    lo = max(0, i1 - half)
    hi = min(masked.size, i1 + half)
    masked[lo:hi] = 0.0
    i2 = int(np.argmax(masked))
    p2, y2 = parabolic_peak(metric, i2)
    first, second = (p1, p2) if p1 <= p2 else (p2, p1)
    e1, e2 = (y1, y2) if p1 <= p2 else (y2, y1)
    noise = float(np.median(metric)) + 1e-18
    snr1 = e1 / noise
    snr2 = e2 / noise
    ok = snr1 > 8.0
    return (
        first,
        second,
        metric,
        {
            "ok": ok,
            "snr1": snr1,
            "snr2": snr2,
            "peak1": first,
            "peak2": second,
        },
    )


def dechirp_sfo(rx, peak, p=None):
    """
    对检测到的扫频段去斜，用相位二次项粗估采样钟偏 δ=(fs_rx-fs_tx)/fs_tx。
    回环时二次项应接近 0。
    """
    d = derived(p)
    i0 = int(np.floor(peak))
    n = d["chirp_len"]
    if i0 < 0 or i0 + n > len(rx):
        return 0.0, {"ok": False, "reason": "扫频窗越界"}
    seg = np.asarray(rx[i0 : i0 + n], dtype=np.float64)
    # 分数延迟：线性插值对齐到峰值
    frac = float(peak - i0)
    if 0.0 < frac < 1.0 and i0 + n + 1 <= len(rx):
        nxt = np.asarray(rx[i0 + 1 : i0 + n + 1], dtype=np.float64)
        seg = (1.0 - frac) * seg + frac * nxt
    phase = chirp_phase_samples(n, d)
    de = seg * np.exp(-1j * phase)
    # 丢掉两端能量较弱的过渡
    cut = max(n // 20, 8)
    z = de[cut:-cut]
    ang = np.unwrap(np.angle(z + 1e-15j))
    x = np.arange(ang.size, dtype=np.float64)
    coef = np.polyfit(x, ang, 2)
    # 相位 ≈ c n²；残余调频率来自钟偏
    mu = (d["end_freq"] - d["start_freq"]) / d["chirp_duration"]
    # dφ/dn² 的标称二次系数为 π μ / fs²，钟偏会把它缩放
    fs = float(d["sample_rate"])
    nom = np.pi * mu / (fs * fs)
    # polyfit 的 c 对应 c*n²，测量二次项 / 标称
    meas = float(coef[0])
    # 若 fs_rx = fs*(1+δ)，有效 μ'≈μ/(1+δ)²，二次项变小
    ratio = meas / (nom + 1e-18)
    # 去斜后理想二次项为 0；把残余二次项映射成小 δ
    delta = float(np.clip(-meas / (nom + 1e-18) * 0.5, -3e-4, 3e-4))
    if abs(delta) < 3e-6:
        delta = 0.0
    return delta, {"ok": True, "quad": meas, "nominal": nom, "ratio": ratio, "sfo": delta}


def clock_scale_from_anchors(first, second, body_start, end_offset, k, p=None):
    """
    用扫频间距和前后训练位置拟合采样钟 scale≈fs_rx/fs_tx。
    做法对齐 AudioModem：整段录音上的观测位置 / 标称位置，而不是去斜二次项。
    """
    d = derived(p)
    n_pre = d["preamble_count"]
    slen = d["symbol_len"]
    n_sym = n_pre + int(k) + n_pre
    expected_gap = d["chirp_len"] + 2 * d["silence_len"] + n_sym * slen
    expected_body = d["chirp_len"] + d["silence_len"]
    expected_end = (n_pre + int(k)) * slen
    gap = float(second) - float(first)
    scale_chirp = gap / float(expected_gap)
    scale_end = float(end_offset) / float(expected_end)
    scale_body = (float(body_start) - float(first)) / float(expected_body)
    scale = float(np.median([scale_chirp, scale_end]))
    if not np.isfinite(scale) or scale <= 0.5 or scale >= 1.5:
        scale = 1.0
    table = np.array(
        [
            [0.0, float(first), 1.0, 0.0],
            [float(expected_body), float(body_start), 1.0, 1.0],
            [float(expected_body + expected_end), float(body_start) + float(end_offset), 1.0, 1.0],
            [float(expected_gap), float(second), 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return scale, {
        "ok": True,
        "status": "preamble",
        "scale": scale,
        "ppm": (scale - 1.0) * 1e6,
        "scale_chirp": float(scale_chirp),
        "scale_end": float(scale_end),
        "scale_body": float(scale_body),
        "expected_gap": float(expected_gap),
        "gap": gap,
        "table": table,
    }


def resample_clock(rx, scale, ppm_skip=2.0):
    """
    AudioModem 同款线性插值：corrected[n] = rx[n * scale]。
    可表示十几 ppm，不再用 resample_poly(1000) 把钟偏四舍五入掉。
    """
    rx = np.asarray(rx, dtype=np.float64)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.5 or scale >= 1.5:
        return rx, False
    if abs(scale - 1.0) * 1e6 < float(ppm_skip):
        return rx, False
    n_out = int(np.floor((len(rx) - 1) / scale))
    if n_out < 16:
        return rx, False
    src = np.arange(n_out, dtype=np.float64) * scale
    y = np.interp(src, np.arange(len(rx), dtype=np.float64), rx)
    return np.asarray(y, dtype=np.float64), True


def apply_sfo(rx, delta, fs):
    """按 δ=(fs_rx-fs_tx)/fs_tx 重采样；内部转到 scale=1+δ。"""
    del fs
    return resample_clock(rx, 1.0 + float(delta))


def coarse_body_start(first_peak, p=None):
    """第一扫频起点 + 扫频长 + 静音 = OFDM 体粗起点。"""
    d = derived(p)
    return float(first_peak) + d["chirp_len"] + d["silence_len"]
