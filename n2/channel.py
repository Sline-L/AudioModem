"""频域 Wiener 平滑信道、公共相位跟踪、轻度 NLMS、条件 Turbo 回灌。"""

from __future__ import annotations

import numpy as np

from params import derived
from phy import active_bins, mirror_bins, slice_qpsk, train_freq_time


def unused_bin_noise(specs, p=None):
    """用数据带下方保护音的中位数估计噪声，避开 14–24 kHz 扬声器截止区。"""
    d = derived(p)
    guard = np.arange(1, d["start_index"])
    mag = np.median(np.abs(specs[:, guard]) ** 2)
    return float(max(mag, 1e-12))


def ls_from_train(specs, train_f, bins):
    """多训练符号的最小二乘：H = mean(Y/X)。"""
    x = train_f[:, bins]
    y = specs[:, bins]
    h = np.mean(y / (x + 1e-12), axis=0)
    return h


def phase_aligned_mean(h_rows):
    """
    各训练符号的 H 先对齐到最后一个的公共相位再平均。
    避免 CPE 把平均信道抵消。
    """
    channels = np.asarray(h_rows, dtype=np.complex128)
    reference = channels[-1]
    aligned = np.empty_like(channels)
    for i, channel in enumerate(channels):
        weight = np.abs(channel) * np.abs(reference)
        phase = np.angle(np.sum(weight * np.conj(channel) * reference))
        aligned[i] = channel * np.exp(1j * phase)
    return np.mean(aligned, axis=0), aligned


def wiener_smooth(h, sigma2, span=7):
    """
    SNR 自适应邻域平滑：高 SNR 信本地 LS，低 SNR 信邻域均值。
    沿频率，不是沿符号做线性插值。
    """
    h = np.asarray(h, dtype=np.complex128)
    span = int(span) | 1
    kernel = np.ones(span, dtype=np.float64) / span
    mr = np.convolve(h.real, kernel, mode="same")
    mi = np.convolve(h.imag, kernel, mode="same")
    hm = mr + 1j * mi
    p = np.abs(h) ** 2
    sig = np.asarray(sigma2, dtype=np.float64)
    if sig.ndim == 0 or sig.size == 1:
        w = p / (p + float(np.ravel(sig)[0]) + 1e-12)
    else:
        w = p / (p + sig.ravel()[: h.size] + 1e-12)
    return w * h + (1.0 - w) * hm


def tone_reliability(h, sigma2):
    """每音可靠性 ρ，高频弱信道自动降权。"""
    h = np.asarray(h, dtype=np.complex128)
    sig = np.asarray(sigma2, dtype=np.float64)
    if sig.ndim == 0 or sig.size == 1:
        sig = float(np.ravel(sig)[0])
        snr = np.abs(h) ** 2 / (sig + 1e-12)
    else:
        snr = np.abs(h) ** 2 / (sig.ravel()[: h.size] + 1e-12)
    rho = snr / (snr + 1.0)
    return np.clip(rho, 0.05, 1.0)


def estimate_from_known(specs, known, p=None):
    """已知训练频域符号 → 相位对齐 LS + Wiener + 每音残差。"""
    d = derived(p)
    bins = active_bins(d)
    mir = mirror_bins(d)
    specs = np.asarray(specs)
    known = np.asarray(known)
    y = specs[:, bins]
    x = known[:, bins]
    h_ls, _ = phase_aligned_mean(y / (x + 1e-12))
    y_m = specs[:, mir]
    x_m = known[:, mir]
    h_m_ls, _ = phase_aligned_mean(y_m / (x_m + 1e-12))
    resid_tone = np.mean(np.abs(y - h_ls * x) ** 2, axis=0)
    guard = unused_bin_noise(specs, d)
    sigma2 = float(np.median(resid_tone))
    sigma2 = float(max(0.75 * sigma2 + 0.25 * min(guard, 4.0 * sigma2 + 1e-12), 1e-12))
    h = wiener_smooth(h_ls, np.maximum(resid_tone, 0.25 * sigma2))
    h_m = wiener_smooth(h_m_ls, sigma2)
    rho = tone_reliability(h, resid_tone)
    return {
        "H": h,
        "H_mirror": h_m,
        "H_ls": h_ls,
        "sigma2": sigma2,
        "sigma2_tone": resid_tone,
        "rho": rho,
    }


def h_and_llr_scale(y, x):
    """
    n3_2：多训练 LS 信道，再用每音残差给出 LLR 尺度。
    y, x: (n_train, n_bins)
    """
    y = np.asarray(y, dtype=np.complex128)
    x = np.asarray(x, dtype=np.complex128)
    h = np.mean(y / (x + 1e-12), axis=0)
    resid = y - x * h
    mag2 = np.maximum(np.abs(h) ** 2, 1e-12)
    carrier_noise = np.mean(np.abs(resid) ** 2, axis=0) / mag2
    mag = np.abs(h)
    valid = mag > max(float(np.max(mag)) * 1e-8, 1e-12)
    noise_var = max(float(np.median(carrier_noise[valid])), 1e-3) if np.any(valid) else 1e-3
    reliability = noise_var / np.maximum(carrier_noise, 1e-12)
    scale = 3.0 * np.clip(np.sqrt(reliability), 0.25, 4.0)
    return h, scale, noise_var, valid


def estimate_preamble_channel(train_specs, p=None):
    """起始训练：LS + Wiener，并给出镜像信道与噪声。"""
    d = derived(p)
    freq, _ = train_freq_time(d)
    n_pre = d["preamble_count"]
    return estimate_from_known(train_specs[:n_pre], freq[:n_pre], d)


def estimate_end_channel(end_specs, p=None):
    """结束训练：与起始训练同一套估计。"""
    d = derived(p)
    freq, _ = train_freq_time(d)
    n_pre = d["preamble_count"]
    return estimate_from_known(end_specs[:n_pre], freq[n_pre : 2 * n_pre], d)


def lerp_h(h0, h1, alpha):
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - a) * h0 + a * h1


def hermitian_gate(z, z_m, sigma_z):
    """
    均衡后 Z 与 conj(Z_mirror) 一致性门控。一致则 g≈1，不一致则压低 LLR。
    不做 MRC 合并。sigma_z 必须是均衡域噪声，不能直接塞 FFT 域 σ²。
    """
    err = np.abs(z - np.conj(z_m)) ** 2
    sig = np.asarray(sigma_z, dtype=np.float64)
    if sig.ndim == 0 or sig.size == 1:
        g = np.exp(-err / (4.0 * float(np.ravel(sig)[0]) + 1e-12))
    else:
        g = np.exp(-err / (4.0 * sig.ravel()[: err.size] + 1e-12))
    return np.clip(g, 0.05, 1.0)


def cpe_and_nlms(y, h, rho, mu=0.12):
    """公共相位误差校正 + 轻度 NLMS 更新 H。"""
    z = y / (h + 1e-12)
    xhat = slice_qpsk(z)
    w = rho
    num = np.sum(w * z * np.conj(xhat))
    theta = np.angle(num) if np.abs(num) > 1e-12 else 0.0
    h = h * np.exp(1j * theta)
    z = y / (h + 1e-12)
    xhat = slice_qpsk(z)
    err = y - h * xhat
    h = h + mu * np.conj(xhat) * err / (np.abs(xhat) ** 2 + 1e-6)
    z = y / (h + 1e-12)
    return z, h, float(theta)


def equalize_symbol(spec, ce, p=None, mu=0.12):
    """均衡一个数据符号，返回 Z、门控、更新后的信道。"""
    d = derived(p)
    bins = active_bins(d)
    mir = mirror_bins(d)
    y = spec[bins]
    y_m = spec[mir]
    z, h_new, theta = cpe_and_nlms(y, ce["H"], ce["rho"], mu=mu)
    h_m = ce["H_mirror"] * np.exp(-1j * theta)
    z_m = y_m / (h_m + 1e-12)
    tone = ce.get("sigma2_tone", ce["sigma2"])
    sigz = np.asarray(tone, dtype=np.float64) / (np.abs(h_new) ** 2 + 1e-12)
    g = hermitian_gate(z, z_m, sigz)
    ce = dict(ce)
    ce["H"] = h_new
    ce["H_mirror"] = 0.9 * h_m + 0.1 * (y_m / (np.conj(z) + 1e-12))
    return z, g, ce, theta


def turbo_refine(data_specs, xhat_grid, ce, p=None):
    """
    用硬判决重建的频域符号当导频，再 LS + Wiener。
    xhat_grid: (K, n_active) 已知/判决 QPSK。
    """
    d = derived(p)
    bins = active_bins(d)
    y = np.stack([s[bins] for s in data_specs], axis=0)
    h_ls = np.mean(y / (xhat_grid + 1e-12), axis=0)
    h = wiener_smooth(h_ls, ce["sigma2"])
    ce = dict(ce)
    ce["H"] = h
    ce["rho"] = tone_reliability(h, ce.get("sigma2_tone", ce["sigma2"]))
    return ce
