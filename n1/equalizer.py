# -*- coding: utf-8 -*-
"""信道估计与均衡：LS、Wiener、首尾插值、CPE+NLMS、MMSE/MRC。"""

from __future__ import annotations

import numpy as np


def active_bins(start_index: int, chunk_size: int) -> np.ndarray:
    return np.arange(start_index, start_index + 4 * chunk_size)


def mirror_bins(n_fft: int, start_index: int, chunk_size: int) -> np.ndarray:
    i = np.arange(4 * chunk_size)
    return n_fft - start_index - i


def ls_channel_estimate(rx_freq: np.ndarray, tx_freq: np.ndarray,
                        bins: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.zeros(rx_freq.shape[-1], dtype=complex)
    x = tx_freq[..., bins]
    y = rx_freq[..., bins]
    h_bins = y / (x + eps)
    if h_bins.ndim == 1:
        h[bins] = h_bins
    else:
        h[bins] = np.mean(h_bins, axis=0)
    return h


def wiener_smooth(h_bins: np.ndarray, sigma2: float, span: int = 7) -> np.ndarray:
    """沿频率的 SNR 自适应平滑（仅对有效子载波向量）。"""
    h = np.asarray(h_bins, dtype=np.complex128)
    span = int(span) | 1
    kernel = np.ones(span, dtype=np.float64) / span
    hm = np.convolve(h.real, kernel, mode="same") + 1j * np.convolve(
        h.imag, kernel, mode="same"
    )
    p = np.abs(h) ** 2
    w = p / (p + float(sigma2) + 1e-12)
    return w * h + (1.0 - w) * hm


def phase_aligned_ls(rx_freq: np.ndarray, tx_freq: np.ndarray,
                     bins: np.ndarray) -> np.ndarray:
    """多训练符号 LS：先对齐公共相位再平均，减轻逐符号 CPE。"""
    h0 = rx_freq[0, bins] / (tx_freq[0, bins] + 1e-12)
    acc = np.zeros_like(h0)
    for i in range(rx_freq.shape[0]):
        hi = rx_freq[i, bins] / (tx_freq[i, bins] + 1e-12)
        ph = np.angle(np.vdot(h0, hi))
        acc += hi * np.exp(-1j * ph)
    h = np.zeros(rx_freq.shape[-1], dtype=complex)
    h[bins] = acc / rx_freq.shape[0]
    return h


def turbo_refine_h(data_specs: np.ndarray, xhat_grid: np.ndarray,
                   bins: np.ndarray, noise_var: float) -> np.ndarray:
    """用判决/重建的 QPSK 网格当导频，再 LS + Wiener（仅有效子载波）。"""
    y = data_specs[:, bins]
    h_ls = np.mean(y / (xhat_grid + 1e-12), axis=0)
    return wiener_smooth(h_ls, noise_var)


def cfo_from_preamble_pair(specs: np.ndarray, train_f: np.ndarray,
                           bins: np.ndarray, sample_rate: float,
                           sym_len: int) -> float:
    """相邻训练符号相位差估残余 CFO（Hz）。"""
    if specs.shape[0] < 2:
        return 0.0
    # 去信道：用 X* Y，相邻符号相关
    a = specs[0, bins] * np.conj(train_f[0, bins])
    b = specs[1, bins] * np.conj(train_f[1, bins])
    prod = np.vdot(a, b)
    # angle ≈ 2π f_cfo * sym_len / fs
    ang = float(np.angle(prod))
    return ang * sample_rate / (2 * np.pi * sym_len)


def interpolate_channel(h_start: np.ndarray, h_end: np.ndarray,
                        num_data: int) -> np.ndarray:
    """首尾 H 线性插值到每个数据符号。"""
    if num_data <= 0:
        return np.zeros((0, h_start.size), dtype=complex)
    if num_data == 1:
        return ((h_start + h_end) * 0.5)[None, :]
    out = np.zeros((num_data, h_start.size), dtype=complex)
    for i in range(num_data):
        a = i / (num_data - 1)
        out[i] = (1 - a) * h_start + a * h_end
    return out


def estimate_noise_var(rx_freq: np.ndarray, tx_freq: np.ndarray,
                       h: np.ndarray, bins: np.ndarray) -> float:
    err = rx_freq[..., bins] - h[bins] * tx_freq[..., bins]
    return float(np.mean(np.abs(err) ** 2) + 1e-12)


def tone_noise_and_llr_scale(rx_freq: np.ndarray, tx_freq: np.ndarray,
                             h: np.ndarray, bins: np.ndarray) -> tuple[float, np.ndarray]:
    """按训练残余估每子载波噪声，并生成 n3_2 风格的 LLR 幅度门控。"""
    err = rx_freq[..., bins] - h[bins] * tx_freq[..., bins]
    # carrier_noise ≈ E[|Y-HX|^2] / |H|^2
    carrier = np.mean(np.abs(err) ** 2, axis=0) / np.maximum(np.abs(h[bins]) ** 2, 1e-12)
    noise_var = float(max(np.median(carrier), 1e-3))
    reliability = noise_var / np.maximum(carrier, 1e-12)
    scale = 3.0 * np.clip(np.sqrt(reliability), 0.25, 4.0)
    return noise_var, scale.astype(np.float64)


def delay_from_h(href: np.ndarray, hmeas: np.ndarray, bins: np.ndarray,
                 n_fft: int) -> float:
    """由 H_end/H_start 相位斜率估残余时延（采样点）。参考 AudioModem / 声学 OFDM。"""
    href = np.asarray(href, dtype=np.complex128)
    hmeas = np.asarray(hmeas, dtype=np.complex128)
    b = np.asarray(bins, dtype=np.float64)
    order = np.argsort(b)
    b, href, hmeas = b[order], href[order], hmeas[order]
    ang = np.unwrap(np.angle(hmeas / (href + 1e-12)))
    w = np.maximum(np.abs(href) * np.abs(hmeas), 1e-12)
    n = max(32, min(len(b), len(b) // 3))
    coef = np.polyfit(b[:n], ang[:n], 1, w=w[:n])
    return float(-coef[0] * n_fft / (2.0 * np.pi))


def apply_symbol_delays(specs: np.ndarray, delays, n_fft: int) -> np.ndarray:
    """每符号频域线性相位，补偿残余采样时延。"""
    specs = np.asarray(specs, dtype=np.complex128)
    k = np.arange(int(n_fft), dtype=np.float64)
    out = np.empty_like(specs)
    for m, delay in enumerate(delays):
        out[m] = specs[m] * np.exp(-2j * np.pi * k * float(delay) / n_fft)
    return out


def slice_qpsk(z: np.ndarray) -> np.ndarray:
    """硬切到 {±1±j}。"""
    return np.sign(np.real(z) + 1e-15) + 1j * np.sign(np.imag(z) + 1e-15)


def cpe_and_nlms(y: np.ndarray, h: np.ndarray, mu: float = 0.12) -> tuple[np.ndarray, np.ndarray, float]:
    """公共相位误差校正 + 轻度 NLMS 更新 H（仅有效子载波向量）。"""
    z = y / (h + 1e-12)
    xhat = slice_qpsk(z)
    w = np.abs(h) ** 2
    num = np.sum(w * z * np.conj(xhat))
    theta = float(np.angle(num)) if np.abs(num) > 1e-12 else 0.0
    h = h * np.exp(1j * theta)
    z = y / (h + 1e-12)
    xhat = slice_qpsk(z)
    err = y - h * xhat
    h = h + mu * np.conj(xhat) * err / (np.abs(xhat) ** 2 + 1e-6)
    z = y / (h + 1e-12)
    return z, h, theta


def mmse_equalize(y: np.ndarray, h: np.ndarray, noise_var: float,
                  eps: float = 1e-12) -> np.ndarray:
    h_pow = np.abs(h) ** 2
    return np.conj(h) * y / (h_pow + noise_var + eps)


def mirror_mrc(eq_pos: np.ndarray, eq_mir: np.ndarray,
               h_pos: np.ndarray, h_mir: np.ndarray,
               noise_var: float, eps: float = 1e-12) -> np.ndarray:
    w_p = np.abs(h_pos) ** 2
    w_m = np.abs(h_mir) ** 2
    return (w_p * eq_pos + w_m * eq_mir) / (w_p + w_m + eps)


def equalize_symbol(y_freq: np.ndarray, h: np.ndarray, bins_pos: np.ndarray,
                    bins_mir: np.ndarray, noise_var: float,
                    track: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """MMSE + 镜像 MRC；可选 CPE/NLMS。返回 (X_hat, 更新后的全长 H)。"""
    h = h.copy()
    y_p = y_freq[bins_pos]
    y_m = y_freq[bins_mir]
    h_p = h[bins_pos]
    h_m = h[bins_mir]

    if track:
        z_p, h_p, _ = cpe_and_nlms(y_p, h_p)
        # 镜像支路：对 conj(Y) 跟踪
        z_m, h_m_c, _ = cpe_and_nlms(np.conj(y_m), np.conj(h_m))
        h_m = np.conj(h_m_c)
        h[bins_pos] = h_p
        h[bins_mir] = h_m
        eq_p = z_p
        eq_m = z_m
    else:
        eq_p = mmse_equalize(y_p, h_p, noise_var)
        eq_m = mmse_equalize(np.conj(y_m), np.conj(h_m), noise_var)

    x_hat = mirror_mrc(eq_p, eq_m, h[bins_pos], h[bins_mir], noise_var)
    return x_hat, h
