"""发射端比特级兼容的物理层原语：扫频、训练、映射、扰码、帧头。"""

from __future__ import annotations

import random
import zlib
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from params import HEADER_MAGIC, SCRAMBLER_SEED, TRAIN_SEED, derived


def linear_chirp(p=None):
    """生成与 TX.log_chirp_gen 相同的线性扫频（二次相位，不是对数扫频）。"""
    d = derived(p)
    n = d["chirp_len"]
    t = np.linspace(0.0, d["chirp_duration"], n, endpoint=False)
    bandwidth = d["end_freq"] - d["start_freq"]
    phase = (
        2.0 * np.pi * d["start_freq"] * t
        + t * t * np.pi * bandwidth / d["chirp_duration"]
    )
    return d["amp"] * np.cos(phase)


def chirp_phase_samples(n, d):
    """去斜用的离散相位，与线性扫频同一公式。"""
    t = np.arange(n, dtype=np.float64) / d["sample_rate"]
    bandwidth = d["end_freq"] - d["start_freq"]
    return (
        2.0 * np.pi * d["start_freq"] * t
        + t * t * np.pi * bandwidth / d["chirp_duration"]
    )


def lfsr_sequence(length, seed=SCRAMBLER_SEED):
    """15 位 LFSR，与 TX.sequence 一致；每个码字应重新调用以复位。"""
    state = int(seed)
    out = np.empty(length, dtype=np.int8)
    for i in range(length):
        out[i] = (state >> 14) & 1
        feedback = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) & 0x7FFF) | feedback
    return out


def train_freq_time(p=None, rand_seed=TRAIN_SEED):
    """
    再生 16 个训练符号（前 8 + 后 8）。
    必须用 CPython random，禁止 NumPy RNG。
    """
    d = derived(p)
    n = d["N"]
    cp = d["CP"]
    start = d["start_index"]
    end = d["end_index"]
    count = 2 * d["preamble_count"]
    random.seed(rand_seed)
    freq = np.zeros((count, n), dtype=np.complex128)
    for i in range(count):
        for j in range(start, end):
            rand_num = random.randint(0, 3)
            freq[i, j] = (
                -2 * (rand_num & 1)
                + 1
                - 2 * ((rand_num >> 1) & 1) * 1j
                + 1j
            )
            freq[i, n - j] = np.conj(freq[i, j])
    time_body = np.fft.ifft(freq, axis=1).real
    with_cp = np.hstack((time_body[:, -cp:], time_body))
    return freq, with_cp


def active_bins(p=None):
    d = derived(p)
    start = d["start_index"]
    return np.arange(start, start + d["n_active"])


def mirror_bins(p=None):
    d = derived(p)
    n = d["N"]
    return n - active_bins(d)


def as_uint8(chunk):
    """把 bytes / 数组转成 uint8 向量。NumPy 2 不能对 Python bytes 直接 asarray。"""
    if isinstance(chunk, np.ndarray):
        return np.asarray(chunk, dtype=np.uint8).ravel()
    if isinstance(chunk, memoryview):
        return np.frombuffer(chunk, dtype=np.uint8).copy()
    return np.frombuffer(bytes(chunk), dtype=np.uint8).copy()


def bytes_to_qpsk_grid(chunk, p=None):
    """按 TX.mapp 把一个 498 字节块放到 N 点频域（含共轭镜像）。"""
    d = derived(p)
    chunk = as_uint8(chunk)
    arr = np.zeros(d["N"], dtype=np.complex128)
    start = d["start_index"]
    n = d["N"]
    for i in range(d["n_active"]):
        d_i = i // 4
        b_i = 3 - (i % 4)
        byte = int(chunk[d_i])
        bit_i = (byte >> (b_i * 2)) & 1
        bit_q = (byte >> (b_i * 2 + 1)) & 1
        arr[start + i] = (1 - 2 * bit_i) + 1j * (1 - 2 * bit_q)
        arr[n - start - i] = np.conj(arr[start + i])
    return arr


def qpsk_llr_scaled(z, scale, p=None):
    """
    n3_2 口径：每个子载波先 Q 后 I，LLR 乘每音尺度后裁剪。
    子载波顺序与 TX 字节打包一致（高位对比特先出）。
    """
    del p
    z = np.asarray(z, dtype=np.complex128).ravel()
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim == 0 or scale.size == 1:
        scale = np.full(z.size, float(np.ravel(scale)[0]))
    else:
        scale = scale.ravel()[: z.size]
    llr = np.empty(z.size * 2, dtype=np.float64)
    llr[0::2] = z.imag * scale
    llr[1::2] = z.real * scale
    return np.clip(llr, -30.0, 30.0)


def qpsk_llr_from_eq(z, noise_var, p=None):
    """
    由均衡后的 Z[k] 得到一个码字的 3984 个 LLR。
    约定 LLR = log P(b=0)/P(b=1)；正值更像 0。
    字节内顺序与 TX 打包一致：先 MSB。
    """
    d = derived(p)
    z = np.asarray(z, dtype=np.complex128).ravel()
    n_act = d["n_active"]
    llr = np.zeros(d["coded_bits"], dtype=np.float64)
    sigma = np.asarray(noise_var, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(n_act, float(sigma) + 1e-12)
    else:
        sigma = np.maximum(sigma.ravel()[:n_act], 1e-12)
    for i in range(n_act):
        d_i = i // 4
        b_i = 3 - (i % 4)
        scale = 2.0 / sigma[i]
        llr_i = scale * z[i].real
        llr_q = scale * z[i].imag
        # 字节位权：bit7 对应 scrambled_bin[8*d]，是第一符号的 Q
        pair = 3 - b_i
        llr[8 * d_i + 2 * pair] = llr_q
        llr[8 * d_i + 2 * pair + 1] = llr_i
    return llr


def descramble_llr(llr, seed=SCRAMBLER_SEED):
    """解扰：扰码比特为 1 时翻转 LLR 符号。每码字单独调用。"""
    llr = np.asarray(llr, dtype=np.float64).ravel()
    scram = lfsr_sequence(llr.size, seed=seed)
    return llr * (1 - 2 * scram.astype(np.float64))


def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size % 8:
        raise ValueError("比特数必须是 8 的倍数")
    packed = np.packbits(bits, bitorder="big")
    return packed.tobytes()


def parse_ph_header(blob):
    """
    解析 249 字节信息块中的 PH 头。
    CRC 覆盖到文件名最后一个字节，不含随后的 0x00。
    成功返回 dict，失败返回 None。
    """
    data = bytes(blob)
    if len(data) < 16 or data[:2] != HEADER_MAGIC:
        return None
    version = data[2]
    file_size = int.from_bytes(data[3:7], "big")
    declared_symbols = int.from_bytes(data[7:11], "big")
    try:
        zero_at = data.index(0, 11)
    except ValueError:
        return None
    name_bytes = data[11:zero_at]
    if zero_at + 5 > len(data):
        return None
    try:
        filename = name_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    crc_store = int.from_bytes(data[zero_at + 1 : zero_at + 5], "big")
    crc_calc = zlib.crc32(data[: 11 + len(name_bytes)]) & 0xFFFFFFFF
    if crc_store != crc_calc:
        return None
    return {
        "version": version,
        "file_size": file_size,
        "declared_symbols": declared_symbols,
        "filename": filename,
        "crc": crc_store,
    }


def search_ph_in_stream(info_bytes, info_block=249):
    """
    在解码后的信息字节流里滑动寻找 CRC 通过的 PH。
    优先按码字边界，再允许逐字节滑动以抗少量对齐误差。
    """
    buf = bytes(info_bytes)
    # 先按信息块边界
    for off in range(0, max(len(buf) - 15, 0) + 1, info_block):
        parsed = parse_ph_header(buf[off : off + info_block])
        if parsed is not None:
            parsed["offset"] = off
            return parsed
    magic = buf.find(HEADER_MAGIC)
    while magic >= 0:
        parsed = parse_ph_header(buf[magic : magic + info_block])
        if parsed is not None:
            parsed["offset"] = magic
            return parsed
        magic = buf.find(HEADER_MAGIC, magic + 1)
    return None


def _wav_to_float(x):
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        if x.dtype == np.int16:
            return x.astype(np.float64) / 32767.0
        info = np.iinfo(x.dtype)
        return x.astype(np.float64) / int(info.max)
    x = x.astype(np.float64)
    peak = np.max(np.abs(x)) if x.size else 1.0
    if peak > 1.5:
        x = x / 32767.0
    return x


def read_wav_channels(path):
    """读取 WAV 全部声道，各路归一到约 [-1, 1]。立体声不强行合成。"""
    fs, data = wavfile.read(str(path))
    x = np.asarray(data)
    if x.ndim == 1:
        return int(fs), [_wav_to_float(x)]
    return int(fs), [_wav_to_float(x[:, i]) for i in range(x.shape[1])]


def read_wav_mono(path):
    """读取 WAV 为 float64 单声道；多声道先取能量最大的一路。"""
    fs, channels = read_wav_channels(path)
    if len(channels) == 1:
        return fs, channels[0]
    power = [float(np.mean(ch * ch)) for ch in channels]
    return fs, channels[int(np.argmax(power))]


def slice_qpsk(z):
    z = np.asarray(z, dtype=np.complex128)
    return np.sign(z.real + 1e-30) + 1j * np.sign(z.imag + 1e-30)


def remap_info_and_parity_to_qpsk(coded_bytes, p=None):
    """把一个码字的 498 字节映射回数据子载波上的 QPSK。"""
    d = derived(p)
    grid = bytes_to_qpsk_grid(coded_bytes, d)
    return grid[active_bins(d)]


def safe_filename(name):
    name = Path(str(name)).name
    if not name or name in {".", ".."}:
        return "recovered.bin"
    return name
