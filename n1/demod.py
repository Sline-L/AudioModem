# -*- coding: utf-8 -*-
"""解调、解扰、LDPC 与 PH 文件头解析。"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np


def sequence(length: int, seed: int = 0x5A4D) -> np.ndarray:
    """与 TX 相同的 15 位 LFSR 加扰序列。"""
    state = int(seed)
    result = np.empty(length, dtype=int)
    for i in range(length):
        result[i] = (state >> 14) & 1
        feedback = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) & 0x7FFF) | feedback
    return result


def symbols_to_byte_llrs(symbols: np.ndarray, chunk_size: int,
                         noise_var: float = 1.0,
                         tone_scale: np.ndarray | None = None) -> np.ndarray:
    """1992 个 QPSK → 与 TX 加扰比特流一致的 3984 个 LLR（MSB-first）。

    TX ``mapp`` 对字节用 **从 LSB 计数** 的位移 ``>>(b_i*2)`` 取 dibit，
    而 LDPC/加扰比特流按 **MSB-first** 打包（``scambled_bin[8*j+0]`` 为字节最高位）。
    因此 LSB 位号 ``p`` 对应线性下标 ``7-p``。

    ``tone_scale``：可选每子载波门控（n3_2 风格），乘到软信息幅度上。
    """
    assert symbols.size == 4 * chunk_size
    llr = np.zeros(chunk_size * 8, dtype=float)
    sigma2 = max(float(noise_var), 1e-12)
    if tone_scale is None:
        scale = np.ones(4 * chunk_size, dtype=np.float64)
    else:
        scale = np.asarray(tone_scale, dtype=np.float64).ravel()
        assert scale.size == 4 * chunk_size
    for i in range(4 * chunk_size):
        d_i = i // 4
        b_i = 3 - (i % 4)
        sym = symbols[i]
        g = float(scale[i]) / sigma2
        # I→bit0@LSB位 b_i*2，Q→bit1@LSB位 b_i*2+1
        p0 = b_i * 2
        p1 = b_i * 2 + 1
        llr[d_i * 8 + (7 - p0)] = 2.0 * np.real(sym) * g
        llr[d_i * 8 + (7 - p1)] = 2.0 * np.imag(sym) * g
    return np.clip(llr, -30.0, 30.0)


def descramble_llr(llr: np.ndarray, seed: int = 0x5A4D) -> np.ndarray:
    """解扰：加扰比特为 1 时翻转 LLR 符号。"""
    scram = sequence(llr.size, seed=seed)
    out = llr.copy()
    out[scram == 1] *= -1
    return out


def llr_to_hard_bits(llr: np.ndarray) -> np.ndarray:
    """LLR→硬比特：LLR>0 → 0，否则 1。"""
    return (llr < 0).astype(int)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """MSB 优先打包。"""
    bits = np.asarray(bits, dtype=int).ravel() & 1
    assert bits.size % 8 == 0
    out = bytearray(bits.size // 8)
    for j in range(len(out)):
        v = 0
        for b in range(8):
            v = (v << 1) | int(bits[8 * j + b])
        out[j] = v
    return bytes(out)


def decode_codeword(llr_scrambled: np.ndarray, coder, use_soft: bool = True):
    """解扰 + LDPC。

    返回 (info_bytes, coded_hard_bits长度N)。
    课程包：`app, it = coder.decode(llr)`，`app<0` → 比特 1。
    """
    llr = descramble_llr(llr_scrambled)
    if not use_soft:
        hard = llr_to_hard_bits(llr)
        llr = np.where(hard == 0, 12.0, -12.0)

    result = coder.decode(llr)
    if isinstance(result, tuple):
        app, _it = result
        hat = (np.asarray(app).ravel() < 0).astype(int)
    else:
        hat = np.asarray(result).ravel().astype(int)

    n = int(getattr(coder, "N", getattr(coder, "n", len(hat))))
    k = int(getattr(coder, "K", getattr(coder, "k", n // 2)))
    coded = hat[:n]
    info_bits = coded[:k]
    return bits_to_bytes(info_bits), coded


def scramble_bits(bits: np.ndarray, seed: int = 0x5A4D) -> np.ndarray:
    """与 TX 相同：码字比特 XOR LFSR（用于 Turbo 重建空中 QPSK）。"""
    bits = np.asarray(bits, dtype=int).ravel() & 1
    return bits ^ sequence(bits.size, seed=seed)


def bytes_to_qpsk(chunk: np.ndarray | bytes, chunk_size: int = 498) -> np.ndarray:
    """TX ``mapp`` 的单符号版本：498 字节 → 1992 个 QPSK（有效正频率）。"""
    data = np.frombuffer(bytes(chunk), dtype=np.uint8)
    if data.size < chunk_size:
        data = np.pad(data, (0, chunk_size - data.size))
    data = data[:chunk_size].astype(int)
    out = np.zeros(4 * chunk_size, dtype=complex)
    for i in range(4 * chunk_size):
        d_i = i // 4
        b_i = 3 - (i % 4)
        out[i] = (
            -2 * ((data[d_i] >> (b_i * 2)) & 1)
            + 1
            + (-2 * ((data[d_i] >> (b_i * 2 + 1)) & 1) * 1j + 1j)
        )
    return out


def rebuild_air_qpsk(coded_bits: np.ndarray, chunk_size: int = 498) -> np.ndarray:
    """未加扰硬判码字 → 加扰 → QPSK（与空中接口一致）。"""
    scrambled = scramble_bits(coded_bits)
    return bytes_to_qpsk(bits_to_bytes(scrambled), chunk_size=chunk_size)


def parse_ph_header(data: bytes) -> dict:
    """解析 PH 文件头。

    布局：PH | ver | file_size(4) | symbol_count(4) | filename | 0x00 | CRC32(4) | 填充
    CRC 覆盖到文件名最后一个字节（不含 0x00）。
    """
    if len(data) < 16 or data[0:2] != b"PH":
        raise ValueError("文件头魔数不是 PH")
    version = data[2]
    file_size = int.from_bytes(data[3:7], "big")
    symbol_count = int.from_bytes(data[7:11], "big")
    end = data.find(b"\x00", 11)
    if end < 0:
        raise ValueError("文件头缺少文件名结束符")
    name_bytes = data[11:end]
    filename = name_bytes.decode("utf-8", errors="replace")
    crc_offset = end + 1
    if crc_offset + 4 > len(data):
        raise ValueError("文件头 CRC 越界")
    crc_rx = int.from_bytes(data[crc_offset : crc_offset + 4], "big")
    crc_calc = zlib.crc32(data[:end]) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        raise ValueError(f"文件头 CRC 失败：rx={crc_rx:08x} calc={crc_calc:08x}")
    if file_size < 0 or file_size > 50_000_000:
        raise ValueError(f"file_size 不合理: {file_size}")
    return {
        "version": version,
        "file_size": file_size,
        "symbol_count": symbol_count,
        "filename": filename,
        "header_info_bytes": len(data),
        "crc_ok": True,
    }


def search_ph_in_stream(info_bytes: bytes, info_block: int = 249) -> tuple[dict, int]:
    """在信息字节流中滑动寻找 CRC 通过的 PH。

    返回 (header_dict, 头起始字节偏移)。
    先按码字对齐尝试，再全流滑动。
    """
    buf = bytes(info_bytes)
    # 1) 码字对齐
    for off in range(0, max(1, len(buf) - info_block + 1), info_block):
        try:
            return parse_ph_header(buf[off : off + info_block]), off
        except ValueError:
            continue
    # 2) 字节滑动（内容里碰巧出现 PH 也必须过 CRC）
    start = 0
    while True:
        magic = buf.find(b"PH", start)
        if magic < 0 or magic + info_block > len(buf):
            break
        try:
            return parse_ph_header(buf[magic : magic + info_block]), magic
        except ValueError:
            start = magic + 1
            continue
    raise ValueError("未找到 CRC 通过的 PH 文件头")


def extract_payload(info_stream: bytes, file_size: int, header_offset: int = 0,
                    header_len: int = 249) -> bytes:
    """从头起始偏移之后跳过一个信息块长度，截取 file_size。"""
    # 标准：头独占首个 249B 信息块；若滑动找到的头不在块首，仍取「头所在块」之后的载荷
    # 对齐情况：offset=0 → payload 从 249 开始
    # 滑动到块内 offset=k：载荷从 (k 所在块末尾) 开始更稳，这里采用
    # payload 起点 = ((header_offset // header_len) + 1) * header_len
    block_end = ((header_offset // header_len) + 1) * header_len
    payload = info_stream[block_end : block_end + file_size]
    if len(payload) < file_size:
        raise ValueError(f"载荷不足：需要 {file_size}，得到 {len(payload)}")
    return payload


def write_recovered(out_dir: str | Path, filename: str, payload: bytes) -> Path:
    """写出还原文件。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 防止路径穿越
    safe = Path(filename).name
    path = out_dir / safe
    path.write_bytes(payload)
    return path
