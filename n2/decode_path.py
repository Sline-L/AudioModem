"""软 QPSK、解扰、LDPC、PH 滑动搜索。LDPC 直接使用仓库内 LDPC/new_ldpc。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from params import derived
from phy import (
    bits_to_bytes,
    descramble_llr,
    lfsr_sequence,
    qpsk_llr_from_eq,
    remap_info_and_parity_to_qpsk,
    search_ph_in_stream,
)

LDPC_ROOT = Path(__file__).resolve().parents[1] / "LDPC" / "new_ldpc"
LDPC_PY = LDPC_ROOT / "py"
if str(LDPC_PY) not in sys.path:
    sys.path.insert(0, str(LDPC_PY))


def make_coder(z=166):
    """与发射端相同：ldpc.code(standard='802.16', rate='1/2', z=...)。"""
    import ldpc

    cwd = os.getcwd()
    os.chdir(LDPC_ROOT)
    try:
        return ldpc.code(standard="802.16", rate="1/2", z=int(z))
    finally:
        os.chdir(cwd)


class LdpcBank:
    """发射端同一套 802.16 码的译码入口。"""

    def __init__(self, p=None):
        d = derived(p)
        self.d = d
        self.coder = make_coder(d["ldpc_z"])

    def decode_codeword(self, llr_scrambled):
        llr = descramble_llr(llr_scrambled)
        app, it = self.coder.decode(llr, dectype="sumprod2")
        hard = (np.asarray(app) < 0).astype(np.uint8)
        return hard[: self.coder.K], hard[: self.coder.N], int(it)


def llr_from_symbol(z, gate, rho, sigma2, h, p=None):
    """可靠性加权的每音噪声：σ_eq² = σ² / (|H|² g)。每音 σ² 时不再重复乘 ρ。"""
    sig = np.asarray(sigma2, dtype=np.float64)
    mag = np.abs(h) ** 2
    if sig.ndim == 0 or sig.size == 1:
        eq_var = (float(np.ravel(sig)[0]) + 1e-12) / (
            mag * np.maximum(rho * gate, 0.05) + 1e-12
        )
    else:
        eq_var = (sig.ravel()[: mag.size] + 1e-12) / (mag * np.maximum(gate, 0.05) + 1e-12)
    return qpsk_llr_from_eq(z, eq_var, p)


def decode_frame(z_list, gate_list, h_list, ce, bank, p=None):
    """逐符号用当时的 H 出 LLR，再按码字译码。"""
    d = derived(p)
    infos = []
    codeds = []
    iters = []
    llrs = []
    noise = ce.get("sigma2_tone", ce["sigma2"])
    for z, g, h in zip(z_list, gate_list, h_list):
        llr = llr_from_symbol(z, g, ce["rho"], noise, h, d)
        info, coded, it = bank.decode_codeword(llr)
        infos.append(info)
        codeds.append(coded)
        iters.append(it)
        llrs.append(llr)
    info_bytes = bits_to_bytes(np.concatenate(infos)) if infos else b""
    return {
        "info_bytes": info_bytes,
        "header": search_ph_in_stream(info_bytes, info_block=d["info_bytes"]),
        "coded_list": codeds,
        "iters": iters,
        "llr_list": llrs,
    }


def scramble_bits(bits):
    """与发射端相同：每个码字用 LFSR 再加扰后再映射。"""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    scram = lfsr_sequence(bits.size)
    return bits ^ scram.astype(np.uint8)


def rebuild_qpsk_from_coded(coded_list, p=None):
    """Turbo 用：未扰码硬判码字先加扰，再映射到空中接口 QPSK。"""
    grids = [
        remap_info_and_parity_to_qpsk(bits_to_bytes(scramble_bits(c)), p)
        for c in coded_list
    ]
    return np.stack(grids, axis=0) if grids else None


def extract_payload(info_bytes, header):
    """载荷从帧头信息块之后开始，长度只信 CRC 通过的 file_size。"""
    if header is None:
        return None
    start = int(header["offset"]) + 249
    size = int(header["file_size"])
    chunk = info_bytes[start : start + size]
    if len(chunk) != size:
        return None
    return chunk
