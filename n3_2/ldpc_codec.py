from __future__ import annotations

import ctypes
import platform
from pathlib import Path

import numpy as np

from lib.ldpc.py import ldpc
from n3_2.modem import scramble_bits


INFO_BITS = 1992
CODE_BITS = 3984


class PortableCode(ldpc.code):
    """IEEE LDPC code that locates its native decoder relative to this file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        long_dtype = np.int32 if ctypes.sizeof(ctypes.c_long) == 4 else np.int64
        self.vdeg = np.ascontiguousarray(self.vdeg, dtype=long_dtype)
        self.cdeg = np.ascontiguousarray(self.cdeg, dtype=long_dtype)
        self.intrlv = np.ascontiguousarray(self.intrlv, dtype=long_dtype)

    def _load_library(self):
        suffix = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
        library = Path(ldpc.__file__).resolve().parents[1] / "bin" / f"c_ldpc{suffix}"
        if not library.exists():
            raise FileNotFoundError(f"LDPC native library not found: {library}")
        return ctypes.CDLL(str(library))


class StandardLdpc:
    """IEEE 802.16 rate-1/2, Z=166 soft-decision block codec."""

    def __init__(self):
        self.code = PortableCode(standard="802.16", rate="1/2", z=166)
        if (self.code.K, self.code.N) != (INFO_BITS, CODE_BITS):
            raise RuntimeError("unexpected IEEE 802.16 LDPC dimensions")
        self._scrambler = scramble_bits(np.zeros(CODE_BITS, dtype=np.uint8))
        self._parity_check = None

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        bits = _binary_bits(info_bits, INFO_BITS, "info_bits")
        packed = np.packbits(bits, bitorder="big")
        encoded = self.code.encode(np.unpackbits(packed, bitorder="big"))
        return np.asarray(encoded, dtype=np.uint8).reshape(CODE_BITS)

    def decode_llr(self, scrambled_llr: np.ndarray) -> tuple[np.ndarray, bool]:
        llr = _llrs(scrambled_llr)
        descrambled = llr * (1.0 - 2.0 * self._scrambler)
        primary = self._decode(descrambled)

        # The public contract is scrambled LLRs.  Retaining the raw candidate also
        # makes noiseless codeword diagnostics usable without a separate API.
        raw = self._decode(llr)
        primary_score = float(np.dot(llr, 1.0 - 2.0 * scramble_bits(primary)))
        raw_score = float(np.dot(llr, 1.0 - 2.0 * raw))
        decoded = primary if primary_score >= raw_score else raw
        return decoded[:INFO_BITS].astype(np.uint8, copy=False), not self.syndrome(decoded).any()

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        word = _binary_bits(bits, CODE_BITS, "bits")
        if self._parity_check is None:
            self._parity_check = self.code.pcmat().astype(np.uint8, copy=False)
        return np.mod(self._parity_check.dot(word), 2).astype(np.uint8)

    def _decode(self, llr: np.ndarray) -> np.ndarray:
        app, _ = self.code.decode(llr, "sumprod2", 0.7)
        return (np.asarray(app) < 0.0).astype(np.uint8)


def _binary_bits(values: np.ndarray, size: int, name: str) -> np.ndarray:
    bits = np.asarray(values)
    if bits.shape != (size,):
        raise ValueError(f"{name} must be a one-dimensional array of {size} bits")
    if bits.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8")
    if np.any(bits > 1):
        raise ValueError(f"{name} must contain only zero or one values")
    return bits


def _llrs(values: np.ndarray) -> np.ndarray:
    llr = np.asarray(values, dtype=float)
    if llr.shape != (CODE_BITS,):
        raise ValueError(f"scrambled_llr must be a one-dimensional array of {CODE_BITS} LLRs")
    if not np.isfinite(llr).all():
        raise ValueError("scrambled_llr must contain finite values")
    return llr
