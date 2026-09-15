from __future__ import annotations

import ctypes
import functools
import platform
from pathlib import Path

import numpy as np

from n3.modem import CODE_BITS, INFO_BYTES, INFO_BITS, qpsk_to_bits, scrambler_sequence


try:
    import lib.ldpc.py.ldpc as _ldpc
except ImportError as exc:  # pragma: no cover - import failure is environment-specific
    raise ImportError("N3 requires the repository lib/ldpc Python package") from exc


class PortableCode(_ldpc.code):
    """Use the checked-in native library independent of the current directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        long_dtype = np.int32 if ctypes.sizeof(ctypes.c_long) == 4 else np.int64
        self.vdeg = np.ascontiguousarray(self.vdeg, dtype=long_dtype)
        self.cdeg = np.ascontiguousarray(self.cdeg, dtype=long_dtype)
        self.intrlv = np.ascontiguousarray(self.intrlv, dtype=long_dtype)

    def _load_library(self):
        suffix = {"Windows": ".dll", "Darwin": ".dylib"}.get(platform.system(), ".so")
        library = Path(_ldpc.__file__).resolve().parents[1] / "bin" / f"c_ldpc{suffix}"
        if not library.exists():
            raise FileNotFoundError(f"LDPC native library not found: {library}")
        return ctypes.CDLL(str(library))


@functools.lru_cache(maxsize=1)
def get_code():
    return PortableCode("802.16", "1/2", 166, "A")


def encode_block(info: bytes) -> np.ndarray:
    if len(info) != INFO_BYTES:
        raise ValueError(f"LDPC information block must be {INFO_BYTES} bytes")
    encoded = np.asarray(get_code().encode(_bytes_to_bits(info)), dtype=np.uint8)
    return encoded ^ scrambler_sequence(CODE_BITS)


def decode_block(llr: np.ndarray) -> tuple[bytes, int]:
    values = np.asarray(llr, dtype=float).ravel()
    if len(values) != CODE_BITS:
        raise ValueError(f"LDPC channel block must contain {CODE_BITS} LLRs")
    unscrambled = values * (1.0 - 2.0 * scrambler_sequence(CODE_BITS))
    app, iterations = get_code().decode(unscrambled, "sumprod2", 0.7)
    bits = (np.asarray(app[:INFO_BITS]) < 0).astype(np.uint8)
    return _bits_to_bytes(bits), int(iterations)


def llr_from_qpsk(symbols: np.ndarray, scale: float | np.ndarray = 3.0) -> np.ndarray:
    values = np.asarray(symbols, dtype=complex).ravel()
    scale_values = np.asarray(scale, dtype=float)
    if scale_values.ndim == 0:
        return float(scale_values) * np.column_stack((values.imag, values.real)).ravel()
    scale_values = scale_values.ravel()
    if len(scale_values) != len(values):
        raise ValueError("per-carrier QPSK scale must match the symbol count")
    return np.column_stack((values.imag * scale_values, values.real * scale_values)).ravel()


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
