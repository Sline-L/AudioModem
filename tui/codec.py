from __future__ import annotations

import math
from pathlib import Path
import struct
import zlib

import numpy as np

from . import legacy_step8 as legacy
from .profiles import ModemProfile

try:
    from reedsolo import RSCodec, ReedSolomonError
except ImportError:  # Encoding without RS remains usable before optional deps are installed.
    RSCodec = None
    ReedSolomonError = Exception


MAGIC = b"AMT1"
VERSION = 1
HEADER_SIZE = 128
RS_DATA = 223
RS_PARITY = 32
RS_CODEWORD = RS_DATA + RS_PARITY


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bytes_from_bits(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    return np.packbits(values[: values.size // 8 * 8], bitorder="big").tobytes()


def rs_encoded_bytes(raw_bytes: int) -> int:
    return math.ceil(raw_bytes / RS_DATA) * RS_CODEWORD


def encoded_bit_length(raw_bytes: int, mode: str) -> int:
    inner_bytes = rs_encoded_bytes(raw_bytes) if mode in ("rs", "rs_conv") else raw_bytes
    if mode in ("conv", "rs_conv"):
        return legacy.encoded_length(inner_bytes)
    return inner_bytes * 8


def _rs_codec():
    if RSCodec is None:
        raise RuntimeError("RS FEC requires: pip install -r tui/requirements.txt")
    return RSCodec(RS_PARITY)


def rs_encode(raw: bytes) -> bytes:
    codec = _rs_codec()
    result = bytearray()
    for start in range(0, len(raw), RS_DATA):
        chunk = raw[start : start + RS_DATA].ljust(RS_DATA, b"\0")
        result.extend(codec.encode(chunk))
    return bytes(result)


def rs_decode(encoded: bytes, raw_bytes: int) -> bytes:
    codec = _rs_codec()
    expected = rs_encoded_bytes(raw_bytes)
    if len(encoded) < expected:
        raise ValueError(f"truncated RS frame: need {expected} bytes, got {len(encoded)}")
    result = bytearray()
    for start in range(0, expected, RS_CODEWORD):
        try:
            decoded = codec.decode(encoded[start : start + RS_CODEWORD])[0]
        except ReedSolomonError as exc:
            raise ValueError(f"Reed-Solomon decode failed: {exc}") from exc
        result.extend(decoded)
    return bytes(result[:raw_bytes])


def fec_encode(raw: bytes, mode: str, seed: int) -> np.ndarray:
    protected = rs_encode(raw) if mode in ("rs", "rs_conv") else raw
    bits = bits_from_bytes(protected)
    coded = legacy.conv_encode(bits) if mode in ("conv", "rs_conv") else bits
    return legacy.interleave(coded, seed)[0]


def fec_decode_values(values: np.ndarray, raw_bytes: int, mode: str) -> bytes:
    protected_bytes = rs_encoded_bytes(raw_bytes) if mode in ("rs", "rs_conv") else raw_bytes
    if mode in ("conv", "rs_conv"):
        bits = legacy.viterbi_decode(values, protected_bytes * 8)
    else:
        bits = (values < 0).astype(np.uint8)
    protected = bytes_from_bits(bits)
    return rs_decode(protected, raw_bytes) if mode in ("rs", "rs_conv") else protected[:raw_bytes]


def fec_decode(llr: np.ndarray, raw_bytes: int, mode: str, seed: int) -> bytes:
    need = encoded_bit_length(raw_bytes, mode)
    if len(llr) < need:
        raise ValueError(f"truncated FEC frame: need {need} LLRs, got {len(llr)}")
    values = legacy.deinterleave(np.asarray(llr[:need]), seed)
    return fec_decode_values(values, raw_bytes, mode)


def header_bytes(path: Path, profile: ModemProfile) -> bytes:
    body = Path(path).read_bytes()
    name = Path(path).name.encode("utf-8")
    if len(name) > 100:
        raise ValueError("UTF-8 filename must be at most 100 bytes")
    blocks = math.ceil(len(body) / profile.block_size)
    profile_tag = int(profile.profile_id[:4], 16)
    first = struct.pack(
        ">4sBBHQHHI100s",
        MAGIC,
        VERSION,
        len(name),
        profile_tag,
        len(body),
        profile.block_size,
        blocks,
        zlib.crc32(body),
        name.ljust(100, b"\0"),
    )
    return first + struct.pack(">I", zlib.crc32(first))


def parse_header(raw: bytes, profile: ModemProfile) -> dict:
    if len(raw) != HEADER_SIZE:
        raise ValueError("invalid TUI header length")
    first, stored_crc = raw[:-4], struct.unpack(">I", raw[-4:])[0]
    if zlib.crc32(first) != stored_crc:
        raise ValueError("header CRC mismatch")
    magic, version, name_len, tag, size, block_size, blocks, file_crc, name_raw = struct.unpack(
        ">4sBBHQHHI100s", first
    )
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported TUI modem header")
    if tag != int(profile.profile_id[:4], 16):
        raise ValueError("recording profile does not match selected profile")
    if name_len > 100 or block_size < 1 or blocks != math.ceil(size / block_size):
        raise ValueError("invalid header fields")
    return {
        "name": Path(name_raw[:name_len].decode("utf-8")).name,
        "file_size": int(size),
        "block_size": int(block_size),
        "block_count": int(blocks),
        "file_crc32": int(file_crc),
        "profile_tag": int(tag),
    }


def data_block_bytes(index: int, chunk: bytes, block_size: int) -> bytes:
    if len(chunk) > block_size:
        raise ValueError("chunk exceeds block size")
    prefix = struct.pack(">HH", index, len(chunk))
    return prefix + chunk.ljust(block_size, b"\0") + struct.pack(">I", zlib.crc32(prefix + chunk))


def parse_data_block(raw: bytes, expected_index: int, block_size: int) -> bytes:
    if len(raw) != block_size + 8:
        raise ValueError("invalid data block length")
    index, size = struct.unpack(">HH", raw[:4])
    if index != expected_index or size > block_size:
        raise ValueError("invalid data block header")
    chunk = raw[4 : 4 + size]
    if zlib.crc32(raw[:4] + chunk) != struct.unpack(">I", raw[-4:])[0]:
        raise ValueError("data block CRC mismatch")
    return chunk


def encoded_frames(path: Path, profile: ModemProfile) -> tuple[list[np.ndarray], list[str], list[bytes]]:
    header = header_bytes(path, profile)
    frames: list[np.ndarray] = []
    labels: list[str] = []
    raw_frames: list[bytes] = []
    for repeat in range(profile.header_repeats):
        frames.append(fec_encode(header, profile.fec, profile.fec_seed + repeat))
        labels.append(f"header{repeat + 1}")
        raw_frames.append(header)
    body = Path(path).read_bytes()
    for index, start in enumerate(range(0, len(body), profile.block_size)):
        raw = data_block_bytes(index, body[start : start + profile.block_size], profile.block_size)
        frames.append(fec_encode(raw, profile.fec, profile.fec_seed + 1000 + index))
        labels.append(f"block{index}")
        raw_frames.append(raw)
    return frames, labels, raw_frames


def map_data(bits: np.ndarray, modulation: str) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if modulation == "bpsk":
        return 1.0 - 2.0 * values
    if len(values) % 2:
        values = np.r_[values, 0]
    pairs = values.reshape(-1, 2)
    return ((1.0 - 2.0 * pairs[:, 0]) + 1j * (1.0 - 2.0 * pairs[:, 1])) / np.sqrt(2)


def demap_llr(equalized_numerator: np.ndarray, variance: np.ndarray, modulation: str) -> np.ndarray:
    if modulation == "bpsk":
        return 2.0 * np.real(equalized_numerator) / variance
    scale = 2.0 * np.sqrt(2.0) / variance
    pair = np.c_[scale * np.real(equalized_numerator), scale * np.imag(equalized_numerator)]
    return pair.ravel()
