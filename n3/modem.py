from __future__ import annotations

import math
import random
import struct
import wave
import zlib
from pathlib import Path

import numpy as np
from scipy import signal


FS = 48_000
N = 8192
CP = 2048
L = N + CP
I16 = 32768.0

BAND_START_BIN = 400
BAND_END_BIN = 2392
ACTIVE_BINS = np.arange(BAND_START_BIN, BAND_END_BIN, dtype=int)
ACTIVE_COUNT = len(ACTIVE_BINS)

INFO_BYTES = 249
CODE_BYTES = 498
INFO_BITS = INFO_BYTES * 8
CODE_BITS = CODE_BYTES * 8
TRAINING_SYMBOLS = 8
TRAILING_TRAINING_SYMBOLS = 8

CHIRP_SECONDS = 3.0
CHIRP_START_HZ = 100.0
CHIRP_END_HZ = 20_000.0
CHIRP_SAMPLES = int(round(CHIRP_SECONDS * FS))
SILENCE_SECONDS = 0.5
SILENCE_SAMPLES = int(round(SILENCE_SECONDS * FS))

MAGIC = b"PH"
VERSION = 0
SCRAMBLER_SEED = 0x5A4D
MAX_FILENAME_BYTES = INFO_BYTES - 16


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bytes_from_bits(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    values = values[: values.size // 8 * 8]
    return np.packbits(values, bitorder="big").tobytes()


def header_bytes(name: str | Path, file_size: int, payload_symbols: int) -> bytes:
    encoded_name = Path(name).name.encode("utf-8")
    if len(encoded_name) > MAX_FILENAME_BYTES:
        raise ValueError(f"UTF-8 filename must be at most {MAX_FILENAME_BYTES} bytes")
    if not 0 <= int(file_size) <= 0xFFFFFFFF:
        raise ValueError("file size must fit in four bytes")
    if not 0 <= int(payload_symbols) <= 0xFFFFFFFF:
        raise ValueError("payload symbol count must fit in four bytes")

    raw = bytearray(INFO_BYTES)
    raw[:2] = MAGIC
    raw[2] = VERSION
    raw[3:7] = int(file_size).to_bytes(4, "big")
    raw[7:11] = int(payload_symbols).to_bytes(4, "big")
    raw[11:11 + len(encoded_name)] = encoded_name
    crc = zlib.crc32(raw[:11 + len(encoded_name)])
    raw[12 + len(encoded_name):16 + len(encoded_name)] = crc.to_bytes(4, "big")
    return bytes(raw)


def parse_header(raw: bytes) -> dict:
    if len(raw) < INFO_BYTES:
        raise ValueError("header is shorter than 249 bytes")
    first = raw[:INFO_BYTES]
    if first[:2] != MAGIC or first[2] != VERSION:
        raise ValueError("unsupported N3 header")
    name_end = first.find(b"\0", 11)
    if name_end < 0:
        raise ValueError("header filename is not terminated")
    if name_end > MAX_FILENAME_BYTES + 11:
        raise ValueError("header filename is too long")
    crc_offset = 12 + (name_end - 11)
    if crc_offset + 4 > INFO_BYTES:
        raise ValueError("header CRC is outside the header")
    stored_crc = int.from_bytes(first[crc_offset:crc_offset + 4], "big")
    if zlib.crc32(first[:11 + (name_end - 11)]) != stored_crc:
        raise ValueError("header CRC mismatch")
    try:
        name = first[11:name_end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("header filename is not UTF-8") from exc
    if not name or name in {".", ".."}:
        raise ValueError("header filename is empty")
    return {
        "name": Path(name).name,
        "file_size": int.from_bytes(first[3:7], "big"),
        "payload_symbols": int.from_bytes(first[7:11], "big"),
        "version": int(first[2]),
        "header_crc32": stored_crc,
    }


def scrambler_sequence(length: int, seed: int = SCRAMBLER_SEED) -> np.ndarray:
    state = int(seed)
    result = np.empty(int(length), dtype=np.uint8)
    for index in range(len(result)):
        result[index] = (state >> 14) & 1
        feedback = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) & 0x7FFF) | feedback
    return result


def qpsk_from_bits(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    if values.size % 2:
        values = np.r_[values, 0]
    pairs = values.reshape(-1, 2)
    pairs = pairs.astype(float)
    return (1 - 2 * pairs[:, 1]) + 1j * (1 - 2 * pairs[:, 0])


def bytes_to_qpsk(data: bytes) -> np.ndarray:
    return qpsk_from_bits(bits_from_bytes(data))


def qpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols, dtype=complex).ravel()
    return np.column_stack((values.imag < 0, values.real < 0)).astype(np.uint8).ravel()


def training_symbols(seed: int = 80) -> tuple[np.ndarray, np.ndarray]:
    saved_state = random.getstate()
    random.seed(seed)
    try:
        freq = np.zeros((TRAINING_SYMBOLS * 2, N), dtype=complex)
        for row in range(len(freq)):
            for bin_index in range(BAND_START_BIN, BAND_END_BIN):
                value = random.randint(0, 3)
                freq[row, bin_index] = (
                    -2 * (value & 1) + 1
                    - 2 * ((value >> 1) & 1) * 1j
                    + 1j
                )
                freq[row, N - bin_index] = np.conj(freq[row, bin_index])
        time = np.fft.ifft(freq, axis=1).real
        with_cp = np.hstack((time[:, -CP:], time))
        return freq, with_cp
    finally:
        random.setstate(saved_state)


def chirp_wave() -> np.ndarray:
    t = np.linspace(0.0, CHIRP_SECONDS, CHIRP_SAMPLES, endpoint=False)
    bandwidth = CHIRP_END_HZ - CHIRP_START_HZ
    phase = 2 * np.pi * CHIRP_START_HZ * t + np.pi * bandwidth * t * t / CHIRP_SECONDS
    return 0.8 * np.cos(phase)


def ofdm_tx(rows: np.ndarray) -> np.ndarray:
    active = np.asarray(rows, dtype=complex)
    if active.ndim == 1:
        active = active.reshape(1, -1)
    if active.shape[1] != ACTIVE_COUNT:
        raise ValueError(f"expected {ACTIVE_COUNT} active carriers, got {active.shape[1]}")
    freq = np.zeros((len(active), N), dtype=complex)
    freq[:, ACTIVE_BINS] = active
    freq[:, N - ACTIVE_BINS] = np.conj(active)
    time = np.fft.ifft(freq, axis=1).real
    with_cp = np.hstack((time[:, -CP:], time))
    return np.concatenate(with_cp)


def ofdm_rx(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float).ravel()
    if len(values) % L:
        raise ValueError("OFDM sample block is not an integer number of symbols")
    rows = values.reshape(-1, L)[:, CP:]
    freq = np.fft.fft(rows, axis=1)
    return freq[:, ACTIVE_BINS]


def payload_symbol_count(file_size: int) -> int:
    return int(math.ceil(2 * int(file_size) / CODE_BYTES))


def coded_symbol_count(file_size: int) -> int:
    combined = INFO_BYTES + int(file_size)
    padded = int(math.ceil(combined / CODE_BYTES) * CODE_BYTES)
    return padded // INFO_BYTES


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != FS:
            raise ValueError(f"expected mono 16-bit {FS} Hz wav: {path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / I16


def write_wav(path: str | Path, samples: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.asarray(samples, dtype=float) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(FS)
        wav.writeframes(pcm.tobytes())


def symbol_frequency_hz() -> np.ndarray:
    return ACTIVE_BINS.astype(float) * FS / N
