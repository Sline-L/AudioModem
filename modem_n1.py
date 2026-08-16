from pathlib import Path
import struct
import wave
import zlib

import numpy as np
from scipy import signal


FS = 48000
N = 4096
CP = 2048
L = N + CP
I16 = 32768.0

BAND_HZ = (2000.0, 7000.0)
ACTIVE_BINS = np.arange(
    int(np.ceil(BAND_HZ[0] * N / FS)),
    int(np.floor(BAND_HZ[1] * N / FS)) + 1,
    dtype=int,
)

MODS = ("bpsk", "qpsk", "qam16")
MAGIC = b"AMS0"
VERSION = 1
HEADER_BODY = struct.Struct(">4sBBHQI120s")
HEADER_SIZE = HEADER_BODY.size + 4


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != FS:
            raise ValueError(f"expected mono 16-bit {FS} Hz wav: {path}")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / I16


def wav_gain(samples):
    peak = np.max(np.abs(samples))
    if peak == 0:
        raise ValueError("empty signal")
    return 0.95 * (I16 - 1) / I16 / peak


def write_wav(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples * wav_gain(samples) * I16, -I16 + 1, I16 - 1).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(FS)
        wav.writeframes(pcm.tobytes())


def bits_per_symbol(mod):
    if mod == "bpsk":
        return 1
    if mod == "qpsk":
        return 2
    if mod == "qam16":
        return 4
    raise ValueError(f"mod must be one of {MODS}")


def bits_from_bytes(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bytes_from_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits[: bits.size // 8 * 8], bitorder="big").tobytes()


def _pam4(a, b):
    return np.select(
        [(a == 0) & (b == 0), (a == 0) & (b == 1), (a == 1) & (b == 1)],
        [3, 1, -1],
        default=-3,
    )


def mod_symbols(data, mod="qpsk"):
    bits = bits_from_bytes(data)
    width = bits_per_symbol(mod)
    if bits.size % width:
        bits = np.r_[bits, np.zeros(width - bits.size % width, dtype=np.uint8)]
    bits = bits.reshape(-1, width)
    if mod == "bpsk":
        return np.where(bits[:, 0], -1.0, 1.0).astype(complex)
    if mod == "qpsk":
        return (np.where(bits[:, 1], -1.0, 1.0) + 1j * np.where(bits[:, 0], -1.0, 1.0)) / np.sqrt(2)
    real = _pam4(bits[:, 2], bits[:, 3])
    imag = _pam4(bits[:, 0], bits[:, 1])
    return (real + 1j * imag) / np.sqrt(10)


def _bits_from_pam4(values):
    out = np.zeros((len(values), 2), dtype=np.uint8)
    out[values < 2, 1] = 1
    out[values < 0, 0] = 1
    out[values < -2, 1] = 0
    return out


def bytes_from_mod(symbols, mod="qpsk"):
    symbols = np.asarray(symbols).ravel()
    if mod == "bpsk":
        bits = (symbols.real < 0).astype(np.uint8)
    elif mod == "qpsk":
        bits = np.c_[symbols.imag < 0, symbols.real < 0].astype(np.uint8).ravel()
    elif mod == "qam16":
        z = symbols * np.sqrt(10)
        bits = np.c_[_bits_from_pam4(z.imag), _bits_from_pam4(z.real)].ravel()
    else:
        raise ValueError(f"mod must be one of {MODS}")
    return bytes_from_bits(bits)


def pack_file(path):
    path = Path(path)
    body = path.read_bytes()
    name = path.name.encode("utf-8")
    if len(name) > 120:
        raise ValueError("UTF-8 filename must be at most 120 bytes")
    first = HEADER_BODY.pack(
        MAGIC,
        VERSION,
        len(name),
        0,
        len(body),
        zlib.crc32(body),
        name.ljust(120, b"\0"),
    )
    return first + struct.pack(">I", zlib.crc32(first)) + body


def unpack_file(data):
    if len(data) < HEADER_SIZE:
        raise ValueError("payload is shorter than header")
    first = data[: HEADER_BODY.size]
    stored_header_crc = struct.unpack(">I", data[HEADER_BODY.size:HEADER_SIZE])[0]
    if zlib.crc32(first) != stored_header_crc:
        raise ValueError("header CRC mismatch")
    magic, version, name_len, _, size, file_crc, raw_name = HEADER_BODY.unpack(first)
    if magic != MAGIC or version != VERSION or name_len > len(raw_name):
        raise ValueError("unsupported n1 header")
    body = data[HEADER_SIZE : HEADER_SIZE + size]
    if len(body) != size:
        raise ValueError("recording ended before full payload")
    if zlib.crc32(body) != file_crc:
        raise ValueError("file CRC mismatch")
    return Path(raw_name[:name_len].decode("utf-8")).name, body


def file_symbols(path, mod="qpsk", bins=ACTIVE_BINS):
    symbols = mod_symbols(pack_file(path), mod)
    rows = int(np.ceil(len(symbols) / len(bins)))
    padded = np.zeros(rows * len(bins), complex)
    padded[: len(symbols)] = symbols
    return padded.reshape(rows, len(bins)), len(symbols)


def random_qpsk(rows, bins=ACTIVE_BINS, seed=2026):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (rows, len(bins), 2), dtype=np.uint8)
    return (np.where(bits[:, :, 1], -1.0, 1.0) + 1j * np.where(bits[:, :, 0], -1.0, 1.0)) / np.sqrt(2)


def ofdm_tx(symbols, bins=ACTIVE_BINS):
    freq = np.zeros((len(symbols), N), complex)
    freq[:, bins] = symbols
    freq[:, N - bins] = np.conj(symbols)
    time = np.fft.ifft(freq, axis=1).real
    return np.c_[time[:, -CP:], time].ravel()


def ofdm_rx(samples, bins=ACTIVE_BINS):
    rows = len(samples) // L
    if rows == 0:
        return np.empty((0, len(bins)), complex)
    time = samples[: rows * L].reshape(rows, L)[:, CP:]
    return np.fft.fft(time, axis=1)[:, bins]


def find_sync(rx, template):
    if len(rx) < len(template):
        raise ValueError("receive wav is shorter than sync template")
    corr = signal.correlate(rx, template, mode="valid", method="fft")
    peak = int(np.argmax(np.abs(corr)))
    energy = np.linalg.norm(rx[peak : peak + len(template)]) * np.linalg.norm(template)
    return peak, float(abs(corr[peak]) / energy) if energy else 0.0


def estimate_channel(received, known):
    rows = min(len(received), len(known))
    if rows == 0:
        raise ValueError("no preamble symbols available for channel estimate")
    return np.mean(received[:rows] / known[:rows], axis=0)


def equalize(received, h):
    return received / np.where(np.abs(h) > 1e-12, h, 1.0)


def profile_meta():
    return {
        "profile": "n1_n4096_cp2048_2k_7k",
        "fs": FS,
        "fft_size": N,
        "cp": CP,
        "symbol_len": L,
        "band_hz": [float(BAND_HZ[0]), float(BAND_HZ[1])],
        "active_bins": int(len(ACTIVE_BINS)),
        "bin_start": int(ACTIVE_BINS[0]),
        "bin_end": int(ACTIVE_BINS[-1]),
        "bin_start_hz": float(ACTIVE_BINS[0] * FS / N),
        "bin_end_hz": float(ACTIVE_BINS[-1] * FS / N),
    }
