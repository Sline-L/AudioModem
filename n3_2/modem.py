import random
import wave
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import chirp

FS = 48000
N = 8192
CP = 2048
L = N + CP
K0 = 400
K1 = 2391
M = K1 - K0 + 1


def header_bytes(name, size, payload_symbols):
    name_bytes = str(name).encode("utf-8")
    if not 0 <= int(size) <= 0xFFFFFFFF or not 0 <= int(payload_symbols) <= 0xFFFFFFFF:
        raise ValueError("header integers must fit uint32")
    if len(name_bytes) + 16 > 249:
        raise ValueError("filename is too long")
    raw = bytearray(249)
    raw[:2] = b"PH"
    raw[2] = 0
    raw[3:7] = int(size).to_bytes(4, "big")
    raw[7:11] = int(payload_symbols).to_bytes(4, "big")
    raw[11:11 + len(name_bytes)] = name_bytes
    end = 11 + len(name_bytes)
    raw[end] = 0
    raw[end + 1:end + 5] = zlib.crc32(raw[:end]).to_bytes(4, "big")
    return bytes(raw)


def parse_header(raw):
    raw = bytes(raw)
    if len(raw) != 249 or raw[:2] != b"PH" or raw[2] != 0:
        raise ValueError("invalid header")
    end = raw.find(b"\0", 11)
    if end < 0 or end + 5 > len(raw):
        raise ValueError("invalid filename terminator")
    crc = int.from_bytes(raw[end + 1:end + 5], "big")
    if crc != zlib.crc32(raw[:end]):
        raise ValueError("header CRC mismatch")
    try:
        name = raw[11:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid filename") from exc
    return {"name": name, "size": int.from_bytes(raw[3:7], "big"), "payload_symbols": int.from_bytes(raw[7:11], "big")}


def scramble_bits(bits, seed=0x5A4D):
    bits = np.asarray(bits, dtype=np.uint8)
    state = int(seed)
    if not 0 < state < (1 << 15):
        raise ValueError("seed must be a nonzero 15-bit value")
    out = np.empty(bits.size, dtype=np.uint8)
    for i in range(bits.size):
        seq = (state >> 14) & 1
        out[i] = bits.flat[i] ^ seq
        feedback = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) & 0x7FFF) | feedback
    return out.reshape(bits.shape)


def training_symbols():
    rng = random.Random(80)
    symbols = np.zeros((16, N), dtype=np.complex128)
    for row in symbols:
        for j in range(M):
            value = rng.randint(0, 3)
            k = K0 + j
            row[k] = -2 * (value & 1) + 1 + 1j * (1 - 2 * ((value >> 1) & 1))
            row[N - k] = np.conj(row[k])
    return symbols


def qpsk_map(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 2:
        raise ValueError("QPSK requires an even number of bits")
    pairs = bits.reshape(-1, 2).astype(np.int8)
    return (1 - 2 * pairs[:, 1]).astype(float) + 1j * (1 - 2 * pairs[:, 0]).astype(float)


def qpsk_llr(symbols, noise_var):
    if noise_var <= 0:
        raise ValueError("noise variance must be positive")
    symbols = np.asarray(symbols)
    out = np.empty(symbols.size * 2, dtype=float)
    out[0::2] = 2 * symbols.imag.ravel() / noise_var
    out[1::2] = 2 * symbols.real.ravel() / noise_var
    return out


def ofdm_time(carriers):
    carriers = np.asarray(carriers, dtype=np.complex128)
    if carriers.shape[-1] != M:
        raise ValueError("carriers must have M columns")
    spectrum = np.zeros(carriers.shape[:-1] + (N,), dtype=np.complex128)
    spectrum[..., K0:K1 + 1] = carriers
    spectrum[..., N - K1:N - K0 + 1] = np.conj(carriers[..., ::-1])
    time = np.fft.ifft(spectrum, axis=-1).real
    return np.concatenate((time[..., -CP:], time), axis=-1)


def linear_chirp():
    t = np.arange(3 * FS, dtype=float) / FS
    return chirp(t, f0=100, f1=20000, t1=3, method="linear")


def read_pcm16_wav(path):
    with wave.open(str(Path(path)), "rb") as stream:
        channels, width, rate, frames = stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()
        if rate != FS or width != 2 or channels not in (1, 2):
            raise ValueError("WAV must be 48 kHz, 16-bit mono or stereo")
        data = np.frombuffer(stream.readframes(frames), dtype="<i2").copy()
    if channels == 2:
        data = data.reshape(-1, 2)
    return rate, data


def write_pcm16_wav(path, samples):
    samples = np.asarray(samples)
    if samples.dtype != np.int16 or samples.ndim not in (1, 2) or (samples.ndim == 2 and samples.shape[1] != 2):
        raise ValueError("samples must be int16 mono or stereo")
    channels = 1 if samples.ndim == 1 else 2
    with wave.open(str(Path(path)), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(FS)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())
