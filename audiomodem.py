from pathlib import Path
import wave

import numpy as np
from scipy import signal

FS, N, CP = 48000, 1024, 128
L = N + CP
I16 = 32768.0
MODS = ("bpsk", "qpsk", "qam16")


def bins(a=8, b=420):
    if a < 1 or b > N // 2 - 1 or a > b:
        raise ValueError("bins must satisfy 1 <= start <= end <= 511")
    return np.arange(a, b + 1)


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != FS:
            raise ValueError(f"expected mono 16-bit {FS} Hz wav: {p}")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / I16


def write_wav(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(x * wav_gain(x) * I16, -I16 + 1, I16 - 1).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(y.tobytes())


def wav_gain(x):
    m = np.max(np.abs(x))
    if m == 0:
        raise ValueError("empty signal")
    return 0.95 * (I16 - 1) / I16 / m


def bits_per_symbol(mod):
    if mod == "bpsk":
        return 1
    if mod == "qpsk":
        return 2
    if mod == "qam16":
        return 4
    raise ValueError(f"mod must be one of {MODS}")


def mod_symbols(data, mod="qpsk"):
    b = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    m = bits_per_symbol(mod)
    if b.size % m:
        b = np.r_[b, np.zeros(m - b.size % m, dtype=np.uint8)]
    b = b.reshape(-1, m)
    if mod == "bpsk":
        return np.where(b[:, 0], -1, 1).astype(complex)
    if mod == "qpsk":
        return np.where(b[:, 1], -1, 1) + 1j * np.where(b[:, 0], -1, 1)
    r = _pam4(b[:, 2], b[:, 3])
    i = _pam4(b[:, 0], b[:, 1])
    return (r + 1j * i) / np.sqrt(10)


def _pam4(a, b):
    return np.select(
        [(a == 0) & (b == 0), (a == 0) & (b == 1), (a == 1) & (b == 1)],
        [3, 1, -1],
        default=-3,
    )


def bytes_from_mod(z, mod="qpsk"):
    z = z.ravel()
    if mod == "bpsk":
        b = (z.real < 0).astype(np.uint8)
    elif mod == "qpsk":
        b = np.c_[z.imag < 0, z.real < 0].astype(np.uint8).ravel()
    elif mod == "qam16":
        r = np.sqrt(np.mean(np.abs(z) ** 2))
        if r > 0:
            z = z / r
        b = np.c_[_bits_from_pam4(z.imag * np.sqrt(10)), _bits_from_pam4(z.real * np.sqrt(10))].ravel()
    else:
        raise ValueError(f"mod must be one of {MODS}")
    return np.packbits(b[: b.size // 8 * 8], bitorder="big").tobytes()


def _bits_from_pam4(v):
    out = np.zeros((len(v), 2), dtype=np.uint8)
    out[v < 2, 1] = 1
    out[v < 0, 0] = 1
    out[v < -2, 1] = 0
    return out


def qpsk(data):
    return mod_symbols(data, "qpsk")


def bytes_from_qpsk(z):
    return bytes_from_mod(z, "qpsk")


def pack(path):
    path = Path(path)
    body = path.read_bytes()
    return path.name.encode() + b"\0" + str(len(body)).encode() + b"\0" + body


def unpack(data):
    i = data.index(0)
    j = data.index(0, i + 1)
    name = data[:i].decode()
    size = int(data[i + 1 : j])
    body = data[j + 1 : j + 1 + size]
    if len(body) != size:
        raise ValueError("short payload")
    return Path(name).name, body


def ofdm_tx(s, k):
    f = np.zeros((len(s), N), complex)
    f[:, k] = s
    f[:, N - k] = np.conj(s)
    x = np.fft.ifft(f, axis=1).real
    return np.c_[x[:, -CP:], x].ravel()


def ofdm_rx(x, k):
    n = len(x) // L
    y = x[: n * L].reshape(n, L)[:, CP:]
    return np.fft.fft(y, axis=1)[:, k]


def preamble_symbols(k, n=32, seed=2026):
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, (n, len(k), 2), dtype=np.uint8)
    return np.where(b[:, :, 1], -1, 1) + 1j * np.where(b[:, :, 0], -1, 1)


def preamble_wave(k, n=32, seed=2026):
    return ofdm_tx(preamble_symbols(k, n, seed), k)


def find_sync(rx, preamble):
    if len(rx) < len(preamble):
        raise ValueError("receive wav is shorter than preamble")
    c = signal.correlate(rx, preamble, mode="valid", method="fft")
    i = int(np.argmax(np.abs(c)))
    e = np.linalg.norm(rx[i : i + len(preamble)]) * np.linalg.norm(preamble)
    return i, float(abs(c[i]) / e) if e else 0.0


def file_symbols(path, k, mod="qpsk"):
    z = mod_symbols(pack(path), mod)
    n = int(np.ceil(len(z) / len(k)))
    s = np.zeros(n * len(k), complex)
    s[: len(z)] = z
    return s.reshape(n, len(k))


def probe_symbols(kind, k, n, seed=2026):
    t = np.arange(n)[:, None]
    f = np.arange(len(k))[None, :]
    if kind == "ones":
        return np.ones((n, len(k)), complex)
    if kind == "chirp":
        return np.exp(2j * np.pi * t * f / max(1, n * len(k)))
    if kind == "step":
        q = np.floor(t * 16 / max(1, n)).astype(int)
        return np.exp(2j * np.pi * q * f / max(1, len(k)))
    if kind == "bandstep":
        parts = min(16, len(k), n)
        band = int(np.ceil(len(k) / parts))
        q = np.minimum(np.arange(n) * parts // n, parts - 1)
        s = np.zeros((n, len(k)), complex)
        for i, j in enumerate(q):
            a = j * band
            b = min(a + band, len(k))
            s[i, a:b] = 1
        return s
    if kind == "random":
        rng = np.random.default_rng(seed)
        b = rng.integers(0, 2, (n, len(k), 2))
        return np.where(b[:, :, 1], -1, 1) + 1j * np.where(b[:, :, 0], -1, 1)
    raise ValueError("kind must be ones, chirp, step, bandstep, or random")
