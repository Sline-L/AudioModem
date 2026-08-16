import struct
import wave
import zlib
from pathlib import Path

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

CHIRP_SECONDS = 0.500
CHIRP_GUARD_SECONDS = 0.030
TRAINING_SYMBOLS = 12
TAIL_TRAINING_SYMBOLS = TRAINING_SYMBOLS
HEADER_SYMBOLS = 9
HEADER_COPIES = 3
HEADER_COPY_SYMBOLS = 3
INTER_FRAME_GAP_SECONDS = 0.050
CHIRP_BAND_HZ = (1000.0, 9000.0)
CHIRP_SAMPLES = int(round(CHIRP_SECONDS * FS))
CHIRP_GUARD_SAMPLES = int(round(CHIRP_GUARD_SECONDS * FS))
INTER_FRAME_GAP_SAMPLES = int(round(INTER_FRAME_GAP_SECONDS * FS))

MODS = ("bpsk", "qpsk", "qam16")
MOD_IDS = {name: index for index, name in enumerate(MODS)}
MAGIC = b"AMN2"
VERSION = 1
HEADER_FLAG_LDPC = 0x01
MAX_NAME_BYTES = 38
HEADER_BODY = struct.Struct(">4sBBBBIIHI38s")
HEADER_SIZE = HEADER_BODY.size + 4
HEADER_MOD = "bpsk"
HEADER_BITS = HEADER_SIZE * 8
HEADER_COPY_BITS = HEADER_COPY_SYMBOLS * len(ACTIVE_BINS)
HEADER_PERM_SEEDS = (0, 4101, 9103)

LDPC_ENABLED_DEFAULT = False
LDPC_STANDARD = "802.11n"
LDPC_RATE = "3/4"
LDPC_Z = 27
LDPC_PTYPE = "A"
LDPC_DECODER = "sumprod2"
LDPC_CORR_FACTOR = 0.7
LDPC_LLR_SCALE = 3.0
LDPC_INTERLEAVER_ENABLED = True
LDPC_INTERLEAVER_SEED = 20260816
_LDPC_CODE = None


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
    return mod_symbols_from_bits(bits_from_bytes(data), mod)


def mod_symbols_from_bits(bits, mod="qpsk"):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
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


def bits_from_mod(symbols, mod="qpsk"):
    symbols = np.asarray(symbols).ravel()
    if mod == "bpsk":
        return (symbols.real < 0).astype(np.uint8)
    if mod == "qpsk":
        return np.c_[symbols.imag < 0, symbols.real < 0].astype(np.uint8).ravel()
    if mod == "qam16":
        z = symbols * np.sqrt(10)
        return np.c_[_bits_from_pam4(z.imag), _bits_from_pam4(z.real)].ravel()
    raise ValueError(f"mod must be one of {MODS}")


def llr_from_symbols(symbols, mod="qpsk"):
    symbols = np.asarray(symbols, dtype=complex).ravel()
    scale = float(LDPC_LLR_SCALE)
    if mod == "bpsk":
        return scale * symbols.real
    if mod == "qpsk":
        return scale * np.c_[symbols.imag * np.sqrt(2), symbols.real * np.sqrt(2)].ravel()
    if mod != "qam16":
        raise ValueError(f"mod must be one of {MODS}")

    labels = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    points = mod_symbols_from_bits(labels.ravel(), "qam16")
    distances = np.abs(symbols[:, None] - points[None, :]) ** 2
    out = np.empty((len(symbols), 4), dtype=float)
    for bit_index in range(4):
        d0 = np.min(distances[:, labels[:, bit_index] == 0], axis=1)
        d1 = np.min(distances[:, labels[:, bit_index] == 1], axis=1)
        out[:, bit_index] = scale * (d1 - d0)
    return out.ravel()


def symbols_from_bytes(data, mod="qpsk", rows=None, bins=ACTIVE_BINS):
    return symbols_from_bits(bits_from_bytes(data), mod, rows=rows, bins=bins)


def symbols_from_bits(bits, mod="qpsk", rows=None, bins=ACTIVE_BINS):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    symbols = mod_symbols_from_bits(bits, mod)
    needed_rows = int(np.ceil(len(symbols) / len(bins)))
    if rows is None:
        rows = needed_rows
    if needed_rows > rows:
        raise ValueError(f"{len(bits)} bits need {needed_rows} OFDM rows, only {rows} available")
    padded = np.zeros(rows * len(bins), complex)
    padded[: len(symbols)] = symbols
    return padded.reshape(rows, len(bins)), len(symbols)


def ldpc_code():
    global _LDPC_CODE
    if _LDPC_CODE is None:
        import lib.ldpc.py.ldpc as ldpc

        _LDPC_CODE = ldpc.code(LDPC_STANDARD, LDPC_RATE, LDPC_Z, LDPC_PTYPE)
    return _LDPC_CODE


def ldpc_meta():
    code = ldpc_code()
    return {
        "ldpc_standard": LDPC_STANDARD,
        "ldpc_rate": LDPC_RATE,
        "ldpc_z": int(LDPC_Z),
        "ldpc_ptype": LDPC_PTYPE,
        "ldpc_decoder": LDPC_DECODER,
        "ldpc_corr_factor": float(LDPC_CORR_FACTOR),
        "ldpc_llr_scale": float(LDPC_LLR_SCALE),
        "ldpc_interleaver_enabled": bool(LDPC_INTERLEAVER_ENABLED),
        "ldpc_interleaver_seed": int(LDPC_INTERLEAVER_SEED),
        "ldpc_K": int(code.K),
        "ldpc_N": int(code.N),
    }


def ldpc_interleaver(length):
    rng = np.random.default_rng(LDPC_INTERLEAVER_SEED)
    return rng.permutation(int(length))


def interleave_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if not LDPC_INTERLEAVER_ENABLED or bits.size == 0:
        return bits
    return bits[ldpc_interleaver(bits.size)]


def deinterleave_bits(bits):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if not LDPC_INTERLEAVER_ENABLED or bits.size == 0:
        return bits
    out = np.empty_like(bits)
    out[ldpc_interleaver(bits.size)] = bits
    return out


def deinterleave_llr(llr):
    llr = np.asarray(llr, dtype=float).ravel()
    if not LDPC_INTERLEAVER_ENABLED or llr.size == 0:
        return llr
    out = np.empty_like(llr)
    out[ldpc_interleaver(llr.size)] = llr
    return out


def ldpc_encode_bits(info_bits):
    code = ldpc_code()
    info_bits = np.asarray(info_bits, dtype=np.uint8).ravel()
    padding = (-len(info_bits)) % code.K
    padded = np.r_[info_bits, np.zeros(padding, dtype=np.uint8)]
    blocks = padded.reshape(-1, code.K) if padded.size else np.empty((0, code.K), dtype=np.uint8)
    encoded = [code.encode(block.astype(int)).astype(np.uint8) for block in blocks]
    coded = np.concatenate(encoded) if encoded else np.empty(0, dtype=np.uint8)
    return coded, {
        "ldpc_blocks": int(len(blocks)),
        "ldpc_padding_bits": int(padding),
        "ldpc_info_bits": int(len(info_bits)),
        "ldpc_coded_bits": int(len(coded)),
    }


def ldpc_decode_bits(llr, info_bit_count):
    code = ldpc_code()
    llr = np.asarray(llr, dtype=float).ravel()
    blocks = len(llr) // code.N
    decoded = []
    iterations = []
    for block in llr[: blocks * code.N].reshape(blocks, code.N):
        app, iteration_count = code.decode(block, LDPC_DECODER, LDPC_CORR_FACTOR)
        decoded.append((app[: code.K] < 0.0).astype(np.uint8))
        iterations.append(int(iteration_count))
    bits = np.concatenate(decoded) if decoded else np.empty(0, dtype=np.uint8)
    return bits[:info_bit_count], {
        "ldpc_decode_iterations": iterations,
        "ldpc_decode_iterations_mean": float(np.mean(iterations)) if iterations else None,
        "ldpc_decode_iterations_max": int(max(iterations)) if iterations else None,
    }


def payload_symbols(path, mod="qpsk", bins=ACTIVE_BINS, ldpc_enabled=LDPC_ENABLED_DEFAULT):
    body = Path(path).read_bytes()
    info_bits = bits_from_bytes(body)
    if not ldpc_enabled:
        rows, mod_symbol_count = symbols_from_bits(info_bits, mod, bins=bins)
        return rows, mod_symbol_count, {
            "ldpc_enabled": False,
            "ldpc_blocks": 0,
            "ldpc_padding_bits": 0,
            "ldpc_info_bits": int(len(info_bits)),
            "ldpc_coded_bits": int(len(info_bits)),
        }
    coded_bits, meta = ldpc_encode_bits(info_bits)
    coded_bits = interleave_bits(coded_bits)
    rows, mod_symbol_count = symbols_from_bits(coded_bits, mod, bins=bins)
    meta["ldpc_enabled"] = True
    return rows, mod_symbol_count, meta


def header_bytes(path, mod, payload_rows, payload_mod_symbols, ldpc_enabled=LDPC_ENABLED_DEFAULT):
    path = Path(path)
    body = path.read_bytes()
    name = path.name.encode("utf-8")
    if len(name) > MAX_NAME_BYTES:
        raise ValueError(f"UTF-8 filename must be at most {MAX_NAME_BYTES} bytes")
    if len(body) > 0xFFFFFFFF:
        raise ValueError("N2 header supports files up to 2^32-1 bytes")
    if payload_rows > 0xFFFF:
        raise ValueError("N2 header supports up to 65535 payload OFDM symbols")
    flags = HEADER_FLAG_LDPC if ldpc_enabled else 0
    first = HEADER_BODY.pack(
        MAGIC,
        VERSION,
        len(name),
        MOD_IDS[mod],
        flags,
        len(body),
        int(payload_rows),
        payload_mod_symbols,
        zlib.crc32(body),
        name.ljust(MAX_NAME_BYTES, b"\0"),
    )
    return first + struct.pack(">I", zlib.crc32(first))


def header_permutation(copy_index):
    if not 0 <= copy_index < HEADER_COPIES:
        raise ValueError("invalid N2 header copy index")
    if copy_index == 0:
        return np.arange(HEADER_COPY_BITS)
    rng = np.random.default_rng(HEADER_PERM_SEEDS[copy_index])
    return rng.permutation(HEADER_COPY_BITS)


def bpsk_symbols_from_bits(bits):
    return np.where(np.asarray(bits, dtype=np.uint8), -1.0, 1.0).astype(complex)


def header_symbols(path, mod, payload_rows, payload_mod_symbols, ldpc_enabled=LDPC_ENABLED_DEFAULT):
    source = np.zeros(HEADER_COPY_BITS, dtype=np.uint8)
    source[:HEADER_BITS] = bits_from_bytes(
        header_bytes(path, mod, payload_rows, payload_mod_symbols, ldpc_enabled=ldpc_enabled)
    )
    copies = []
    for copy_index in range(HEADER_COPIES):
        tx_bits = source[header_permutation(copy_index)]
        copies.append(bpsk_symbols_from_bits(tx_bits).reshape(HEADER_COPY_SYMBOLS, len(ACTIVE_BINS)))
    return np.vstack(copies)


def parse_header(raw):
    if len(raw) < HEADER_SIZE:
        raise ValueError("header is shorter than N2 header")
    first = raw[: HEADER_BODY.size]
    stored_crc = struct.unpack(">I", raw[HEADER_BODY.size:HEADER_SIZE])[0]
    if zlib.crc32(first) != stored_crc:
        raise ValueError("header CRC mismatch")
    magic, version, name_len, mod_id, flags, size, payload_rows, payload_mod_symbols, file_crc, raw_name = HEADER_BODY.unpack(first)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported N2 header")
    if name_len > len(raw_name) or mod_id >= len(MODS):
        raise ValueError("invalid N2 header fields")
    return {
        "name": Path(raw_name[:name_len].decode("utf-8")).name,
        "mod": MODS[mod_id],
        "file_size": int(size),
        "payload_symbols": int(payload_rows),
        "payload_mod_symbols": int(payload_mod_symbols),
        "file_crc32": int(file_crc),
        "ldpc_enabled": bool(flags & HEADER_FLAG_LDPC),
        "flags": int(flags),
    }


def header_copy_bytes(equalized_symbols):
    copies = np.asarray(equalized_symbols).reshape(HEADER_COPIES, HEADER_COPY_SYMBOLS, len(ACTIVE_BINS))
    raw_copies = []
    bit_copies = []
    crc_ok = []
    for copy_index, copy in enumerate(copies):
        rx_bits = bits_from_mod(copy.ravel(), HEADER_MOD)[:HEADER_COPY_BITS]
        depermuted = np.empty(HEADER_COPY_BITS, dtype=np.uint8)
        depermuted[header_permutation(copy_index)] = rx_bits
        header_bits = depermuted[:HEADER_BITS]
        raw = bytes_from_bits(header_bits)[:HEADER_SIZE]
        raw_copies.append(raw)
        bit_copies.append(header_bits)
        try:
            parse_header(raw)
            crc_ok.append(True)
        except ValueError:
            crc_ok.append(False)
    return raw_copies, np.asarray(bit_copies, dtype=np.uint8), crc_ok


def vote_header(equalized_symbols):
    raw_copies, bit_copies, crc_ok = header_copy_bytes(equalized_symbols)
    votes = np.sum(bit_copies, axis=0)
    voted_bits = (votes >= 2).astype(np.uint8)
    voted = bytes_from_bits(voted_bits)[:HEADER_SIZE]
    disagreements = int(np.count_nonzero((votes != 0) & (votes != HEADER_COPIES)))
    return {
        "raw": voted,
        "header": parse_header(voted),
        "raw_copies": raw_copies,
        "copy_crc_ok": crc_ok,
        "bit_disagreements": disagreements,
    }


def random_qpsk(rows, bins=ACTIVE_BINS, seed=2026):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (rows, len(bins), 2), dtype=np.uint8)
    return (np.where(bits[:, :, 1], -1.0, 1.0) + 1j * np.where(bits[:, :, 0], -1.0, 1.0)) / np.sqrt(2)


def training_symbols(seed=3026):
    return random_qpsk(TRAINING_SYMBOLS, seed=seed)


def chirp_wave():
    t = np.arange(CHIRP_SAMPLES) / FS
    wave_data = signal.chirp(
        t,
        f0=CHIRP_BAND_HZ[0],
        f1=CHIRP_BAND_HZ[1],
        t1=CHIRP_SECONDS,
        method="linear",
    )
    fade = min(int(round(0.005 * FS)), CHIRP_SAMPLES // 2)
    if fade:
        edge = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        wave_data[:fade] *= edge
        wave_data[-fade:] *= edge[::-1]
    return wave_data.astype(float)


def find_chirp(rx, template, start=0, end=None):
    if end is None:
        end = len(rx)
    start = max(0, int(start))
    end = min(len(rx), int(end))
    if end - start < len(template):
        raise ValueError("search window is shorter than chirp template")
    section = rx[start:end]
    corr = signal.correlate(section, template, mode="valid", method="fft")
    energy = signal.fftconvolve(section * section, np.ones(len(template)), mode="valid")
    denom = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(template)
    score = np.divide(np.abs(corr), denom, out=np.zeros_like(denom), where=denom > 0)
    peak = int(np.argmax(score))
    delta = 0.0
    if 0 < peak < len(score) - 1:
        left, middle, right = score[peak - 1 : peak + 2]
        curvature = left - 2.0 * middle + right
        if abs(curvature) > 1e-12:
            delta = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(start + peak + delta), float(np.clip(score[peak], 0.0, 1.0))


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


def phase_correct(symbols, symbol_starts, reference_start, epsilon, bins=ACTIVE_BINS):
    symbols = np.asarray(symbols, dtype=complex)
    symbol_starts = np.asarray(symbol_starts, dtype=float)
    drift = epsilon * (symbol_starts - reference_start)
    ramp = np.exp(2j * np.pi * drift[:, None] * bins[None, :] / N)
    return symbols * ramp


def estimate_channel(received, known):
    rows = min(len(received), len(known))
    if rows == 0:
        raise ValueError("no training symbols available for channel estimate")
    return np.mean(received[:rows] / known[:rows], axis=0)


def equalize(received, h):
    return received / np.where(np.abs(h) > 1e-12, h, 1.0)


def training_phase_fit(received, known, h, bins=ACTIVE_BINS):
    rows = min(len(received), len(known))
    x = bins.astype(float)
    fits = []
    for row, ref in zip(received[:rows], known[:rows]):
        error = row / np.where(np.abs(h * ref) > 1e-12, h * ref, 1.0)
        phase = np.unwrap(np.angle(error))
        slope, intercept = np.polyfit(x, phase, 1)
        residual = phase - (slope * x + intercept)
        delta_n = -slope * N / (2.0 * np.pi)
        fits.append([intercept, slope, delta_n, float(np.sqrt(np.mean(residual * residual)))])
    return np.asarray(fits)


def decode_payload(symbols, header, ldpc_enabled=None):
    symbols = np.asarray(symbols).ravel()[: header["payload_mod_symbols"]]
    if ldpc_enabled is None:
        ldpc_enabled = bool(header.get("ldpc_enabled", False))
    if ldpc_enabled:
        info_bits = int(header["file_size"]) * 8
        llr = llr_from_symbols(symbols, header["mod"])
        decoded_bits, _ = ldpc_decode_bits(llr, info_bits)
        body = bytes_from_bits(decoded_bits)[: header["file_size"]]
    else:
        raw = bytes_from_mod(symbols, header["mod"])
        body = raw[: header["file_size"]]
    if len(body) != header["file_size"]:
        raise ValueError("decoded payload is shorter than header file size")
    if zlib.crc32(body) != header["file_crc32"]:
        raise ValueError("file CRC mismatch")
    return body


def frame_sample_counts(payload_rows, tail_training=False):
    training_samples = TRAINING_SYMBOLS * L
    tail_training_samples = TAIL_TRAINING_SYMBOLS * L if tail_training else 0
    header_samples = HEADER_SYMBOLS * L
    payload_samples = int(payload_rows) * L
    tail_training_start = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + training_samples + header_samples + payload_samples
    gap_start = tail_training_start + tail_training_samples
    tail_start = gap_start + INTER_FRAME_GAP_SAMPLES
    total = tail_start + CHIRP_SAMPLES
    return {
        "chirp_samples": CHIRP_SAMPLES,
        "guard_samples": CHIRP_GUARD_SAMPLES,
        "training_samples": training_samples,
        "tail_training_enabled": bool(tail_training),
        "tail_training_symbols": TAIL_TRAINING_SYMBOLS if tail_training else 0,
        "tail_training_samples": tail_training_samples,
        "header_samples": header_samples,
        "payload_samples": payload_samples,
        "tail_training_start": tail_training_start if tail_training else None,
        "gap_start": gap_start,
        "gap_samples": INTER_FRAME_GAP_SAMPLES,
        "tail_chirp_start": tail_start,
        "D_tx": tail_start,
        "total_samples": total,
    }


def profile_meta(tail_training=False, ldpc_enabled=LDPC_ENABLED_DEFAULT):
    band_start = int(round(BAND_HZ[0]))
    band_end = int(round(BAND_HZ[1]))
    meta = {
        "profile": f"n2_chirp_tail_n{N}_cp{CP}_{band_start}hz_{band_end}hz",
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
        "chirp_seconds": CHIRP_SECONDS,
        "chirp_samples": CHIRP_SAMPLES,
        "chirp_band_hz": [float(CHIRP_BAND_HZ[0]), float(CHIRP_BAND_HZ[1])],
        "chirp_guard_seconds": CHIRP_GUARD_SECONDS,
        "chirp_guard_samples": CHIRP_GUARD_SAMPLES,
        "training_symbols": TRAINING_SYMBOLS,
        "tail_training_enabled": bool(tail_training),
        "tail_training_symbols": TAIL_TRAINING_SYMBOLS if tail_training else 0,
        "header_symbols": HEADER_SYMBOLS,
        "header_copies": HEADER_COPIES,
        "header_copy_symbols": HEADER_COPY_SYMBOLS,
        "header_size": HEADER_SIZE,
        "header_bits": HEADER_BITS,
        "header_copy_bits": HEADER_COPY_BITS,
        "header_permutation_used": True,
        "header_permutation_seeds": [int(seed) for seed in HEADER_PERM_SEEDS],
        "max_name_bytes": MAX_NAME_BYTES,
        "inter_frame_gap_seconds": INTER_FRAME_GAP_SECONDS,
        "inter_frame_gap_samples": INTER_FRAME_GAP_SAMPLES,
        "header_mod": HEADER_MOD,
    }
    meta.update(ldpc_meta())
    meta["ldpc_enabled"] = bool(ldpc_enabled)
    return meta
