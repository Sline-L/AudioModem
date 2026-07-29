#!/usr/bin/env python3
"""Compact Week 2 OFDM/QPSK modulation pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import wave

import numpy as np


FFT_SIZE = 1024
CP_LEN = 32
SYMBOL_LEN = FFT_SIZE + CP_LEN
DATA_BINS = np.arange(1, 512)
INT16_LIMIT = 32767
QPSK_LEVEL = 1.0
SAMPLE_RATE = 48000


def t01_payload_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.iterdir()
        if path.is_file()
        and path.name != "recovered_files.csv"
        and not path.stem.endswith("_fixed")
    )


def t02_channel_response(channel_path: Path) -> np.ndarray:
    taps = np.loadtxt(channel_path, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"bad channel file: {channel_path}")
    return np.fft.fft(taps, FFT_SIZE)[DATA_BINS]


def t02b_data_bins(bin_start: int, bin_end: int) -> np.ndarray:
    if bin_start < 1 or bin_end > FFT_SIZE // 2 - 1 or bin_start > bin_end:
        raise ValueError("bin range must satisfy 1 <= bin_start <= bin_end <= 511")
    return np.arange(bin_start, bin_end + 1)


def t02c_channel_response(channel_path: Path, bins: np.ndarray) -> np.ndarray:
    taps = np.loadtxt(channel_path, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"bad channel file: {channel_path}")
    return np.fft.fft(taps, FFT_SIZE)[bins]


def t03_qpsk_symbols(data: bytes, mapping: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")
    if bits.size % 2:
        bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2)

    if mapping == "main":
        imag = np.where(pairs[:, 0] == 1, -QPSK_LEVEL, QPSK_LEVEL)
        real = np.where(pairs[:, 1] == 1, -QPSK_LEVEL, QPSK_LEVEL)
    elif mapping == "alt":
        real = np.where(pairs[:, 0] == 1, -QPSK_LEVEL, QPSK_LEVEL)
        imag = np.where(pairs[:, 1] == 1, -QPSK_LEVEL, QPSK_LEVEL)
    else:
        raise ValueError(f"unknown mapping: {mapping}")
    return real + 1j * imag


def t04_write_wav(samples: np.ndarray, output: Path) -> None:
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        raise ValueError("empty modulated signal")
    pcm = np.clip(samples / peak * 0.95 * INT16_LIMIT, -INT16_LIMIT, INT16_LIMIT)
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.astype("<i2").tobytes())


def f01_parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Modulate files into OFDM/QPSK WAVs.")
    parser.add_argument("input", nargs="?", type=Path, default=here / "source")
    parser.add_argument("-c", "--channel", type=Path, default=here / "channel.csv")
    parser.add_argument("-o", "--output-dir", type=Path, default=here / "data")
    parser.add_argument("-m", "--mapping", choices=["main", "alt"], default="main")
    parser.add_argument("--name-prefix", default="files/")
    parser.add_argument("--bin-start", type=int, default=1)
    parser.add_argument("--bin-end", type=int, default=511)
    return parser.parse_args()


def f02_modulate_one(
    payload_path: Path,
    channel_response: np.ndarray,
    mapping: str,
    name_prefix: str,
    bins: np.ndarray,
) -> np.ndarray:
    #Step 1: Read the original file and build the filename\0size\0payload byte stream.
    payload = payload_path.read_bytes()
    header_name = f"{name_prefix}{payload_path.name}".encode("utf-8")
    data = header_name + b"\0" + str(len(payload)).encode("ascii") + b"\0" + payload

    #Step 2: Convert bytes into QPSK constellation symbols with the selected mapping.
    qpsk = t03_qpsk_symbols(data, mapping)

    #Step 3: Pad QPSK symbols so every OFDM symbol carries the selected active bins.
    symbol_count = int(np.ceil(qpsk.size / bins.size))
    padded = np.zeros(symbol_count * bins.size, dtype=np.complex128)
    padded[: qpsk.size] = qpsk
    active_symbols = padded.reshape(symbol_count, bins.size)

    #Step 4: Apply the channel frequency response, the inverse of demodulator equalization.
    active_bins = active_symbols * channel_response

    #Step 5: Build a conjugate-symmetric FFT frame so the IFFT output is real audio.
    fft_frames = np.zeros((symbol_count, FFT_SIZE), dtype=np.complex128)
    fft_frames[:, bins] = active_bins
    fft_frames[:, FFT_SIZE - bins] = np.conj(active_bins)

    #Step 6: Run IFFT to create time-domain OFDM symbols.
    time_symbols = np.fft.ifft(fft_frames, axis=1).real

    #Step 7: Add the 32-sample cyclic prefix and flatten symbols into one waveform.
    with_cp = np.concatenate((time_symbols[:, -CP_LEN:], time_symbols), axis=1)
    return with_cp.reshape(-1)


def f03_write_summary(rows: list[tuple[str, str, int, str]], output_dir: Path) -> None:
    with (output_dir / "modulated_files.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_file", "output_wav", "ofdm_symbols", "mapping"])
        writer.writerows(rows)


def f04_run() -> None:
    args = f01_parse_args()
    try:
        payload_files = t01_payload_files(args.input)
        if not payload_files:
            raise ValueError(f"no payload files found: {args.input}")
        bins = t02b_data_bins(args.bin_start, args.bin_end)
        channel_response = t02c_channel_response(args.channel, bins)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for index, payload_path in enumerate(payload_files, start=1):
            samples = f02_modulate_one(payload_path, channel_response, args.mapping, args.name_prefix, bins)
            output_path = args.output_dir / f"file{index:02d}.wav"
            t04_write_wav(samples, output_path)
            rows.append((str(payload_path), str(output_path), samples.size // SYMBOL_LEN, args.mapping))

        f03_write_summary(rows, args.output_dir)
        print(f"Modulated {len(rows)} file(s) to {args.output_dir}")
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    f04_run()
