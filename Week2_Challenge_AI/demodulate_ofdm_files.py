#!/usr/bin/env python3
"""Demodulate the Week 2 OFDM WAV files and extract their payload files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np


FFT_SIZE = 1024
CP_LEN = 32
SYMBOL_LEN = FFT_SIZE + CP_LEN
DATA_BINS = np.arange(1, 512)
INT16_SCALE = 32768.0


@dataclass(frozen=True)
class DemodulatedFile:
    source: Path
    filename: str
    payload: bytes
    mapping: str
    symbols: int
    constellation_values: int


def read_wav_samples(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        compression = wav_file.getcomptype()
        frames = wav_file.getnframes()
        frame_bytes = wav_file.readframes(frames)

    if channels != 1:
        raise ValueError(f"{wav_path} must be mono, got {channels} channels")
    if sample_width != 2:
        raise ValueError(f"{wav_path} must be 16-bit PCM, got {sample_width * 8}-bit")
    if compression != "NONE":
        raise ValueError(f"{wav_path} must be uncompressed PCM, got {compression}")

    return np.frombuffer(frame_bytes, dtype="<i2").astype(np.float64) / INT16_SCALE


def read_channel(channel_path: Path) -> np.ndarray:
    taps = np.loadtxt(channel_path, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"{channel_path} must contain one FIR tap per line")
    return taps


def iter_wav_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.wav"))


def equalized_constellation(samples: np.ndarray, channel: np.ndarray) -> np.ndarray:
    symbol_count = samples.size // SYMBOL_LEN
    if symbol_count == 0:
        raise ValueError("input is shorter than one OFDM symbol")

    usable = samples[: symbol_count * SYMBOL_LEN]
    symbols = usable.reshape(symbol_count, SYMBOL_LEN)[:, CP_LEN:]
    received = np.fft.fft(symbols, axis=1)

    channel_response = np.fft.fft(channel, FFT_SIZE)
    active_response = channel_response[DATA_BINS]
    min_response = np.min(np.abs(active_response))
    if min_response < 1e-8:
        raise ValueError(f"channel response is too close to zero: {min_response:g}")

    return received[:, DATA_BINS] / active_response


def constellation_to_bit_pairs(values: np.ndarray, mapping: str) -> list[str]:
    real_positive = values.real >= 0
    imag_positive = values.imag >= 0

    if mapping == "main":
        quadrant_bits = {
            (True, True): "00",
            (False, True): "01",
            (False, False): "11",
            (True, False): "10",
        }
    elif mapping == "alt":
        quadrant_bits = {
            (True, True): "00",
            (False, True): "10",
            (False, False): "11",
            (True, False): "01",
        }
    else:
        raise ValueError(f"unknown mapping: {mapping}")

    return [
        quadrant_bits[(bool(real), bool(imag))]
        for real, imag in zip(real_positive.ravel(), imag_positive.ravel())
    ]


def bit_pairs_to_bytes(bit_pairs: list[str]) -> bytes:
    byte_count = len(bit_pairs) // 4
    output = bytearray(byte_count)

    for byte_index in range(byte_count):
        bits = "".join(bit_pairs[byte_index * 4 : byte_index * 4 + 4])
        output[byte_index] = int(bits, 2)

    return bytes(output)


def read_null_terminated(data: bytes, start: int) -> tuple[str, int]:
    end = data.find(b"\0", start)
    if end == -1:
        raise ValueError("missing null terminator in header")
    try:
        return data[start:end].decode("utf-8"), end + 1
    except UnicodeDecodeError as exc:
        raise ValueError("header string is not valid UTF-8") from exc


def extract_payload(data: bytes) -> tuple[str, bytes]:
    filename, offset = read_null_terminated(data, 0)
    size_text, offset = read_null_terminated(data, offset)

    try:
        file_size = int(size_text)
    except ValueError as exc:
        raise ValueError(f"header file size is not an integer: {size_text!r}") from exc

    if not filename:
        raise ValueError("header filename is empty")
    if file_size < 0:
        raise ValueError(f"header file size must be non-negative: {file_size}")

    end = offset + file_size
    if end > len(data):
        raise ValueError(
            f"payload is shorter than header says: need {file_size} bytes, "
            f"only {len(data) - offset} available"
        )

    return filename, data[offset:end]


def decode_with_mapping(
    wav_path: Path, channel: np.ndarray, mapping: str
) -> DemodulatedFile:
    samples = read_wav_samples(wav_path)
    constellation = equalized_constellation(samples, channel)
    bit_pairs = constellation_to_bit_pairs(constellation, mapping)
    data = bit_pairs_to_bytes(bit_pairs)
    filename, payload = extract_payload(data)

    return DemodulatedFile(
        source=wav_path,
        filename=filename,
        payload=payload,
        mapping=mapping,
        symbols=samples.size // SYMBOL_LEN,
        constellation_values=constellation.size,
    )


def demodulate_file(
    wav_path: Path, channel: np.ndarray, mapping: str
) -> DemodulatedFile:
    mappings = ["main", "alt"] if mapping == "auto" else [mapping]
    errors: list[str] = []

    for candidate in mappings:
        try:
            return decode_with_mapping(wav_path, channel, candidate)
        except ValueError as exc:
            errors.append(f"{candidate}: {exc}")

    raise ValueError("; ".join(errors))


def safe_output_path(output_dir: Path, filename: str) -> Path:
    return output_dir / Path(filename).name


def write_demodulated_file(result: DemodulatedFile, output_dir: Path) -> Path:
    output_path = safe_output_path(output_dir, result.filename)
    output_path.write_bytes(result.payload)
    return output_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Demodulate Week 2 OFDM WAV files and extract their payloads."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=script_dir / "data",
        help="Input WAV file or directory. Default: Week2_Challenge/data",
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=Path,
        default=script_dir / "channel.csv",
        help="FIR impulse response CSV. Default: Week2_Challenge/channel.csv",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        default=script_dir / "extracted",
        help="Directory for extracted files. Default: Week2_Challenge/extracted",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        choices=["auto", "main", "alt"],
        default="auto",
        help="QPSK Gray mapping to use. Default: auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel = read_channel(args.channel)
    wav_files = iter_wav_files(args.input)

    if not wav_files:
        raise SystemExit(f"No .wav files found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for wav_path in wav_files:
        try:
            result = demodulate_file(wav_path, channel, args.mapping)
            output_path = write_demodulated_file(result, args.output_dir)
        except ValueError as exc:
            failures += 1
            print(f"{wav_path}: error: {exc}")
            continue

        print(
            f"{wav_path} -> {output_path} "
            f"({len(result.payload)} bytes, "
            f"{result.symbols} OFDM symbols, "
            f"{result.constellation_values} constellation values, "
            f"mapping={result.mapping})"
        )

    if failures:
        raise SystemExit(f"{failures} file(s) failed")


if __name__ == "__main__":
    main()
