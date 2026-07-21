#!/usr/bin/env python3
"""Recover payload files from OFDM FFT or constellation .npy files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FFT_SIZE = 1024
DATA_BINS = np.arange(1, 512)


@dataclass(frozen=True)
class RecoveredFile:
    source: Path
    output: Path
    filename: str
    size_bytes: int
    header_bytes: int
    mapping: str
    mode: str


def iter_npy_files(input_path: Path, channel: np.ndarray | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    pattern = "*.fft.npy" if channel is not None else "*.constellation.npy"
    return sorted(input_path.glob(pattern))


def read_channel(channel_path: Path | None) -> np.ndarray | None:
    if channel_path is None:
        return None

    taps = np.loadtxt(channel_path, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"{channel_path} must contain one FIR tap per line")
    return taps


def constellation_from_npy(npy_path: Path, channel: np.ndarray | None) -> tuple[np.ndarray, str]:
    values = np.load(npy_path)
    if not np.iscomplexobj(values):
        raise ValueError(f"{npy_path} must contain complex FFT or constellation values")

    if npy_path.name.endswith(".fft.npy"):
        if values.ndim != 2 or values.shape[1] != FFT_SIZE:
            raise ValueError(f"{npy_path} must have shape (symbols, 1024), got {values.shape}")

        active_values = values[:, DATA_BINS]
        if channel is None:
            return active_values, "raw_fft_bins_1_to_511"

        channel_response = np.fft.fft(channel, FFT_SIZE)
        active_response = channel_response[DATA_BINS]
        min_response = np.min(np.abs(active_response))
        if min_response < 1e-8:
            raise ValueError(f"channel response is too close to zero: {min_response:g}")
        return active_values / active_response, "equalized_fft_bins_1_to_511"

    if values.ndim != 2 or values.shape[1] != DATA_BINS.size:
        raise ValueError(
            f"{npy_path} must have shape (symbols, 511) for constellation data, "
            f"got {values.shape}"
        )
    return values, "constellation_npy"


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


def extract_payload(data: bytes) -> tuple[str, bytes, int]:
    filename, offset = read_null_terminated(data, 0)
    size_text, offset = read_null_terminated(data, offset)

    if not filename:
        raise ValueError("header filename is empty")

    try:
        file_size = int(size_text)
    except ValueError as exc:
        raise ValueError(f"header file size is not an integer: {size_text!r}") from exc

    if file_size < 0:
        raise ValueError(f"header file size must be non-negative: {file_size}")

    end = offset + file_size
    if end > len(data):
        raise ValueError(
            f"payload is shorter than header says: need {file_size} bytes, "
            f"only {len(data) - offset} available"
        )

    return filename, data[offset:end], offset


def decode_payload_with_mapping(
    npy_path: Path, channel: np.ndarray | None, mapping: str
) -> tuple[str, bytes, int, str]:
    constellation, mode = constellation_from_npy(npy_path, channel)
    bit_pairs = constellation_to_bit_pairs(constellation, mapping)
    data = bit_pairs_to_bytes(bit_pairs)
    filename, payload, header_bytes = extract_payload(data)
    return filename, payload, header_bytes, mode


def decode_payload(
    npy_path: Path, channel: np.ndarray | None, mapping: str
) -> tuple[str, bytes, int, str, str]:
    mappings = ["main", "alt"] if mapping == "auto" else [mapping]
    errors: list[str] = []

    for candidate in mappings:
        try:
            filename, payload, header_bytes, mode = decode_payload_with_mapping(
                npy_path, channel, candidate
            )
            return filename, payload, header_bytes, mode, candidate
        except ValueError as exc:
            errors.append(f"{candidate}: {exc}")

    raise ValueError("; ".join(errors))


def safe_output_path(output_dir: Path, filename: str) -> Path:
    name = Path(filename).name
    if not name:
        raise ValueError("header filename is empty")
    return output_dir / name


def recover_file(
    npy_path: Path, channel: np.ndarray | None, output_dir: Path, mapping: str
) -> RecoveredFile:
    filename, payload, header_bytes, mode, selected_mapping = decode_payload(
        npy_path, channel, mapping
    )
    output_path = safe_output_path(output_dir, filename)
    output_path.write_bytes(payload)

    return RecoveredFile(
        source=npy_path,
        output=output_path,
        filename=filename,
        size_bytes=len(payload),
        header_bytes=header_bytes,
        mapping=selected_mapping,
        mode=mode,
    )


def write_summary(results: list[RecoveredFile], output_dir: Path) -> Path:
    summary_path = output_dir / "recovered_files.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source",
                "output",
                "mode",
                "mapping",
                "filename",
                "size_bytes",
                "header_bytes",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.source,
                    result.output,
                    result.mode,
                    result.mapping,
                    result.filename,
                    result.size_bytes,
                    result.header_bytes,
                ]
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Recover final payload files from .constellation.npy files, or from "
            ".fft.npy files when a channel impulse response is supplied."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=script_dir / "ofdm_stage1_outputs",
        help="Input .npy file or directory. Default: Week2_Challenge/ofdm_stage1_outputs",
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=Path,
        help="Optional FIR impulse response file. Use this with .fft.npy inputs.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=script_dir / "recovered_from_npy",
        help="Output directory. Default: Week2_Challenge/recovered_from_npy",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        choices=["auto", "main", "alt"],
        default="auto",
        help="QPSK Gray mapping. Default: auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel = read_channel(args.channel)
    npy_files = iter_npy_files(args.input, channel)

    if not npy_files:
        raise SystemExit(f"No matching .npy files found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RecoveredFile] = []
    failures = 0
    for npy_path in npy_files:
        try:
            result = recover_file(npy_path, channel, args.output_dir, args.mapping)
        except ValueError as exc:
            failures += 1
            print(f"{npy_path}: error: {exc}")
            continue

        results.append(result)
        print(
            f"{npy_path} -> {result.output} "
            f"({result.size_bytes} bytes, mapping={result.mapping}, mode={result.mode})"
        )

    if results:
        summary_path = write_summary(results, args.output_dir)
        print(f"summary_written: {summary_path}")

    if failures:
        raise SystemExit(f"{failures} file(s) failed")


if __name__ == "__main__":
    main()
