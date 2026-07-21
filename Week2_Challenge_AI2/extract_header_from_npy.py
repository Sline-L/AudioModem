#!/usr/bin/env python3
"""Extract filename and size headers from OFDM FFT or constellation .npy files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FFT_SIZE = 1024
DATA_BINS = np.arange(1, 512)


@dataclass(frozen=True)
class HeaderResult:
    source: Path
    output: Path
    filename: str
    size_bytes: int
    mapping: str
    mode: str
    header_bytes: int


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


def parse_header(data: bytes) -> tuple[str, int, int]:
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

    return filename, file_size, offset


def decode_header_with_mapping(
    npy_path: Path, channel: np.ndarray | None, mapping: str
) -> HeaderResult:
    constellation, mode = constellation_from_npy(npy_path, channel)
    bit_pairs = constellation_to_bit_pairs(constellation, mapping)
    data = bit_pairs_to_bytes(bit_pairs)
    filename, file_size, header_bytes = parse_header(data)

    return HeaderResult(
        source=npy_path,
        output=Path(),
        filename=filename,
        size_bytes=file_size,
        mapping=mapping,
        mode=mode,
        header_bytes=header_bytes,
    )


def decode_header(npy_path: Path, channel: np.ndarray | None, mapping: str) -> HeaderResult:
    mappings = ["main", "alt"] if mapping == "auto" else [mapping]
    errors: list[str] = []

    for candidate in mappings:
        try:
            return decode_header_with_mapping(npy_path, channel, candidate)
        except ValueError as exc:
            errors.append(f"{candidate}: {exc}")

    raise ValueError("; ".join(errors))


def header_output_name(npy_path: Path) -> str:
    name = npy_path.name
    for suffix in [".constellation.npy", ".fft.npy"]:
        if name.endswith(suffix):
            return name.removesuffix(suffix) + ".header.txt"
    return npy_path.stem + ".header.txt"


def write_header_file(result: HeaderResult, output_dir: Path) -> HeaderResult:
    output_path = output_dir / header_output_name(result.source)
    output_path.write_text(
        "\n".join(
            [
                f"source: {result.source}",
                f"mode: {result.mode}",
                f"mapping: {result.mapping}",
                f"filename: {result.filename}",
                f"size_bytes: {result.size_bytes}",
                f"header_bytes: {result.header_bytes}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return HeaderResult(
        source=result.source,
        output=output_path,
        filename=result.filename,
        size_bytes=result.size_bytes,
        mapping=result.mapping,
        mode=result.mode,
        header_bytes=result.header_bytes,
    )


def write_summary(results: list[HeaderResult], output_dir: Path) -> Path:
    summary_path = output_dir / "headers.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source",
                "header_txt",
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
            "Extract filename\\0size\\0 headers from .constellation.npy files, "
            "or from .fft.npy files when a channel impulse response is supplied."
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
        default=script_dir / "extracted_headers",
        help="Output directory. Default: Week2_Challenge/extracted_headers",
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

    results: list[HeaderResult] = []
    failures = 0
    for npy_path in npy_files:
        try:
            result = decode_header(npy_path, channel, args.mapping)
            result = write_header_file(result, args.output_dir)
        except ValueError as exc:
            failures += 1
            print(f"{npy_path}: error: {exc}")
            continue

        results.append(result)
        print(
            f"{npy_path} -> {result.output} "
            f"(filename={result.filename}, size={result.size_bytes}, "
            f"mapping={result.mapping}, mode={result.mode})"
        )

    if results:
        summary_path = write_summary(results, args.output_dir)
        print(f"summary_written: {summary_path}")

    if failures:
        raise SystemExit(f"{failures} file(s) failed")


if __name__ == "__main__":
    main()
