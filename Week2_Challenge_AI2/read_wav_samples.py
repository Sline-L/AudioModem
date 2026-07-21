#!/usr/bin/env python3
"""Read PCM samples from a WAV file and export them as raw binary."""

from __future__ import annotations

import argparse
import csv
from array import array
from pathlib import Path
import sys
import wave


def pcm_samples(wav_path: Path) -> tuple[list[int], dict[str, int | str | float]]:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        compression = wav_file.getcomptype()
        frame_bytes = wav_file.readframes(frames)

    if sample_width != 2:
        raise ValueError(
            f"only 16-bit PCM WAV is supported by this script, got {sample_width * 8}-bit"
        )
    if compression != "NONE":
        raise ValueError(f"only uncompressed PCM WAV is supported, got {compression}")

    samples = array("h")
    samples.frombytes(frame_bytes)

    metadata = {
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "duration_sec": frames / sample_rate,
        "compression": compression,
        "sample_count": len(samples),
    }
    return samples.tolist(), metadata


def write_binary_samples(samples: list[int], output: Path) -> None:
    """Write headerless signed 16-bit little-endian PCM samples."""
    sample_array = array("h", samples)
    if sys.byteorder != "little":
        sample_array.byteswap()
    output.write_bytes(sample_array.tobytes())


def write_csv(samples: list[int], metadata: dict[str, int | str | float], output: Path) -> None:
    channels = int(metadata["channels"])
    sample_rate = int(metadata["sample_rate_hz"])

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if channels == 1:
            writer.writerow(["index", "time_sec", "sample_bits"])
            for index, amplitude in enumerate(samples):
                writer.writerow([index, index / sample_rate, signed_16bit_to_bits(amplitude)])
            return

        header = ["frame", "time_sec"] + [
            f"channel_{channel}_bits" for channel in range(channels)
        ]
        writer.writerow(header)

        for frame in range(0, len(samples), channels):
            frame_index = frame // channels
            row = [
                frame_index,
                frame_index / sample_rate,
                *[signed_16bit_to_bits(value) for value in samples[frame : frame + channels]],
            ]
            writer.writerow(row)


def signed_16bit_to_bits(value: int) -> str:
    return format(value & 0xFFFF, "016b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read WAV PCM samples and export a headerless signed 16-bit "
            "little-endian binary file."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/file01.wav"),
        help="Input WAV file. Default: data/file01.wav",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output binary path. Default: same path with .samples.bin suffix",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional CSV debug output path. Not used for OFDM decoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".samples.bin")
    samples, metadata = pcm_samples(args.input)
    write_binary_samples(samples, output)

    if args.csv_output:
        write_csv(samples, metadata, args.csv_output)

    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("binary_format: signed 16-bit little-endian PCM samples, no WAV header")
    print(f"min_amplitude: {min(samples)}")
    print(f"max_amplitude: {max(samples)}")
    print(f"first_16_samples: {samples[:16]}")
    print(f"binary_written: {output}")
    if args.csv_output:
        print(f"csv_written: {args.csv_output}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
