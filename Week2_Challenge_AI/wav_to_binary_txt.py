#!/usr/bin/env python3
"""Convert WAV PCM audio frames to binary text files."""

from __future__ import annotations

import argparse
from pathlib import Path
import wave


def wav_frames_to_bits(wav_path: Path) -> tuple[str, dict[str, int | str]]:
    """Return audio-frame bytes as a continuous 0/1 string and WAV metadata."""
    with wave.open(str(wav_path), "rb") as wav_file:
        metadata = {
            "channels": wav_file.getnchannels(),
            "sample_width_bytes": wav_file.getsampwidth(),
            "sample_rate_hz": wav_file.getframerate(),
            "frames": wav_file.getnframes(),
            "compression": wav_file.getcomptype(),
        }
        frame_bytes = wav_file.readframes(wav_file.getnframes())

    bits = "".join(f"{byte:08b}" for byte in frame_bytes)
    return bits, metadata


def convert_file(wav_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = wav_path.with_suffix(".txt")

    bits, metadata = wav_frames_to_bits(wav_path)
    output_path.write_text(bits + "\n", encoding="ascii")

    duration = metadata["frames"] / metadata["sample_rate_hz"]
    print(
        f"{wav_path} -> {output_path} "
        f"({metadata['channels']} ch, "
        f"{metadata['sample_width_bytes'] * 8}-bit, "
        f"{metadata['sample_rate_hz']} Hz, "
        f"{duration:.3f} s, "
        f"{len(bits)} bits)"
    )
    return output_path


def iter_wav_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.wav"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WAV audio PCM frames to binary .txt files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data",
        type=Path,
        help="WAV file or directory containing .wav files. Default: data",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .txt path. Only valid when input is one WAV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    wav_files = iter_wav_files(input_path)

    if not wav_files:
        raise SystemExit(f"No .wav files found: {input_path}")

    if args.output and len(wav_files) != 1:
        raise SystemExit("--output can only be used with one WAV input file")

    for wav_path in wav_files:
        convert_file(wav_path, args.output)


if __name__ == "__main__":
    main()
