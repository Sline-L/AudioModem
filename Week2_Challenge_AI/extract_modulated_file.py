#!/usr/bin/env python3
"""Extract a file from demodulated binary data with a null-terminated header."""

from __future__ import annotations

import argparse
from pathlib import Path


def read_bit_text(path: Path) -> str:
    bits = "".join(path.read_text(encoding="ascii").split())
    invalid = set(bits) - {"0", "1"}
    if invalid:
        raise ValueError(f"{path} contains non-binary characters: {sorted(invalid)}")
    if len(bits) % 8 != 0:
        raise ValueError(f"bit count must be divisible by 8, got {len(bits)}")
    return bits


def bits_to_bytes(bits: str) -> bytes:
    return bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))


def read_null_terminated(data: bytes, start: int) -> tuple[str, int]:
    end = data.find(b"\0", start)
    if end == -1:
        raise ValueError("missing null terminator in header")
    try:
        value = data[start:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "header string is not valid UTF-8; make sure the input is demodulated "
            "file data, not raw WAV PCM samples"
        ) from exc
    return value, end + 1


def safe_output_path(output_dir: Path, filename: str) -> Path:
    name = Path(filename).name
    if not name:
        raise ValueError("header filename is empty")
    return output_dir / name


def extract_payload(data: bytes) -> tuple[str, bytes]:
    filename, offset = read_null_terminated(data, 0)
    size_text, offset = read_null_terminated(data, offset)

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

    return filename, data[offset:end]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw file data from a demodulated 0/1 text file whose header is "
            "filename\\0size\\0."
        )
    )
    parser.add_argument("input", type=Path, help="Text file containing demodulated 0/1 bits")
    parser.add_argument(
        "-d",
        "--output-dir",
        type=Path,
        default=Path("extracted"),
        help="Directory to write the extracted file. Default: extracted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bits = read_bit_text(args.input)
    data = bits_to_bytes(bits)
    filename, payload = extract_payload(data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = safe_output_path(args.output_dir, filename)
    output_path.write_bytes(payload)

    print(f"filename: {filename}")
    print(f"size: {len(payload)} bytes")
    print(f"written: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
