#!/usr/bin/env python3
"""Run the full Week 2 OFDM recovery pipeline by orchestrating helper scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_header_from_npy import (
    decode_header,
    write_header_file,
    write_summary as write_header_summary,
)
from ofdm_stage1_fft import (
    process_bin_file,
    read_channel,
    write_summary as write_ofdm_summary,
)
from read_wav_samples import pcm_samples, write_binary_samples, write_csv
from recover_files_from_npy import (
    recover_file,
    write_summary as write_recovery_summary,
)


def iter_wav_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.wav"))


def export_samples(wav_path: Path, output_dir: Path, write_csv_output: bool) -> Path:
    samples, metadata = pcm_samples(wav_path)
    bin_path = output_dir / f"{wav_path.stem}.samples.bin"
    write_binary_samples(samples, bin_path)

    if write_csv_output:
        csv_path = output_dir / f"{wav_path.stem}.samples.csv"
        write_csv(samples, metadata, csv_path)

    return bin_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run WAV sample export, OFDM FFT/constellation processing, header "
            "extraction, and final file recovery."
        )
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
        help="Channel impulse response CSV. Default: Week2_Challenge/channel.csv",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=script_dir,
        help="Root directory for output folders. Default: Week2_Challenge",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        choices=["auto", "main", "alt"],
        default="auto",
        help="QPSK Gray mapping. Default: auto",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also export large per-sample CSV files.",
    )
    parser.add_argument(
        "--write-txt",
        action="store_true",
        help="Also export large text versions of normalized/no-CP/FFT/constellation arrays.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=50000,
        help="Maximum constellation points drawn per SVG. Default: 50000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_files = iter_wav_files(args.input)
    if not wav_files:
        raise SystemExit(f"No .wav files found: {args.input}")

    channel = read_channel(args.channel)
    if channel is None:
        raise SystemExit(f"Channel file is required: {args.channel}")

    exported_dir = args.output_root / "exported_samples"
    ofdm_dir = args.output_root / "ofdm_stage1_outputs"
    headers_dir = args.output_root / "extracted_headers"
    recovered_dir = args.output_root / "recovered_from_main"
    for directory in [exported_dir, ofdm_dir, headers_dir, recovered_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    ofdm_results = []
    header_results = []
    recovered_results = []
    failures = 0

    for wav_path in wav_files:
        try:
            bin_path = export_samples(wav_path, exported_dir, args.write_csv)
            ofdm_result = process_bin_file(
                bin_path,
                ofdm_dir,
                channel,
                args.max_plot_points,
                write_txt=args.write_txt,
            )
            header_result = decode_header(
                ofdm_result.fft_npy_path,
                channel,
                args.mapping,
            )
            header_result = write_header_file(header_result, headers_dir)
            recovered_result = recover_file(
                ofdm_result.fft_npy_path,
                channel,
                recovered_dir,
                args.mapping,
            )
        except ValueError as exc:
            failures += 1
            print(f"{wav_path}: error: {exc}")
            continue

        ofdm_results.append(ofdm_result)
        header_results.append(header_result)
        recovered_results.append(recovered_result)
        print(
            f"{wav_path} -> {recovered_result.output} "
            f"({recovered_result.size_bytes} bytes, "
            f"{ofdm_result.symbols} OFDM symbols, "
            f"mapping={recovered_result.mapping})"
        )

    if ofdm_results:
        print(f"ofdm_summary_written: {write_ofdm_summary(ofdm_results, ofdm_dir)}")
    if header_results:
        print(f"header_summary_written: {write_header_summary(header_results, headers_dir)}")
    if recovered_results:
        print(
            f"recovery_summary_written: "
            f"{write_recovery_summary(recovered_results, recovered_dir)}"
        )

    if failures:
        raise SystemExit(f"{failures} file(s) failed")


if __name__ == "__main__":
    main()
