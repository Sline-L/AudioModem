#!/usr/bin/env python3
"""Normalize PCM sample binaries, remove OFDM CP, run FFT, and map constellation points."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FFT_SIZE = 1024
CP_LEN = 32
SYMBOL_LEN = FFT_SIZE + CP_LEN
DATA_BINS = np.arange(1, 512)
INT16_SCALE = 32768.0


@dataclass(frozen=True)
class ProcessedBin:
    source: Path
    normalized_npy_path: Path
    normalized_txt_path: Path
    no_cp_npy_path: Path
    no_cp_txt_path: Path
    fft_npy_path: Path
    fft_txt_path: Path
    constellation_npy_path: Path
    constellation_txt_path: Path
    constellation_svg_path: Path
    constellation_mode: str
    samples: int
    symbols: int
    discarded_samples: int


def iter_bin_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.samples.bin"))


def load_and_normalize(bin_path: Path) -> np.ndarray:
    samples = np.fromfile(bin_path, dtype="<i2")
    if samples.size == 0:
        raise ValueError(f"{bin_path} contains no int16 samples")
    return samples.astype(np.float64) / INT16_SCALE


def remove_cyclic_prefix(normalized: np.ndarray) -> tuple[np.ndarray, int, int]:
    symbol_count = normalized.size // SYMBOL_LEN
    if symbol_count == 0:
        raise ValueError(
            f"need at least {SYMBOL_LEN} samples for one OFDM symbol, got {normalized.size}"
        )

    usable_samples = symbol_count * SYMBOL_LEN
    discarded_samples = normalized.size - usable_samples
    symbols = normalized[:usable_samples].reshape(symbol_count, SYMBOL_LEN)
    no_cp = symbols[:, CP_LEN:]

    return no_cp, symbol_count, discarded_samples


def read_channel(channel_path: Path | None) -> np.ndarray | None:
    if channel_path is None:
        return None
    taps = np.loadtxt(channel_path, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"{channel_path} must contain one FIR tap per line")
    return taps


def active_constellation(fft_values: np.ndarray, channel: np.ndarray | None) -> tuple[np.ndarray, str]:
    active_values = fft_values[:, DATA_BINS]
    if channel is None:
        return active_values, "raw_fft_bins_1_to_511"

    channel_response = np.fft.fft(channel, FFT_SIZE)
    active_response = channel_response[DATA_BINS]
    min_response = np.min(np.abs(active_response))
    if min_response < 1e-8:
        raise ValueError(f"channel response is too close to zero: {min_response:g}")

    return active_values / active_response, "equalized_bins_1_to_511"


def constellation_plot_points(values: np.ndarray, max_points: int) -> np.ndarray:
    points = values.ravel()
    if points.size <= max_points:
        return points

    step = int(np.ceil(points.size / max_points))
    return points[::step]


def write_constellation_svg(values: np.ndarray, output: Path, title: str, max_points: int) -> None:
    points = constellation_plot_points(values, max_points)
    real = points.real
    imag = points.imag

    limit = float(np.percentile(np.abs(np.concatenate([real, imag])), 99.5))
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    limit *= 1.1

    size = 900
    margin = 70
    plot_size = size - 2 * margin

    def x_coord(value: float) -> float:
        return margin + ((value + limit) / (2 * limit)) * plot_size

    def y_coord(value: float) -> float:
        return margin + ((limit - value) / (2 * limit)) * plot_size

    axis_x = x_coord(0.0)
    axis_y = y_coord(0.0)
    circles = "\n".join(
        (
            f'<circle cx="{x_coord(float(x)):.2f}" cy="{y_coord(float(y)):.2f}" '
            'r="1.4" fill="#2563eb" fill-opacity="0.28" />'
        )
        for x, y in zip(real, imag)
    )

    ideal = [
        (limit / 2, limit / 2, "00"),
        (-limit / 2, limit / 2, "01"),
        (-limit / 2, -limit / 2, "11"),
        (limit / 2, -limit / 2, "10"),
    ]
    ideal_marks = "\n".join(
        (
            f'<circle cx="{x_coord(x):.2f}" cy="{y_coord(y):.2f}" r="6" '
            'fill="none" stroke="#dc2626" stroke-width="2" />'
            f'<text x="{x_coord(x) + 10:.2f}" y="{y_coord(y) - 10:.2f}" '
            'font-size="18" fill="#991b1b">'
            f"{label}</text>"
        )
        for x, y, label in ideal
    )

    output.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
<rect width="100%" height="100%" fill="white" />
<text x="{margin}" y="38" font-size="24" font-family="sans-serif" fill="#111827">{title}</text>
<text x="{margin}" y="62" font-size="14" font-family="sans-serif" fill="#4b5563">showing {points.size} of {values.size} points; red labels are ideal QPSK Gray quadrants</text>
<line x1="{margin}" y1="{axis_y:.2f}" x2="{size - margin}" y2="{axis_y:.2f}" stroke="#9ca3af" stroke-width="1" />
<line x1="{axis_x:.2f}" y1="{margin}" x2="{axis_x:.2f}" y2="{size - margin}" stroke="#9ca3af" stroke-width="1" />
<rect x="{margin}" y="{margin}" width="{plot_size}" height="{plot_size}" fill="none" stroke="#374151" stroke-width="1.5" />
{circles}
{ideal_marks}
<text x="{size - margin - 28}" y="{axis_y - 8:.2f}" font-size="16" font-family="sans-serif" fill="#374151">I</text>
<text x="{axis_x + 8:.2f}" y="{margin + 18}" font-size="16" font-family="sans-serif" fill="#374151">Q</text>
</svg>
""",
        encoding="utf-8",
    )


def write_array_text(values: np.ndarray, output: Path) -> None:
    if np.iscomplexobj(values):
        rows = np.column_stack((values.real.ravel(), values.imag.ravel()))
        header = f"shape={values.shape} columns=real imag"
        np.savetxt(output, rows, fmt="%.18e", header=header)
        return

    if values.ndim == 1:
        header = f"shape={values.shape}"
        np.savetxt(output, values, fmt="%.18e", header=header)
        return

    header = f"shape={values.shape}"
    np.savetxt(output, values, fmt="%.18e", header=header)


def process_bin_file(
    bin_path: Path,
    output_dir: Path,
    channel: np.ndarray | None,
    max_plot_points: int,
    write_txt: bool = True,
) -> ProcessedBin:
    normalized = load_and_normalize(bin_path)
    no_cp, symbol_count, discarded_samples = remove_cyclic_prefix(normalized)
    fft_values = np.fft.fft(no_cp, axis=1)
    constellation, constellation_mode = active_constellation(fft_values, channel)

    stem = bin_path.name.removesuffix(".samples.bin")
    normalized_npy_path = output_dir / f"{stem}.normalized.npy"
    normalized_txt_path = output_dir / f"{stem}.normalized.txt"
    no_cp_npy_path = output_dir / f"{stem}.no_cp.npy"
    no_cp_txt_path = output_dir / f"{stem}.no_cp.txt"
    fft_npy_path = output_dir / f"{stem}.fft.npy"
    fft_txt_path = output_dir / f"{stem}.fft.txt"
    constellation_npy_path = output_dir / f"{stem}.constellation.npy"
    constellation_txt_path = output_dir / f"{stem}.constellation.txt"
    constellation_svg_path = output_dir / f"{stem}.constellation.svg"

    np.save(normalized_npy_path, normalized)
    np.save(no_cp_npy_path, no_cp)
    np.save(fft_npy_path, fft_values)
    np.save(constellation_npy_path, constellation)
    if write_txt:
        write_array_text(normalized, normalized_txt_path)
        write_array_text(no_cp, no_cp_txt_path)
        write_array_text(fft_values, fft_txt_path)
        write_array_text(constellation, constellation_txt_path)
    write_constellation_svg(
        constellation,
        constellation_svg_path,
        f"{stem} constellation ({constellation_mode})",
        max_plot_points,
    )

    return ProcessedBin(
        source=bin_path,
        normalized_npy_path=normalized_npy_path,
        normalized_txt_path=normalized_txt_path,
        no_cp_npy_path=no_cp_npy_path,
        no_cp_txt_path=no_cp_txt_path,
        fft_npy_path=fft_npy_path,
        fft_txt_path=fft_txt_path,
        constellation_npy_path=constellation_npy_path,
        constellation_txt_path=constellation_txt_path,
        constellation_svg_path=constellation_svg_path,
        constellation_mode=constellation_mode,
        samples=normalized.size,
        symbols=symbol_count,
        discarded_samples=discarded_samples,
    )


def write_summary(results: list[ProcessedBin], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source",
                "samples",
                "ofdm_symbols",
                "discarded_samples",
                "normalized_npy",
                "normalized_txt",
                "no_cp_npy",
                "no_cp_txt",
                "fft_npy",
                "fft_txt",
                "constellation_npy",
                "constellation_txt",
                "constellation_svg",
                "constellation_mode",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.source,
                    result.samples,
                    result.symbols,
                    result.discarded_samples,
                    result.normalized_npy_path,
                    result.normalized_txt_path,
                    result.no_cp_npy_path,
                    result.no_cp_txt_path,
                    result.fft_npy_path,
                    result.fft_txt_path,
                    result.constellation_npy_path,
                    result.constellation_txt_path,
                    result.constellation_svg_path,
                    result.constellation_mode,
                ]
            )
    return summary_path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Process exported signed 16-bit PCM .samples.bin files: normalize, "
            "remove the first 32 cyclic-prefix samples of each 1056-sample OFDM "
            "symbol, run a 1024-point FFT, then export bins 1..511 as constellation points."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=script_dir / "exported_samples",
        help="Input .samples.bin file or directory. Default: Week2_Challenge/exported_samples",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=script_dir / "ofdm_stage1_outputs",
        help="Output directory. Default: Week2_Challenge/ofdm_stage1_outputs",
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=Path,
        help=(
            "Optional FIR impulse response file. If supplied, constellation points "
            "are channel-equalized before saving and plotting."
        ),
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
    channel = read_channel(args.channel)
    bin_files = iter_bin_files(args.input)
    if not bin_files:
        raise SystemExit(f"No .samples.bin files found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProcessedBin] = []
    for bin_path in bin_files:
        try:
            result = process_bin_file(
                bin_path,
                args.output_dir,
                channel,
                args.max_plot_points,
            )
        except ValueError as exc:
            print(f"{bin_path}: error: {exc}")
            continue

        results.append(result)
        print(
            f"{bin_path} -> {result.constellation_svg_path} "
            f"({result.samples} samples, {result.symbols} OFDM symbols, "
            f"{result.discarded_samples} discarded, "
            f"constellation={result.constellation_mode})"
        )

    if not results:
        raise SystemExit("No files were processed successfully")

    summary_path = write_summary(results, args.output_dir)
    print(f"summary_written: {summary_path}")


if __name__ == "__main__":
    main()
