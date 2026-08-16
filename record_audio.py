"""跨平台麦克风录音工具，用于替代 Linux 的 ``pw-record``。"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path


DEFAULT_RATE = 48_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_COUNT = 400_000


def configure_stdio() -> None:
    """避免部分 Windows 终端因系统编码无法输出中文。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def positive_int(value: str) -> int:
    """把命令行参数转换为正整数。"""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是整数：{value}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def positive_float(value: str) -> float:
    """把命令行参数转换为正浮点数。"""
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是数字：{value}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用麦克风录制 WAV（固定为 48 kHz、单声道、s16）。"
    )
    parser.add_argument("output", nargs="?", type=Path, help="输出 WAV 文件路径")
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--sample-count",
        type=positive_int,
        help=f"录制的每声道采样点数；未指定长度时默认 {DEFAULT_SAMPLE_COUNT}",
    )
    duration.add_argument(
        "--seconds", type=positive_float, help="录音时长（秒），例如 --seconds 10"
    )
    parser.add_argument("--device", help="输入设备编号或设备名称")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    parser.add_argument("--force", action="store_true", help="允许覆盖已经存在的输出文件")
    return parser


def resolve_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def resolve_sample_count(args: argparse.Namespace) -> int:
    if args.seconds is not None:
        return max(1, round(args.seconds * DEFAULT_RATE))
    if args.sample_count is not None:
        return args.sample_count
    return DEFAULT_SAMPLE_COUNT


def write_wav(path: Path, samples, rate: int, channels: int) -> None:
    """把 int16 NumPy 数组写成标准 PCM WAV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.tobytes())


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import sounddevice as sd
    except ImportError:
        parser.exit(
            1,
            "错误：缺少 sounddevice。请运行：python -m pip install sounddevice\n",
        )

    if args.list_devices:
        print(sd.query_devices())
        return 0

    if args.output is None:
        parser.error("必须提供输出 WAV 文件路径（除非使用 --list-devices）")
    if args.output.suffix.lower() != ".wav":
        parser.error("输出文件必须使用 .wav 扩展名")
    if args.output.exists() and not args.force:
        parser.error(f"文件已存在：{args.output}；如需覆盖，请添加 --force")

    sample_count = resolve_sample_count(args)
    seconds = sample_count / DEFAULT_RATE
    device = resolve_device(args.device)

    print(
        f"开始录音：{seconds:.3f} 秒，{DEFAULT_RATE} Hz，{DEFAULT_CHANNELS} 声道，"
        f"s16，共 {sample_count} 个采样点"
    )
    try:
        samples = sd.rec(
            sample_count,
            samplerate=DEFAULT_RATE,
            channels=DEFAULT_CHANNELS,
            dtype="int16",
            device=device,
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\n录音已取消，未写入文件。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"录音失败：{exc}", file=sys.stderr)
        return 1

    write_wav(args.output, samples, DEFAULT_RATE, DEFAULT_CHANNELS)
    print(f"录音完成：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
