# -*- coding: utf-8 -*-
"""n1 接收端命令行。

在仓库根运行：

    python n1/run_rx.py --wav data/r1.wav --out-dir run/n1/r1

含义：解析 data/r1.wav，结果全部写到 run/n1/r1/（含还原文件与 metrics.json）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_N1 = Path(__file__).resolve().parent
_ROOT = _N1.parent
if str(_N1) not in sys.path:
    sys.path.insert(0, str(_N1))

from ofdm_rx import Ofdm_rx

DIR_DATA = _ROOT / "data"
DIR_SOURCE = _ROOT / "source"
if not DIR_SOURCE.is_dir() and (_ROOT / "sourse").is_dir():
    DIR_SOURCE = _ROOT / "sourse"
DIR_RUN = _ROOT / "run"
DIR_N1_RUN = DIR_RUN / "n1"

_ROOT_MARKERS = {
    "data": DIR_DATA,
    "source": _ROOT / "source",
    "sourse": _ROOT / "sourse",
    "run": DIR_RUN,
}


def _tokens(raw: str) -> list[str]:
    s = str(raw).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = s[2:]
    s = s.lstrip("/")
    return [p for p in s.split("/") if p and p not in (".", "..")]


def _after_marker(raw: str):
    parts = _tokens(raw)
    names = [p.lower() for p in parts]
    for i, name in enumerate(names):
        if name in _ROOT_MARKERS:
            rest = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path()
            return _ROOT_MARKERS[name], rest
    return None, None


def resolve_wav(wav_arg: str | None) -> Path:
    """解析 --wav，优先落到仓库根 data/。"""
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    if wav_arg is None:
        wavs = sorted(DIR_DATA.glob("*.wav")) + sorted(DIR_DATA.glob("*.WAV"))
        if len(wavs) == 1:
            return wavs[0]
        if not wavs:
            raise FileNotFoundError(f"找不到 WAV，请放到 {DIR_DATA} 或指定 --wav")
        names = ", ".join(p.name for p in wavs)
        raise FileNotFoundError(f"{DIR_DATA} 中有多个 WAV，请用 --wav 指定：{names}")

    raw = str(wav_arg)
    base, rest = _after_marker(raw)
    if base is not None:
        path = base / rest if rest.parts else base
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"找不到接收音频：{path}")

    path = Path(raw)
    if path.is_file():
        return path.resolve()
    if (DIR_DATA / path.name).is_file():
        return (DIR_DATA / path.name).resolve()
    if (_ROOT / path).is_file():
        return (_ROOT / path).resolve()
    raise FileNotFoundError(f"找不到接收音频：{wav_arg}（已查 {DIR_DATA}）")


def resolve_out_dir(name: str | None, wav_stem: str | None = None) -> Path:
    """解析 --out-dir。

    - 默认：run/n1/<wav_stem>，例如 r1.wav → run/n1/r1
    - 显式 run/n1/r1 → 仓库根下该目录（不再套一层 stem）
    """
    DIR_N1_RUN.mkdir(parents=True, exist_ok=True)
    if not name:
        dest = DIR_N1_RUN / (wav_stem or "out")
        dest.mkdir(parents=True, exist_ok=True)
        return dest.resolve()

    base, rest = _after_marker(name)
    if base is not None:
        dest = base / rest if rest.parts else base
        dest.mkdir(parents=True, exist_ok=True)
        return dest.resolve()

    path = Path(name)
    if path.is_absolute():
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    # 相对路径：相对仓库根（与从根执行 python n1/run_rx.py 一致）
    dest = (_ROOT / path).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _guess_source(name_hint: str | None = None) -> Path | None:
    if not DIR_SOURCE.is_dir():
        return None
    files = [p for p in DIR_SOURCE.iterdir() if p.is_file()]
    if name_hint:
        for p in files:
            if p.name == name_hint:
                return p
    if len(files) == 1:
        return files[0]
    return None


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="OFDM n1 接收端",
        epilog=(
            "示例（在仓库根执行）：  "
            "python n1/run_rx.py --wav data/r1.wav --out-dir run/n1/r1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wav",
        nargs="*",
        default=None,
        help="接收 WAV（相对仓库根，如 data/r1.wav；可多个）",
    )
    parser.add_argument(
        "--out-dir",
        "--out",
        default=None,
        help="输出目录（如 run/n1/r1；默认 run/n1/<wav名>）",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="可选原始文件，用于 file_match（默认按帧头名在 source/ 查找）",
    )
    args = parser.parse_args(argv)

    wav_args = args.wav
    if not wav_args:
        try:
            wav_args = [str(resolve_wav(None))]
        except FileNotFoundError as exc:
            print(exc)
            return 1

    many = len(wav_args) > 1
    results = []
    base_out = None

    for wav in wav_args:
        try:
            wav_path = resolve_wav(wav)
        except FileNotFoundError as exc:
            print(exc)
            return 1

        if many:
            # 多文件：--out-dir 为父目录（默认 run/n1），每文件一个子目录
            parent = resolve_out_dir(args.out_dir or "run/n1", wav_stem=None)
            out_dir = parent / wav_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            base_out = parent
        else:
            # 单文件：--out-dir run/n1/r1 即最终目录，不再套一层
            out_dir = resolve_out_dir(args.out_dir, wav_stem=wav_path.stem)

        source = Path(args.source) if args.source else None
        rx = Ofdm_rx()
        try:
            path = rx.rx(
                str(wav_path),
                out_dir=str(out_dir),
                source_path=str(source) if source else None,
            )
            print(f"recovered={path}")
            print(f"out_dir={out_dir}")
            print(f"metrics={out_dir / 'metrics.json'}")
        except Exception as exc:
            print(f"failed: {exc}")
            print(f"out_dir={out_dir}")
            print(f"metrics={out_dir / 'metrics.json'}")
            if not many:
                return 1
        results.append(rx.metrics)

    if many and results and base_out is not None:
        fields = [
            "input",
            "out",
            "stage",
            "K",
            "clock_error_ppm",
            "cfo_hz",
            "train_mse",
            "header_ok",
            "turbo",
            "recovered_name",
            "file_match",
            "error",
        ]
        csv_path = base_out / "batch_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        (base_out / "batch_summary.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"batch_summary={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
