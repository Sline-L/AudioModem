"""n2 接收端命令行。在 n2/ 下运行，路径相对仓库根的 data/ 与 run/。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

N2 = Path(__file__).resolve().parent
if str(N2) not in sys.path:
    sys.path.insert(0, str(N2))

from paths import (  # noqa: E402
    DATA_DIR,
    N2_RUN_DIR,
    SOURCE_DIR,
    ensure_dirs,
    resolve_run_dir,
    resolve_wav,
)
from phy import read_wav_channels  # noqa: E402
from receiver import recover_from_channels  # noqa: E402
from report import format_console_line, write_run_outputs  # noqa: E402


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ensure_dirs()
    parser = argparse.ArgumentParser(
        description="OFDM n2 接收端",
        epilog=(
            "在 n2 目录下运行，例如：  "
            "python run_rx.py --wav ..\\data\\r1.wav --out-dir ..\\run\\n2\\r1"
        ),
    )
    parser.add_argument(
        "--wav",
        default=None,
        help=f"接收 WAV（默认找 {DATA_DIR} 下唯一的 wav）",
    )
    parser.add_argument(
        "--out-dir",
        "--out",
        default=str(N2_RUN_DIR),
        help=f"输出目录（默认: {N2_RUN_DIR}）",
    )
    parser.add_argument(
        "--source-dir",
        default=str(SOURCE_DIR),
        help=f"未做 OFDM 的原文件目录（默认: {SOURCE_DIR}）",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="对照原文件（只用于 file_match / BER，不参与解调）",
    )
    args = parser.parse_args(argv)
    try:
        wav_path = resolve_wav(args.wav)
        out_dir = resolve_run_dir(args.out_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    source_path = Path(args.source) if args.source else None
    fs, channels = read_wav_channels(wav_path)
    result = recover_from_channels(channels, fs=fs)
    result.diagnostics["wav"] = str(wav_path)
    result.artifacts["wav"] = str(wav_path)
    result.artifacts["out_dir"] = str(out_dir)
    if source_path is None and result.filename:
        cand = Path(args.source_dir) / result.filename
        if cand.is_file():
            source_path = cand
    source_match = None
    if source_path is not None and source_path.is_file() and result.payload:
        source_match = source_path.read_bytes() == result.payload
    metrics = write_run_outputs(
        out_dir,
        result,
        source_path=source_path,
        source_match=source_match,
    )
    print(format_console_line(metrics, wav_path.name))
    recovered = metrics.get("recovered_name")
    out_path = metrics.get("out_path") or result.out_path
    if recovered and metrics.get("header_ok"):
        print(f"recovered={recovered} -> {out_path}")
    elif metrics.get("error"):
        print(f"decode_error={metrics['error']}")
    print(f"wrote {Path(out_dir) / 'metrics.json'}")
    named = bool(metrics.get("header_ok") and recovered and not str(recovered).endswith(".partial"))
    return 0 if result.ok or named else 1


if __name__ == "__main__":
    raise SystemExit(main())
