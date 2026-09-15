"""N3.1 receiver entry point with explicit header diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from n3.modem import INFO_BYTES, bits_from_bytes, header_bytes, parse_header, payload_symbol_count
from n3.rx import recover_recording as _recover_recording

from n3_1 import PROFILE


def _bit_error_metrics(received: bytes, expected: bytes) -> tuple[int | None, float | None]:
    if len(received) != len(expected):
        return None, None
    errors = int(np.count_nonzero(bits_from_bytes(received) != bits_from_bytes(expected)))
    return errors, float(errors / (len(expected) * 8))


def write_header_diagnostics(out: str | Path, source: str | Path | None = None) -> dict:
    """Write parse and optional truth-comparison evidence for the decoded header."""
    output = Path(out)
    raw_path = output / "decoded_header.bin"
    raw = raw_path.read_bytes() if raw_path.exists() else b""
    debug: dict[str, object] = {
        "header_available": len(raw) == INFO_BYTES,
        "header_bytes": len(raw),
        "header_parse_ok": False,
        "header_error": None,
        "expected_header_bit_errors": None,
        "expected_header_bit_error_rate": None,
    }
    try:
        header = parse_header(raw)
        debug["header_parse_ok"] = True
        debug["header"] = header
    except ValueError as exc:
        debug["header_error"] = str(exc)

    if source is not None and Path(source).exists() and len(raw) == INFO_BYTES:
        truth_path = Path(source)
        truth = truth_path.read_bytes()
        expected = header_bytes(
            truth_path.name,
            len(truth),
            payload_symbol_count(len(truth)),
        )
        errors, rate = _bit_error_metrics(raw, expected)
        debug["expected_header_bit_errors"] = errors
        debug["expected_header_bit_error_rate"] = rate

    (output / "header_debug.json").write_text(
        json.dumps(debug, indent=2), encoding="utf-8"
    )
    return debug


def recover_recording(
    receive: str | Path,
    out: str | Path,
    source: str | Path | None = None,
) -> dict:
    """Recover a frame and always attach header-level diagnostic evidence."""
    output = Path(out)
    metrics = _recover_recording(receive, output, source)
    debug = write_header_diagnostics(output, source)
    metrics["standard_profile"] = PROFILE
    metrics["header_debug"] = debug
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="recover N3.1 standard-compatible WAVs")
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    options = args()
    failed = False
    many = len(options.input) > 1
    for receive in options.input:
        destination = options.out / receive.stem if many else options.out
        metrics = recover_recording(receive, destination, options.source)
        print(
            f"{receive}: header_ok={metrics['header_ok']} "
            f"file_match={metrics['file_match']} profile={PROFILE}"
        )
        failed = failed or not metrics["header_ok"] or (
            options.source is not None and not metrics["file_match"]
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
