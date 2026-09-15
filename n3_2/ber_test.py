"""Clean-loopback BER check for the standalone N3.2 modem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rx import run_rx
from .tx import run_tx


def run_ber_test(source: Path, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wav = run_tx(Path(source), out / "tx.wav")
    recovered = run_rx(wav, out / "rx", Path(source))
    original, result = Path(source).read_bytes(), recovered.read_bytes()
    limit = min(len(original), len(result))
    errors = sum(a != b for a, b in zip(original[:limit], result[:limit])) + abs(len(original) - len(result))
    report = {"payload_bytes": len(original), "payload_byte_errors": errors,
              "payload_byte_error_rate": errors / len(original) if original else 0.0}
    (out / "ber.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="N3.2 clean loopback BER test")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_ber_test(args.input, args.out), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
