"""N3.1 transmitter entry point using the verified N3 PHY implementation."""

from __future__ import annotations

import json
from pathlib import Path

from n3.tx import args, build_frame, information_blocks
from n3.tx import main as _main
from n3.tx import write_transmission as _write_transmission

from n3_1 import PROFILE


def write_transmission(input_path: str | Path, output_path: str | Path) -> dict:
    """Write a reference-compatible frame and identify it as N3.1 in metadata."""
    output = Path(output_path)
    meta = _write_transmission(input_path, output)
    meta["profile"] = PROFILE
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main() -> int:
    """Keep the fixed N3 command-line contract while emitting N3.1 metadata."""
    options = args()
    write_transmission(options.input, options.out)
    print(f"wrote {options.out} profile={PROFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
