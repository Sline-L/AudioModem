from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent


def experiment_id(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.") or "experiment"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{stamp}_{safe}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values)
    path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def recent_manifests(limit: int = 50) -> list[dict[str, Any]]:
    paths = sorted((ROOT / "run").glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    results = []
    for path in paths[:limit]:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            item["manifest_path"] = str(path)
            results.append(item)
        except (OSError, json.JSONDecodeError):
            continue
    return results
