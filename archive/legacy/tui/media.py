from __future__ import annotations

import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import wave

import numpy as np


FS = 48_000
I16 = 32768.0
_child: subprocess.Popen | None = None


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def parse_wpctl_status(text: str) -> dict[str, list[dict[str, str]]]:
    result = {"sources": [], "sinks": []}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.endswith("Sources:") or line.endswith("Sinks:"):
            section = "sources" if line.endswith("Sources:") else "sinks"
            continue
        if line.endswith(":") and line not in ("Sources:", "Sinks:"):
            section = ""
        if section not in result:
            continue
        cleaned = re.sub(r"^[│├└─*\s]+", "", raw).strip()
        match = re.match(r"(\d+)\.\s+(.+?)(?:\s+\[|$)", cleaned)
        if match:
            result[section].append({"id": match.group(1), "name": match.group(2).strip()})
    return result


def discover_devices() -> dict:
    try:
        run = subprocess.run(["wpctl", "status"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"sources": [], "sinks": [], "error": str(exc)}
    devices = parse_wpctl_status(run.stdout)
    devices["error"] = run.stderr.strip() if run.returncode else ""
    return devices


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def _stop_child(*_args) -> None:
    global _child
    if _child and _child.poll() is None:
        _child.terminate()


def record(path: Path, seconds: float, target: str = "", volume: float = 1.0) -> dict:
    global _child
    if seconds <= 0:
        raise ValueError("recording duration must be positive")
    samples_needed = round(seconds * FS)
    command = [
        "pw-record", "--raw", "--rate", str(FS), "--channels", "1", "--format", "s16",
        "--volume", str(volume), "--sample-count", str(samples_needed),
    ]
    if target:
        command += ["--target", target]
    command.append("-")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_term = signal.signal(signal.SIGTERM, _stop_child)
    started = time.monotonic()
    peak = energy = 0.0
    clipping = count = 0
    emit("command", command=command)
    try:
        _child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(FS)
            while count < samples_needed:
                raw = _child.stdout.read(min(9600, (samples_needed - count) * 2))
                if not raw:
                    break
                if len(raw) % 2:
                    raw = raw[:-1]
                wav.writeframesraw(raw)
                values = np.frombuffer(raw, dtype="<i2").astype(float) / I16
                if not len(values):
                    continue
                count += len(values)
                window_peak = float(np.max(np.abs(values)))
                window_rms = float(np.sqrt(np.mean(values * values)))
                peak = max(peak, window_peak)
                energy += float(np.sum(values * values))
                clipping += int(np.count_nonzero(np.abs(values) >= 0.99))
                emit(
                    "meter", elapsed=count / FS, remaining=max(0.0, seconds - count / FS),
                    peak=window_peak, rms=window_rms, clipping=clipping,
                    clipping_fraction=clipping / count,
                )
        returncode = _child.wait(timeout=5)
        stderr = _child.stderr.read().decode(errors="replace").strip()
        if count < samples_needed:
            raise RuntimeError(stderr or f"pw-record ended early at {count / FS:.3f}s")
        if returncode:
            raise RuntimeError(stderr or f"pw-record exited {returncode}")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        if _child and _child.poll() is None:
            _child.terminate()
        _child = None
    result = {
        "path": str(path), "duration": count / FS, "samples": count,
        "peak": peak, "rms": math.sqrt(energy / count) if count else 0.0,
        "clipping_samples": clipping, "clipping_fraction": clipping / count if count else 0.0,
        "target": target or "default", "volume": volume,
        "elapsed_wall": time.monotonic() - started,
    }
    emit("result", status="success", **result)
    return result


def play(path: Path, target: str = "", volume: float = 1.0) -> dict:
    global _child
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    command = ["pw-play", "--volume", str(volume)]
    if target:
        command += ["--target", target]
    command.append(str(path))
    emit("command", command=command)
    started = time.monotonic()
    old_term = signal.signal(signal.SIGTERM, _stop_child)
    try:
        _child = subprocess.Popen(command, stderr=subprocess.PIPE)
        duration = wav_seconds(path)
        while _child.poll() is None:
            elapsed = time.monotonic() - started
            emit("progress", stage="playback", elapsed=elapsed, remaining=max(0.0, duration - elapsed))
            time.sleep(0.2)
        stderr = _child.stderr.read().decode(errors="replace").strip()
        if _child.returncode:
            raise RuntimeError(stderr or f"pw-play exited {_child.returncode}")
    finally:
        signal.signal(signal.SIGTERM, old_term)
        if _child and _child.poll() is None:
            _child.terminate()
        _child = None
    result = {"path": str(path), "duration": wav_seconds(path), "target": target or "default", "volume": volume}
    emit("result", status="success", **result)
    return result
