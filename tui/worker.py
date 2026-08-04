from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .dsp import decode_file, encode_file, emit
from .media import discover_devices, play, record
from .profiles import ModemProfile


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="AudioModem TUI structured worker")
    commands = root.add_subparsers(dest="action", required=True)
    enc = commands.add_parser("encode")
    enc.add_argument("source", type=Path); enc.add_argument("--profile", type=Path, required=True)
    enc.add_argument("--out", type=Path, required=True)
    dec = commands.add_parser("decode")
    dec.add_argument("recording", type=Path); dec.add_argument("--profile", type=Path, required=True)
    dec.add_argument("--source", type=Path); dec.add_argument("--out", type=Path, required=True)
    rec = commands.add_parser("record")
    rec.add_argument("--out", type=Path, required=True); rec.add_argument("--seconds", type=float, required=True)
    rec.add_argument("--target", default=""); rec.add_argument("--volume", type=float, default=1.0)
    pla = commands.add_parser("play")
    pla.add_argument("wav", type=Path); pla.add_argument("--target", default="")
    pla.add_argument("--volume", type=float, default=1.0)
    commands.add_parser("devices")
    return root


def _compat_encode(a, profile: ModemProfile) -> None:
    command = [
        sys.executable, "-m", "tui.compat_tx", str(a.source),
        "--noise-seconds", str(profile.noise_seconds), "--sync-symbols", str(profile.sync_symbols),
        "--sync-seed", str(profile.sync_seed), "--preamble-symbols", str(profile.preamble_symbols),
        "--preamble-repeats", str(profile.preamble_repeats), "--preamble-seed", str(profile.preamble_seed),
        "--block-size", str(profile.block_size), "--header-repeats", str(profile.header_repeats),
        "--pilot-seed", str(profile.pilot_seed), "--fec-seed", str(profile.fec_seed),
        "--timing-anchor-interval", str(profile.timing_anchor_interval),
        "--timing-anchor-symbols", str(profile.timing_anchor_symbols),
        "--timing-anchor-seed", str(profile.timing_anchor_seed),
        "--payload-start-anchor-symbols", str(profile.payload_start_anchor_symbols),
        "--payload-start-anchor-seed", str(profile.payload_start_anchor_seed),
        "--tail-seconds", str(profile.tail_seconds), "--out", str(a.out),
    ]
    emit("command", command=command)
    run = subprocess.run(command, text=True, capture_output=True)
    if run.stdout: emit("log", text=run.stdout)
    if run.returncode: raise RuntimeError(run.stderr or f"compat transmitter exited {run.returncode}")
    meta = a.out.with_suffix(".meta.json")
    emit("result", status="success", wav=str(a.out), meta=str(meta))


def _compat_decode(a, profile: ModemProfile) -> None:
    command = [
        sys.executable, "-m", "tui.compat_rx", str(a.recording),
        "--noise-seconds", str(profile.noise_seconds), "--sync-symbols", str(profile.sync_symbols),
        "--sync-correlation-symbols", str(profile.sync_correlation_symbols), "--sync-seed", str(profile.sync_seed),
        "--preamble-symbols", str(profile.preamble_symbols), "--preamble-repeats", str(profile.preamble_repeats),
        "--preamble-seed", str(profile.preamble_seed), "--header-repeats", str(profile.header_repeats),
        "--pilot-seed", str(profile.pilot_seed), "--fec-seed", str(profile.fec_seed),
        "--channel-alpha", str(profile.channel_alpha), "--timing-anchor-interval", str(profile.timing_anchor_interval),
        "--timing-anchor-symbols", str(profile.timing_anchor_symbols), "--timing-anchor-seed", str(profile.timing_anchor_seed),
        "--payload-start-anchor-symbols", str(profile.payload_start_anchor_symbols),
        "--payload-start-anchor-seed", str(profile.payload_start_anchor_seed),
        "--payload-start-anchor-h-alpha", str(profile.payload_start_anchor_h_alpha),
        "--anchor-h-alpha", str(profile.anchor_h_alpha), "--anchor-min-score", str(profile.anchor_min_score),
        "--training-anchor-symbols", str(profile.training_anchor_symbols),
        "--training-anchor-step", str(profile.training_anchor_step), "--clock-search", str(profile.clock_search),
        "--payload-search", str(profile.payload_search), "--phase-slope", profile.phase_slope,
        "--slope-window", str(profile.slope_window), "--slope-clip", str(profile.slope_clip), "--out", str(a.out),
    ]
    if a.source: command += ["--source", str(a.source)]
    emit("command", command=command)
    run = subprocess.run(command, text=True, capture_output=True)
    if run.stdout: emit("log", text=run.stdout)
    if run.returncode: raise RuntimeError(run.stderr or f"compat receiver exited {run.returncode}")
    metrics_path = a.out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status = "exact" if metrics["file_match"] else "crc_ok" if metrics["file_crc_ok"] else "partial" if metrics["header_ok"] else "failed"
    emit("result", status=status, metrics=str(metrics_path), file_match=metrics["file_match"])


def main() -> None:
    a = parser().parse_args()
    try:
        if a.action == "devices":
            emit("result", status="success", **discover_devices()); return
        if a.action == "record": record(a.out, a.seconds, a.target, a.volume); return
        if a.action == "play": play(a.wav, a.target, a.volume); return
        profile = ModemProfile.load(a.profile)
        if a.action == "encode":
            _compat_encode(a, profile) if profile.protocol == "step8_compatible" else encode_file(a.source, a.out, profile)
        elif a.action == "decode":
            _compat_decode(a, profile) if profile.protocol == "step8_compatible" else decode_file(a.recording, a.out, profile, a.source)
    except Exception as exc:
        emit("error", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
