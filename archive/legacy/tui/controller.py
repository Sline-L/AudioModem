from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import sys
from typing import Awaitable, Callable

from .manifest import ROOT, experiment_id, now, write_manifest
from .media import wav_seconds
from .profiles import ModemProfile


EventHandler = Callable[[dict], None]


@dataclass
class Job:
    kind: str
    label: str
    profile: ModemProfile | None = None
    id: str = field(init=False)
    run_dir: Path = field(init=False)
    manifest_path: Path = field(init=False)
    processes: set[asyncio.subprocess.Process] = field(default_factory=set)
    cancelled: bool = False

    def __post_init__(self) -> None:
        self.id = experiment_id(self.label)
        self.run_dir = ROOT / "run" / self.id
        self.manifest_path = self.run_dir / "manifest.json"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        values = {
            "experiment_id": self.id, "kind": self.kind, "label": self.label,
            "status": "created", "created_at": now(), "artifacts": [], "commands": [], "stages": [],
        }
        if self.profile:
            profile_path = self.run_dir / "profile.json"
            profile_path.write_text(json.dumps(self.profile.to_dict(), indent=2), encoding="utf-8")
            values.update({"profile": self.profile.to_dict(), "profile_path": str(profile_path)})
        write_manifest(self.manifest_path, values)

    @property
    def profile_path(self) -> Path:
        return self.run_dir / "profile.json"

    def update(self, **values) -> None:
        write_manifest(self.manifest_path, values)

    def cancel(self) -> None:
        self.cancelled = True
        for process in tuple(self.processes):
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        self.update(status="cancelled", ended_at=now())


class ExperimentController:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.manifest_lock = asyncio.Lock()

    def create(self, kind: str, label: str, profile: ModemProfile | None = None) -> Job:
        job = Job(kind, label, profile)
        self.jobs[job.id] = job
        return job

    async def command(self, job: Job, arguments: list[str], on_event: EventHandler) -> dict:
        command = [sys.executable, "-m", "tui.worker", *arguments]
        async with self.manifest_lock:
            current = json.loads(job.manifest_path.read_text(encoding="utf-8"))
            commands = current.get("commands", []) + [command]
            stages = current.get("stages", []) + [{"action": arguments[0], "status": "running", "started_at": now()}]
            stage_index = len(stages) - 1
            job.update(status="running", started_at=current.get("started_at", now()), command=command,
                       commands=commands, stages=stages)
        on_event({"event": "command", "command": command, "job_id": job.id})
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT.parent,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        job.processes.add(process)
        result: dict = {}

        async def read_stdout() -> None:
            nonlocal result
            assert process.stdout
            while line := await process.stdout.readline():
                text = line.decode(errors="replace").rstrip()
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    event = {"event": "log", "text": text}
                event["job_id"] = job.id
                on_event(event)
                if event.get("event") in ("result", "error"):
                    result = event

        stdout_task = asyncio.create_task(read_stdout())
        assert process.stderr
        stderr_task = asyncio.create_task(process.stderr.read())
        await process.wait()
        await stdout_task
        stderr = (await stderr_task).decode(errors="replace").strip()
        job.processes.discard(process)
        if stderr:
            on_event({"event": "log", "text": stderr, "job_id": job.id})
        if process.returncode:
            if job.cancelled:
                job.update(status="cancelled", ended_at=now())
                raise asyncio.CancelledError
            error = result.get("error") or stderr or f"worker exited {process.returncode}"
            job.update(status="failed", ended_at=now(), error=error)
            raise RuntimeError(error)
        async with self.manifest_lock:
            current = json.loads(job.manifest_path.read_text(encoding="utf-8"))
            stages = current.get("stages", [])
            if stage_index < len(stages):
                stages[stage_index].update({"status": result.get("status", "success"), "ended_at": now(), "result": result})
            job.update(status=result.get("status", "success"), ended_at=now(), result=result, stages=stages)
        return result

    async def encode(self, job: Job, source: Path, filename: str, on_event: EventHandler) -> tuple[Path, dict]:
        tx_dir = ROOT / "data" / "tx" / job.id
        tx_dir.mkdir(parents=True, exist_ok=False)
        out = tx_dir / filename
        if out.suffix.lower() != ".wav": out = out.with_suffix(".wav")
        result = await self.command(job, ["encode", str(source), "--profile", str(job.profile_path), "--out", str(out)], on_event)
        artifacts = [str(path) for path in tx_dir.iterdir()]
        job.update(source=str(Path(source).resolve()), tx_wav=str(out), artifacts=artifacts)
        return out, result

    async def decode(self, job: Job, recording: Path, source: Path | None, on_event: EventHandler) -> dict:
        decode_dir = job.run_dir / "decode"
        args = ["decode", str(recording), "--profile", str(job.profile_path), "--out", str(decode_dir)]
        if source: args += ["--source", str(source)]
        result = await self.command(job, args, on_event)
        metrics_path = decode_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        artifacts = [str(path) for path in decode_dir.iterdir()] if decode_dir.exists() else []
        job.update(recording=str(Path(recording).resolve()), source=str(Path(source).resolve()) if source else None,
                   metrics=metrics, metrics_path=str(metrics_path), artifacts=artifacts)
        return metrics

    async def record(self, job: Job, seconds: float, filename: str, target: str,
                     volume: float, on_event: EventHandler) -> tuple[Path, dict]:
        rx_dir = ROOT / "data" / "rx" / job.id
        rx_dir.mkdir(parents=True, exist_ok=False)
        out = rx_dir / filename
        if out.suffix.lower() != ".wav": out = out.with_suffix(".wav")
        result = await self.command(job, ["record", "--out", str(out), "--seconds", str(seconds),
                                                  "--target", target, "--volume", str(volume)], on_event)
        job.update(recording=str(out), capture=result, artifacts=[str(out)])
        return out, result

    async def play(self, job: Job, wav: Path, target: str, volume: float, on_event: EventHandler) -> dict:
        result = await self.command(job, ["play", str(wav), "--target", target, "--volume", str(volume)], on_event)
        job.update(tx_wav=str(Path(wav).resolve()), playback=result)
        return result

    async def pipeline(self, job: Job, source: Path, tx_filename: str, rx_filename: str,
                       duration: float, pre_roll: float, input_target: str, output_target: str,
                       record_volume: float, play_volume: float, on_event: EventHandler) -> dict:
        tx_wav, _ = await self.encode(job, source, tx_filename, on_event)
        tx_seconds = wav_seconds(tx_wav)
        if duration < pre_roll + tx_seconds:
            raise ValueError(f"recording needs at least {pre_roll + tx_seconds:.3f}s")
        tail = duration - pre_roll - tx_seconds
        if tail < 0.25:
            on_event({"event": "warning", "text": f"tail margin is only {tail:.3f}s", "job_id": job.id})
        rx_dir = ROOT / "data" / "rx" / job.id
        rx_dir.mkdir(parents=True, exist_ok=False)
        rx_wav = (rx_dir / rx_filename).with_suffix(".wav")
        record_task = asyncio.create_task(self.command(
            job, ["record", "--out", str(rx_wav), "--seconds", str(duration),
                  "--target", input_target, "--volume", str(record_volume)], on_event
        ))
        try:
            await asyncio.sleep(pre_roll)
            if record_task.done():
                await record_task
            await self.command(job, ["play", str(tx_wav), "--target", output_target,
                                     "--volume", str(play_volume)], on_event)
            capture = await record_task
        except BaseException:
            if not record_task.done():
                job.cancel()
                await asyncio.gather(record_task, return_exceptions=True)
            raise
        job.update(recording=str(rx_wav), capture=capture, pre_roll=pre_roll,
                   requested_duration=duration, tail_margin=tail)
        metrics = await self.decode(job, rx_wav, source, on_event)
        return metrics
