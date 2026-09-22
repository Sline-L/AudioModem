# -*- coding: utf-8 -*-
"""接收结果输出（风格参考 AudioModem rx_step8：metrics.json + 一行摘要 + 诊断图）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def build_metrics(
    *,
    wav_path: str | Path,
    out_dir: str | Path,
    stage: str,
    debug: dict,
    header: dict | None = None,
    header_ok: bool = False,
    header_offset: int | None = None,
    recovered_path: str | Path | None = None,
    recovered_name: str = "",
    file_match: bool | None = None,
    source_path: str | Path | None = None,
    error: str = "",
    turbo: bool = False,
) -> dict:
    """组装与 AudioModem 类似的 metrics 字典。"""
    sfo = float(debug.get("sfo", 1.0) or 1.0)
    noise_var = debug.get("noise_var")
    h_abs_mean = debug.get("mean_abs_h")
    train_mse = debug.get("train_mse")
    metrics = {
        "input": str(wav_path),
        "out": str(out_dir),
        "stage": stage,
        "error": error,
        "chirp_peaks": debug.get("chirp_peaks"),
        "body_start": debug.get("body_start"),
        "timing_peak": debug.get("timing_peak"),
        "K": debug.get("K"),
        "sfo": sfo,
        "clock_error_ppm": float((sfo - 1.0) * 1e6),
        "cfo_hz": debug.get("cfo_hz"),
        "timing_offset": debug.get("timing_offset"),
        "residual_delay_samples": debug.get("residual_delay_samples"),
        "noise_var": noise_var,
        "train_mse": train_mse,
        "mean_abs_h": h_abs_mean,
        "median_abs_llr": debug.get("median_abs_llr"),
        "eq_mse_hard": debug.get("eq_mse_hard"),
        "turbo": bool(turbo or debug.get("turbo")),
        "header_ok": bool(header_ok),
        "header_offset": header_offset,
        "header": header,
        "recovered_name": recovered_name,
        "recovered_path": str(recovered_path) if recovered_path else None,
        "file_match": file_match,
        "source": str(source_path) if source_path else None,
        "profile": {
            "N": 8192,
            "CP": 2048,
            "chunk_size": 498,
            "ldpc": "802.16 rate=1/2 z=166",
            "modulation": "QPSK Hermitian",
        },
    }
    return metrics


def write_metrics(out_dir: str | Path, metrics: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def print_summary(metrics: dict) -> None:
    """一行控制台摘要（对齐 AudioModem / n2）。"""
    wav = Path(metrics.get("input") or "?")
    ppm = metrics.get("clock_error_ppm")
    ppm_s = f"{ppm:+.4f}" if ppm is not None else "nan"
    cfo = metrics.get("cfo_hz")
    cfo_s = f"{cfo:+.3f}" if cfo is not None else "nan"
    mse = metrics.get("train_mse")
    mse_s = f"{mse:.3e}" if mse is not None else "nan"
    name = metrics.get("recovered_name") or "-"
    match = metrics.get("file_match")
    match_s = "n/a" if match is None else str(bool(match))
    print(
        f"{wav.name}: stage={metrics.get('stage')} K={metrics.get('K')} "
        f"ppm={ppm_s} cfo_hz={cfo_s} train_mse={mse_s} "
        f"header_ok={metrics.get('header_ok')} turbo={metrics.get('turbo')} "
        f"recovered={name} file_match={match_s}"
    )
    if metrics.get("error"):
        print(f"decode_error={metrics['error']}")


def save_diagnostics(
    out_dir: str | Path,
    *,
    h_start: np.ndarray | None = None,
    h_end: np.ndarray | None = None,
    bins_pos: np.ndarray | None = None,
    x_eq: list | np.ndarray | None = None,
    stream: bytes | None = None,
    sample_rate: float = 48000.0,
    n_fft: int = 8192,
) -> None:
    """写出 H.npy、部分字节流、|H|/星座诊断图。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if h_start is not None:
        np.save(out_dir / "H_start.npy", h_start)
    if h_end is not None:
        np.save(out_dir / "H_end.npy", h_end)
    if stream is not None:
        (out_dir / "info_stream.partial.bin").write_bytes(stream)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if h_start is not None and bins_pos is not None:
        freq = np.asarray(bins_pos) * sample_rate / n_fft
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(freq, np.abs(h_start[bins_pos]), marker=".", markersize=2)
        ax[0].set_ylabel("|H_start|")
        ax[0].grid(alpha=0.25)
        if h_end is not None:
            ax[1].plot(freq, np.abs(h_end[bins_pos]), marker=".", markersize=2, color="tab:orange")
            ax[1].set_ylabel("|H_end|")
        else:
            ax[1].plot(freq, np.angle(h_start[bins_pos]), marker=".", markersize=2)
            ax[1].set_ylabel("arg(H_start)")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "channel.png", dpi=150)
        plt.close(fig)

    if x_eq is not None and len(x_eq) > 0:
        # 取前几个符号画星座
        pts = np.concatenate([np.asarray(z).ravel() for z in list(x_eq)[: min(4, len(x_eq))]])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pts.real, pts.imag, s=4, alpha=0.35)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("Equalized constellation (first symbols)")
        fig.tight_layout()
        fig.savefig(out_dir / "constellation.png", dpi=150)
        plt.close(fig)
