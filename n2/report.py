"""把一次接收写成 AudioModem 风格的 run 目录：metrics、图、npy、恢复文件。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from params import derived
from phy import HEADER_MAGIC, parse_ph_header, safe_filename


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return None
    try:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            if obj.size > 64 and obj.ndim >= 1:
                return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _save_npy(out: Path, name: str, value):
    if value is None:
        return
    arr = np.asarray(value)
    np.save(out / name, arr)


def _try_plots(out: Path, art: dict, d: dict):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    bins = np.asarray(art.get("bins", np.arange(d["start_index"], d["start_index"] + d["n_active"])))
    freq = bins * d["sample_rate"] / d["N"]
    h = art.get("H")
    noise = art.get("training_noise")
    if h is not None and noise is not None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(freq, np.abs(h))
        ax[0].set_ylabel("|H|")
        ax[0].grid(alpha=0.25)
        ax[1].semilogy(freq, np.maximum(np.asarray(noise), 1e-15), label="preamble residual")
        end_n = art.get("end_noise")
        if end_n is not None:
            ax[1].semilogy(freq, np.maximum(np.asarray(end_n), 1e-15), label="end residual")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Noise power")
        ax[1].legend()
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "channel_and_noise.png", dpi=150)
        plt.close(fig)

    cpe = art.get("cpe")
    z = art.get("z")
    if cpe is not None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(np.unwrap(np.asarray(cpe, dtype=np.float64)))
        ax[0].set_ylabel("CPE (rad)")
        ax[0].grid(alpha=0.25)
        if z is not None and len(z):
            evm = []
            for row in np.asarray(z):
                xhat = np.sign(row.real + 1e-30) + 1j * np.sign(row.imag + 1e-30)
                evm.append(float(np.sqrt(np.mean(np.abs(row - xhat) ** 2) / 2.0)))
            ax[1].plot(evm)
            ax[1].set_ylabel("QPSK EVM")
        ax[1].set_xlabel("Data OFDM symbol")
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "phase_tracking.png", dpi=150)
        plt.close(fig)

    table = art.get("clock_table")
    if table is not None:
        table = np.asarray(table, dtype=np.float64)
        if table.ndim == 2 and table.shape[0] >= 2 and table.shape[1] >= 2:
            nom, obs = table[:, 0], table[:, 1]
            kind = table[:, 3] if table.shape[1] > 3 else np.zeros(len(nom))
            coef = np.polyfit(nom, obs, 1)
            fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            seconds = nom / d["sample_rate"]
            ax[0].scatter(seconds[kind < 0.5], (obs - nom)[kind < 0.5], label="chirp")
            ax[0].scatter(seconds[kind >= 0.5], (obs - nom)[kind >= 0.5], label="preamble")
            fit = coef[1] + (coef[0] - 1.0) * nom
            ax[0].plot(seconds, fit, color="black", linewidth=1.2, label="clock fit")
            ax[0].set_ylabel("Observed - nominal (samples)")
            ax[0].legend()
            ax[0].grid(alpha=0.25)
            residual = obs - (coef[0] * nom + coef[1])
            ax[1].scatter(seconds, residual)
            ax[1].axhline(0.0, color="black", linewidth=1)
            ax[1].set_xlabel("Nominal time from first chirp (s)")
            ax[1].set_ylabel("Fit residual (samples)")
            ax[1].grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / "clock_fit.png", dpi=150)
            plt.close(fig)


def write_run_outputs(out_dir, result, source_path=None, source_match=None):
    """
    对照 AudioModem rx_step8 的 run 目录：
    metrics.json、summary.csv、decoded_header.bin、H.npy、图、恢复文件或 .partial。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = derived()
    art = dict(result.artifacts or {})
    diag = result.diagnostics or {}
    info = art.get("info_bytes", b"") or b""
    if isinstance(info, str):
        info = info.encode("latin1", errors="replace")
    elif isinstance(info, np.ndarray):
        info = np.asarray(info, dtype=np.uint8).tobytes()
    elif not isinstance(info, (bytes, bytearray)):
        info = bytes(info) if info else b""
    header_raw = art.get("header_raw") or b""
    if isinstance(header_raw, np.ndarray):
        header_raw = np.asarray(header_raw, dtype=np.uint8).tobytes()
    elif not isinstance(header_raw, (bytes, bytearray)):
        header_raw = bytes(header_raw) if header_raw else b""

    header = diag.get("header") if isinstance(diag.get("header"), dict) else {}
    decoded_hdr = bytes(header_raw) if header_raw else b""
    if not decoded_hdr:
        off = int(header.get("offset") or 0)
        decoded_hdr = info[off : off + d["info_bytes"]] if info else b""
    (out / "decoded_header.bin").write_bytes(decoded_hdr)
    parsed = parse_ph_header(decoded_hdr)
    header_ok = parsed is not None or bool(header.get("filename"))
    if result.stage == "header" and diag.get("reason") == "PH 头 CRC 未通过":
        header_ok = False

    recovered_name = result.filename or (header.get("filename") if isinstance(header, dict) else "")
    recovered_name = safe_filename(recovered_name) if recovered_name else "recovered.bin"
    payload = result.payload or b""
    size_ok = bool(payload) and (
        not result.file_size or len(payload) == int(result.file_size)
    )
    # 协议里只有帧头 CRC。有合法头且按 file_size 写出，就保留原文件名，避免被看成没解出来。
    write_named = bool(header_ok and size_ok)
    if write_named:
        target = out / recovered_name
        target.write_bytes(payload)
    else:
        body = info[d["info_bytes"] :] if info[:2] == HEADER_MAGIC else info
        target = out / (recovered_name + ".partial")
        target.write_bytes(payload or body or info)
    art["out_path"] = str(target)
    result.out_path = str(target)
    file_crc_ok = header_ok

    _save_npy(out, "H.npy", art.get("H"))
    _save_npy(out, "H_end.npy", art.get("H_end"))
    _save_npy(out, "H_ls.npy", art.get("H_ls"))
    _save_npy(out, "H_track.npy", art.get("H_track"))
    _save_npy(out, "training_noise.npy", art.get("training_noise"))
    _save_npy(out, "end_noise.npy", art.get("end_noise"))
    _save_npy(out, "cpe.npy", art.get("cpe"))
    _save_npy(out, "rx_llr.npy", art.get("llr"))
    _save_npy(out, "clock_anchors.npy", art.get("clock_table"))

    iters = np.asarray(art.get("iters", []), dtype=np.int32)
    llr = art.get("llr")
    h = art.get("H")
    noise = art.get("training_noise")
    bins = np.asarray(art.get("bins", np.arange(d["start_index"], d["start_index"] + d["n_active"])))

    with (out / "ldpc_symbols.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "iters", "converged", "median_abs_llr"])
        for i, it in enumerate(iters.tolist() if iters.size else []):
            med = ""
            if llr is not None and i < len(llr):
                med = float(np.median(np.abs(llr[i])))
            writer.writerow([i, int(it), bool(int(it) < 200), med])

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin", "freq_hz", "abs_h", "training_noise", "mean_abs_llr"])
        mean_llr = None
        if llr is not None and np.asarray(llr).size:
            arr = np.asarray(llr)
            # 每音 2 bit；按子载波聚合
            if arr.ndim == 2:
                n_sym, n_bits = arr.shape
                n_act = len(bins)
                if n_bits == n_act * 2:
                    mag = np.mean(np.abs(arr).reshape(n_sym, n_act, 2), axis=(0, 2))
                    mean_llr = mag
        for i, bin_index in enumerate(bins):
            writer.writerow(
                [
                    int(bin_index),
                    float(bin_index * d["sample_rate"] / d["N"]),
                    float(np.abs(h[i])) if h is not None and i < len(h) else "",
                    float(noise[i]) if noise is not None and i < len(noise) else "",
                    float(mean_llr[i]) if mean_llr is not None and i < len(mean_llr) else "",
                ]
            )

    _try_plots(out, art, d)

    clock = diag.get("clock") or {}
    ldpc_ok = int(np.count_nonzero(iters < 200)) if iters.size else 0
    ldpc_total = int(iters.size)
    used_llr = np.asarray(llr) if llr is not None else np.array([])
    metrics = {
        "input": art.get("wav", diag.get("wav", "")),
        "out": str(out),
        "ok": bool(result.ok),
        "stage": result.stage,
        "sync_start": float((diag.get("lock") or {}).get("start", 0.0) or 0.0),
        "sync_score": float((diag.get("lock") or {}).get("snr", 0.0) or 0.0),
        "clock_status": clock.get("status", ""),
        "clock_scale": clock.get("after", clock).get("scale")
        if isinstance(clock.get("after"), dict)
        else clock.get("scale"),
        "clock_error_ppm": clock.get("ppm"),
        "residual_ppm": clock.get("residual_ppm"),
        "resampled": bool(diag.get("resampled")),
        "cfo_hz": diag.get("cfo_hz"),
        "K": art.get("K", (diag.get("end") or {}).get("K")),
        "residual_delay_samples": diag.get("residual_delay_samples"),
        "mean_abs_h": float(np.mean(np.abs(h))) if h is not None else None,
        "median_training_noise": float(np.median(noise)) if noise is not None else None,
        "median_abs_llr": float(np.median(np.abs(used_llr))) if used_llr.size else None,
        "turbo": bool(diag.get("turbo")),
        "header_ok": bool(header_ok),
        "header": diag.get("header")
        or (
            {
                "filename": parsed["filename"],
                "file_size": parsed["file_size"],
                "declared_symbols": parsed["declared_symbols"],
            }
            if parsed
            else None
        ),
        "blocks_ok": ldpc_ok,
        "blocks_total": ldpc_total,
        "payload_complete": bool(diag.get("payload_complete", result.ok)),
        "file_crc_ok": bool(file_crc_ok),
        "file_match": source_match if source_path else "n/a",
        "source_path": str(source_path) if source_path else None,
        "recovered_name": recovered_name if write_named else recovered_name + ".partial",
        "out_path": str(target),
        "file_size": result.file_size,
        "error": "" if result.ok else diag.get("reason", result.stage),
        "capture": diag.get("capture"),
        "lock": diag.get("lock"),
        "end": diag.get("end"),
    }
    if parsed is not None and not header_ok:
        metrics["header_raw_guess"] = {
            "filename": parsed["filename"],
            "file_size": parsed["file_size"],
        }
    (out / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    art["metrics"] = metrics
    result.artifacts = art
    return metrics


def format_console_line(metrics, wav_name=""):
    """一行摘要：先报恢复文件，避免 file_match=n/a 被看成失败。"""
    name = wav_name or Path(str(metrics.get("input") or "")).name or "rx"
    sync = metrics.get("sync_score")
    ppm = metrics.get("clock_error_ppm")
    coded = metrics.get("coded_bit_error_rate")
    blocks_ok = metrics.get("blocks_ok")
    blocks_total = metrics.get("blocks_total")
    ber = f"{coded:.6%}" if isinstance(coded, float) else "n/a"
    ppm_s = f"{ppm:+.4f}" if isinstance(ppm, (int, float)) and ppm is not None else "n/a"
    sync_s = f"{sync:.6f}" if isinstance(sync, (int, float)) else "n/a"
    match = metrics.get("file_match")
    if match is True:
        match_s = "True"
    elif match is False:
        match_s = "False"
    else:
        match_s = "n/a"
    recovered = metrics.get("recovered_name") or "-"
    return (
        f"{name}: recovered={recovered} header_ok={metrics.get('header_ok')} "
        f"ldpc={blocks_ok}/{blocks_total} complete={metrics.get('payload_complete')} "
        f"file_match={match_s} ppm={ppm_s} sync={sync_s} coded_BER={ber}"
    )
