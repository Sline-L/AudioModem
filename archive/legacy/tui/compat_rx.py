from pathlib import Path
import argparse
import csv
import json
import zlib

import matplotlib.pyplot as plt
import numpy as np

from .legacy_step8 import (
    ACTIVE_BINS,
    BLOCK_SIZE,
    FEC_SEED,
    FS,
    HEADER_REPEATS,
    HEADER_SIZE,
    L,
    N,
    PILOT_SEED,
    PAYLOAD_START_ANCHOR_SEED,
    PAYLOAD_START_ANCHOR_SYMBOLS,
    TIMING_ANCHOR_INTERVAL,
    TIMING_ANCHOR_SEED,
    TIMING_ANCHOR_SYMBOLS,
    bytes_from_bits,
    coded_frames,
    correct_clock,
    deinterleave,
    detect_clock_anchors,
    encoded_length,
    estimate_training,
    estimate_anchor_channel,
    find_sync,
    noise_variance,
    ofdm_rx,
    ofdm_tx,
    parse_data_block,
    parse_header,
    payload_llrs_anchored,
    payload_start_anchor_symbols,
    correlate_near,
    profile_meta,
    read_wav,
    training_symbols,
    viterbi_decode,
)

TUI_ROOT = Path(__file__).resolve().parent


def args():
    p = argparse.ArgumentParser(description="recover a Step8 periodic-clock-anchor recording")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument(
        "--source",
        type=Path,
        default=None,
    )
    p.add_argument("--noise-seconds", type=float, default=0.5)
    p.add_argument("--sync-symbols", type=int, default=64)
    p.add_argument("--sync-correlation-symbols", type=int, default=8)
    p.add_argument("--sync-seed", type=int, default=7026)
    p.add_argument("--preamble-symbols", type=int, default=128)
    p.add_argument("--preamble-repeats", type=int, default=2)
    p.add_argument("--preamble-seed", type=int, default=8026)
    p.add_argument("--header-repeats", type=int, default=HEADER_REPEATS)
    p.add_argument("--pilot-seed", type=int, default=PILOT_SEED)
    p.add_argument("--fec-seed", type=int, default=FEC_SEED)
    p.add_argument("--channel-alpha", type=float, default=0.35)
    p.add_argument("--timing-anchor-interval", type=int, default=TIMING_ANCHOR_INTERVAL)
    p.add_argument("--timing-anchor-symbols", type=int, default=TIMING_ANCHOR_SYMBOLS)
    p.add_argument("--timing-anchor-seed", type=int, default=TIMING_ANCHOR_SEED)
    p.add_argument("--payload-start-anchor-symbols", type=int, default=PAYLOAD_START_ANCHOR_SYMBOLS)
    p.add_argument("--payload-start-anchor-seed", type=int, default=PAYLOAD_START_ANCHOR_SEED)
    p.add_argument("--payload-start-anchor-h-alpha", type=float, default=0.5)
    p.add_argument("--anchor-h-alpha", type=float, default=0.5)
    p.add_argument("--anchor-min-score", type=float, default=0.12)
    p.add_argument("--training-anchor-symbols", type=int, default=8)
    p.add_argument("--training-anchor-step", type=int, default=32)
    p.add_argument("--clock-search", type=int, default=128)
    p.add_argument("--payload-search", type=int, default=16)
    p.add_argument("--phase-slope", choices=("off", "slow"), default="off")
    p.add_argument("--slope-window", type=int, default=64)
    p.add_argument("--slope-clip", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=TUI_ROOT / "run" / "step8_compatible")
    return p.parse_args()


def validate(a):
    positive = (
        "sync_symbols",
        "preamble_symbols",
        "preamble_repeats",
        "header_repeats",
        "timing_anchor_interval",
        "timing_anchor_symbols",
        "training_anchor_symbols",
        "training_anchor_step",
        "clock_search",
        "slope_window",
    )
    for name in positive:
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if not 1 <= a.sync_correlation_symbols <= a.sync_symbols:
        raise SystemExit("--sync-correlation-symbols must be between 1 and --sync-symbols")
    if a.training_anchor_symbols > a.sync_symbols + a.preamble_symbols * a.preamble_repeats:
        raise SystemExit("--training-anchor-symbols exceeds the complete training sequence")
    if a.payload_start_anchor_symbols < 0:
        raise SystemExit("--payload-start-anchor-symbols must be >= 0")
    for name in ("channel_alpha", "anchor_h_alpha", "payload_start_anchor_h_alpha"):
        if not 0 <= getattr(a, name) <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not 0 <= a.anchor_min_score <= 1:
        raise SystemExit("--anchor-min-score must be between 0 and 1")
    if a.payload_search < 0 or a.slope_clip < 0:
        raise SystemExit("search radius and slope clip must be non-negative")


def output_dir(base, receive, many):
    if not many:
        return base
    suffix = receive.stem.rsplit("_", 1)[-1]
    return base / (suffix if suffix.isdigit() else receive.stem)


def decode_header(llr, repeats, fec_seed):
    coded_len = encoded_length(HEADER_SIZE)
    need = repeats * coded_len
    if len(llr) < need:
        raise ValueError(f"recording ends inside header: need {need} LLRs, got {len(llr)}")
    combined = np.zeros(coded_len)
    for repeat in range(repeats):
        part = llr[repeat * coded_len : (repeat + 1) * coded_len]
        combined += deinterleave(part, fec_seed + repeat)
    bits = viterbi_decode(combined, HEADER_SIZE * 8)
    raw = bytes_from_bits(bits)
    return parse_header(raw), raw, need


def decode_blocks(llr, pos, header, fec_seed, source_bytes=None):
    block_size = header["block_size"]
    raw_bytes = block_size + 8
    coded_len = encoded_length(raw_bytes)
    chunks = []
    rows = []
    post_errors = 0
    post_total = 0
    for index in range(header["block_count"]):
        start = pos + index * coded_len
        part = llr[start : start + coded_len]
        expected_size = min(block_size, header["file_size"] - index * block_size)
        if len(part) < coded_len:
            rows.append({"block": index, "crc_ok": False, "bytes": expected_size, "error": "recording truncated"})
            chunks.append(None)
            continue
        deinterleaved = deinterleave(part, fec_seed + 1000 + index)
        raw = bytes_from_bits(viterbi_decode(deinterleaved, raw_bytes * 8))
        if source_bytes is not None:
            got = raw[4 : 4 + expected_size]
            truth = source_bytes[index * block_size : index * block_size + expected_size]
            xor = np.frombuffer(got, dtype=np.uint8) ^ np.frombuffer(truth, dtype=np.uint8)
            post_errors += int(np.unpackbits(xor, bitorder="big").sum())
            post_total += 8 * expected_size
        try:
            chunk = parse_data_block(raw, index, block_size)
            chunks.append(chunk)
            rows.append({"block": index, "crc_ok": True, "bytes": len(chunk), "error": ""})
        except Exception as exc:
            chunks.append(None)
            rows.append({"block": index, "crc_ok": False, "bytes": expected_size, "error": str(exc)})
    return chunks, rows, pos + header["block_count"] * coded_len, post_errors, post_total


def save_bin_summary(path, h, training_var, silence_var, llr, llr_bins):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "freq_hz", "abs_h", "training_noise", "silence_noise", "mean_abs_llr", "llr_count"])
        for index, bin_index in enumerate(ACTIVE_BINS):
            values = np.abs(llr[llr_bins == bin_index])
            writer.writerow(
                [
                    int(bin_index),
                    float(bin_index * FS / N),
                    float(abs(h[index])),
                    float(training_var[index]),
                    float(silence_var[index]),
                    float(np.mean(values)) if len(values) else 0.0,
                    int(len(values)),
                ]
            )


def source_coded_ber(llr, source, block_size, header_repeats, fec_seed):
    frames, _ = coded_frames(source, block_size, header_repeats, fec_seed)
    truth = np.concatenate(frames)
    count = min(len(llr), len(truth))
    errors = int(np.count_nonzero((llr[:count] < 0).astype(np.uint8) != truth[:count]))
    errors += max(0, len(truth) - count)
    return errors, len(truth), errors / len(truth) if len(truth) else 0.0


def choose_payload_start(rx, expected, h, training_var, silence_var, a):
    best = (np.inf, expected, 0)
    for delta in range(-a.payload_search, a.payload_search + 1):
        start = expected + delta
        if start < 0:
            continue
        y = ofdm_rx(rx[start : start + 24 * L])
        if len(y) < 8:
            continue
        track = payload_llrs_anchored(
            y,
            h,
            training_var,
            silence_var,
            a.pilot_seed,
            channel_alpha=a.channel_alpha,
            interval=a.timing_anchor_interval,
            anchor_rows=a.timing_anchor_symbols,
            anchor_seed=a.timing_anchor_seed,
            anchor_h_alpha=0.0,
            phase_slope=a.phase_slope,
            slope_window=a.slope_window,
            slope_clip=a.slope_clip,
        )
        signal_power = max(float(np.median(np.abs(h) ** 2)), 1e-12)
        score = float(np.median(track["pilot_residual"]) / signal_power)
        if score < best[0]:
            best = (score, start, delta)
    return best[1], best[2], best[0]


def save_clock_plot(path, table, intercept, scale):
    nominal = table[:, 0]
    observed = table[:, 1]
    score = table[:, 2]
    residual = table[:, 3]
    used = table[:, 4].astype(bool)
    payload = table[:, 5].astype(bool)
    seconds = nominal / FS
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].scatter(seconds[~payload], observed[~payload] - nominal[~payload], c="tab:blue", label="training")
    ax[0].scatter(seconds[payload], observed[payload] - nominal[payload], c=score[payload], cmap="viridis", label="payload")
    ax[0].plot(seconds, intercept + (scale - 1.0) * nominal, color="black", linewidth=1.2, label="clock fit")
    ax[0].set_ylabel("Observed - nominal (samples)")
    ax[0].legend()
    ax[1].scatter(seconds[~used], residual[~used], marker="x", color="tab:red", label="rejected")
    ax[1].scatter(seconds[used], residual[used], c=score[used], cmap="viridis", label="used")
    ax[1].axhline(0.0, color="black", linewidth=1)
    ax[1].set_xlabel("Nominal time from sync (s)")
    ax[1].set_ylabel("Fit residual (samples)")
    ax[1].legend()
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_tracking_plots(out, h, training_var, silence_var, track, consumed):
    freq = ACTIVE_BINS * FS / N
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(freq, np.abs(h), marker=".")
    ax[0].set_ylabel("|H|")
    ax[1].semilogy(freq, np.maximum(training_var, 1e-15), label="preamble residual")
    ax[1].semilogy(freq, np.maximum(silence_var, 1e-15), label="noise-only")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Noise power")
    ax[1].legend()
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "channel_and_noise.png", dpi=150)
    plt.close(fig)

    symbols = np.arange(len(track["cpe"]))
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(symbols, np.unwrap(track["cpe"]))
    ax[0].set_ylabel("CPE (rad)")
    ax[1].plot(symbols, track["phase_slope_raw"], alpha=0.4, label="raw")
    ax[1].plot(symbols, track["phase_slope"], label="applied")
    ax[1].set_ylabel("Slope (rad/bin)")
    ax[1].legend()
    ax[2].semilogy(symbols, np.maximum(track["pilot_residual"], 1e-15))
    ax[2].set_ylabel("Pilot residual")
    ax[2].set_xlabel("Logical payload OFDM symbol")
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Step8 tracking, consumed coded bits={consumed}")
    fig.tight_layout()
    fig.savefig(out / "phase_tracking.png", dpi=150)
    plt.close(fig)


def run_one(receive, out, a, sync, preamble_blocks):
    rx = read_wav(receive)
    sync_template = ofdm_tx(sync[: a.sync_correlation_symbols])
    coarse_sync_start, sync_score = find_sync(rx, sync_template)
    training = np.vstack([sync, *preamble_blocks])
    clock_intercept, clock_scale, clock_table, clock_status, initial_clock = detect_clock_anchors(
        rx,
        training,
        coarse_sync_start,
        a.training_anchor_symbols,
        a.training_anchor_step,
        a.timing_anchor_interval,
        a.timing_anchor_symbols,
        a.timing_anchor_seed,
        a.payload_start_anchor_symbols,
        a.payload_start_anchor_seed,
        a.clock_search,
        a.anchor_min_score,
    )
    corrected = correct_clock(rx, clock_intercept, clock_scale)

    noise_samples = int(round(a.noise_seconds * FS))
    noise_end = max(0, int(round(clock_intercept)))
    noise_start = max(0, noise_end - noise_samples)
    silence_var = noise_variance(rx[noise_start:noise_end])

    sync_end = len(sync) * L
    preamble_total = sum(map(len, preamble_blocks))
    preamble_end = sync_end + preamble_total * L
    sync_y = ofdm_rx(corrected[:sync_end])[: len(sync)]
    preamble_y = ofdm_rx(corrected[sync_end:preamble_end])[:preamble_total]
    y_blocks = [sync_y]
    x_blocks = [sync]
    pos = 0
    for block in preamble_blocks:
        y_blocks.append(preamble_y[pos : pos + len(block)])
        x_blocks.append(block)
        pos += len(block)
    if any(len(y) != len(x) for y, x in zip(y_blocks, x_blocks)):
        raise ValueError("recording ends inside Step8 preamble")
    h, h_blocks, training_var = estimate_training(y_blocks, x_blocks)
    h_training = h.copy()

    start_anchor_h = np.empty((0, len(ACTIVE_BINS)), complex)
    start_anchor_score = None
    start_anchor_start = None
    start_anchor_status = "disabled"
    if a.payload_start_anchor_symbols:
        start_known = payload_start_anchor_symbols(
            a.payload_start_anchor_symbols, a.payload_start_anchor_seed
        )
        start_template = ofdm_tx(start_known)
        start_anchor_position, start_anchor_score = correlate_near(
            corrected, start_template, preamble_end, a.payload_search
        )
        if start_anchor_score >= a.anchor_min_score:
            start_anchor_start = int(round(start_anchor_position))
            start_anchor_status = "detected"
        else:
            start_anchor_start = preamble_end
            start_anchor_status = "structural_fallback"
        start_y = ofdm_rx(
            corrected[
                start_anchor_start : start_anchor_start + a.payload_start_anchor_symbols * L
            ]
        )
        start_h = estimate_anchor_channel(start_y, start_known, h)
        h = (
            (1.0 - a.payload_start_anchor_h_alpha) * h
            + a.payload_start_anchor_h_alpha * start_h
        )
        start_anchor_h = start_h[None, :]
        payload_start = start_anchor_start + a.payload_start_anchor_symbols * L
        payload_delta = start_anchor_start - preamble_end
        alignment_score = 1.0 - start_anchor_score
    else:
        payload_start, payload_delta, alignment_score = choose_payload_start(
            corrected, preamble_end, h, training_var, silence_var, a
        )
    payload_y = ofdm_rx(corrected[payload_start:])
    track = payload_llrs_anchored(
        payload_y,
        h,
        training_var,
        silence_var,
        a.pilot_seed,
        channel_alpha=a.channel_alpha,
        interval=a.timing_anchor_interval,
        anchor_rows=a.timing_anchor_symbols,
        anchor_seed=a.timing_anchor_seed,
        anchor_h_alpha=a.anchor_h_alpha,
        phase_slope=a.phase_slope,
        slope_window=a.slope_window,
        slope_clip=a.slope_clip,
    )
    llr = track["llr"]

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "H.npy", h_training)
    np.save(out / "H_payload_start_anchor.npy", start_anchor_h)
    np.save(out / "H_training_blocks.npy", h_blocks)
    np.save(out / "H_anchor_track.npy", track["anchor_H"])
    np.save(out / "training_noise.npy", training_var)
    np.save(out / "silence_noise.npy", silence_var)
    np.save(out / "rx_llr.npy", llr)
    np.save(out / "pilot_H_track.npy", track["H"])
    np.save(out / "cpe.npy", track["cpe"])
    np.save(out / "phase_slope.npy", track["phase_slope"])
    np.save(out / "phase_slope_raw.npy", track["phase_slope_raw"])
    np.save(out / "clock_anchors.npy", clock_table)
    save_clock_plot(out / "clock_fit.png", clock_table, clock_intercept, clock_scale)

    header = None
    header_raw = b""
    chunks = []
    block_rows = []
    consumed = 0
    post_errors = 0
    post_total = 0
    error = ""
    file_match = False
    file_crc_ok = False
    recovered_name = ""
    source = a.source if a.source and a.source.exists() else None
    source_bytes = source.read_bytes() if source else None
    try:
        header, header_raw, consumed = decode_header(llr, a.header_repeats, a.fec_seed)
        recovered_name = header["name"]
        chunks, block_rows, consumed, post_errors, post_total = decode_blocks(
            llr, consumed, header, a.fec_seed, source_bytes
        )
        complete = all(chunk is not None for chunk in chunks)
        body = b"".join(
            chunk
            if chunk is not None
            else b"\0" * min(header["block_size"], header["file_size"] - i * header["block_size"])
            for i, chunk in enumerate(chunks)
        )[: header["file_size"]]
        file_crc_ok = complete and zlib.crc32(body) == header["file_crc32"]
        target = out / (recovered_name if file_crc_ok else recovered_name + ".partial")
        target.write_bytes(body)
        file_match = bool(source_bytes is not None and file_crc_ok and body == source_bytes)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    (out / "decoded_header.bin").write_bytes(header_raw)

    with (out / "blocks.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["block", "crc_ok", "bytes", "error"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(block_rows)

    coded_errors = coded_total = 0
    coded_ber = None
    if source is not None:
        block_size = header["block_size"] if header else BLOCK_SIZE
        coded_errors, coded_total, coded_ber = source_coded_ber(
            llr, source, block_size, a.header_repeats, a.fec_seed
        )
    used = min(consumed, len(llr))
    save_bin_summary(out / "summary.csv", h, training_var, silence_var, llr[:used], track["bins"][:used])
    save_tracking_plots(out, h, training_var, silence_var, track, used)

    accepted = clock_table[:, 4].astype(bool)
    payload_rows = clock_table[:, 5] == 1
    fit_residual = clock_table[accepted, 3]
    metrics = {
        "input": str(receive),
        "out": str(out),
        "coarse_sync_start": int(coarse_sync_start),
        "sync_start": float(clock_intercept),
        "sync_score": float(sync_score),
        "clock_status": clock_status,
        "initial_clock_scale": float(initial_clock["scale"]),
        "initial_clock_error_ppm": float((initial_clock["scale"] - 1.0) * 1e6),
        "clock_scale": float(clock_scale),
        "clock_error_ppm": float((clock_scale - 1.0) * 1e6),
        "clock_anchor_candidates": int(len(clock_table)),
        "clock_anchor_used": int(np.count_nonzero(accepted)),
        "clock_anchor_rejected": int(len(clock_table) - np.count_nonzero(accepted)),
        "payload_anchor_candidates": int(np.count_nonzero(payload_rows)),
        "payload_anchor_used": int(np.count_nonzero(accepted & payload_rows)),
        "clock_anchor_mean_score": float(np.mean(clock_table[accepted, 2])),
        "clock_fit_residual_rms": float(np.sqrt(np.mean(fit_residual**2))) if len(fit_residual) else None,
        "clock_fit_residual_max_abs": float(np.max(np.abs(fit_residual))) if len(fit_residual) else None,
        "noise_samples_used": int(noise_end - noise_start),
        "payload_start_corrected": int(payload_start),
        "payload_start": float(clock_intercept + payload_start * clock_scale),
        "structural_payload_start": int(preamble_end),
        "payload_start_anchor_start_corrected": start_anchor_start,
        "payload_start_anchor_score": start_anchor_score,
        "payload_start_anchor_status": start_anchor_status,
        "payload_start_anchor_symbols": int(a.payload_start_anchor_symbols),
        "payload_start_anchor_seed": int(a.payload_start_anchor_seed),
        "payload_start_anchor_h_alpha": float(a.payload_start_anchor_h_alpha),
        "payload_delta": int(payload_delta),
        "payload_alignment_score": float(alignment_score),
        "payload_physical_symbols_received": int(len(payload_y)),
        "payload_logical_symbols_processed": int(track["logical_symbols"]),
        "payload_anchors_consumed": int(track["anchors_consumed"]),
        "llr_count": int(len(llr)),
        "coded_bits_consumed": int(consumed),
        "coded_bit_errors": int(coded_errors),
        "coded_bit_total": int(coded_total),
        "coded_bit_error_rate": coded_ber,
        "post_fec_bit_errors": int(post_errors),
        "post_fec_bit_total": int(post_total),
        "post_fec_bit_error_rate": float(post_errors / post_total) if post_total else None,
        "header_ok": header is not None,
        "header": header,
        "blocks_ok": int(sum(bool(row.get("crc_ok")) for row in block_rows)),
        "blocks_total": int(len(block_rows)),
        "file_crc_ok": bool(file_crc_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "mean_abs_h": float(np.mean(np.abs(h))),
        "median_training_noise": float(np.median(training_var)),
        "median_silence_noise": float(np.median(silence_var)),
        "median_abs_llr": float(np.median(np.abs(llr[:used]))) if used else None,
        "median_pilot_residual": float(np.median(track["pilot_residual"])) if len(track["pilot_residual"]) else None,
        "channel_alpha": float(a.channel_alpha),
        "anchor_h_alpha": float(a.anchor_h_alpha),
        "phase_slope_mode": a.phase_slope,
        "slope_window": int(a.slope_window),
        "slope_clip": float(a.slope_clip),
        "timing_anchor_interval": int(a.timing_anchor_interval),
        "timing_anchor_symbols": int(a.timing_anchor_symbols),
        "timing_anchor_seed": int(a.timing_anchor_seed),
        "error": error,
    }
    metrics.update(profile_meta())
    metrics.update(
        {
            "clock_status": clock_status,
            "clock_error_ppm": float((clock_scale - 1.0) * 1e6),
            "header_repeats": int(a.header_repeats),
            "timing_anchor_interval": int(a.timing_anchor_interval),
            "timing_anchor_symbols": int(a.timing_anchor_symbols),
            "timing_anchor_seed": int(a.timing_anchor_seed),
            "payload_start_anchor_symbols": int(a.payload_start_anchor_symbols),
            "payload_start_anchor_seed": int(a.payload_start_anchor_seed),
            "payload_start_anchor_h_alpha": float(a.payload_start_anchor_h_alpha),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"{receive}: sync={sync_score:.6f} clock={clock_status} "
        f"ppm={metrics['clock_error_ppm']:+.4f} initial_ppm={metrics['initial_clock_error_ppm']:+.4f} "
        f"coded_BER={coded_ber if coded_ber is not None else float('nan'):.6%} "
        f"blocks={metrics['blocks_ok']}/{metrics['blocks_total']} file_match={file_match}"
    )
    if post_total:
        print(f"post_FEC_BER={post_errors / post_total:.6%} file_crc_ok={file_crc_ok}")
    if clock_status != "full":
        print("warning=payload timing anchors insufficient; using training-only clock estimate")
    if error:
        print(f"decode_error={error}")
    return metrics


def main():
    a = args()
    validate(a)
    sync = training_symbols(a.sync_symbols, a.sync_seed)
    preamble_blocks = [
        training_symbols(a.preamble_symbols, a.preamble_seed + i) for i in range(a.preamble_repeats)
    ]
    many = len(a.input) > 1
    results = []
    for receive in a.input:
        results.append(run_one(receive, output_dir(a.out, receive, many), a, sync, preamble_blocks))
    if many:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = [
            "input",
            "sync_score",
            "clock_status",
            "initial_clock_error_ppm",
            "clock_error_ppm",
            "payload_anchor_used",
            "coded_bit_error_rate",
            "post_fec_bit_error_rate",
            "header_ok",
            "blocks_ok",
            "blocks_total",
            "file_crc_ok",
            "file_match",
            "error",
        ]
        with (a.out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {a.out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
