from pathlib import Path
import argparse
import csv
import json
import zlib

import matplotlib.pyplot as plt
import numpy as np

from step7_modem import (
    ACTIVE_BINS,
    BLOCK_SIZE,
    FEC_SEED,
    FS,
    HEADER_REPEATS,
    HEADER_SIZE,
    L,
    N,
    PILOT_SEED,
    bytes_from_bits,
    coded_frames,
    deinterleave,
    encoded_length,
    estimate_training,
    find_sync,
    noise_variance,
    ofdm_rx,
    ofdm_tx,
    parse_data_block,
    parse_header,
    payload_llrs,
    profile_meta,
    read_wav,
    training_symbols,
    viterbi_decode,
)


def args():
    p = argparse.ArgumentParser(description="recover a Step7 adaptive-pilot, soft-FEC recording")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument("--source", type=Path, default=Path("data/step6_newexp/file.jpeg"))
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
    p.add_argument("--clock-anchor-symbols", type=int, default=8)
    p.add_argument("--clock-anchor-step", type=int, default=32)
    p.add_argument("--clock-search", type=int, default=128)
    p.add_argument("--payload-search", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("runs/step7_adaptive_fec"))
    return p.parse_args()


def validate(a):
    for name in ("sync_symbols", "preamble_symbols", "preamble_repeats", "header_repeats"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if not 1 <= a.sync_correlation_symbols <= a.sync_symbols:
        raise SystemExit("--sync-correlation-symbols must be between 1 and --sync-symbols")
    if not 0 <= a.channel_alpha <= 1:
        raise SystemExit("--channel-alpha must be between 0 and 1")
    if a.payload_search < 0:
        raise SystemExit("--payload-search must be >= 0")
    if a.clock_anchor_symbols < 2 or a.clock_anchor_step < 1 or a.clock_search < 1:
        raise SystemExit("clock tracking parameters must be positive")


def output_dir(base, receive, many):
    if not many:
        return base
    suffix = receive.stem.rsplit("_", 1)[-1]
    return base / (suffix if suffix.isdigit() else receive.stem)


def find_sync_near(rx, template, expected, radius):
    lo = max(0, int(round(expected)) - radius)
    hi = min(len(rx) - len(template), int(round(expected)) + radius)
    if hi < lo:
        raise ValueError("recording is too short for timing anchor")
    local, score = find_sync(rx[lo : hi + len(template)], template)
    return lo + local, score


def estimate_clock(rx, training, first_start, anchor_symbols, anchor_step, radius):
    anchors = []
    for symbol in range(0, len(training) - anchor_symbols + 1, anchor_step):
        template = ofdm_tx(training[symbol : symbol + anchor_symbols])
        expected = first_start + symbol * L
        start, score = find_sync_near(rx, template, expected, radius)
        anchors.append((symbol, start, score))
    if len(anchors) < 3:
        raise ValueError("not enough clock anchors in training sequence")
    nominal = np.asarray([symbol * L for symbol, _, _ in anchors], dtype=float)
    observed = np.asarray([start for _, start, _ in anchors], dtype=float)
    weights = np.maximum(np.asarray([score for _, _, score in anchors], dtype=float), 1e-3) ** 2
    design = np.c_[np.ones(len(nominal)), nominal]
    lhs = design.T @ (weights[:, None] * design)
    rhs = design.T @ (weights * observed)
    intercept, scale = np.linalg.solve(lhs, rhs)
    if not 0.995 <= scale <= 1.005:
        raise ValueError(f"implausible sample-clock scale: {scale:.8f}")
    return float(intercept), float(scale), np.asarray(anchors, dtype=float)


def correct_clock(rx, intercept, scale):
    length = int(np.floor((len(rx) - intercept - 1) / scale))
    if length <= 0:
        raise ValueError("recording ends before synchronized signal")
    source = intercept + np.arange(length) * scale
    return np.interp(source, np.arange(len(rx)), rx)


def choose_payload_start(rx, expected, h, training_var, silence_var, a):
    best = (np.inf, expected, 0)
    for delta in range(-a.payload_search, a.payload_search + 1):
        start = expected + delta
        if start < 0:
            continue
        y = ofdm_rx(rx[start : start + 24 * L])
        if len(y) < 8:
            continue
        track = payload_llrs(
            y,
            h,
            training_var,
            silence_var,
            a.pilot_seed,
            alpha=a.channel_alpha,
        )
        signal_power = max(float(np.median(np.abs(h) ** 2)), 1e-12)
        score = float(np.median(track["pilot_residual"]) / signal_power)
        if score < best[0]:
            best = (score, start, delta)
    return best[1], best[2], best[0]


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
        if len(part) < coded_len:
            rows.append({"block": index, "crc_ok": False, "error": "recording truncated"})
            chunks.append(None)
            continue
        deinterleaved = deinterleave(part, fec_seed + 1000 + index)
        raw = bytes_from_bits(viterbi_decode(deinterleaved, raw_bytes * 8))
        expected_size = min(block_size, header["file_size"] - index * block_size)
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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "freq_hz", "abs_h", "training_noise", "silence_noise", "mean_abs_llr", "llr_count"])
        for i, k in enumerate(ACTIVE_BINS):
            values = np.abs(llr[llr_bins == k])
            w.writerow(
                [
                    int(k),
                    float(k * FS / N),
                    float(abs(h[i])),
                    float(training_var[i]),
                    float(silence_var[i]),
                    float(np.mean(values)) if len(values) else 0.0,
                    int(len(values)),
                ]
            )


def save_plots(out, h, training_var, silence_var, track, consumed):
    freq = ACTIVE_BINS * FS / N
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(freq, np.abs(h), marker=".")
    ax[0].set_ylabel("|H|")
    ax[0].grid(alpha=0.25)
    ax[1].semilogy(freq, np.maximum(training_var, 1e-15), label="preamble residual")
    ax[1].semilogy(freq, np.maximum(silence_var, 1e-15), label="noise-only")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Noise power")
    ax[1].grid(alpha=0.25)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "channel_and_noise.png", dpi=150)
    plt.close(fig)

    symbols = np.arange(len(track["cpe"]))
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(symbols, np.unwrap(track["cpe"]))
    ax[0].set_ylabel("CPE (rad)")
    ax[1].plot(symbols, track["phase_slope"])
    ax[1].set_ylabel("Slope (rad/bin)")
    ax[2].semilogy(symbols, np.maximum(track["pilot_residual"], 1e-15))
    ax[2].set_ylabel("Pilot residual")
    ax[2].set_xlabel("Payload OFDM symbol")
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Step7 tracking, consumed coded bits={consumed}")
    fig.tight_layout()
    fig.savefig(out / "phase_tracking.png", dpi=150)
    plt.close(fig)


def source_coded_ber(llr, source, block_size, header_repeats, fec_seed):
    frames, _ = coded_frames(source, block_size, header_repeats, fec_seed)
    truth = np.concatenate(frames)
    n = min(len(llr), len(truth))
    errors = int(np.count_nonzero((llr[:n] < 0).astype(np.uint8) != truth[:n]))
    errors += max(0, len(truth) - n)
    return errors, len(truth), errors / len(truth) if len(truth) else 0.0


def run_one(receive, out, a, sync, preamble_blocks):
    rx = read_wav(receive)
    sync_wave = ofdm_tx(sync)
    sync_template = ofdm_tx(sync[: a.sync_correlation_symbols])
    coarse_sync_start, sync_score = find_sync(rx, sync_template)
    training = np.vstack([sync, *preamble_blocks])
    clock_intercept, clock_scale, clock_anchors = estimate_clock(
        rx,
        training,
        coarse_sync_start,
        a.clock_anchor_symbols,
        a.clock_anchor_step,
        a.clock_search,
    )
    corrected = correct_clock(rx, clock_intercept, clock_scale)
    sync_start = 0
    sync_end = len(sync_wave)

    noise_samples = int(round(a.noise_seconds * FS))
    noise_end = max(0, int(round(clock_intercept)))
    noise_start = max(0, noise_end - noise_samples)
    silence_var = noise_variance(rx[noise_start:noise_end])

    preamble_total = sum(map(len, preamble_blocks))
    preamble_end = sync_end + preamble_total * L
    sync_y = ofdm_rx(corrected[sync_start:sync_end])[: len(sync)]
    preamble_y = ofdm_rx(corrected[sync_end:preamble_end])[:preamble_total]
    y_blocks = [sync_y]
    x_blocks = [sync]
    pos = 0
    for block in preamble_blocks:
        y_blocks.append(preamble_y[pos : pos + len(block)])
        x_blocks.append(block)
        pos += len(block)
    if any(len(y) != len(x) for y, x in zip(y_blocks, x_blocks)):
        raise ValueError("recording ends inside Step7 preamble")
    h, h_blocks, training_var = estimate_training(y_blocks, x_blocks)

    payload_start, payload_delta, alignment_score = choose_payload_start(
        corrected, preamble_end, h, training_var, silence_var, a
    )
    payload_y = ofdm_rx(corrected[payload_start:])
    track = payload_llrs(
        payload_y,
        h,
        training_var,
        silence_var,
        a.pilot_seed,
        alpha=a.channel_alpha,
    )
    llr = track["llr"]

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "H.npy", h)
    np.save(out / "H_training_blocks.npy", h_blocks)
    np.save(out / "training_noise.npy", training_var)
    np.save(out / "silence_noise.npy", silence_var)
    np.save(out / "rx_llr.npy", llr)
    np.save(out / "pilot_H_track.npy", track["H"])
    np.save(out / "cpe.npy", track["cpe"])
    np.save(out / "phase_slope.npy", track["phase_slope"])
    np.save(out / "clock_anchors.npy", clock_anchors)

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
        body = b"".join(chunk if chunk is not None else b"\0" * min(
            header["block_size"], header["file_size"] - i * header["block_size"]
        ) for i, chunk in enumerate(chunks))[: header["file_size"]]
        file_crc_ok = complete and zlib.crc32(body) == header["file_crc32"]
        target = out / (recovered_name if file_crc_ok else recovered_name + ".partial")
        target.write_bytes(body)
        file_match = bool(source_bytes is not None and file_crc_ok and body == source_bytes)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    (out / "decoded_header.bin").write_bytes(header_raw)

    with (out / "blocks.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["block", "crc_ok", "bytes", "error"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(block_rows)

    coded_errors = coded_total = 0
    coded_ber = None
    if source is not None:
        block_size = header["block_size"] if header else BLOCK_SIZE
        coded_errors, coded_total, coded_ber = source_coded_ber(
            llr, source, block_size, a.header_repeats, a.fec_seed
        )
    used = min(consumed, len(llr))
    save_bin_summary(
        out / "summary.csv",
        h,
        training_var,
        silence_var,
        llr[:used],
        track["bins"][:used],
    )
    save_plots(out, h, training_var, silence_var, track, used)

    metrics = {
        "input": str(receive),
        "out": str(out),
        "coarse_sync_start": int(coarse_sync_start),
        "sync_start": float(clock_intercept),
        "sync_score": float(sync_score),
        "sync_correlation_symbols": int(a.sync_correlation_symbols),
        "clock_scale": float(clock_scale),
        "clock_error_ppm": float((clock_scale - 1.0) * 1e6),
        "clock_anchor_count": int(len(clock_anchors)),
        "clock_anchor_mean_score": float(np.mean(clock_anchors[:, 2])),
        "noise_samples_used": int(noise_end - noise_start),
        "payload_start_corrected": int(payload_start),
        "payload_start": float(clock_intercept + payload_start * clock_scale),
        "structural_payload_start": int(preamble_end),
        "payload_delta": int(payload_delta),
        "payload_alignment_score": float(alignment_score),
        "payload_symbols_received": int(len(payload_y)),
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
        "error": error,
    }
    metrics.update(profile_meta())
    metrics["header_repeats"] = int(a.header_repeats)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"{receive}: sync={sync_score:.6f} delta={payload_delta} "
        f"coded_BER={coded_ber if coded_ber is not None else float('nan'):.6%} "
        f"header_ok={metrics['header_ok']} blocks={metrics['blocks_ok']}/{metrics['blocks_total']} "
        f"file_match={file_match}"
    )
    if post_total:
        print(f"post_FEC_BER={post_errors / post_total:.6%} file_crc_ok={file_crc_ok}")
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
            "payload_delta",
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
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"wrote {a.out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
