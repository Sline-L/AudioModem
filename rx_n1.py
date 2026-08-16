from pathlib import Path
import argparse
import csv
import json

import numpy as np

from modem_n1 import (
    ACTIVE_BINS,
    CHIRP_GUARD_SAMPLES,
    CHIRP_SAMPLES,
    FS,
    HEADER_MOD,
    HEADER_SIZE,
    HEADER_SYMBOLS,
    HEADER_COPIES,
    HEADER_COPY_SYMBOLS,
    INTER_FRAME_GAP_SAMPLES,
    L,
    N,
    TRAINING_SYMBOLS,
    bytes_from_mod,
    bytes_from_bits,
    chirp_wave,
    decode_payload,
    equalize,
    estimate_channel,
    find_chirp,
    frame_sample_counts,
    ofdm_rx,
    parse_header,
    phase_correct,
    profile_meta,
    read_wav,
    training_phase_fit,
    training_symbols,
    header_copy_bytes,
)


def args():
    p = argparse.ArgumentParser(description="recover an N1 chirp/training/header/payload/tail-chirp recording")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--training-seed", type=int, default=3026)
    p.add_argument("--tail-search-seconds", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=Path("runs/n1"))
    return p.parse_args()


def validate(a):
    if a.tail_search_seconds < 0:
        raise SystemExit("--tail-search-seconds must be >= 0")


def output_dir(base, receive, many):
    if not many:
        return base
    suffix = receive.stem.rsplit("_", 1)[-1]
    return base / (suffix if suffix.isdigit() else receive.stem)


def save_summary(path, h, training_error):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "freq_hz", "abs_h", "training_error"])
        for index, bin_index in enumerate(ACTIVE_BINS):
            writer.writerow(
                [
                    int(bin_index),
                    float(bin_index * FS / N),
                    float(abs(h[index])),
                    float(training_error[index]),
                ]
            )


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def symbol_block(rx, front_start, offset, rows):
    start = int(round(front_start + offset))
    return ofdm_rx(rx[start : start + rows * L])[:rows], start


def block_starts(front_start, offset, rows):
    start = float(front_start + offset)
    return start + np.arange(rows) * L


def decode_header_pass(rx, front_start, training, epsilon=0.0):
    reference = float(front_start)
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    header_offset = training_offset + TRAINING_SYMBOLS * L

    train_y, _ = symbol_block(rx, front_start, training_offset, TRAINING_SYMBOLS)
    train_y = phase_correct(
        train_y,
        block_starts(front_start, training_offset, len(train_y)),
        reference,
        epsilon,
    )
    h = estimate_channel(train_y, training)
    training_error = np.mean(np.abs(train_y - training[: len(train_y)] * h) ** 2, axis=0)
    phase_fit = training_phase_fit(train_y, training, h)

    header_y, header_start = symbol_block(rx, front_start, header_offset, HEADER_SYMBOLS)
    header_y = phase_correct(
        header_y,
        block_starts(front_start, header_offset, len(header_y)),
        reference,
        epsilon,
    )
    header_eq = equalize(header_y, h)
    raw_copies, bit_copies, copy_crc_ok = header_copy_bytes(header_eq)
    votes = np.sum(bit_copies, axis=0)
    voted_bits = (votes >= 2).astype(np.uint8)
    header_raw = bytes_from_bits(voted_bits)[:HEADER_SIZE]
    diagnostics = {
        "copy_crc_ok": copy_crc_ok,
        "bit_disagreements": int(np.count_nonzero((votes != 0) & (votes != HEADER_COPIES))),
        "raw_copies": raw_copies,
    }
    try:
        header = parse_header(header_raw)
    except Exception as exc:
        exc.header_raw = header_raw
        exc.header_diag = diagnostics
        raise
    return header, header_raw, h, training_error, phase_fit, header_start, diagnostics


def find_tail(rx, chirp, front_start, d_tx, search_seconds):
    expected = float(front_start + d_tx)
    radius = int(round(search_seconds * FS))
    start = max(0, int(round(expected)) - radius)
    end = min(len(rx), int(round(expected)) + radius + len(chirp))
    try:
        return find_chirp(rx, chirp, start, end)
    except ValueError:
        return find_chirp(rx, chirp, int(round(front_start + CHIRP_SAMPLES)), None)


def run_one(receive, out, a, training, chirp):
    rx = read_wav(receive)
    front_start, front_score = find_chirp(rx, chirp)

    header = None
    header_raw = b""
    h = np.zeros(len(ACTIVE_BINS), complex)
    training_error = np.full(len(ACTIVE_BINS), np.nan)
    phase_fit = np.empty((0, 4))
    tail_start = None
    tail_score = 0.0
    d_tx = None
    d_rx = None
    epsilon = 0.0
    header_ok = False
    header_diag = {
        "copy_crc_ok": [],
        "bit_disagreements": None,
        "raw_copies": [],
    }
    error = ""

    try:
        header, header_raw, h, training_error, phase_fit, _, header_diag = decode_header_pass(rx, front_start, training)
        counts = frame_sample_counts(header["payload_symbols"])
        d_tx = float(counts["D_tx"])
        tail_start, tail_score = find_tail(rx, chirp, front_start, d_tx, a.tail_search_seconds)
        d_rx = float(tail_start - front_start)
        epsilon = (d_rx - d_tx) / d_tx
        header, header_raw, h, training_error, phase_fit, _, header_diag = decode_header_pass(
            rx, front_start, training, epsilon
        )
        header_ok = True
    except Exception as exc:
        header_raw = getattr(exc, "header_raw", header_raw)
        header_diag = getattr(exc, "header_diag", header_diag)
        error = f"{type(exc).__name__}: {exc}"
        counts = frame_sample_counts(header["payload_symbols"] if header else 0)

    out.mkdir(parents=True, exist_ok=True)
    (out / "decoded_header.bin").write_bytes(header_raw)
    np.save(out / "H.npy", h)
    np.save(out / "training_phase_fit.npy", phase_fit)
    save_summary(out / "summary.csv", h, training_error)

    recovered_name = ""
    recovered_bytes = b""
    file_crc_ok = False
    file_match = False
    payload_count = 0
    payload_eq = np.empty(0, complex)

    if header_ok and not error:
        try:
            payload_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES + (TRAINING_SYMBOLS + HEADER_SYMBOLS) * L
            payload_y, _ = symbol_block(rx, front_start, payload_offset, header["payload_symbols"])
            payload_y = phase_correct(
                payload_y,
                block_starts(front_start, payload_offset, len(payload_y)),
                float(front_start),
                epsilon,
            )
            payload_eq = equalize(payload_y, h).ravel()[: header["payload_mod_symbols"]]
            recovered_bytes = decode_payload(payload_eq, header)
            recovered_name = header["name"]
            (out / recovered_name).write_bytes(recovered_bytes)
            file_crc_ok = True
            if a.source and a.source.exists():
                file_match = recovered_bytes == a.source.read_bytes()
            payload_count = int(len(payload_y))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raw = bytes_from_mod(payload_eq, header["mod"]) if header else b""
            (out / "decoded_payload.bin").write_bytes(raw)
    else:
        (out / "decoded_payload.bin").write_bytes(b"")

    np.save(out / "payload_symbols.npy", payload_eq)
    metrics = {
        "input": str(receive),
        "out": str(out),
        "front_chirp_start": float(front_start),
        "front_chirp_score": float(front_score),
        "tail_chirp_start": float(tail_start) if tail_start is not None else None,
        "tail_chirp_score": float(tail_score),
        "D_tx": d_tx,
        "D_rx": d_rx,
        "sfo_epsilon": float(epsilon),
        "sfo_ppm": float(epsilon * 1e6),
        "training_start": int(round(front_start + CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES)),
        "payload_ofdm_symbols_received": int(payload_count),
        "header_ok": bool(header_ok),
        "header": header,
        "header_copies": int(HEADER_COPIES),
        "header_copy_symbols": int(HEADER_COPY_SYMBOLS),
        "header_size": int(HEADER_SIZE),
        "header_copy_crc_ok": [bool(value) for value in header_diag["copy_crc_ok"]],
        "header_bit_disagreements": header_diag["bit_disagreements"],
        "header_vote_used": True,
        "file_crc_ok": bool(file_crc_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "recovered_bytes": int(len(recovered_bytes)),
        "mean_abs_h": float(np.mean(np.abs(h))) if h.size else None,
        "median_training_error": finite_median(training_error),
        "error": error,
    }
    metrics.update(profile_meta())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"{receive}: front={front_score:.6f} tail={tail_score:.6f} "
        f"sfo_ppm={epsilon * 1e6:+.4f} header_ok={metrics['header_ok']} "
        f"file_crc_ok={file_crc_ok} file_match={file_match}"
    )
    if error:
        print(f"decode_error={error}")
    return metrics


def main():
    a = args()
    validate(a)
    training = training_symbols(a.training_seed)
    chirp = chirp_wave()
    many = len(a.input) > 1
    results = []
    for receive in a.input:
        results.append(run_one(receive, output_dir(a.out, receive, many), a, training, chirp))
    if many:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = [
            "input",
            "front_chirp_score",
            "tail_chirp_score",
            "sfo_ppm",
            "header_ok",
            "file_crc_ok",
            "file_match",
            "error",
        ]
        with (a.out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {a.out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
