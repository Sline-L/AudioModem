from pathlib import Path
import argparse
import csv
import json
import zlib

import numpy as np
from scipy import signal

from modem_n1 import (
    ACTIVE_BINS,
    CHIRP_GUARD_SAMPLES,
    CHIRP_SAMPLES,
    FS,
    HEADER_BITS,
    HEADER_MOD,
    HEADER_SIZE,
    HEADER_SYMBOLS,
    HEADER_COPIES,
    HEADER_COPY_SYMBOLS,
    HEADER_PERM_SEEDS,
    INTER_FRAME_GAP_SAMPLES,
    L,
    N,
    TRAINING_SYMBOLS,
    bits_from_bytes,
    bits_from_mod,
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
    header_bytes,
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
        exc.h = h
        exc.training_error = training_error
        exc.phase_fit = phase_fit
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


def chirp_candidates(rx, chirp, max_peaks=24, min_spacing=None):
    if min_spacing is None:
        min_spacing = max(5000, len(chirp) // 2)
    corr = signal.correlate(rx, chirp, mode="valid", method="fft")
    energy = signal.fftconvolve(rx * rx, np.ones(len(chirp)), mode="valid")
    denom = np.sqrt(np.maximum(energy, 0.0)) * np.linalg.norm(chirp)
    score = np.divide(np.abs(corr), denom, out=np.zeros_like(denom), where=denom > 0)
    selected = []
    for index in np.argsort(score)[::-1]:
        index = int(index)
        if all(abs(index - other[0]) >= min_spacing for other in selected):
            selected.append((index, float(np.clip(score[index], 0.0, 1.0))))
        if len(selected) >= max_peaks:
            break
    return sorted(selected, key=lambda item: item[0])


def find_chirp_pair(rx, chirp):
    candidates = chirp_candidates(rx, chirp)
    fixed_part = fixed_part_samples()
    best = None
    for front_index, (front_start, front_score) in enumerate(candidates):
        for tail_start, tail_score in candidates[front_index + 1 :]:
            d_rx = float(tail_start - front_start)
            payload_symbols = int(round((d_rx - fixed_part) / L))
            if payload_symbols < 0:
                continue
            counts = frame_sample_counts(payload_symbols)
            d_tx = float(counts["D_tx"])
            grid_error = abs(d_rx - d_tx)
            epsilon = (d_rx - d_tx) / d_tx if d_tx else 0.0
            if grid_error > 500:
                continue
            if abs(epsilon * 1e6) > 300:
                continue
            if front_start + d_tx + CHIRP_SAMPLES > len(rx):
                continue
            candidate = {
                "front_start": float(front_start),
                "front_score": float(front_score),
                "tail_start": float(tail_start),
                "tail_score": float(tail_score),
                "payload_symbols": int(payload_symbols),
                "counts": counts,
                "D_rx": float(d_rx),
                "D_tx": float(d_tx),
                "epsilon": float(epsilon),
                "grid_error": float(grid_error),
                "candidates": candidates,
            }
            key = (-(front_score + tail_score), grid_error, front_start)
            if best is None or key < best[0]:
                best = (key, candidate)
    if best:
        return best[1]
    front_start, front_score = find_chirp(rx, chirp)
    return {
        "front_start": float(front_start),
        "front_score": float(front_score),
        "tail_start": None,
        "tail_score": 0.0,
        "payload_symbols": None,
        "counts": None,
        "D_rx": None,
        "D_tx": None,
        "epsilon": 0.0,
        "grid_error": None,
        "candidates": candidates,
    }


def score_failed_header(header_diag):
    disagreements = header_diag.get("bit_disagreements")
    if disagreements is None:
        disagreements = HEADER_COPIES * HEADER_SIZE * 8
    copy_crc_ok = header_diag.get("copy_crc_ok", [])
    crc_bonus = 1000 * sum(bool(value) for value in copy_crc_ok)
    return float(disagreements - crc_bonus)


def search_header_epsilon(rx, front_start, training, ppm_min=-80, ppm_max=80, ppm_step=5):
    best = None
    attempts = []
    scores = []
    for ppm in range(ppm_min, ppm_max + ppm_step, ppm_step):
        epsilon = ppm * 1e-6
        try:
            result = decode_header_pass(rx, front_start, training, epsilon)
            header, header_raw, h, training_error, phase_fit, header_start, header_diag = result
            score = -10000.0 - 1000.0 * sum(bool(value) for value in header_diag["copy_crc_ok"])
            attempt = {
                "ppm": int(ppm),
                "epsilon": float(epsilon),
                "score": float(score),
                "header_ok": True,
                "header": header,
                "header_raw": header_raw,
                "h": h,
                "training_error": training_error,
                "phase_fit": phase_fit,
                "header_start": header_start,
                "header_diag": header_diag,
            }
            attempts.append(attempt)
            scores.append(
                {
                    "ppm": int(ppm),
                    "score": float(score),
                    "header_ok": True,
                    "bit_disagreements": int(header_diag["bit_disagreements"]),
                    "copy_crc_ok": [bool(value) for value in header_diag["copy_crc_ok"]],
                }
            )
            if best is None or (not best["header_ok"], abs(ppm), score) < (
                not best["header_ok"],
                abs(best["ppm"]),
                best["score"],
            ):
                best = attempt
        except Exception as exc:
            header_diag = getattr(exc, "header_diag", {"copy_crc_ok": [], "bit_disagreements": None})
            score = score_failed_header(header_diag)
            attempt = {
                "ppm": int(ppm),
                "epsilon": float(epsilon),
                "score": float(score),
                "header_ok": False,
                "header": None,
                "header_raw": getattr(exc, "header_raw", b""),
                "h": getattr(exc, "h", np.zeros(len(ACTIVE_BINS), complex)),
                "training_error": getattr(exc, "training_error", np.full(len(ACTIVE_BINS), np.nan)),
                "phase_fit": getattr(exc, "phase_fit", np.empty((0, 4))),
                "header_start": None,
                "header_diag": header_diag,
                "error": f"{type(exc).__name__}: {exc}",
            }
            attempts.append(attempt)
            scores.append(
                {
                    "ppm": int(ppm),
                    "score": float(score),
                    "header_ok": False,
                    "bit_disagreements": header_diag.get("bit_disagreements"),
                    "copy_crc_ok": [bool(value) for value in header_diag.get("copy_crc_ok", [])],
                }
            )
            if best is None or (not best["header_ok"], score, abs(ppm)) < (
                not best["header_ok"],
                best["score"],
                abs(best["ppm"]),
            ):
                best = attempt
    if attempts and not any(attempt["header_ok"] for attempt in attempts):
        min_score = min(attempt["score"] for attempt in attempts)
        close = [attempt for attempt in attempts if attempt["score"] <= min_score + 2.0]
        best = min(close, key=lambda attempt: (abs(attempt["ppm"] + 10), abs(attempt["ppm"])))
    return best, scores


def refine_tail_from_epsilon(rx, chirp, front_start, d_tx, epsilon, radius_samples=20):
    if d_tx is None:
        return None, 0.0, None
    expected = float(front_start + d_tx * (1.0 + epsilon))
    start = max(0, int(round(expected - radius_samples)))
    end = min(len(rx), int(round(expected + radius_samples + len(chirp))))
    try:
        tail_start, tail_score = find_chirp(rx, chirp, start, end)
    except ValueError:
        return None, 0.0, None
    refined_epsilon = ((tail_start - front_start) - d_tx) / d_tx if d_tx else None
    return tail_start, tail_score, refined_epsilon


def fixed_part_samples():
    return (
        CHIRP_SAMPLES
        + CHIRP_GUARD_SAMPLES
        + TRAINING_SYMBOLS * L
        + HEADER_SYMBOLS * L
        + INTER_FRAME_GAP_SAMPLES
    )


def estimate_payload_from_tail(rx, chirp, front_start):
    fixed_part = fixed_part_samples()
    search_start = int(round(front_start + fixed_part - L // 2))
    tail_start, tail_score = find_chirp(rx, chirp, search_start, None)
    d_rx = float(tail_start - front_start)
    payload_symbols = max(0, int(round((d_rx - fixed_part) / L)))
    counts = frame_sample_counts(payload_symbols)
    d_tx = float(counts["D_tx"])
    epsilon = (d_rx - d_tx) / d_tx if d_tx else 0.0
    return payload_symbols, counts, tail_start, tail_score, d_rx, d_tx, epsilon


def hard_bits(symbols, mod):
    return bits_from_mod(symbols, mod)


def bit_error_count(received, truth):
    count = min(len(received), len(truth))
    if count == 0:
        return 0, 0, None
    errors = int(np.count_nonzero(received[:count] != truth[:count]))
    return errors, int(count), float(errors / count)


def byte_error_count(received, truth):
    count = min(len(received), len(truth))
    if count == 0:
        return 0, 0, None
    received_bytes = np.frombuffer(received[:count], dtype=np.uint8)
    truth_bytes = np.frombuffer(truth[:count], dtype=np.uint8)
    errors = int(np.count_nonzero(received_bytes != truth_bytes))
    return errors, int(count), float(errors / count)


def truth_header_from_source(source, payload_rows, mod="qpsk"):
    source_bytes = source.read_bytes()
    payload_mod_symbols = int(np.ceil(len(source_bytes) * 8 / {
        "bpsk": 1,
        "qpsk": 2,
        "qam16": 4,
    }[mod]))
    return header_bytes(source, mod, payload_rows, payload_mod_symbols)


def ber_diagnostics(rx, front_start, training, epsilon, source, payload_rows, header, h=None):
    if source is None or not source.exists() or payload_rows is None:
        return {}

    source_bytes = source.read_bytes()
    mod = header["mod"] if header else "qpsk"
    payload_rows = int(payload_rows)
    payload_mod_symbols = int(np.ceil(len(source_bytes) * 8 / {"bpsk": 1, "qpsk": 2, "qam16": 4}[mod]))

    reference = float(front_start)
    training_offset = CHIRP_SAMPLES + CHIRP_GUARD_SAMPLES
    header_offset = training_offset + TRAINING_SYMBOLS * L
    payload_offset = header_offset + HEADER_SYMBOLS * L

    train_y, _ = symbol_block(rx, front_start, training_offset, TRAINING_SYMBOLS)
    train_y = phase_correct(
        train_y,
        block_starts(front_start, training_offset, len(train_y)),
        reference,
        epsilon,
    )
    if h is not None and h.size and np.any(np.abs(h) > 1e-12):
        diag_h = h
    else:
        diag_h = estimate_channel(train_y, training)

    header_y, _ = symbol_block(rx, front_start, header_offset, HEADER_SYMBOLS)
    header_y = phase_correct(
        header_y,
        block_starts(front_start, header_offset, len(header_y)),
        reference,
        epsilon,
    )
    header_eq = equalize(header_y, diag_h)
    _, header_bit_copies, header_copy_crc_ok = header_copy_bytes(header_eq)
    votes = np.sum(header_bit_copies, axis=0)
    voted_header_bits = (votes >= 2).astype(np.uint8)
    truth_header_bits = bits_from_bytes(truth_header_from_source(source, payload_rows, mod))[:HEADER_BITS]
    header_vote_errors, header_vote_bits, header_vote_ber = bit_error_count(voted_header_bits, truth_header_bits)
    header_copy_errors = [bit_error_count(bits, truth_header_bits)[0] for bits in header_bit_copies]
    header_copy_bers = [errors / HEADER_BITS for errors in header_copy_errors]

    payload_y, _ = symbol_block(rx, front_start, payload_offset, payload_rows)
    payload_y = phase_correct(
        payload_y,
        block_starts(front_start, payload_offset, len(payload_y)),
        reference,
        epsilon,
    )
    payload_eq = equalize(payload_y, diag_h).ravel()[:payload_mod_symbols]
    payload_bits = hard_bits(payload_eq, mod)[: len(source_bytes) * 8]
    truth_payload_bits = bits_from_bytes(source_bytes)
    payload_errors, payload_bits_count, payload_ber = bit_error_count(payload_bits, truth_payload_bits)
    decoded_payload = bytes_from_bits(payload_bits)[: len(source_bytes)]
    payload_byte_errors, payload_bytes_count, payload_byte_error_rate = byte_error_count(decoded_payload, source_bytes)

    overall_errors = header_vote_errors + payload_errors
    overall_bits = header_vote_bits + payload_bits_count
    return {
        "ber_available": True,
        "ber_source": str(source),
        "ber_mod_assumed": mod,
        "ber_payload_symbols_used": int(payload_rows),
        "ber_payload_mod_symbols_used": int(payload_mod_symbols),
        "ber_header_copy_crc_ok": [bool(value) for value in header_copy_crc_ok],
        "ber_header_copy_bit_errors": [int(value) for value in header_copy_errors],
        "ber_header_copy_ber": [float(value) for value in header_copy_bers],
        "ber_header_vote_bit_errors": int(header_vote_errors),
        "ber_header_vote_bits": int(header_vote_bits),
        "ber_header_vote_ber": header_vote_ber,
        "ber_header_bit_disagreements": int(np.count_nonzero((votes != 0) & (votes != HEADER_COPIES))),
        "ber_payload_bit_errors": int(payload_errors),
        "ber_payload_bits": int(payload_bits_count),
        "ber_payload_ber": payload_ber,
        "ber_payload_byte_errors": int(payload_byte_errors),
        "ber_payload_bytes": int(payload_bytes_count),
        "ber_payload_byte_error_rate": payload_byte_error_rate,
        "ber_overall_useful_bit_errors": int(overall_errors),
        "ber_overall_useful_bits": int(overall_bits),
        "ber_overall_useful_ber": float(overall_errors / overall_bits) if overall_bits else None,
        "ber_payload_crc32": int(zlib.crc32(decoded_payload)),
        "ber_truth_crc32": int(zlib.crc32(source_bytes)),
    }


def run_one(receive, out, a, training, chirp):
    rx = read_wav(receive)
    sync = find_chirp_pair(rx, chirp)
    front_start = sync["front_start"]
    front_score = sync["front_score"]

    header = None
    header_raw = b""
    h = np.zeros(len(ACTIVE_BINS), complex)
    training_error = np.full(len(ACTIVE_BINS), np.nan)
    phase_fit = np.empty((0, 4))
    tail_start = sync["tail_start"]
    tail_score = sync["tail_score"]
    d_tx = sync["D_tx"]
    d_rx = sync["D_rx"]
    chirp_epsilon = sync["epsilon"]
    epsilon = 0.0
    payload_symbols_guess = sync["payload_symbols"]
    payload_symbols_from_tail = sync["payload_symbols"] if sync["tail_start"] is not None else None
    payload_symbols_header_match = None
    sync_mode = "pair_sync" if sync["tail_start"] is not None else "sync_fallback"
    tail_search_mode = sync_mode
    sync_grid_error = sync["grid_error"]
    sync_candidates = sync["candidates"]
    epsilon_search_scores = []
    selected_sfo_source = "header_search"
    refined_tail_start = None
    refined_tail_score = 0.0
    refined_chirp_epsilon = None
    fixed_part = fixed_part_samples()
    header_ok = False
    header_diag = {
        "copy_crc_ok": [],
        "bit_disagreements": None,
        "raw_copies": [],
    }
    error = ""

    try:
        if sync_mode == "pair_sync":
            counts = sync["counts"]
        else:
            (
                payload_symbols_guess,
                counts,
                tail_start,
                tail_score,
                d_rx,
                d_tx,
                chirp_epsilon,
            ) = estimate_payload_from_tail(rx, chirp, front_start)
            payload_symbols_from_tail = payload_symbols_guess
            tail_search_mode = "sync_fallback"

        selected, epsilon_search_scores = search_header_epsilon(rx, front_start, training)
        epsilon = selected["epsilon"]
        header = selected["header"]
        header_raw = selected["header_raw"]
        h = selected["h"]
        training_error = selected["training_error"]
        phase_fit = selected["phase_fit"]
        header_diag = selected["header_diag"]
        header_ok = bool(selected["header_ok"])
        error = "" if header_ok else selected.get("error", "ValueError: header CRC mismatch")
        if header_ok:
            payload_symbols_header_match = payload_symbols_guess == header["payload_symbols"]
            counts = frame_sample_counts(header["payload_symbols"])
            d_tx = float(counts["D_tx"])
        refined_tail_start, refined_tail_score, refined_chirp_epsilon = refine_tail_from_epsilon(
            rx, chirp, front_start, d_tx, epsilon
        )
    except Exception as exc:
        try:
            if sync_mode != "pair_sync":
                header, header_raw, h, training_error, phase_fit, _, header_diag = decode_header_pass(
                    rx, front_start, training
                )
                counts = frame_sample_counts(header["payload_symbols"])
                d_tx = float(counts["D_tx"])
                tail_start, tail_score = find_tail(rx, chirp, front_start, d_tx, a.tail_search_seconds)
                d_rx = float(tail_start - front_start)
                chirp_epsilon = (d_rx - d_tx) / d_tx if d_tx else 0.0
                payload_symbols_guess = header["payload_symbols"]
                payload_symbols_from_tail = None
                tail_search_mode = "header_fallback"
                selected, epsilon_search_scores = search_header_epsilon(rx, front_start, training)
                epsilon = selected["epsilon"]
                header = selected["header"]
                header_raw = selected["header_raw"]
                h = selected["h"]
                training_error = selected["training_error"]
                phase_fit = selected["phase_fit"]
                header_diag = selected["header_diag"]
                header_ok = bool(selected["header_ok"])
                error = "" if header_ok else selected.get("error", "ValueError: header CRC mismatch")
                if header_ok:
                    payload_symbols_header_match = payload_symbols_guess == header["payload_symbols"]
                    counts = frame_sample_counts(header["payload_symbols"])
                refined_tail_start, refined_tail_score, refined_chirp_epsilon = refine_tail_from_epsilon(
                    rx, chirp, front_start, d_tx, epsilon
                )
            else:
                raise exc
        except Exception as fallback_exc:
            header_raw = getattr(fallback_exc, "header_raw", header_raw)
            header_diag = getattr(fallback_exc, "header_diag", header_diag)
            error = f"{type(fallback_exc).__name__}: {fallback_exc}"
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

    ber_payload_rows = header["payload_symbols"] if header else payload_symbols_guess
    ber_metrics = {}
    try:
        ber_metrics = ber_diagnostics(
            rx,
            front_start,
            training,
            epsilon,
            a.source,
            ber_payload_rows,
            header,
            h,
        )
    except Exception as exc:
        ber_metrics = {
            "ber_available": False,
            "ber_error": f"{type(exc).__name__}: {exc}",
        }

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
        "chirp_sfo_epsilon": float(chirp_epsilon),
        "chirp_sfo_ppm": float(chirp_epsilon * 1e6),
        "sfo_epsilon": float(epsilon),
        "sfo_ppm": float(epsilon * 1e6),
        "selected_sfo_epsilon": float(epsilon),
        "selected_sfo_ppm": float(epsilon * 1e6),
        "selected_sfo_source": selected_sfo_source,
        "epsilon_search_ppm": int(round(epsilon * 1e6)),
        "epsilon_search_scores": epsilon_search_scores,
        "refined_tail_chirp_start": float(refined_tail_start) if refined_tail_start is not None else None,
        "refined_tail_chirp_score": float(refined_tail_score),
        "refined_chirp_sfo_epsilon": float(refined_chirp_epsilon) if refined_chirp_epsilon is not None else None,
        "refined_chirp_sfo_ppm": float(refined_chirp_epsilon * 1e6) if refined_chirp_epsilon is not None else None,
        "fixed_part_samples": int(fixed_part),
        "payload_symbols_guess": int(payload_symbols_guess) if payload_symbols_guess is not None else None,
        "payload_symbols_from_tail": int(payload_symbols_from_tail) if payload_symbols_from_tail is not None else None,
        "payload_symbols_header_match": payload_symbols_header_match,
        "tail_search_mode": tail_search_mode,
        "sync_mode": sync_mode,
        "sync_grid_error_samples": float(sync_grid_error) if sync_grid_error is not None else None,
        "sync_front_score": float(front_score),
        "sync_tail_score": float(tail_score),
        "sync_candidates": [
            {"start": int(start), "score": float(score)}
            for start, score in sync_candidates
        ],
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
        "header_permutation_used": True,
        "header_permutation_seeds": [int(seed) for seed in HEADER_PERM_SEEDS],
        "file_crc_ok": bool(file_crc_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "recovered_bytes": int(len(recovered_bytes)),
        "mean_abs_h": float(np.mean(np.abs(h))) if h.size else None,
        "median_training_error": finite_median(training_error),
        "error": error,
    }
    metrics.update(ber_metrics)
    metrics.update(profile_meta())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"{receive}: front={front_score:.6f} tail={tail_score:.6f} "
        f"sfo_ppm={epsilon * 1e6:+.4f} header_ok={metrics['header_ok']} "
        f"file_crc_ok={file_crc_ok} file_match={file_match}"
    )
    if metrics.get("ber_available"):
        print(
            f"ber_header_vote={metrics['ber_header_vote_ber']:.6f} "
            f"ber_payload={metrics['ber_payload_ber']:.6f} "
            f"ber_overall={metrics['ber_overall_useful_ber']:.6f}"
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
