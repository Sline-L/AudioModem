from pathlib import Path
import argparse
import csv
import json

import numpy as np

from step6_modem import (
    ACTIVE_BINS,
    DATA_BINS,
    FS,
    L,
    MODS,
    N,
    PILOT_BINS,
    bytes_from_mod,
    decode_payload,
    estimate_h,
    file_symbols_with_pilots,
    find_sync,
    ofdm_rx,
    ofdm_tx,
    phase_aligned_mean,
    profile_meta,
    read_wav,
    training_blocks,
    unpack,
    wav_gain,
)
from audiomodem import pack


def args():
    p = argparse.ArgumentParser(description="recover a Step 6 N=512, CP=256 overlap-band recording")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument("--source", type=Path, default=Path("data/step6_newexp/file.jpeg"))
    p.add_argument("--mod", choices=MODS, default="bpsk")
    p.add_argument("--probe-symbols", type=int, default=256)
    p.add_argument("--probe-repeats", type=int, default=3)
    p.add_argument("--probe-seed", type=int, default=2026)
    p.add_argument("--sync-symbols", type=int, default=128)
    p.add_argument("--sync-repeats", type=int, default=3)
    p.add_argument("--sync-seed", type=int, default=3026)
    p.add_argument("--pilot-seed", type=int, default=2027)
    p.add_argument("--pilot-smooth", type=float, default=0.25)
    p.add_argument("--sync-search", type=int, default=4096)
    p.add_argument("--payload-search", type=int, default=96)
    p.add_argument("--out", type=Path, default=Path("runs/step6_newexp/overlap_n512_cp256"))
    return p.parse_args()


def find_sync_near(rx, tx, expected, radius):
    lo = max(0, expected - radius)
    hi = min(len(rx) - len(tx), expected + radius)
    if hi < lo:
        raise ValueError("receive WAV is too short for file preamble")
    local_start, score = find_sync(rx[lo : hi + len(tx)], tx)
    return lo + local_start, score


def split_h_blocks(y, x_blocks):
    hs = []
    pos = 0
    for x in x_blocks:
        hs.append(estimate_h(y[pos : pos + len(x)], x))
        pos += len(x)
    return np.asarray(hs)


def pilot_alignment_score(rx, start, payload_symbols, pilots, h):
    if start < 0:
        return np.inf
    score_symbols = min(payload_symbols, len(pilots), 64)
    y = ofdm_rx(rx[start : start + score_symbols * L])
    if len(y) < min(score_symbols, 32):
        return np.inf
    n = min(len(y), score_symbols)
    got = y[:n, np.isin(ACTIVE_BINS, PILOT_BINS)] / h[np.isin(ACTIVE_BINS, PILOT_BINS)]
    return float(np.mean(np.abs(got - pilots[:n]) ** 2))


def choose_payload_start(rx, expected, payload_symbols, pilots, h, radius):
    best = (np.inf, expected, 0)
    for delta in range(-radius, radius + 1):
        start = expected + delta
        score = pilot_alignment_score(rx, start, payload_symbols, pilots, h)
        if score < best[0]:
            best = (score, start, delta)
    return best[1], best[2], best[0]


def decoded_metrics(decoded, source):
    truth = pack(source)
    n = min(len(decoded), len(truth))
    got = np.frombuffer(decoded[:n], dtype=np.uint8)
    ref = np.frombuffer(truth[:n], dtype=np.uint8)
    bit_errors = int(np.unpackbits(got ^ ref, bitorder="big").sum())
    byte_errors = int(np.count_nonzero(got != ref))
    if len(decoded) < len(truth):
        missing = len(truth) - len(decoded)
        bit_errors += 8 * missing
        byte_errors += missing
    return {
        "decoded_len": int(len(decoded)),
        "truth_len": int(len(truth)),
        "byte_errors": byte_errors,
        "bit_errors": bit_errors,
        "bit_error_rate": float(bit_errors / (8 * len(truth))) if truth else 0.0,
    }


def save_h_summary(path, h_probe, h_sync, h_combined):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "freq_hz", "role", "abs_h_probe", "abs_h_sync", "abs_h_combined", "phase_h"])
        for i, k in enumerate(ACTIVE_BINS):
            role = "pilot" if k in PILOT_BINS else "data"
            w.writerow(
                [
                    int(k),
                    float(k * FS / N),
                    role,
                    float(abs(h_probe[i])),
                    float(abs(h_sync[i])),
                    float(abs(h_combined[i])),
                    float(np.angle(h_combined[i])),
                ]
            )


def output_dir(base, receive, many):
    if not many:
        return base
    suffix = receive.stem.rsplit("_", 1)[-1]
    return base / (suffix if suffix.isdigit() else receive.stem)


def run_one(receive, out, a, probe_blocks, sync_blocks, payload, pilots, gain):
    rx = read_wav(receive)
    probe_x = [x * gain for x in probe_blocks]
    sync_x = [x * gain for x in sync_blocks]
    probe_wave = ofdm_tx(np.vstack(probe_blocks))
    sync_wave = ofdm_tx(np.vstack(sync_blocks))

    probe_start, probe_score = find_sync(rx, probe_wave)
    probe_end = probe_start + len(probe_wave)
    probe_y = ofdm_rx(rx[probe_start:probe_end])[: sum(map(len, probe_blocks))]
    h_probe_blocks = split_h_blocks(probe_y, probe_x)
    h_probe, h_probe_aligned = phase_aligned_mean(h_probe_blocks)

    sync_start, sync_score = find_sync_near(rx, sync_wave, probe_end, a.sync_search)
    sync_end = sync_start + len(sync_wave)
    sync_y = ofdm_rx(rx[sync_start:sync_end])[: sum(map(len, sync_blocks))]
    h_sync_blocks = split_h_blocks(sync_y, sync_x)
    h_sync, h_sync_aligned = phase_aligned_mean(h_sync_blocks)
    h_combined, h_all_aligned = phase_aligned_mean(np.vstack([h_probe_blocks, h_sync_blocks]), h_sync)

    payload_start, payload_delta, pilot_score = choose_payload_start(
        rx, sync_end, len(payload), pilots * gain, h_combined, a.payload_search
    )
    payload_y = ofdm_rx(rx[payload_start : payload_start + len(payload) * L])[: len(payload)]
    z, pilot_h, pilot_residuals = decode_payload(payload_y, pilots * gain, h_combined, a.pilot_smooth)
    decoded = bytes_from_mod(z, a.mod)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "H.npy", h_combined)
    np.save(out / "H_probe.npy", h_probe)
    np.save(out / "H_sync.npy", h_sync)
    np.save(out / "H_probe_repeats.npy", h_probe_aligned)
    np.save(out / "H_sync_repeats.npy", h_sync_aligned)
    np.save(out / "H_training_all.npy", h_all_aligned)
    np.save(out / "pilot_H.npy", pilot_h)
    np.save(out / "rx_symbols.npy", z)
    (out / "decoded_raw.bin").write_bytes(decoded)
    save_h_summary(out / "summary.csv", h_probe, h_sync, h_combined)

    header_ok = False
    file_match = False
    recovered_name = ""
    error = ""
    try:
        name, body = unpack(decoded)
        header_ok = True
        recovered_name = name
        (out / name).write_bytes(body)
        file_match = body == a.source.read_bytes()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    metrics = {
        "input": str(receive),
        "out": str(out),
        "probe_start": int(probe_start),
        "probe_sync_score": float(probe_score),
        "file_sync_start": int(sync_start),
        "file_sync_score": float(sync_score),
        "payload_start": int(payload_start),
        "structural_payload_start": int(sync_end),
        "payload_delta": int(payload_delta),
        "payload_repeats": 1,
        "probe_repeats": int(a.probe_repeats),
        "sync_repeats": int(a.sync_repeats),
        "pilot_alignment_score": float(pilot_score),
        "pilot_residual": float(np.mean(pilot_residuals)) if len(pilot_residuals) else None,
        "pilot_smooth": float(a.pilot_smooth),
        "mean_abs_h_probe": float(np.mean(np.abs(h_probe))),
        "mean_abs_h_sync": float(np.mean(np.abs(h_sync))),
        "mean_abs_h": float(np.mean(np.abs(h_combined))),
        "header_ok": bool(header_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "first16_hex": decoded[:16].hex(),
        "error": error,
    }
    metrics.update(decoded_metrics(decoded, a.source))
    metrics.update(profile_meta())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"{receive}: probe_sync={probe_score:.6f} file_sync={sync_score:.6f} "
        f"payload_delta={payload_delta} BER={metrics['bit_error_rate']:.6%} "
        f"header_ok={header_ok} file_match={file_match}"
    )
    if error:
        print(f"decode_error={error}")
    return metrics


def main():
    a = args()
    if not 0 <= a.pilot_smooth <= 1:
        raise SystemExit("--pilot-smooth must be between 0 and 1")
    if not a.source.exists():
        raise SystemExit(f"source file does not exist: {a.source}")
    probe_blocks = training_blocks(ACTIVE_BINS, a.probe_symbols, a.probe_repeats, a.probe_seed)
    sync_blocks = training_blocks(ACTIVE_BINS, a.sync_symbols, a.sync_repeats, a.sync_seed)
    payload, pilots = file_symbols_with_pilots(a.source, a.mod, a.pilot_seed)
    raw = np.r_[ofdm_tx(np.vstack(probe_blocks)), ofdm_tx(np.vstack(sync_blocks)), ofdm_tx(payload)]
    gain = wav_gain(raw)

    many = len(a.input) > 1
    results = []
    for receive in a.input:
        results.append(
            run_one(
                receive,
                output_dir(a.out, receive, many),
                a,
                probe_blocks,
                sync_blocks,
                payload,
                pilots,
                gain,
            )
        )
    if many:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = [
            "input",
            "probe_sync_score",
            "file_sync_score",
            "payload_delta",
            "mean_abs_h",
            "pilot_residual",
            "bit_errors",
            "bit_error_rate",
            "header_ok",
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
