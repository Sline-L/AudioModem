from pathlib import Path
import argparse
import csv
import json

import numpy as np

from modem_n1 import (
    ACTIVE_BINS,
    FS,
    L,
    MODS,
    N,
    bytes_from_mod,
    equalize,
    estimate_channel,
    find_sync,
    ofdm_rx,
    ofdm_tx,
    profile_meta,
    random_qpsk,
    read_wav,
    unpack_file,
)


def args():
    p = argparse.ArgumentParser(description="recover a n1 sync + preamble + data recording")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--noise-seconds", type=float, default=0.5)
    p.add_argument("--sync-symbols", type=int, default=16)
    p.add_argument("--sync-seed", type=int, default=2026)
    p.add_argument("--preamble-symbols", type=int, default=64)
    p.add_argument("--preamble-seed", type=int, default=3026)
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--out", type=Path, default=Path("runs/n1"))
    return p.parse_args()


def validate(a):
    for name in ("sync_symbols", "preamble_symbols"):
        if getattr(a, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if a.noise_seconds < 0:
        raise SystemExit("--noise-seconds must be >= 0")


def output_dir(base, receive, many):
    if not many:
        return base
    suffix = receive.stem.rsplit("_", 1)[-1]
    return base / (suffix if suffix.isdigit() else receive.stem)


def save_summary(path, h, preamble_error):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bin", "freq_hz", "abs_h", "preamble_error"])
        for index, bin_index in enumerate(ACTIVE_BINS):
            writer.writerow(
                [
                    int(bin_index),
                    float(bin_index * FS / N),
                    float(abs(h[index])),
                    float(preamble_error[index]),
                ]
            )


def run_one(receive, out, a, sync, preamble):
    rx = read_wav(receive)
    sync_template = ofdm_tx(sync)
    sync_start, sync_score = find_sync(rx, sync_template)

    sync_samples = len(sync) * L
    preamble_samples = len(preamble) * L
    preamble_start = sync_start + sync_samples
    payload_start = preamble_start + preamble_samples

    preamble_y = ofdm_rx(rx[preamble_start : preamble_start + preamble_samples])[: len(preamble)]
    h = estimate_channel(preamble_y, preamble)
    preamble_error = np.mean(np.abs(preamble_y - preamble[: len(preamble_y)] * h) ** 2, axis=0)

    payload_y = ofdm_rx(rx[payload_start:])
    payload_eq = equalize(payload_y, h).ravel()
    raw = bytes_from_mod(payload_eq, a.mod)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "H.npy", h)
    np.save(out / "payload_symbols.npy", payload_eq)
    save_summary(out / "summary.csv", h, preamble_error)

    recovered_name = ""
    recovered_bytes = b""
    file_ok = False
    file_match = False
    error = ""
    try:
        recovered_name, recovered_bytes = unpack_file(raw)
        target = out / recovered_name
        target.write_bytes(recovered_bytes)
        file_ok = True
        if a.source and a.source.exists():
            file_match = recovered_bytes == a.source.read_bytes()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (out / "decoded_payload.bin").write_bytes(raw)

    metrics = {
        "input": str(receive),
        "out": str(out),
        "sync_start": int(sync_start),
        "sync_score": float(sync_score),
        "preamble_start": int(preamble_start),
        "payload_start": int(payload_start),
        "payload_ofdm_symbols_received": int(len(payload_y)),
        "mod": a.mod,
        "mean_abs_h": float(np.mean(np.abs(h))),
        "median_preamble_error": float(np.median(preamble_error)),
        "file_ok": bool(file_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "recovered_bytes": int(len(recovered_bytes)),
        "error": error,
    }
    metrics.update(profile_meta())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"{receive}: sync={sync_score:.6f} payload_symbols={len(payload_y)} "
        f"file_ok={file_ok} file_match={file_match}"
    )
    if error:
        print(f"decode_error={error}")
    return metrics


def main():
    a = args()
    validate(a)
    sync = random_qpsk(a.sync_symbols, seed=a.sync_seed)
    preamble = random_qpsk(a.preamble_symbols, seed=a.preamble_seed)
    many = len(a.input) > 1
    results = []
    for receive in a.input:
        results.append(run_one(receive, output_dir(a.out, receive, many), a, sync, preamble))
    if many:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = ["input", "sync_score", "file_ok", "file_match", "recovered_name", "error"]
        with (a.out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {a.out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
