from pathlib import Path
import argparse
import csv
import json

import numpy as np

from audiomodem import (
    FS,
    L,
    MODS,
    bins,
    bytes_from_mod,
    file_symbols,
    find_sync,
    ofdm_rx,
    ofdm_tx,
    pack,
    preamble_symbols,
    preamble_wave,
    probe_symbols,
    read_wav,
    unpack,
    wav_gain,
)
from fband import (
    fband_profile,
    file_symbols_with_comb_pilots,
    interp_h_from_comb_pilots,
    profile_meta,
)


def normalized_score(rx, tx, start):
    if start < 0 or start + len(tx) > len(rx):
        return 0.0
    e = np.linalg.norm(rx[start : start + len(tx)]) * np.linalg.norm(tx)
    c = np.vdot(tx, rx[start : start + len(tx)])
    return float(abs(c) / e) if e else 0.0


def find_sync_limited(rx, tx, max_start=None):
    if max_start is not None:
        if max_start < 0:
            max_start = 0
        rx = rx[: max_start + len(tx)]
    return find_sync(rx, tx)


def find_sync_near(rx, tx, expected, radius):
    if radius < 0:
        tail = rx[expected:]
        start, score = find_sync(tail, tx)
        return expected + start, score
    best = (expected, normalized_score(rx, tx, expected))
    a = max(0, expected - radius)
    b = min(len(rx) - len(tx), expected + radius)
    for start in range(a, b + 1):
        score = normalized_score(rx, tx, start)
        if score > best[1]:
            best = (start, score)
    return best


def args():
    p = argparse.ArgumentParser(description="recover file from one-shot probe + file OFDM wav")
    p.add_argument("input", nargs="+", type=Path)
    p.add_argument("--source", type=Path, default=Path("data/step2_file/exp2.txt"))
    p.add_argument("--bins", nargs=2, type=int, default=(8, 150), metavar=("START", "END"))
    p.add_argument("--fband-profile", choices=["conservative", "trimmed"], default=None)
    p.add_argument("--mod", choices=MODS, default="qpsk")
    p.add_argument("--probe-kind", choices=["ones", "chirp", "step", "bandstep", "random"], default="ones")
    p.add_argument("--probe-symbols", type=int, default=256)
    p.add_argument("--probe-seed", type=int, default=2026)
    p.add_argument("--sync-symbols", type=int, default=32)
    p.add_argument("--sync-seed", type=int, default=2026)
    p.add_argument("--pilot-interval", type=int, default=0)
    p.add_argument("--pilot-len", type=int, default=1)
    p.add_argument("--pilot-kind", choices=["ones", "chirp", "step", "bandstep", "random"], default="random")
    p.add_argument("--pilot-seed", type=int, default=2027)
    p.add_argument("--payload-repeats", type=int, default=1)
    p.add_argument("--payload-search", type=int, default=40)
    p.add_argument("--file-sync-search", type=int, default=8192)
    p.add_argument("--file-sync-mode", choices=["near", "global", "best"], default="best")
    p.add_argument("--pilot-smooth", type=int, default=0)
    p.add_argument("--repeat-combine", choices=["hard", "soft"], default="hard")
    p.add_argument("--out", type=Path, default=Path("runs/step2_file/combo_8_150"))
    return p.parse_args()


def estimate_h(y, x):
    mask = np.abs(x) > 0
    h_each = np.divide(y, x, out=np.full_like(y, np.nan + 1j * np.nan), where=mask)
    seen = np.any(mask, axis=0)
    h = np.full(x.shape[1], np.nan + 1j * np.nan)
    h[seen] = np.nanmean(h_each[:, seen], axis=0)
    return h


def pilot_count(data_symbols, interval, pilot_len):
    return (data_symbols // interval) * pilot_len if interval > 0 else 0


def frame_payload(data, pilots, interval, pilot_len):
    if interval <= 0:
        return data
    parts = []
    pilot_i = 0
    for i in range(0, len(data), interval):
        chunk = data[i : i + interval]
        parts.append(chunk)
        if len(chunk) == interval:
            parts.append(pilots[pilot_i : pilot_i + pilot_len])
            pilot_i += pilot_len
    return np.vstack(parts) if parts else data


def smooth_h_series(hs, radius):
    if radius <= 0 or len(hs) == 0:
        return hs
    out = []
    for i in range(len(hs)):
        a = max(0, i - radius)
        b = min(len(hs), i + radius + 1)
        out.append(np.nanmean(hs[a:b], axis=0))
    return np.asarray(out, dtype=complex)


def pilot_score_and_decode(y, h0, data_symbols, pilots, interval, pilot_len, pilot_smooth=0):
    if pilot_smooth > 0 and interval > 0 and len(pilots) > 0:
        blocks = []
        pilot_h = []
        residuals = []
        pos = 0
        data_i = 0
        pilot_i = 0
        h = h0.copy()
        while data_i < data_symbols and pos < len(y):
            take = min(interval, data_symbols - data_i)
            blocks.append(y[pos : pos + take])
            pos += take
            data_i += take
            if take == interval and pilot_i + pilot_len <= len(pilots) and pos + pilot_len <= len(y):
                pilot_y = y[pos : pos + pilot_len]
                pilot_x = pilots[pilot_i : pilot_i + pilot_len]
                pilot_eq = pilot_y / h
                residuals.append(float(np.mean(np.abs(pilot_eq - pilot_x) ** 2)))
                h = estimate_h(pilot_y, pilot_x)
                pilot_h.append(h.copy())
                pos += pilot_len
                pilot_i += pilot_len
        pilot_h = smooth_h_series(np.asarray(pilot_h, dtype=complex), pilot_smooth)
        data = []
        for i, block in enumerate(blocks):
            if len(pilot_h):
                h = pilot_h[min(i, len(pilot_h) - 1)]
            data.extend([yy / h for yy in block])
        data = np.asarray(data, dtype=complex).reshape(-1, len(h0)) if data else np.empty((0, len(h0)), complex)
        score = float(np.mean(residuals)) if residuals else np.nan
        return data, pilot_h, score

    h = h0.copy()
    data = []
    pilot_h = []
    residuals = []
    pos = 0
    data_i = 0
    pilot_i = 0
    if interval <= 0 or len(pilots) == 0:
        return y / h, np.empty((0, len(h)), complex), np.nan
    while data_i < data_symbols and pos < len(y):
        take = min(interval, data_symbols - data_i)
        block = []
        for _ in range(take):
            block.append(y[pos])
            pos += 1
            data_i += 1
            if pos >= len(y):
                break
        use_h = h
        if take == interval and pilot_i + pilot_len <= len(pilots) and pos + pilot_len <= len(y):
            pilot_y = y[pos : pos + pilot_len]
            pilot_x = pilots[pilot_i : pilot_i + pilot_len]
            pilot_eq = pilot_y / h
            residuals.append(float(np.mean(np.abs(pilot_eq - pilot_x) ** 2)))
            h = estimate_h(pilot_y, pilot_x)
            use_h = h
            pilot_h.append(h.copy())
            pilot_i += pilot_len
            pos += pilot_len
        for yy in block:
            data.append(yy / use_h)
    data = np.asarray(data, dtype=complex).reshape(-1, len(h)) if data else np.empty((0, len(h)), complex)
    pilot_h = np.asarray(pilot_h, dtype=complex).reshape(-1, len(h)) if pilot_h else np.empty((0, len(h)), complex)
    score = float(np.mean(residuals)) if residuals else np.nan
    return data, pilot_h, score


def symbols_to_bytes(z, mod):
    return bytes_from_mod(z, mod)


def bytes_from_bpsk_soft(z_repeats):
    if len(z_repeats) == 1:
        return bytes_from_mod(z_repeats[0], "bpsk")
    n = min(z.size for z in z_repeats)
    soft = np.sum([z.ravel()[:n].real for z in z_repeats], axis=0)
    bits = (soft < 0).astype(np.uint8)
    return np.packbits(bits[: bits.size // 8 * 8], bitorder="big").tobytes()


def comb_pilot_residual(y, h0, comb_pilots, pilot_idx):
    n = min(len(y), len(comb_pilots))
    if n == 0 or len(pilot_idx) == 0:
        return np.nan
    eq = y[:n, pilot_idx] / h0[pilot_idx]
    return float(np.mean(np.abs(eq - comb_pilots[:n]) ** 2))


def comb_pilot_decode(y, h0, payload_info, pilot_smooth=0):
    profile = payload_info["fband_profile"]
    data_symbols = payload_info["data_symbols"]
    comb_pilots = payload_info["comb_pilots"]
    n = min(len(y), data_symbols, len(comb_pilots))
    if n == 0:
        return np.empty((0, len(profile["data_idx"])), complex), np.empty((0, len(h0)), complex), np.nan

    k = payload_info["k"]
    pilot_idx = profile["pilot_idx"]
    data_idx = profile["data_idx"]
    hs = []
    residuals = []
    for i in range(n):
        h = h0.copy()
        residuals.append(float(np.mean(np.abs(y[i, pilot_idx] / h0[pilot_idx] - comb_pilots[i]) ** 2)))
        h[pilot_idx] = y[i, pilot_idx] / comb_pilots[i]
        h[data_idx] = interp_h_from_comb_pilots(k, h, pilot_idx, data_idx, profile["ranges"])
        hs.append(h)
    hs = np.asarray(hs, dtype=complex)
    hs = smooth_h_series(hs, pilot_smooth)
    z = y[:n, data_idx] / hs[:, data_idx]
    return z, hs, float(np.mean(residuals)) if residuals else np.nan


def vote_repeat_bytes(decoded_repeats):
    if len(decoded_repeats) == 1:
        return decoded_repeats[0]
    n = min(map(len, decoded_repeats))
    bits = np.stack(
        [np.unpackbits(np.frombuffer(d[:n], dtype=np.uint8), bitorder="big") for d in decoded_repeats]
    )
    voted = (np.sum(bits, axis=0) >= (len(decoded_repeats) // 2 + 1)).astype(np.uint8)
    return np.packbits(voted, bitorder="big").tobytes()


def decoded_metrics(decoded, source):
    if not source.exists():
        return None
    truth = pack(source)
    n = min(len(decoded), len(truth))
    got = np.frombuffer(decoded[:n], dtype=np.uint8)
    ref = np.frombuffer(truth[:n], dtype=np.uint8)
    bit_errors = int(np.unpackbits(got ^ ref, bitorder="big").sum())
    if len(decoded) < len(truth):
        bit_errors += 8 * (len(truth) - len(decoded))
    byte_errors = int(np.count_nonzero(got != ref))
    if len(decoded) < len(truth):
        byte_errors += len(truth) - len(decoded)
    return {
        "decoded_len": int(len(decoded)),
        "truth_len": int(len(truth)),
        "byte_errors": byte_errors,
        "bit_errors": bit_errors,
        "bit_error_rate": float(bit_errors / (8 * len(truth))) if truth else 0.0,
    }


def save_h_summary(path, k, h, y, x):
    ym = np.sqrt(np.nanmean(np.abs(y) ** 2, axis=0))
    xm = np.sqrt(np.nanmean(np.abs(x) ** 2, axis=0))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin", "freq_hz", "abs_y", "abs_x", "real_h", "imag_h", "abs_h", "phase_h"])
        for kk, yy, xx, hh in zip(k, ym, xm, h):
            w.writerow([int(kk), kk * FS / 1024, abs(yy), abs(xx), hh.real, hh.imag, abs(hh), np.angle(hh)])


def choose_payload_start(rx, k, p0, h0, payload_info, interval, pilot_len, repeats, search):
    if payload_info.get("comb_pilots") is not None:
        if payload_info["data_symbols"] == 0:
            return p0, 0, np.nan
        best = (np.inf, p0, 0)
        for d in range(-search, search + 1):
            p = p0 + d
            if p < 0:
                continue
            y = ofdm_rx(rx[p:], k)
            scores = []
            for r in range(repeats):
                a = r * payload_info["framed_symbols_once"]
                b = a + payload_info["framed_symbols_once"]
                if b > len(y):
                    break
                score = comb_pilot_residual(
                    y[a:b],
                    h0,
                    payload_info["comb_pilots"],
                    payload_info["fband_profile"]["pilot_idx"],
                )
                if np.isfinite(score):
                    scores.append(score)
            score = float(np.mean(scores)) if scores else np.nan
            if np.isfinite(score) and score < best[0]:
                best = (score, p, d)
        return best[1], best[2], best[0]

    data_symbols = payload_info["data_symbols"]
    pilots = payload_info["pilots"]
    framed_symbols_once = payload_info["framed_symbols_once"]
    if interval <= 0 or len(pilots) == 0 or data_symbols == 0:
        return p0, 0, np.nan
    best = (np.inf, p0, 0)
    for d in range(-search, search + 1):
        p = p0 + d
        if p < 0:
            continue
        y = ofdm_rx(rx[p:], k)
        scores = []
        for r in range(repeats):
            a = r * framed_symbols_once
            b = a + framed_symbols_once
            if b > len(y):
                break
            _, _, score = pilot_score_and_decode(y[a:b], h0, data_symbols, pilots, interval, pilot_len)
            if np.isfinite(score):
                scores.append(score)
        score = float(np.mean(scores)) if scores else np.nan
        if np.isfinite(score) and score < best[0]:
            best = (score, p, d)
    return best[1], best[2], best[0]


def choose_file_sync(rx, sync_wave, expected, radius, mode):
    near_start, near_score = find_sync_near(rx, sync_wave, expected, radius)
    if mode == "near":
        return near_start, near_score, "near", near_start, near_score, None, None
    global_start, global_score = find_sync(rx, sync_wave)
    if mode == "global":
        return global_start, global_score, "global", near_start, near_score, global_start, global_score
    if global_score > near_score:
        return global_start, global_score, "global", near_start, near_score, global_start, global_score
    return near_start, near_score, "near", near_start, near_score, global_start, global_score


def run_one(receive, out, a, k, probe_x, probe_wave, sync_x, sync_wave, combo_gain, payload_info, combo_samples):
    rx = read_wav(receive)
    max_probe_start = len(rx) - combo_samples if len(rx) >= combo_samples else None
    probe_start, probe_score = find_sync_limited(rx, probe_wave, max_probe_start)
    probe_end = probe_start + len(probe_wave)
    probe_y = ofdm_rx(rx[probe_start:probe_end], k)[: len(probe_x)]
    h_probe = estimate_h(probe_y, probe_x)

    (
        file_sync_start,
        file_sync_score,
        file_sync_source,
        file_sync_near_start,
        file_sync_near_score,
        file_sync_global_start,
        file_sync_global_score,
    ) = choose_file_sync(rx, sync_wave, probe_end, a.file_sync_search, a.file_sync_mode)
    sync_y = ofdm_rx(rx[file_sync_start : file_sync_start + len(sync_wave)], k)[: len(sync_x)]
    h_sync = estimate_h(sync_y, sync_x)
    structural_payload_start = file_sync_start + len(sync_wave)
    payload_start, payload_delta, pilot_score = choose_payload_start(
        rx,
        k,
        structural_payload_start,
        h_sync,
        payload_info,
        a.pilot_interval,
        a.pilot_len,
        a.payload_repeats,
        a.payload_search,
    )
    payload_y = ofdm_rx(rx[payload_start:], k)
    z_repeats = []
    decoded_repeats = []
    pilot_h_parts = []
    pilot_residuals = []
    if payload_info.get("comb_pilots") is not None:
        for r in range(a.payload_repeats):
            a0 = r * payload_info["framed_symbols_once"]
            b0 = a0 + payload_info["framed_symbols_once"]
            yr = payload_y[a0:b0]
            zr, phr, score = comb_pilot_decode(yr, h_sync, payload_info, a.pilot_smooth)
            z_repeats.append(zr)
            decoded_repeats.append(symbols_to_bytes(zr, a.mod))
            if len(phr):
                pilot_h_parts.append(phr)
            if np.isfinite(score):
                pilot_residuals.append(score)
        z = z_repeats[0] if z_repeats else np.empty((0, len(payload_info["fband_profile"]["data_idx"])), complex)
        pilot_h = np.vstack(pilot_h_parts) if pilot_h_parts else np.empty((0, len(k)), complex)
        pilot_residual = float(np.mean(pilot_residuals)) if pilot_residuals else np.nan
        if a.repeat_combine == "soft" and a.mod == "bpsk":
            decoded = bytes_from_bpsk_soft(z_repeats)
        else:
            decoded = vote_repeat_bytes(decoded_repeats)
    elif a.payload_repeats > 1 or (a.pilot_interval > 0 and len(payload_info["pilots"]) > 0):
        for r in range(a.payload_repeats):
            a0 = r * payload_info["framed_symbols_once"]
            b0 = a0 + payload_info["framed_symbols_once"]
            yr = payload_y[a0:b0]
            zr, phr, score = pilot_score_and_decode(
                yr,
                h_sync,
                payload_info["data_symbols"],
                payload_info["pilots"],
                a.pilot_interval,
                a.pilot_len,
                a.pilot_smooth,
            )
            z_repeats.append(zr)
            decoded_repeats.append(symbols_to_bytes(zr, a.mod))
            if len(phr):
                pilot_h_parts.append(phr)
            if np.isfinite(score):
                pilot_residuals.append(score)
        z = z_repeats[0] if z_repeats else np.empty((0, len(k)), complex)
        pilot_h = np.vstack(pilot_h_parts) if pilot_h_parts else np.empty((0, len(k)), complex)
        pilot_residual = float(np.mean(pilot_residuals)) if pilot_residuals else np.nan
        if a.repeat_combine == "soft" and a.mod == "bpsk":
            decoded = bytes_from_bpsk_soft(z_repeats)
        else:
            decoded = vote_repeat_bytes(decoded_repeats)
    else:
        z = payload_y / h_sync
        z_repeats = [z]
        decoded_repeats = [symbols_to_bytes(z, a.mod)]
        pilot_h = np.empty((0, len(k)), complex)
        pilot_residual = np.nan
        decoded = decoded_repeats[0]

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "H.npy", h_sync)
    np.save(out / "H_probe.npy", h_probe)
    np.save(out / "H_sync.npy", h_sync)
    np.save(out / "pilot_H.npy", pilot_h)
    np.save(out / "Y_probe.npy", probe_y)
    np.save(out / "Y_probe_theory.npy", probe_x)
    np.save(out / "Y_sync.npy", sync_y)
    np.save(out / "Y_sync_theory.npy", sync_x)
    np.save(out / "rx_symbols.npy", z)
    for i, zr in enumerate(z_repeats, 1):
        np.save(out / f"rx_symbols_repeat{i}.npy", zr)
        (out / f"decoded_raw_repeat{i}.bin").write_bytes(decoded_repeats[i - 1])
    (out / "decoded_raw.bin").write_bytes(decoded)
    save_h_summary(out / "summary.csv", k, h_sync, sync_y, sync_x)

    header_ok = False
    file_match = False
    recovered_name = ""
    recovered_bytes = 0
    error = ""
    try:
        name, body = unpack(decoded)
        header_ok = True
        recovered_name = name
        recovered_bytes = len(body)
        (out / name).write_bytes(body)
        if a.source.exists():
            file_match = body == a.source.read_bytes()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    metrics = decoded_metrics(decoded, a.source)
    repeat_metrics = [decoded_metrics(d, a.source) for d in decoded_repeats]
    mean_abs_h = float(np.nanmean(np.abs(h_sync[np.isfinite(h_sync)])))
    result = {
        "input": str(receive),
        "out": str(out),
        "probe_start": int(probe_start),
        "probe_sync_score": float(probe_score),
        "file_sync_start": int(file_sync_start),
        "file_sync_score": float(file_sync_score),
        "file_sync_source": file_sync_source,
        "file_sync_near_start": int(file_sync_near_start),
        "file_sync_near_score": float(file_sync_near_score),
        "file_sync_global_start": None if file_sync_global_start is None else int(file_sync_global_start),
        "file_sync_global_score": None if file_sync_global_score is None else float(file_sync_global_score),
        "payload_start": int(payload_start),
        "structural_payload_start": int(structural_payload_start),
        "payload_delta": int(payload_delta),
        "pilot_score": None if not np.isfinite(pilot_score) else float(pilot_score),
        "pilot_residual": None if not np.isfinite(pilot_residual) else float(pilot_residual),
        "mean_abs_h": mean_abs_h,
        "header_ok": bool(header_ok),
        "file_match": bool(file_match),
        "recovered_name": recovered_name,
        "recovered_bytes": int(recovered_bytes),
        "first16_hex": decoded[:16].hex(),
        "error": error,
        "combo_gain": float(combo_gain),
        "mod": a.mod,
        "sync_symbols": int(a.sync_symbols),
        "pilot_interval": int(a.pilot_interval),
        "pilot_len": int(a.pilot_len),
        "pilot_smooth": int(a.pilot_smooth),
        "pilot_symbols": int(len(payload_info["pilots"])),
        "comb_pilot_bins": int(len(payload_info.get("comb_pilot_bins", []))),
        "data_symbols": int(payload_info["data_symbols"]),
        "payload_repeats": int(a.payload_repeats),
        "repeat_combine": a.repeat_combine,
        "payload_symbols_once": int(payload_info["framed_symbols_once"]),
        "combo_samples": int(combo_samples),
    }
    if metrics:
        result.update(metrics)
    result.update(profile_meta(payload_info.get("fband_profile")))
    result["repeat_bit_error_rates"] = [
        None if m is None else m["bit_error_rate"] for m in repeat_metrics
    ]
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{receive}: probe_sync={probe_score:.6f} file_sync={file_sync_score:.6f} "
        f"payload_delta={payload_delta} mean_abs_h={mean_abs_h:.6g} "
        f"header_ok={header_ok} file_match={file_match}"
    )
    if error:
        print(f"decode_error={error}")
    return result


def run_out_dir(base, receive, many):
    if not many:
        return base
    stem = receive.stem
    label = stem.rsplit("_", 1)[-1] if stem.rsplit("_", 1)[-1].isdigit() else stem
    return base / label


def main():
    a = args()
    if a.pilot_interval < 0:
        raise SystemExit("--pilot-interval must be >= 0")
    if a.pilot_len < 1:
        raise SystemExit("--pilot-len must be >= 1")
    if a.payload_repeats < 1:
        raise SystemExit("--payload-repeats must be >= 1")
    profile = fband_profile(a.fband_profile)
    k = profile["k"] if profile else bins(*a.bins)
    probe_raw = probe_symbols(a.probe_kind, k, a.probe_symbols, a.probe_seed)
    if profile:
        if a.pilot_interval > 0:
            raise SystemExit("--fband-profile uses per-symbol comb pilots; leave --pilot-interval at 0")
        if a.source.exists():
            payload, comb_pilots = file_symbols_with_comb_pilots(
                a.source,
                k,
                profile["data_idx"],
                profile["pilot_idx"],
                a.mod,
                a.pilot_seed,
                a.pilot_kind,
            )
        else:
            payload = np.zeros((0, len(k)), complex)
            comb_pilots = np.zeros((0, len(profile["pilot_idx"])), complex)
        n_pilots = 0
        pilots = np.empty((0, len(k)), complex)
        framed_payload_once = payload
    else:
        payload = file_symbols(a.source, k, a.mod) if a.source.exists() else np.zeros((0, len(k)), complex)
        comb_pilots = None
        n_pilots = pilot_count(len(payload), a.pilot_interval, a.pilot_len)
        pilots = probe_symbols(a.pilot_kind, k, n_pilots, a.pilot_seed)
        framed_payload_once = frame_payload(payload, pilots, a.pilot_interval, a.pilot_len)
    framed_payload = np.vstack([framed_payload_once] * a.payload_repeats)
    probe_wave = ofdm_tx(probe_raw, k)
    sync_raw = preamble_symbols(k, a.sync_symbols, a.sync_seed)
    sync_wave = preamble_wave(k, a.sync_symbols, a.sync_seed)
    raw_parts = [probe_wave, sync_wave]
    if len(framed_payload):
        raw_parts.append(ofdm_tx(framed_payload, k))
    combo_gain = wav_gain(np.r_[tuple(raw_parts)])
    probe_x = probe_raw * combo_gain
    sync_x = sync_raw * combo_gain
    payload_info = {
        "k": k,
        "fband_profile": profile,
        "data_symbols": int(len(payload)),
        "pilots": pilots * combo_gain,
        "comb_pilots": None if comb_pilots is None else comb_pilots * combo_gain,
        "comb_pilot_bins": [] if profile is None else profile["pilot_bins"],
        "framed_symbols_once": int(len(framed_payload_once)),
    }

    many = len(a.input) > 1
    results = []
    for receive in a.input:
        out = run_out_dir(a.out, receive, many)
        combo_samples = int(sum(len(part) for part in raw_parts))
        results.append(
            run_one(receive, out, a, k, probe_x, probe_wave, sync_x, sync_wave, combo_gain, payload_info, combo_samples)
        )

    if many:
        a.out.mkdir(parents=True, exist_ok=True)
        fields = [
            "input",
            "probe_sync_score",
            "file_sync_score",
            "payload_start",
            "payload_delta",
            "pilot_score",
            "pilot_residual",
            "mean_abs_h",
            "header_ok",
            "file_match",
            "byte_errors",
            "bit_errors",
            "bit_error_rate",
            "first16_hex",
            "error",
        ]
        with (a.out / "batch_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"wrote {a.out / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
