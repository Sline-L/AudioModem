"""n2 接收机编排。同步/取样对齐 AudioModem n3_2：SFO 放在插值网格，帧头 CRC 选择候选。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from align import (
    best_sfo_near,
    extract_active_sfo,
    find_end_preamble,
    matched_start,
    training_peaks,
)
from capture import coarse_body_start, find_two_chirps
from channel import h_and_llr_scale, lerp_h
from decode_path import LdpcBank, extract_payload
from params import derived
from phy import (
    active_bins,
    bits_to_bytes,
    parse_ph_header,
    qpsk_llr_scaled,
    read_wav_channels,
    safe_filename,
    slice_qpsk,
    train_freq_time,
)


@dataclass
class RxResult:
    ok: bool
    stage: str
    filename: str = ""
    payload: bytes = b""
    file_size: int = 0
    out_path: str = ""
    diagnostics: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


def coded_symbol_count(file_size, info_block=249, coded_bytes=498):
    """含帧头信息块在内的数据 OFDM 符号数，与 n3_2 / TX 补零一致。"""
    return 2 * ((int(info_block) + int(file_size) + int(coded_bytes) - 1) // int(coded_bytes))


def _lock_frame(rx, d):
    first, second, _, cap = find_two_chirps(rx, d)
    if first is None or not cap.get("ok"):
        return None
    coarse = coarse_body_start(first, d)
    lock = matched_start(rx, coarse, d)
    if lock is None or lock["snr"] < 4.0:
        return None
    end = find_end_preamble(rx, lock["start"], d)
    if end is None or end["K"] < 1 or end["K"] > 20000:
        return None
    return {
        "first": first,
        "second": second,
        "cap": cap,
        "lock": lock,
        "end": end,
    }


def _decode_symbol(z, scale, bank):
    llr = qpsk_llr_scaled(z, scale)
    info, coded, it = bank.decode_codeword(llr)
    return info, coded, it, llr


def _try_header(rx, start, sfo, known_front, bank, d):
    try:
        y = extract_active_sfo(rx, start, 9, sfo, d)
    except ValueError:
        return None
    h, scale, noise_var, valid = h_and_llr_scale(y[:8], known_front)
    if not np.any(valid):
        return None
    z = np.zeros_like(y[8])
    z[valid] = y[8, valid] / h[valid]
    info, coded, it, llr = _decode_symbol(z, scale, bank)
    raw = bits_to_bytes(info)
    header = parse_ph_header(raw)
    if header is None:
        return None
    expect = (int(header["file_size"]) + 248) // 249
    if header["declared_symbols"] != expect:
        return None
    if not header["filename"]:
        return None
    header["offset"] = 0
    score = float(np.sum(np.abs(np.mean(y[:8] / (known_front + 1e-12), axis=0)) ** 2))
    return {
        "header": header,
        "h": h,
        "scale": scale,
        "noise_var": noise_var,
        "raw": raw,
        "it": it,
        "llr": llr,
        "score": score,
        "start": float(start),
        "sfo": float(sfo),
    }


def _trust_sfo(sfo_end, sfo_chirp, n_sym_raw):
    """两个钟偏不能取中位数：两个数的 median 等于均值，会把 -0 和离谱 Chirp 平均成几十万 ppm。"""
    aligned = abs(float(n_sym_raw) - round(float(n_sym_raw))) < 0.08
    end_ok = np.isfinite(sfo_end) and abs(sfo_end) < 5e-3
    chirp_ok = np.isfinite(sfo_chirp) and abs(sfo_chirp) < 5e-3
    if aligned and end_ok:
        if chirp_ok and abs(sfo_end - sfo_chirp) < 80e-6:
            return 0.5 * (sfo_end + sfo_chirp)
        return float(sfo_end)
    if chirp_ok:
        return float(sfo_chirp)
    if end_ok:
        return float(sfo_end)
    return 0.0


def _header_jobs(rx, d, fr, known_front, include_train=False):
    """锁帧起点 + 训练峰，钟偏中心用可靠值并额外试 0 ppm。"""
    offsets_lock = (0, -128, 128, -256, 256, -64, 64)
    offsets_train = (0, -128, 128, -256, 256)
    ppm_local = (0, -10, 10, -20, 20, -5, 5, -30, 30)
    jobs = []
    seen = []

    def add(start, sfo, tag):
        start = float(start)
        sfo = float(sfo)
        if not np.isfinite(start) or not np.isfinite(sfo) or abs(sfo) > 5e-3:
            return
        key = (round(start / 8.0), round(sfo * 1e8))
        if key in seen:
            return
        seen.append(key)
        jobs.append((start, sfo, tag))

    sfo_main = 0.0
    if fr is not None:
        n_pre = d["preamble_count"]
        slen = d["symbol_len"]
        k_hint = int(fr["end"]["K"])
        expected_end = (n_pre + k_hint) * slen
        sfo_end = float(fr["end"]["end_offset"]) / float(expected_end) - 1.0
        gap = float(fr["second"]) - float(fr["first"])
        expected_gap = d["chirp_len"] + 2 * d["silence_len"] + (2 * n_pre + k_hint) * slen
        sfo_chirp = gap / expected_gap - 1.0
        sfo_main = _trust_sfo(sfo_end, sfo_chirp, fr["end"]["n_sym_raw"])
        base = float(fr["lock"]["start"])
        centers = [sfo_main]
        if abs(sfo_main) > 40e-6:
            centers.append(0.0)
        for offset in offsets_lock:
            for center in centers:
                for ppm in ppm_local:
                    add(base + offset, center + ppm * 1e-6, "lock")

    peaks = []
    if include_train:
        peaks = training_peaks(rx, d, n_peaks=4)
        for peak in peaks:
            scored = best_sfo_near(rx, peak["start"], known_front, d)
            if scored is None:
                continue
            _, sfo = scored
            for offset in offsets_train:
                add(peak["start"] + offset, sfo, "train")
    return jobs, sfo_main, peaks


def _equalize_once(spec, h, scale, bank):
    mag = np.abs(h)
    valid = mag > max(float(np.max(mag)) * 1e-8, 1e-12)
    z = np.zeros_like(spec)
    z[valid] = spec[valid] / h[valid]
    xhat = slice_qpsk(z)
    weight = mag * mag
    num = np.sum(weight * z * np.conj(xhat))
    theta = float(np.angle(num)) if np.abs(num) > 1e-12 else 0.0
    z = z * np.exp(-1j * theta)
    info, coded, it, llr = _decode_symbol(z, scale, bank)
    return info, int(it), llr, z, theta


def _lerp_scale(a, b, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - alpha) * np.asarray(a) + alpha * np.asarray(b)


def _decode_payload_symbol(spec, h_joint, scale_joint, h_front, scale_front, h_tail, scale_tail, alpha, bank):
    """先用前后训练插值 + CPE；不收敛再换 H。"""
    first_h = lerp_h(h_front, h_tail, alpha)
    first_s = _lerp_scale(scale_front, scale_tail, alpha)
    info, it, llr, z, theta = _equalize_once(spec, first_h, first_s, bank)
    best = (it, -float(np.median(np.abs(llr))), info, llr, z, theta)
    if it < 200:
        return best
    alts = [
        (h_joint, scale_joint),
        (h_front, scale_front),
        (h_tail, scale_tail),
    ]
    for h, scale in alts:
        info, it, llr, z, theta = _equalize_once(spec, h, scale, bank)
        cand = (it, -float(np.median(np.abs(llr))), info, llr, z, theta)
        if cand[:2] < best[:2]:
            best = cand
        if it < 200:
            break
    return best


def _payload_sfo_search(rx, start, sfo0, k, known, d):
    """前后训练联合残差，1 ppm 网格搜 payload 钟偏（n3_2）。"""
    n_pre = d["preamble_count"]
    known_front = known[:n_pre]
    known_tail = known[n_pre : 2 * n_pre]
    slen = d["symbol_len"]
    best = None
    for sfo in sfo0 + np.arange(-50.0, 50.1, 1.0) * 1e-6:
        try:
            front = extract_active_sfo(rx, start, n_pre, sfo, d)
            tail_start = float(start) + (n_pre + k) * slen * (1.0 + sfo)
            tail = extract_active_sfo(rx, tail_start, n_pre, sfo, d)
        except ValueError:
            continue
        joint_y = np.vstack((front, tail))
        joint_x = np.vstack((known_front, known_tail))
        h, scale, noise_var, _ = h_and_llr_scale(joint_y, joint_x)
        err = np.mean(np.abs(joint_y - joint_x * h) ** 2, axis=0)
        score = float(np.median(err / np.maximum(np.abs(h) ** 2, 1e-12)))
        item = (score, float(sfo), h, scale, noise_var)
        if best is None or score < best[0]:
            best = item
    return best


def recover_array(rx, fs=None, p=None, out_dir=None):
    """核心恢复流程，输入已是单声道浮点波形。"""
    del out_dir
    d = derived(p)
    if fs is not None and int(fs) != int(d["sample_rate"]):
        d = derived({**d, "sample_rate": int(d["sample_rate"])})
    diag = {}
    art = {}
    rx = np.asarray(rx, dtype=np.float64).ravel()
    if rx.size < d["chirp_len"] * 2:
        return RxResult(False, "capture", diagnostics={"reason": "波形过短"})

    fr = _lock_frame(rx, d)
    first, second, _, cap = find_two_chirps(rx, d)
    diag["capture"] = fr["cap"] if fr is not None else cap
    if fr is not None:
        diag["lock"] = fr["lock"]
        diag["end"] = fr["end"]
    else:
        diag["lock"] = None
        diag["end"] = None

    slen = d["symbol_len"]
    n_pre = d["preamble_count"]
    freq, _ = train_freq_time(d)
    bins = active_bins(d)
    known = freq[:, bins]
    known_front = known[:n_pre]

    try:
        bank = LdpcBank(d)
    except Exception as exc:
        diag["ldpc_init"] = str(exc)
        return RxResult(False, "ldpc", diagnostics=diag, artifacts=art)

    jobs, sfo_main, peaks = _header_jobs(rx, d, fr, known_front, include_train=False)
    diag["clock"] = {
        "status": "preamble" if fr is not None else "training",
        "scale": 1.0 + sfo_main,
        "ppm": sfo_main * 1e6,
        "n_jobs": len(jobs),
        "n_train_peaks": len(peaks),
    }
    if fr is not None:
        k_hint = int(fr["end"]["K"])
        expected_end = (n_pre + k_hint) * slen
        diag["clock"]["sfo_end"] = float(fr["end"]["end_offset"]) / float(expected_end) - 1.0
        art["clock_table"] = np.array(
            [
                [0.0, float(fr["first"]), 1.0, 0.0],
                [float(d["chirp_len"] + d["silence_len"]), float(fr["lock"]["start"]), 1.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif first is not None:
        art["clock_table"] = np.array(
            [[0.0, float(first), 1.0, 0.0]],
            dtype=np.float64,
        )

    found = None
    attempts = []
    for start_try, sfo_try, tag in jobs:
        hit = _try_header(rx, start_try, sfo_try, known_front, bank, d)
        attempts.append({"start": start_try, "sfo": float(sfo_try), "tag": tag, "ok": hit is not None})
        if hit is None:
            continue
        if found is None or hit["score"] > found["score"]:
            found = hit
        break
    if found is None:
        extra, _, peaks = _header_jobs(rx, d, fr, known_front, include_train=True)
        train_jobs = [job for job in extra if job[2] == "train"]
        for start_try, sfo_try, tag in train_jobs:
            hit = _try_header(rx, start_try, sfo_try, known_front, bank, d)
            attempts.append({"start": start_try, "sfo": float(sfo_try), "tag": tag, "ok": hit is not None})
            if hit is None:
                continue
            found = hit
            break
        diag["clock"]["n_train_peaks"] = len(peaks)
    diag["header_attempts"] = len(attempts)
    if found is None:
        diag["reason"] = "PH 头 CRC 未通过"
        stage = "header" if jobs else ("capture" if first is None else "align")
        return RxResult(False, stage, diagnostics=diag, artifacts=art)

    header = found["header"]
    start = found["start"]
    sfo = found["sfo"]
    header_raw = bytes(found["raw"])
    k = coded_symbol_count(header["file_size"], d["info_bytes"], d["chunk_size"])
    diag["header"] = {
        "filename": header["filename"],
        "file_size": header["file_size"],
        "declared_symbols": header["declared_symbols"],
        "K": k,
        "offset": 0,
        "crc_ok": True,
    }
    art["header_raw"] = header_raw

    searched = _payload_sfo_search(rx, start, sfo, k, known, d)
    if searched is None:
        diag["reason"] = "尾训练越界，无法细化钟偏"
        return RxResult(False, "align", diagnostics=diag, artifacts=art)
    score, payload_sfo, h, scale, noise_var = searched
    diag["clock"]["payload_sfo"] = payload_sfo
    diag["clock"]["payload_ppm"] = payload_sfo * 1e6
    diag["clock"]["train_residual"] = score
    diag["sigma2"] = noise_var
    diag["mean_rho"] = float(np.mean(np.abs(h) / (np.abs(h) + 1e-12)))
    art["H"] = h
    art["training_noise"] = np.asarray(scale)

    try:
        body = extract_active_sfo(rx, start, n_pre + k + n_pre, payload_sfo, d)
        has_tail = True
    except ValueError:
        try:
            body = extract_active_sfo(rx, start, n_pre + k, payload_sfo, d)
        except ValueError as exc:
            diag["fft_error"] = str(exc)
            return RxResult(False, "align", diagnostics=diag, artifacts=art)
        has_tail = False
    data = body[n_pre : n_pre + k]
    payload_specs = data[1:] if data.shape[0] > 1 else data[:0]
    h_front, scale_front, _, _ = h_and_llr_scale(body[:n_pre], known[:n_pre])
    if has_tail and body.shape[0] >= n_pre + k + n_pre:
        h_tail, scale_tail, _, _ = h_and_llr_scale(
            body[n_pre + k : n_pre + k + n_pre], known[n_pre : 2 * n_pre]
        )
    else:
        h_tail, scale_tail = h, scale
    art["H"] = h
    art["H_end"] = h_tail
    art["training_noise"] = np.asarray(scale)

    infos, llrs, iters = [], [], []
    z_list = []
    cpe = []
    for i, spec in enumerate(payload_specs):
        alpha = (i + 1) / float(max(k - 1, 1))
        it, _neg, info, llr, z, theta = _decode_payload_symbol(
            spec, h, scale, h_front, scale_front, h_tail, scale_tail, alpha, bank
        )
        infos.append(info)
        llrs.append(llr)
        iters.append(it)
        z_list.append(z)
        cpe.append(theta)
    payload_bytes = bits_to_bytes(np.concatenate(infos)) if infos else b""
    info_bytes = header_raw + payload_bytes
    payload = extract_payload(info_bytes, header)
    payload_complete = bool(payload) and bool(iters) and all(int(it) < 200 for it in iters)
    art["llr"] = np.stack(llrs, axis=0) if llrs else np.zeros((0, 1))
    art["z"] = np.stack(z_list, axis=0) if z_list else np.zeros((0, d["n_active"]))
    art["info_bytes"] = info_bytes
    art["iters"] = np.asarray(iters, dtype=np.int32)
    art["bins"] = bins
    art["K"] = k
    art["cpe"] = np.asarray(cpe, dtype=np.float64)
    diag["ldpc_iters"] = iters
    diag["payload_complete"] = payload_complete
    diag["turbo"] = False
    diag["cfo_hz"] = 0.0
    diag["resampled"] = False
    diag["payload_sfo"] = payload_sfo

    name = safe_filename(header["filename"])
    if payload is None:
        diag["reason"] = "载荷长度与 file_size 不一致"
        return RxResult(
            False,
            "ldpc",
            filename=name,
            file_size=int(header["file_size"]),
            diagnostics=diag,
            artifacts=art,
        )
    if not payload_complete:
        diag["reason"] = "部分 LDPC 码字未收敛，已按帧头长度写出文件"
        return RxResult(
            False,
            "ldpc",
            filename=name,
            payload=payload,
            file_size=header["file_size"],
            diagnostics=diag,
            artifacts=art,
        )
    return RxResult(
        True,
        "ok",
        filename=name,
        payload=payload,
        file_size=header["file_size"],
        diagnostics=diag,
        artifacts=art,
    )


def _result_key(result):
    iters = result.diagnostics.get("ldpc_iters") or []
    ok_n = sum(int(it) < 200 for it in iters)
    header = 1 if (result.diagnostics.get("header") or {}).get("filename") else 0
    return (int(result.ok), int(bool(result.payload)), header, ok_n)


def recover_from_channels(channels, fs=None, p=None):
    """立体声逐路试，取 header/LDPC 更好的一路。成功则提前停。"""
    order = sorted(range(len(channels)), key=lambda i: -float(np.mean(np.square(channels[i]))))
    best = None
    for index in order:
        result = recover_array(channels[index], fs=fs, p=p)
        result.diagnostics["channel_index"] = index
        if best is None or _result_key(result) > _result_key(best):
            best = result
        if result.ok:
            break
    return best


def recover_wav(wav_path, out_dir=None, p=None, source_path=None, source_match=None):
    """从 WAV 恢复文件。out_dir 非空时写一次结果，避免重复覆盖口径。"""
    fs, channels = read_wav_channels(wav_path)
    result = recover_from_channels(channels, fs=fs, p=p)
    result.diagnostics["wav"] = str(wav_path)
    result.artifacts["wav"] = str(wav_path)
    result.artifacts["out_dir"] = str(out_dir) if out_dir is not None else ""
    if out_dir is not None:
        from report import write_run_outputs

        write_run_outputs(
            Path(out_dir),
            result,
            source_path=source_path,
            source_match=source_match,
        )
        result.out_path = result.artifacts.get("out_path", "")
    return result
