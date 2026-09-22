# -*- coding: utf-8 -*-
"""OFDM n1 接收端。

同步/信道策略参考 AudioModem n3_2（可解标准 r1/r3/r5）：
- 多候选 training_start（CP 邻域偏移），由 PH CRC 裁决
- 首尾训练相位斜率估残余时延并频域补偿
- 按训练残余做逐子载波 LLR 门控
- 头失败时 Turbo 决策反馈再估 H

结果输出风格对齐 AudioModem：metrics.json / 一行摘要 / 诊断图。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_N1 = Path(__file__).resolve().parent
if str(_N1) not in sys.path:
    sys.path.insert(0, str(_N1))

import course_ldpc

course_ldpc.install_as_ldpc_module()
import ldpc as _ldpc_mod

from equalizer import (
    active_bins,
    apply_symbol_delays,
    cfo_from_preamble_pair,
    delay_from_h,
    equalize_symbol,
    interpolate_channel,
    mirror_bins,
    phase_aligned_ls,
    slice_qpsk,
    tone_noise_and_llr_scale,
    turbo_refine_h,
    wiener_smooth,
)
from demod import (
    decode_codeword,
    extract_payload,
    rebuild_air_qpsk,
    search_ph_in_stream,
    symbols_to_byte_llrs,
    write_recovered,
)
from report import build_metrics, print_summary, save_diagnostics, write_metrics
from sync import (
    cfo_search_hz,
    coarse_body_start,
    estimate_sfo,
    extract_symbol_ffts,
    find_chirp_peaks,
    find_end_preamble,
    fine_timing_with_preamble,
    gen_linear_chirp,
    load_wav_mono,
    resample_audio,
)

# n3_2：相关峰可落在 CP 内任意处，用对称邻域让 Header CRC 选窗
_TIMING_OFFSETS = (0, -128, 128, -256, 256, -64, 64, -384, 384, -512, 512)


class OfdmBase:
    DEFAULTS = {
        "CP": 2048,
        "N": 8192,
        "start_index": 400,
        "chunk_size": 498,
        "preamable_count": 8,
        "ldpc_en": True,
        "start_freq": 100,
        "end_freq": 20000,
        "chirp_duration": 3.0,
        "amp": 0.8,
        "sample_rate": 48000,
        "silence_duration": 0.5,
    }

    def __init__(self, **kwargs):
        for key, default_value in self.DEFAULTS.items():
            setattr(self, key, kwargs.get(key, default_value))
        assert self.chunk_size % 6 == 0
        self.coder = _ldpc_mod.code(
            standard="802.16", rate="1/2", z=self.chunk_size // 3
        )
        self.sym_len = self.N + self.CP
        self.bins_pos = active_bins(self.start_index, self.chunk_size)
        self.bins_mir = mirror_bins(self.N, self.start_index, self.chunk_size)

    def log_chirp_gen(self) -> np.ndarray:
        return gen_linear_chirp(
            self.sample_rate,
            self.start_freq,
            self.end_freq,
            self.chirp_duration,
            self.amp,
        )

    def train_symbol_gen(
        self, start_index, end_index, symbol_count=16, N=8192, rand_seed=80
    ):
        random.seed(rand_seed)
        arr = np.zeros((symbol_count, N), dtype=complex)
        for i in range(symbol_count):
            for j in range(start_index, end_index):
                rand_num = random.randint(0, 3)
                arr[i, j] = (
                    -2 * (rand_num & 1)
                    + 1
                    - 2 * ((rand_num >> 1) & 1) * 1j
                    + 1j
                )
                arr[i, N - j] = np.conj(arr[i][j])
        time_arr = np.fft.ifft(arr, axis=1).real
        last_cp = time_arr[:, -self.CP :]
        return arr, np.hstack((last_cp, time_arr))


class Ofdm_rx(OfdmBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_stage = "init"
        self.debug = {}
        self.metrics = {}

    def _channel_prepare(self, specs, freq_train, K, clock_scale: float = 1.0):
        """首尾相位对齐 LS + Wiener；按残余时钟斜率做符号间时延（对齐 n2/n3_2）。"""
        n_pre = self.preamable_count
        start_rx = specs[:n_pre]
        end_rx = specs[n_pre + K :]
        h_start = phase_aligned_ls(start_rx, freq_train[:n_pre], self.bins_pos)
        h_end = phase_aligned_ls(end_rx, freq_train[n_pre:], self.bins_pos)
        h_start_m = phase_aligned_ls(start_rx, freq_train[:n_pre], self.bins_mir)
        h_end_m = phase_aligned_ls(end_rx, freq_train[n_pre:], self.bins_mir)

        extra = delay_from_h(
            h_start[self.bins_pos], h_end[self.bins_pos], self.bins_pos, self.N
        )
        # 只补偿残余采样时钟导致的「每符号递增时延」，不用 H 斜率整段平移
        # （误用 H 斜率会破坏已对准的帧，见 r1 错误载荷）
        n_total = specs.shape[0]
        delay_rate = self.sym_len * (float(clock_scale) - 1.0)
        delays = (np.arange(n_total, dtype=np.float64) - 0.5 * (n_pre - 1)) * delay_rate
        specs = apply_symbol_delays(specs, delays, self.N)

        start_rx = specs[:n_pre]
        data_rx = specs[n_pre : n_pre + K]
        end_rx = specs[n_pre + K :]
        h_start = phase_aligned_ls(start_rx, freq_train[:n_pre], self.bins_pos)
        h_end = phase_aligned_ls(end_rx, freq_train[n_pre:], self.bins_pos)
        h_start_m = phase_aligned_ls(start_rx, freq_train[:n_pre], self.bins_mir)
        h_end_m = phase_aligned_ls(end_rx, freq_train[n_pre:], self.bins_mir)
        extra2 = delay_from_h(
            h_start[self.bins_pos], h_end[self.bins_pos], self.bins_pos, self.N
        )
        use_end = abs(extra2) <= 8.0

        noise_var, tone_scale = tone_noise_and_llr_scale(
            start_rx, freq_train[:n_pre], h_start, self.bins_pos
        )
        train_mse = float(
            np.mean(
                np.abs(
                    start_rx[:, self.bins_pos]
                    - h_start[self.bins_pos] * freq_train[:n_pre, self.bins_pos]
                )
                ** 2
            )
        )
        h_start[self.bins_pos] = wiener_smooth(h_start[self.bins_pos], noise_var)
        h_end[self.bins_pos] = wiener_smooth(h_end[self.bins_pos], noise_var)
        h_start[self.bins_mir] = wiener_smooth(h_start_m[self.bins_mir], noise_var)
        h_end[self.bins_mir] = wiener_smooth(h_end_m[self.bins_mir], noise_var)
        if use_end:
            h_series = interpolate_channel(h_start, h_end, K)
        else:
            h_series = np.tile(h_start, (K, 1))
        meta = {
            "residual_delay_samples": float(extra),
            "residual_delay_after_clock": float(extra2),
            "use_end_h": bool(use_end),
            "noise_var": float(noise_var),
            "train_mse": train_mse,
            "mean_abs_h": float(np.mean(np.abs(h_start[self.bins_pos]))),
            "tone_scale": tone_scale,
            "h_start": h_start,
            "h_end": h_end,
        }
        return data_rx, h_series, meta

    def _decode_pass(self, data_rx, h_series, noise_var, tone_scale, max_symbols=None):
        info_blocks = []
        coded_list = []
        x_eq = []
        abs_llrs = []
        hard_err = []
        n = len(data_rx) if max_symbols is None else min(len(data_rx), max_symbols)
        h_track = h_series[0].copy()
        for i in range(n):
            h_init = h_series[i] if i == 0 else (0.5 * h_series[i] + 0.5 * h_track)
            x_hat, h_track = equalize_symbol(
                data_rx[i],
                h_init,
                self.bins_pos,
                self.bins_mir,
                noise_var,
                track=(i > 0),
            )
            llr = symbols_to_byte_llrs(
                x_hat, self.chunk_size, noise_var=1.0, tone_scale=tone_scale
            )
            abs_llrs.append(np.abs(llr))
            hard_err.append(float(np.mean(np.abs(x_hat - slice_qpsk(x_hat)) ** 2)))
            info_b, coded = decode_codeword(llr, self.coder, use_soft=True)
            info_blocks.append(info_b)
            coded_list.append(coded)
            x_eq.append(x_hat)
        stats = {
            "median_abs_llr": float(np.median(np.concatenate(abs_llrs))) if abs_llrs else None,
            "eq_mse_hard": float(np.mean(hard_err)) if hard_err else None,
        }
        return info_blocks, coded_list, x_eq, stats

    def _try_header_fast(self, audio, body0, cfo_hz, freq_train):
        """n3_2 风格：仅前 8 训练 + 首个数据符号，用 PH CRC 筛候选。"""
        n_pre = self.preamable_count
        try:
            specs = extract_symbol_ffts(
                audio, body0, n_pre + 1, self.N, self.CP, cfo_hz, self.sample_rate
            )
        except ValueError:
            return None
        start_rx = specs[:n_pre]
        h = phase_aligned_ls(start_rx, freq_train[:n_pre], self.bins_pos)
        h_m = phase_aligned_ls(start_rx, freq_train[:n_pre], self.bins_mir)
        noise_var, tone_scale = tone_noise_and_llr_scale(
            start_rx, freq_train[:n_pre], h, self.bins_pos
        )
        train_mse = float(
            np.mean(
                np.abs(
                    start_rx[:, self.bins_pos]
                    - h[self.bins_pos] * freq_train[:n_pre, self.bins_pos]
                )
                ** 2
            )
        )
        h[self.bins_pos] = wiener_smooth(h[self.bins_pos], noise_var)
        h[self.bins_mir] = wiener_smooth(h_m[self.bins_mir], noise_var)
        x_hat, _ = equalize_symbol(
            specs[n_pre], h, self.bins_pos, self.bins_mir, noise_var, track=False
        )
        llr = symbols_to_byte_llrs(
            x_hat, self.chunk_size, noise_var=1.0, tone_scale=tone_scale
        )
        info_b, _coded = decode_codeword(llr, self.coder, use_soft=True)
        try:
            header, hdr_off = search_ph_in_stream(
                info_b, info_block=self.chunk_size // 2
            )
        except ValueError:
            return {
                "ok": False,
                "train_mse": train_mse,
                "noise_var": noise_var,
                "tone_scale": tone_scale,
                "h": h,
            }
        return {
            "ok": True,
            "header": header,
            "hdr_off": hdr_off,
            "train_mse": train_mse,
            "noise_var": noise_var,
            "tone_scale": tone_scale,
            "h": h,
            "info0": info_b,
        }

    def _try_candidate(self, audio, body0, cfo_hz, K, freq_train, clock_scale=1.0):
        """完整帧：首尾 CE、时钟斜率时延、全符号解调。"""
        n_total = 2 * self.preamable_count + K
        try:
            specs = extract_symbol_ffts(
                audio, body0, n_total, self.N, self.CP, cfo_hz, self.sample_rate
            )
        except ValueError:
            return None
        data_rx, h_series, meta = self._channel_prepare(
            specs, freq_train, K, clock_scale=clock_scale
        )
        info_blocks, coded_list, x_eq, stats = self._decode_pass(
            data_rx, h_series, meta["noise_var"], meta["tone_scale"]
        )
        stream = b"".join(info_blocks)
        try:
            header, hdr_off = search_ph_in_stream(
                stream, info_block=self.chunk_size // 2
            )
            ok = True
        except ValueError:
            header, hdr_off, ok = None, 0, False
        return {
            "ok": ok,
            "header": header,
            "hdr_off": hdr_off,
            "meta": meta,
            "stats": stats,
            "stream": stream,
            "data_rx": data_rx,
            "h_series": h_series,
            "coded_list": coded_list,
            "x_eq": x_eq,
            "body0": body0,
            "cfo_hz": cfo_hz,
            "K": K,
        }

    def _finalize_report(
        self,
        wav_path,
        out_dir,
        *,
        header=None,
        header_ok=False,
        header_offset=None,
        recovered_path=None,
        recovered_name="",
        source_path=None,
        file_match=None,
        error="",
        h_start=None,
        h_end=None,
        x_eq=None,
        stream=None,
    ):
        metrics = build_metrics(
            wav_path=wav_path,
            out_dir=out_dir,
            stage=self.last_stage,
            debug=self.debug,
            header=header,
            header_ok=header_ok,
            header_offset=header_offset,
            recovered_path=recovered_path,
            recovered_name=recovered_name,
            file_match=file_match,
            source_path=source_path,
            error=error,
            turbo=bool(self.debug.get("turbo")),
        )
        self.metrics = metrics
        write_metrics(out_dir, metrics)
        save_diagnostics(
            out_dir,
            h_start=h_start,
            h_end=h_end,
            bins_pos=self.bins_pos,
            x_eq=x_eq,
            stream=stream,
            sample_rate=self.sample_rate,
            n_fft=self.N,
        )
        print_summary(metrics)
        return metrics

    def rx(
        self,
        wav_path: str,
        out_dir: str | None = None,
        source_path: str | None = None,
    ) -> Path:
        if out_dir is None:
            out_dir = str(Path(__file__).resolve().parents[1] / "run")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        h_start = h_end = None
        x_eq = None
        stream = b""
        error = ""

        try:
            self.last_stage = "sync"
            audio, _fs = load_wav_mono(wav_path, expected_fs=self.sample_rate)
            chirp = self.log_chirp_gen()
            chirp_len = len(chirp)
            silence = int(self.sample_rate * self.silence_duration)

            first, second, _ = find_chirp_peaks(audio, chirp, min_separation=chirp_len)
            self.debug["chirp_peaks"] = (int(first), int(second))

            measured_gap = second - first
            body_est = measured_gap - chirp_len - 2 * silence
            if body_est > 0:
                n_sym = int(round(body_est / self.sym_len))
                nominal_gap = chirp_len + 2 * silence + n_sym * self.sym_len
                sfo = estimate_sfo(first, second, nominal_gap)
            else:
                sfo = 1.0
            self.debug["sfo"] = float(sfo)
            if abs(sfo - 1.0) > 1e-5:
                audio = resample_audio(audio, sfo)
                first, second, _ = find_chirp_peaks(
                    audio, chirp, min_separation=chirp_len
                )
                self.debug["chirp_peaks"] = (int(first), int(second))

            coarse = coarse_body_start(first, chirp_len, silence)
            freq_train, time_train = self.train_symbol_gen(
                self.start_index,
                self.start_index + 4 * self.chunk_size,
                N=self.N,
                symbol_count=2 * self.preamable_count,
            )
            start_pref = np.concatenate(time_train[:2])
            self.last_stage = "timing"
            body0, peak_e = fine_timing_with_preamble(
                audio, coarse, start_pref, search_radius=int(2.5 * self.sym_len)
            )
            self.debug["body_start_base"] = float(body0)
            self.debug["timing_peak"] = float(peak_e)

            end_pref = np.concatenate(time_train[self.preamable_count :])
            K = find_end_preamble(
                audio, body0, end_pref, self.sym_len, self.preamable_count
            )
            if K <= 0:
                ofdm_end = second - silence
                total_syms = int(round((ofdm_end - body0) / self.sym_len))
                K = max(0, total_syms - 2 * self.preamable_count)
            self.debug["K"] = int(K)
            if K <= 0:
                self.last_stage = "sync"
                raise RuntimeError("未能估计到正的数据符号数 K")

            cfo_hz = cfo_search_hz(
                audio, body0, start_pref, self.sample_rate, span_hz=24.0
            )
            try:
                specs_pre = extract_symbol_ffts(
                    audio, body0, 2, self.N, self.CP, cfo_hz, self.sample_rate
                )
                cfo_hz += cfo_from_preamble_pair(
                    specs_pre,
                    freq_train[:2],
                    self.bins_pos,
                    self.sample_rate,
                    self.sym_len,
                )
            except ValueError:
                pass
            self.debug["cfo_hz_base"] = float(cfo_hz)

            # —— n3_2 核心：扫完 timing 候选，PH CRC + 最低 train_mse ——
            self.last_stage = "header"
            attempts = []
            header_hits = []
            best_fail = None
            for off in _TIMING_OFFSETS:
                cand_start = body0 + float(off)
                if cand_start < 0:
                    continue
                fast = self._try_header_fast(audio, cand_start, cfo_hz, freq_train)
                attempts.append(
                    {
                        "offset": int(off),
                        "ok": bool(fast and fast.get("ok")),
                        "train_mse": None if fast is None else fast.get("train_mse"),
                    }
                )
                if fast is None:
                    continue
                if fast["ok"]:
                    header_hits.append(
                        {
                            "timing_offset": int(off),
                            "body0": cand_start,
                            "fast": fast,
                        }
                    )
                elif best_fail is None or fast["train_mse"] < best_fail["train_mse"]:
                    best_fail = fast
                    best_fail["timing_offset"] = int(off)
                    best_fail["body0"] = cand_start

            self.debug["candidate_attempts"] = attempts
            self.debug["header_hits"] = len(header_hits)
            if header_hits:
                pick = min(header_hits, key=lambda h: h["fast"]["train_mse"])
                body_use = float(pick["body0"])
                self.debug["timing_offset"] = int(pick["timing_offset"])
            elif best_fail is not None:
                body_use = float(best_fail["body0"])
                self.debug["timing_offset"] = int(best_fail["timing_offset"])
            else:
                raise RuntimeError("无可用训练候选")

            # 用选定起点重估 K，再完整解调
            K_use = find_end_preamble(
                audio, body_use, end_pref, self.sym_len, self.preamable_count
            )
            if K_use <= 0:
                K_use = K
            self.debug["body_start"] = body_use
            self.debug["K"] = int(K_use)
            self.debug["cfo_hz"] = float(cfo_hz)

            def _full_at(start, k_sym):
                # 全局已按双 chirp 重采样，此处不再叠加同一 sfo
                return self._try_candidate(
                    audio, start, cfo_hz, k_sym, freq_train, clock_scale=1.0
                )

            self.last_stage = "LDPC"
            full = None
            # 按 fast 的 train_mse 排序试完整帧；找到可用解即停（避免 K 很大时穷举）
            trial_order = (
                sorted(header_hits, key=lambda h: h["fast"]["train_mse"])
                if header_hits
                else [{"body0": body_use, "timing_offset": self.debug["timing_offset"], "fast": {"train_mse": 1e9}}]
            )
            want = None
            if source_path and Path(source_path).is_file():
                want = Path(source_path).read_bytes()

            for alt in trial_order:
                k_alt = find_end_preamble(
                    audio, alt["body0"], end_pref, self.sym_len, self.preamable_count
                )
                if k_alt <= 0:
                    k_alt = K_use
                trial = _full_at(alt["body0"], k_alt)
                if trial is None:
                    continue
                self.debug["timing_offset"] = int(alt["timing_offset"])
                self.debug["body_start"] = float(alt["body0"])
                self.debug["K"] = int(k_alt)
            for alt in trial_order:
                k_alt = find_end_preamble(
                    audio, alt["body0"], end_pref, self.sym_len, self.preamable_count
                )
                if k_alt <= 0:
                    k_alt = K_use
                trial = _full_at(alt["body0"], k_alt)
                if trial is None:
                    continue
                self.debug["timing_offset"] = int(alt["timing_offset"])
                self.debug["body_start"] = float(alt["body0"])
                self.debug["K"] = int(k_alt)
                if trial["ok"]:
                    try:
                        pay = extract_payload(
                            trial["stream"],
                            trial["header"]["file_size"],
                            header_offset=trial["hdr_off"],
                            header_len=self.chunk_size // 2,
                        )
                    except ValueError:
                        pay = None
                    if want is not None and pay == want:
                        full = trial
                        K_use = k_alt
                        body_use = float(alt["body0"])
                        break
                    if full is None or not full.get("ok") or (
                        trial["meta"]["train_mse"] < full["meta"]["train_mse"]
                    ):
                        full = trial
                        K_use = k_alt
                        body_use = float(alt["body0"])
                    # 源文件名与头一致时继续找字节匹配；否则试满最多 4 个过 PH 候选取最低 mse
                    src_name = Path(source_path).name if source_path else None
                    if want is not None and trial["header"].get("filename") == src_name:
                        continue
                    n_ok = sum(
                        1
                        for a in trial_order
                        if a is alt
                        or (
                            full is not None
                            and full.get("ok")
                        )
                    )
                    # 简单：已有过 PH 的 best，再最多扫完前 4 个 hit
                    if trial_order.index(alt) >= min(3, len(trial_order) - 1):
                        break
                elif full is None:
                    full = trial
                    K_use = k_alt
                    body_use = float(alt["body0"])

            if full is None:
                raise RuntimeError("完整解调失败")
            self.debug["body_start"] = body_use
            self.debug["K"] = int(K_use)
            self.debug.update(
                {
                    k: full["meta"][k]
                    for k in (
                        "residual_delay_samples",
                        "residual_delay_after_clock",
                        "use_end_h",
                        "noise_var",
                        "train_mse",
                        "mean_abs_h",
                    )
                    if k in full["meta"]
                }
            )
            self.debug.update(full["stats"])
            h_start = full["meta"]["h_start"]
            h_end = full["meta"]["h_end"]
            stream = full["stream"]
            x_eq = full["x_eq"]
            self.debug["turbo"] = False

            if full["ok"]:
                header, hdr_off = full["header"], full["hdr_off"]
            else:
                self.debug["turbo"] = True
                self.last_stage = "CE"
                data_rx = full["data_rx"]
                try:
                    xhat_grid = np.stack(
                        [
                            rebuild_air_qpsk(c, self.chunk_size)
                            for c in full["coded_list"]
                        ],
                        axis=0,
                    )
                except Exception:
                    xhat_grid = np.stack(
                        [slice_qpsk(z) for z in full["x_eq"]], axis=0
                    )
                noise_var = full["meta"]["noise_var"]
                h_ref = turbo_refine_h(data_rx, xhat_grid, self.bins_pos, noise_var)
                h_ref_m = turbo_refine_h(
                    data_rx, np.conj(xhat_grid), self.bins_mir, noise_var
                )
                h_flat = np.zeros(self.N, dtype=complex)
                h_flat[self.bins_pos] = h_ref
                h_flat[self.bins_mir] = h_ref_m
                h_series2 = np.tile(h_flat, (K_use, 1))
                self.last_stage = "LDPC"
                info_blocks, _, x_eq, stats = self._decode_pass(
                    data_rx, h_series2, noise_var, full["meta"]["tone_scale"]
                )
                self.debug.update(stats)
                stream = b"".join(info_blocks)
                header, hdr_off = search_ph_in_stream(
                    stream, info_block=self.chunk_size // 2
                )
                h_start = full["meta"]["h_start"]
                h_end = full["meta"]["h_end"]

            payload = extract_payload(
                stream,
                header["file_size"],
                header_offset=hdr_off,
                header_len=self.chunk_size // 2,
            )

            path = write_recovered(out_dir, header["filename"], payload)
            self.last_stage = "done"
            self.debug["header"] = header
            self.debug["header_offset"] = hdr_off

            # 未指定 --source 时，按帧头文件名在 source/ 或 sourse/ 查找对照
            if not source_path:
                root = Path(__file__).resolve().parents[1]
                for folder in ("source", "sourse"):
                    cand = root / folder / Path(header["filename"]).name
                    if cand.is_file():
                        source_path = str(cand)
                        break

            file_match = None
            if source_path and Path(source_path).is_file():
                file_match = Path(source_path).read_bytes() == payload

            self._finalize_report(
                wav_path,
                out_dir,
                header=header,
                header_ok=True,
                header_offset=hdr_off,
                recovered_path=path,
                recovered_name=header["filename"],
                source_path=source_path,
                file_match=file_match,
                h_start=h_start,
                h_end=h_end,
                x_eq=x_eq,
                stream=stream,
            )
            return path

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if stream:
                (Path(out_dir) / "recovered.partial.bin").write_bytes(stream)
            self._finalize_report(
                wav_path,
                out_dir,
                header=self.debug.get("header"),
                header_ok=False,
                header_offset=self.debug.get("header_offset"),
                error=error,
                source_path=source_path,
                file_match=False if source_path else None,
                h_start=h_start,
                h_end=h_end,
                x_eq=x_eq,
                stream=stream if stream else None,
            )
            raise
