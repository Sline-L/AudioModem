#!/usr/bin/env python3
"""Compact Week 2 OFDM/QPSK recovery pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import wave

import numpy as np


FFT_SIZE = 1024
CP_LEN = 32
SYMBOL_LEN = FFT_SIZE + CP_LEN
DATA_BINS = np.arange(1, 512)
INT16_SCALE = 32768.0

'''
如果传入的是单个文件，就返回这个文件
如果传入的是目录，就找目录下所有 .wav 文件并排序
默认 ./data/*.wav
'''
def t01_wav_files(input_path: Path) -> list[Path]:
    return [input_path] if input_path.is_file() else sorted(input_path.glob("*.wav"))

'''
读取 channel.csv，并把时域 channel 转成频域响应
'''
def t02_channel_response(channel_path: Path) -> np.ndarray:
    taps = np.loadtxt(channel_path, dtype=np.float64)
    if taps.ndim != 1 or taps.size == 0:
        raise ValueError(f"bad channel file: {channel_path}")
    response = np.fft.fft(taps, FFT_SIZE)[DATA_BINS]
    if np.min(np.abs(response)) < 1e-8:
        raise ValueError("channel response is too close to zero")
    return response

'''
读取 WAV 音频样本，并归一化
'''
def t03_read_wav_samples(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != "NONE":
            raise ValueError(f"unsupported WAV format: {wav_path}")
        raw = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size < SYMBOL_LEN:
        raise ValueError(f"too few samples: {wav_path}")
    return samples.astype(np.float64) / INT16_SCALE


'''
将星座图点转换为字节
'''
def t04_constellation_to_bytes(values: np.ndarray, mapping: str) -> bytes:
    if mapping == "main":
        bits = np.column_stack((values.imag.ravel() < 0, values.real.ravel() < 0))
    elif mapping == "alt":
        bits = np.column_stack((values.real.ravel() < 0, values.imag.ravel() < 0))
    else:
        raise ValueError(f"unknown mapping: {mapping}")
    flat_bits = bits.astype(np.uint8).ravel()
    return np.packbits(flat_bits[: flat_bits.size // 8 * 8], bitorder="big").tobytes()

'''
从 bytes 里读取一个以 \0 结尾的字符串
'''
def t05_read_c_string(data: bytes, start: int) -> tuple[str, int]:
    end = data.find(b"\0", start)
    if end < 0:
        raise ValueError("missing header terminator")
    return data[start:end].decode("utf-8"), end + 1

'''
解析完整数据流，提取文件名和文件内容
'''
def t06_extract_payload(data: bytes) -> tuple[str, bytes, int]:
    filename, offset = t05_read_c_string(data, 0)
    size_text, offset = t05_read_c_string(data, offset)
    if not filename:
        raise ValueError("empty header filename")
    size = int(size_text)
    payload = data[offset : offset + size]
    if len(payload) != size:
        raise ValueError("payload is shorter than header size")
    return filename, payload, offset

'''
处理命令行参数
'''
def f01_parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Recover files from OFDM WAV inputs.")
    parser.add_argument("input", nargs="?", type=Path, default=here / "data")
    parser.add_argument("-c", "--channel", type=Path, default=here / "channel.csv")
    parser.add_argument("-o", "--output-dir", type=Path, default=here / "recovered")
    parser.add_argument("-m", "--mapping", choices=["auto", "main", "alt"], default="auto")
    return parser.parse_args()

'''
恢复单个 WAV 里的文件
'''
def f02_recover_one(wav_path: Path, channel_response: np.ndarray, mapping: str) -> tuple[str, Path, int, str]:
    #Step 1: Read 16-bit PCM samples from WAV and normalize them to float values.
    normalized = t03_read_wav_samples(wav_path)

    #Step 2: Split samples into complete OFDM symbols and remove the 32-sample CP.
    symbol_count = normalized.size // SYMBOL_LEN
    symbols = normalized[: symbol_count * SYMBOL_LEN].reshape(symbol_count, SYMBOL_LEN)
    no_cp = symbols[:, CP_LEN:]

    #Step 3: Run a 1024-point FFT for each symbol and keep data bins 1..511.
    active_bins = np.fft.fft(no_cp, axis=1)[:, DATA_BINS]

    #Step 4: Equalize each active bin by dividing by the channel frequency response.
    constellation = active_bins / channel_response

    #Step 5: Decode QPSK quadrants into bytes; auto tries both supported mappings.
    errors: list[str] = []
    for selected_mapping in (["main", "alt"] if mapping == "auto" else [mapping]):
        try:
            data = t04_constellation_to_bytes(constellation, selected_mapping)

            #Step 6: Parse the filename\0size\0 header and slice out the payload.
            filename, payload, _header_bytes = t06_extract_payload(data)

            #Step 7: Return the recovered payload and the mapping that worked.
            return Path(filename).name, payload, len(payload), selected_mapping
        except (UnicodeDecodeError, ValueError) as exc:
            errors.append(f"{selected_mapping}: {exc}")
    raise ValueError("; ".join(errors))

'''
写恢复结果汇总表
'''
def f03_write_summary(rows: list[tuple[str, str, int, str]], output_dir: Path) -> None:
    with (output_dir / "recovered_files.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_wav", "output_file", "size_bytes", "mapping"])
        writer.writerows(rows)

'''
主程序调度函数
'''
def f04_run() -> None:
    args = f01_parse_args()
    try:
        wav_files = t01_wav_files(args.input)
        if not wav_files:
            raise ValueError(f"no WAV files found: {args.input}")
        channel_response = t02_channel_response(args.channel)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for wav_path in wav_files:
            name, payload, size, used_mapping = f02_recover_one(wav_path, channel_response, args.mapping)
            output_path = args.output_dir / name
            output_path.write_bytes(payload)
            rows.append((str(wav_path), str(output_path), size, used_mapping))

        f03_write_summary(rows, args.output_dir)
        print(f"Recovered {len(rows)} file(s) to {args.output_dir}")
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    f04_run()
