from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from n3.ldpc_codec import encode_block
from n3.modem import (
    ACTIVE_BINS,
    CODE_BYTES,
    FS,
    INFO_BYTES,
    L,
    SILENCE_SAMPLES,
    TRAINING_SYMBOLS,
    chirp_wave,
    coded_symbol_count,
    header_bytes,
    ofdm_tx,
    payload_symbol_count,
    training_symbols,
    write_wav,
)


def information_blocks(path: str | Path) -> tuple[list[bytes], dict]:
    source = Path(path)
    body = source.read_bytes()
    payload_count = payload_symbol_count(len(body))
    header = header_bytes(source.name, len(body), payload_count)
    combined = header + body
    padded_size = int(math.ceil(len(combined) / CODE_BYTES) * CODE_BYTES)
    padded = combined.ljust(padded_size, b"\0")
    blocks = [padded[offset:offset + INFO_BYTES] for offset in range(0, len(padded), INFO_BYTES)]
    return blocks, {
        "input_bytes": len(body),
        "payload_information_symbols": payload_count,
        "information_blocks": len(blocks),
        "coded_symbols": len(blocks),
        "padded_bytes": padded_size,
        "header": header,
    }


def build_frame(path: str | Path) -> tuple[np.ndarray, dict, np.ndarray]:
    source = Path(path)
    blocks, block_meta = information_blocks(source)
    coded_rows = np.asarray([encode_block(block) for block in blocks], dtype=np.uint8)
    qpsk_rows = np.empty((len(coded_rows), len(ACTIVE_BINS)), dtype=complex)
    for index, bits in enumerate(coded_rows):
        pairs = bits.reshape(-1, 2).astype(float)
        qpsk_rows[index] = (1 - 2 * pairs[:, 1]) + 1j * (1 - 2 * pairs[:, 0])

    training_freq, _ = training_symbols()
    leading_wave = ofdm_tx(training_freq[:TRAINING_SYMBOLS, ACTIVE_BINS])
    data_wave = ofdm_tx(qpsk_rows)
    trailing_wave = ofdm_tx(training_freq[TRAINING_SYMBOLS:, ACTIVE_BINS])
    region = np.concatenate((leading_wave, data_wave, trailing_wave))
    peak = float(np.max(region))
    if peak <= 0:
        raise ValueError("OFDM region has no positive peak")
    gain = 0.8 / peak
    region = gain * region
    chirp = chirp_wave()
    silence = np.zeros(SILENCE_SAMPLES)
    frame = np.concatenate((chirp, silence, region, silence, chirp))

    training_samples = TRAINING_SYMBOLS * L
    header_start = len(chirp) + len(silence) + training_samples
    payload_start = header_start + L
    tail_training_start = payload_start + max(0, len(blocks) - 1) * L
    gap_start = tail_training_start + TRAINING_SYMBOLS * L
    tail_chirp_start = len(frame) - len(chirp)
    meta = {
        "profile": "n3_example_tx",
        "input": str(source),
        "input_bytes": int(len(source.read_bytes())),
        "sample_rate": FS,
        "fft_size": 8192,
        "cyclic_prefix": 2048,
        "active_bin_start": int(ACTIVE_BINS[0]),
        "active_bin_end": int(ACTIVE_BINS[-1]),
        "active_subcarriers": int(len(ACTIVE_BINS)),
        "modulation": "qpsk",
        "ldpc_standard": "802.16",
        "ldpc_rate": "1/2",
        "ldpc_z": 166,
        "training_symbols": TRAINING_SYMBOLS,
        "trailing_training_symbols": TRAINING_SYMBOLS,
        "chirp_samples": len(chirp),
        "silence_samples": len(silence),
        "front_chirp_start_sample": 0,
        "training_start_sample": len(chirp) + len(silence),
        "header_start_sample": int(header_start),
        "payload_start_sample": int(payload_start),
        "tail_training_start_sample": int(tail_training_start),
        "gap_start_sample": int(gap_start),
        "tail_chirp_start_sample": int(tail_chirp_start),
        "total_samples": int(len(frame)),
        "seconds": float(len(frame) / FS),
        "normalization_gain": gain,
    }
    meta.update({key: value for key, value in block_meta.items() if key != "header"})
    return frame, meta, training_freq


def write_transmission(input_path: str | Path, output_path: str | Path) -> dict:
    frame, meta, training_freq = build_frame(input_path)
    output = Path(output_path)
    write_wav(output, frame)
    blocks, block_meta = information_blocks(input_path)
    np.save(output.with_suffix(".training.npy"), training_freq)
    output.with_suffix(".header.bin").write_bytes(block_meta["header"])
    meta["out"] = str(output)
    meta["training_sidecar"] = str(output.with_suffix(".training.npy"))
    meta["header_sidecar"] = str(output.with_suffix(".header.bin"))
    output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="write a Standardization/example_tx.py-compatible N3 WAV")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    options = args()
    if not options.input.exists():
        raise SystemExit(f"input file does not exist: {options.input}")
    meta = write_transmission(options.input, options.out)
    print(
        f"wrote {options.out} seconds={meta['seconds']:.3f} "
        f"active_bins={meta['active_subcarriers']} coded_symbols={meta['coded_symbols']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
