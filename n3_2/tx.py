"""Strict standalone transmitter for the N3.2 standard acoustic frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from n3_2.ldpc_codec import StandardLdpc
from n3_2.modem import (
    CP,
    FS,
    K0,
    K1,
    L,
    M,
    N,
    header_bytes,
    linear_chirp,
    ofdm_time,
    qpsk_map,
    scramble_bits,
    training_symbols,
    write_pcm16_wav,
)


INFO_BYTES = 249
CODE_BYTES = 498
TRAINING_PER_EDGE = 8


def build_frame(source: bytes, name: str) -> tuple[np.ndarray, bytes, dict]:
    """Build the exact standard frame for *source* and its transmitted name."""
    source = bytes(source)
    payload_symbols = math.ceil(len(source) / INFO_BYTES)
    header = header_bytes(name, len(source), payload_symbols)
    stream = header + source
    stream += bytes((-len(stream)) % CODE_BYTES)
    info_blocks = np.frombuffer(stream, dtype=np.uint8).reshape(-1, INFO_BYTES)

    codec = StandardLdpc()
    carriers = np.empty((len(info_blocks), M), dtype=np.complex128)
    for index, block in enumerate(info_blocks):
        info_bits = np.unpackbits(block, bitorder="big")
        coded_bits = codec.encode(info_bits)
        carriers[index] = qpsk_map(scramble_bits(coded_bits))

    training = training_symbols()
    training_carriers = training[:, K0:K1 + 1]
    chirp = linear_chirp()
    silence = np.zeros(FS // 2, dtype=float)
    front_training = ofdm_time(training_carriers[:TRAINING_PER_EDGE]).ravel()
    payload = ofdm_time(carriers).ravel()
    tail_training = ofdm_time(training_carriers[TRAINING_PER_EDGE:]).ravel()
    frame = np.concatenate(
        (chirp, silence, front_training, payload, tail_training, silence, chirp)
    )

    meta = {
        "name": str(name),
        "input_bytes": len(source),
        "payload_symbols": payload_symbols,
        "information_blocks": len(info_blocks),
        "coded_symbols": len(info_blocks),
        "sample_rate": FS,
        "fft_size": N,
        "cyclic_prefix": CP,
        "ofdm_symbol_samples": L,
        "active_bin_start": K0,
        "active_bin_end": K1,
        "active_subcarriers": M,
        "modulation": "qpsk",
        "ldpc_standard": "802.16",
        "ldpc_rate": "1/2",
        "ldpc_z": 166,
        "training_symbols": TRAINING_PER_EDGE,
        "trailing_training_symbols": TRAINING_PER_EDGE,
        "chirp_samples": len(chirp),
        "silence_samples": len(silence),
        "total_samples": len(frame),
        "seconds": len(frame) / FS,
    }
    return frame, header, meta


def run_tx(source: Path, out: Path | None = None) -> Path:
    """Encode *source* to a PCM16 standard frame and its sidecar files."""
    source = Path(source)
    output = Path(out) if out is not None else Path("data/n3_2") / f"{source.stem}.wav"
    frame, header, meta = build_frame(source.read_bytes(), source.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.rint(frame * 32767), -32768, 32767).astype(np.int16)
    write_pcm16_wav(output, pcm)

    training = training_symbols()
    np.save(output.with_suffix(".training.npy"), training)
    output.with_suffix(".header.bin").write_bytes(header)
    meta.update(
        {
            "out": str(output),
            "training_sidecar": str(output.with_suffix(".training.npy")),
            "header_sidecar": str(output.with_suffix(".header.bin")),
        }
    )
    output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return output


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="write a strict N3.2 standard WAV")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    options = _args()
    if not options.input.is_file():
        raise SystemExit(f"input file does not exist: {options.input}")
    output = run_tx(options.input, options.out)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
