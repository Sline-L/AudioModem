from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from n3.modem import ACTIVE_BINS, CP, L, N, SILENCE_SAMPLES, TRAINING_SYMBOLS, CHIRP_SAMPLES, write_wav
from n3.tx import build_frame


def inject_payload_bit_errors(
    frame: np.ndarray,
    coded_symbols: int,
    ber: float,
    seed: int = 20260913,
) -> tuple[np.ndarray, dict]:
    """Flip a deterministic fraction of payload QPSK bits in an N3 frame.

    The Header and all training symbols remain untouched. Errors are injected
    after LDPC coding and scrambling, before QPSK demodulation.
    """
    if not 0.0 <= float(ber) <= 1.0:
        raise ValueError("ber must be between 0 and 1")
    if int(coded_symbols) < 2:
        raise ValueError("the frame must contain a Header and at least one payload symbol")

    values = np.asarray(frame, dtype=float).copy()
    data_start = CHIRP_SAMPLES + SILENCE_SAMPLES + TRAINING_SYMBOLS * L
    data_stop = data_start + int(coded_symbols) * L
    if data_stop > len(values):
        raise ValueError("frame does not contain the requested data symbols")

    rows = values[data_start:data_stop].reshape(int(coded_symbols), L)
    freq = np.fft.fft(rows[:, CP:], axis=1)
    payload = freq[1:, ACTIVE_BINS]
    total_bits = int(payload.size * 2)
    error_count = int(round(float(ber) * total_bits))
    rng = np.random.default_rng(seed)
    selected = rng.choice(total_bits, size=error_count, replace=False)

    for bit_index in selected:
        symbol_index, component = divmod(int(bit_index), 2)
        row_index, carrier_index = divmod(symbol_index, len(ACTIVE_BINS))
        value = payload[row_index, carrier_index]
        imag_bit = int(value.imag < 0.0)
        real_bit = int(value.real < 0.0)
        if component == 0:
            imag_bit ^= 1
        else:
            real_bit ^= 1
        amplitude = max((abs(float(value.real)) + abs(float(value.imag))) / 2.0, 1e-12)
        replacement = amplitude * ((1 - 2 * real_bit) + 1j * (1 - 2 * imag_bit))
        absolute_bin = int(ACTIVE_BINS[carrier_index])
        freq[row_index + 1, absolute_bin] = replacement
        freq[row_index + 1, N - absolute_bin] = np.conj(replacement)

    modified_rows = np.fft.ifft(freq, axis=1).real
    modified_with_cp = np.hstack((modified_rows[:, -CP:], modified_rows)).ravel()
    values[data_start:data_stop] = modified_with_cp
    metadata = {
        "requested_ber": float(ber),
        "injected_bit_errors": error_count,
        "payload_coded_bits": total_bits,
        "actual_ber": float(error_count / total_bits) if total_bits else 0.0,
        "seed": int(seed),
        "header_untouched": True,
        "training_untouched": True,
    }
    return values, metadata


def generate(input_path: str | Path, output_path: str | Path, ber: float, seed: int) -> dict:
    frame, meta, _ = build_frame(input_path)
    modified, injection = inject_payload_bit_errors(frame, meta["coded_symbols"], ber, seed)
    output = Path(output_path)
    write_wav(output, modified)
    sidecar = {
        "profile": "n3_example_tx",
        "input": str(Path(input_path)),
        "out": str(output),
        "coded_symbols": int(meta["coded_symbols"]),
        **injection,
    }
    output.with_suffix(".ber.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description="inject controlled payload BER into an N3 WAV")
    parser.add_argument("input", type=Path)
    parser.add_argument("--ber", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    metadata = generate(options.input, options.out, options.ber, options.seed)
    print(
        f"wrote {options.out}: injected_ber={metadata['actual_ber']:.6f} "
        f"errors={metadata['injected_bit_errors']}/{metadata['payload_coded_bits']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
