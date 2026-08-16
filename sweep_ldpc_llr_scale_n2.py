from pathlib import Path
import argparse
import csv
import contextlib
import io
import types

import modem_n2
import rx_n2


DEFAULT_START = 0.4
DEFAULT_STOP = 4.0
DEFAULT_STEP = 0.2


def args():
    p = argparse.ArgumentParser(
        description="sweep N2 LDPC_LLR_SCALE and write BER-only CSV diagnostics"
    )
    p.add_argument("input", nargs="+", type=Path, help="recorded wav file(s) to decode")
    p.add_argument("--source", type=Path, required=True, help="original source file for BER calculation")
    p.add_argument("--out", type=Path, default=Path("runs/n2_llr_sweep"))
    p.add_argument("--csv", type=Path, default=None, help="summary CSV path")
    p.add_argument("--start", type=float, default=DEFAULT_START)
    p.add_argument("--stop", type=float, default=DEFAULT_STOP)
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
    p.add_argument("--training-seed", type=int, default=3026)
    p.add_argument("--tail-training", action="store_true")
    p.add_argument("--no-ldpc", action="store_true", help="decode without LDPC for comparison")
    p.add_argument("--tail-search-seconds", type=float, default=0.5)
    return p.parse_args()


def scale_values(start, stop, step):
    if step <= 0:
        raise SystemExit("--step must be > 0")
    count = int(round((stop - start) / step))
    if count < 0:
        raise SystemExit("--stop must be >= --start")
    return [round(start + index * step, 10) for index in range(count + 1)]


def safe_name(path):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)


def output_dir(base, receive, scale, many):
    scale_name = f"llr_{scale:.1f}".replace(".", "p")
    if many:
        return base / safe_name(receive) / scale_name
    return base / scale_name


def metrics_row(receive, scale, metrics):
    return {
        "input": str(receive),
        "ldpc_llr_scale": f"{scale:.1f}",
        "ber_overall_useful": percent(metrics.get("ber_overall_useful_ber")),
        "ber_payload": percent(metrics.get("ber_payload_ber")),
        "ber_ldpc_off": percent(metrics.get("ber_ldpc_off")),
        "ber_ldpc_off_deinterleaved": percent(metrics.get("ber_ldpc_off_deinterleaved")),
        "ber_ldpc_on": percent(metrics.get("ber_ldpc_on")),
    }


def percent(value):
    if value is None or value == "":
        return ""
    return f"{float(value) * 100:.2f}%"


def main():
    a = args()
    rx_args = types.SimpleNamespace(
        input=a.input,
        source=a.source,
        training_seed=a.training_seed,
        tail_training=a.tail_training,
        ldpc=not a.no_ldpc,
        tail_search_seconds=a.tail_search_seconds,
        out=a.out,
    )
    rx_n2.validate(rx_args)
    training = rx_n2.training_symbols(a.training_seed)
    chirp = rx_n2.chirp_wave()
    scales = scale_values(a.start, a.stop, a.step)
    many = len(a.input) > 1
    rows = []

    for scale in scales:
        modem_n2.LDPC_LLR_SCALE = float(scale)
        for receive in a.input:
            run_dir = output_dir(a.out, receive, scale, many)
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = rx_n2.run_one(receive, run_dir, rx_args, training, chirp)
            rows.append(metrics_row(receive, scale, metrics))

    csv_path = a.csv if a.csv is not None else a.out / "ber_ldpc_llr_scale_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input",
        "ldpc_llr_scale",
        "ber_overall_useful",
        "ber_payload",
        "ber_ldpc_off",
        "ber_ldpc_off_deinterleaved",
        "ber_ldpc_on",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
