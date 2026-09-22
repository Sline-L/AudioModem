"""与发射端 standard/example_tx.py 对齐的物理层常量。"""

TX_DEFAULTS = {
    "CP": 2048,
    "N": 8192,
    "start_index": 400,
    "chunk_size": 498,
    "preamable_count": 8,
    "ldpc_en": True,
    "start_freq": 100.0,
    "end_freq": 20000.0,
    "chirp_duration": 3.0,
    "amp": 0.8,
    "sample_rate": 48000,
    "silence_duration": 0.5,
}

# 训练符号必须使用与发射端相同的 CPython random 种子
TRAIN_SEED = 80
SCRAMBLER_SEED = 0x5A4D
HEADER_MAGIC = b"PH"


def derived(params=None):
    """由基本参数推出的派生量。"""
    p = dict(TX_DEFAULTS if params is None else params)
    n = int(p["N"])
    cp = int(p["CP"])
    chunk = int(p["chunk_size"])
    start = int(p["start_index"])
    fs = int(p["sample_rate"])
    return {
        **p,
        "symbol_len": n + cp,
        "n_active": 4 * chunk,
        "end_index": start + 4 * chunk,
        "info_bytes": chunk // 2,
        "ldpc_z": chunk // 3,
        "coded_bits": chunk * 8,
        "info_bits": (chunk // 2) * 8,
        "chirp_len": int(fs * p["chirp_duration"]),
        "silence_len": int(fs * p["silence_duration"]),
        "preamble_count": int(p["preamable_count"]),
    }
