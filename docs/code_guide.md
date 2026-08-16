# Simple-Main Code Guide / Simple-Main 代码指南

This branch intentionally resets the runnable modem to the simplest useful
frame:

```text
leading silence + sync + preamble + data + trailing silence
```

No error-correction code, comb pilots, start anchors, periodic anchors, or
clock-fit anchors are used.

本分支把当前主线协议回退成最小可用结构：只有前置静音、同步头、preamble 和 data，外加尾部静音。
不使用纠错码、comb pilot、start anchor、周期 anchor 或 anchor-based clock fit。

## 1. Modules / 模块

### `step8_modem.py`

Self-contained simple-main modem implementation:

- mono 16-bit 48 kHz WAV I/O;
- OFDM with `N=4096`, `CP=2048`, symbol length `6144`;
- active band from 2 kHz to 7 kHz, implemented as bins `171..597`;
- deterministic random-QPSK sync and preamble symbols;
- BPSK, QPSK, or QAM16 payload modulation;
- one compact file header with filename, byte length and whole-file CRC32;
- one preamble channel estimate, followed by direct payload equalization.

### `tx_step8.py`

Reads one file and writes a transmit WAV plus deterministic sidecars:

```text
*.sync.npy
*.preamble.npy
*.meta.json
```

### `rx_step8.py`

Accepts one or more WAV recordings, finds the sync header, estimates channel
from the preamble, directly demodulates payload symbols, and writes:

```text
metrics.json
summary.csv
H.npy
payload_symbols.npy
recovered file, when CRC passes
decoded_payload.bin, when decode fails
```

## 2. Generate / 生成发送音频

Default small text file:

```bash
python tx_step8.py
```

Any file:

```bash
python tx_step8.py data/source/file15.txt --out data/simple_main/file15.wav
```

Important transmitter options:

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `input` | `data/source/file16_test.txt` | source file / 源文件 |
| `--noise-seconds` | `0.5` | leading silence |
| `--sync-symbols` | `16` | sync header OFDM symbols |
| `--sync-seed` | `2026` | sync seed |
| `--preamble-symbols` | `64` | channel-estimation preamble symbols |
| `--preamble-seed` | `3026` | preamble seed |
| `--mod` | `qpsk` | `bpsk`, `qpsk`, or `qam16` |
| `--tail-seconds` | `0.25` | trailing silence |
| `--out` | `data/simple_main/simple_main.wav` | transmit WAV path |

## 3. Decode / 解码

Offline loopback:

```bash
python rx_step8.py data/simple_main/simple_main.wav \
  --source data/source/file16_test.txt \
  --out runs/simple_main/offline
```

Recorded WAV:

```bash
python rx_step8.py data/rx/receive.wav \
  --source data/source/file16_test.txt \
  --out runs/simple_main/recording
```

Batch:

```bash
python rx_step8.py data/rx/r1.wav data/rx/r2.wav --out runs/simple_main/batch
```

Important receiver options:

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `--source` | none | optional truth file for exact match reporting |
| `--noise-seconds` | `0.5` | kept for metadata compatibility |
| `--sync-symbols` | `16` | must match transmitter |
| `--sync-seed` | `2026` | must match transmitter |
| `--preamble-symbols` | `64` | must match transmitter |
| `--preamble-seed` | `3026` | must match transmitter |
| `--mod` | `qpsk` | must match transmitter |
| `--out` | `runs/simple_main` | output directory |

## 4. Checks / 检查

Compile-check:

```bash
python -m py_compile step8_modem.py tx_step8.py rx_step8.py
```

Offline file loopback:

```bash
python tx_step8.py data/source/file16_test.txt --out data/simple_main/simple_main.wav
python rx_step8.py data/simple_main/simple_main.wav \
  --source data/source/file16_test.txt \
  --out runs/simple_main/offline
cmp data/source/file16_test.txt runs/simple_main/offline/file16_test.txt
```

For this branch, success is judged by:

```text
sync_score -> file_ok -> file_match
```
