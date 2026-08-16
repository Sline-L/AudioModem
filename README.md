# AudioModem

AudioModem 是一个面向真实声学信道实验的 OFDM 文件传输项目。当前 `simple-main`
分支刻意回到最小协议，用来重新建立稳定基线。

AudioModem is an OFDM file-transfer project for real acoustic-channel tests.
The current `simple-main` branch is intentionally minimal so experiments can
restart from a clean baseline.

## Current Pipeline / 当前主线

```text
48 kHz, N=4096, CP=2048
active band 2 kHz to 7 kHz, bins 171-597
frame = leading silence + sync + preamble + data + trailing silence
payload mod = bpsk, qpsk, or qam16
file header = filename + byte length + whole-file CRC32
```

This branch does not use error-correction code, comb pilots, start anchors,
periodic anchors, or anchor-based sample-clock fitting.

本分支不使用纠错码、comb pilot、start anchor、周期 anchor 或基于 anchor 的采样时钟拟合。

## Quick Start / 快速开始

```bash
pip install -r requirements.txt
python tx_step8.py data/source/file16_test.txt --out data/simple_main/simple_main.wav
python rx_step8.py data/simple_main/simple_main.wav \
  --source data/source/file16_test.txt \
  --out runs/simple_main/offline
cmp data/source/file16_test.txt runs/simple_main/offline/file16_test.txt
```

## Layout / 目录

- `step8_modem.py`: simple-main WAV I/O, OFDM, modulation, packing and sync helpers.
- `tx_step8.py`: transmitter CLI for `silence + sync + preamble + data`.
- `rx_step8.py`: receiver CLI with sync, one preamble H estimate and direct demodulation.
- `data/source/`: source payload files.
- `data/simple_main/`: generated transmit WAVs and sidecars, ignored by Git.
- `runs/simple_main/`: generated receive/analysis outputs, ignored by Git.
- `docs/code_guide.md`: current command and output guide.
- `archive/legacy/`: old active-tree files moved out of the main layout.
- `archive/experiments/`, `archive/runs/`: earlier archived experiment snapshots and results.

## Documentation / 文档

- [Documentation index / 文档索引](docs/README.md)
- [Code guide / 代码指南](docs/code_guide.md)
