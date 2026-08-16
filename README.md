# AudioModem

AudioModem 是一个面向真实声学信道实验的 OFDM 文件传输项目。当前 `n1`
版本使用首尾 chirp、固定 training/header 和变长 payload，作为无 pilot、无 FEC 的新基线。

AudioModem is an OFDM file-transfer project for real acoustic-channel tests.
The current n1 version uses front/tail chirps, fixed training/header symbols,
and variable-length payload as a no-pilot, no-FEC baseline.

## Current Pipeline / 当前主线

```text
150 ms front chirp, 1k-9k
30 ms silence guard
8 training OFDM symbols
9 header OFDM symbols, 3 repeated copies x 3 symbols
variable payload OFDM symbols, determined by file length
50 ms inter-frame gap
150 ms tail chirp, 1k-9k
```

The OFDM data band remains:

```text
48 kHz, N=4096, CP=2048
active data band 2 kHz to 7 kHz, bins 171-597
payload mod = bpsk, qpsk, or qam16
```

The receiver uses the two chirps to estimate sampling drift, then applies
open-loop linear phase correction to training, header and payload symbols.

接收端利用首尾 chirp 估计采样漂移，并对 training、header、payload 做开环线性相位修正。

## Quick Start / 快速开始

```bash
pip install -r requirements.txt
python tx_n1.py data/source/file16_test.txt --out data/n1/n1.wav
python rx_n1.py data/n1/n1.wav \
  --source data/source/file16_test.txt \
  --out runs/n1/offline
cmp data/source/file16_test.txt runs/n1/offline/file16_test.txt
```

## Layout / 目录

- `modem_n1.py`: N1 WAV I/O, chirp sync, OFDM, header, modulation and phase helpers.
- `tx_n1.py`: transmitter CLI for the chirp/training/header/payload/tail-chirp frame.
- `rx_n1.py`: receiver CLI with chirp SFO estimate, channel estimate and file recovery.
- `data/source/`: source payload files.
- `data/n1/`: generated transmit WAVs and sidecars, ignored by Git.
- `runs/n1/`: generated receive/analysis outputs, ignored by Git.
- `docs/code_guide.md`: current command and output guide.
- `archive/legacy/`: old active-tree files moved out of the main layout.
- `archive/experiments/`, `archive/runs/`: earlier archived experiment snapshots and results.

## Documentation / 文档

- [Documentation index / 文档索引](docs/README.md)
- [Code guide / 代码指南](docs/code_guide.md)
