# AudioModem

AudioModem 是一个面向真实声学信道实验的 OFDM 文件传输项目。当前 `n2`
版本基于 `n1` 的首尾 chirp、固定 training/header 和变长 payload，在 payload
链路上增加了可选 LDPC，并默认对 LDPC coded bits 做随机交织。

AudioModem is an OFDM file-transfer project for real acoustic-channel tests.
The current n2 version uses the n1 front/tail chirp, training/header, and
variable payload frame, with optional LDPC coding and coded-bit interleaving on
the payload.

## Current Pipeline / 当前主线

```text
500 ms front chirp, 1k-9k
30 ms silence guard
12 front training OFDM symbols
9 header OFDM symbols, 3 permuted copies x 3 symbols
variable payload OFDM symbols, determined by file length
optional 12 tail training OFDM symbols, enabled by --tail-training
50 ms inter-frame gap
500 ms tail chirp, 1k-9k
```

The OFDM data band remains:

```text
48 kHz, N=4096, CP=2048
active data band 2 kHz to 7 kHz, bins 171-597
payload mod = bpsk, qpsk, or qam16
optional payload LDPC with coded-bit interleaver, enabled by --ldpc
```

The receiver enumerates front/tail chirp pairs, refines the OFDM start with the
known front training waveform, and shortlists the strongest joint
candidates. It then searches SFO within `+/-5 ppm` at `0.25 ppm` resolution
around an in-range chirp estimate, or performs a `[-80, +80] ppm` coarse search
first when needed. Training consistency and the three header copies jointly
select the final pair and SFO before open-loop payload correction.

接收端枚举首尾 chirp pair，用已知前置 training 细化 OFDM 起点并筛选候选；
随后在粗 ppm 附近以 0.25 ppm 精调，并结合 training 一致性和三份 header 的可靠度
选择最终 pair/ppm。Header 使用三份 BPSK copy：原顺序、permutation A、permutation B。
如果发送和接收同时加 `--tail-training`，payload 后会额外插入同一组 training；
接收端假设 `H` 不随时间变化，把前后两段共 24 个 training symbols
一起全局平均来估计 `H[k]`。

## Quick Start / 快速开始

```bash
pip install -r requirements.txt
python tx_n1.py data/source/file16_test.txt --out data/n1/n1.wav
python rx_n1.py data/n1/n1.wav \
  --source data/source/file16_test.txt \
  --out runs/n1/offline
cmp data/source/file16_test.txt runs/n1/offline/file16_test.txt
```

N2 with LDPC:

```bash
python tx_n2.py data/source/file16_test.txt --ldpc --out data/n2/n2_ldpc_interleaved.wav
python rx_n2.py data/n2/n2_ldpc_interleaved.wav \
  --ldpc \
  --source data/source/file16_test.txt \
  --out runs/n2/ldpc_interleaved
cmp data/source/file16_test.txt runs/n2/ldpc_interleaved/file16_test.txt
```

## Layout / 目录

- `modem_n1.py`, `tx_n1.py`, `rx_n1.py`: archived active N1 baseline.
- `modem_n2.py`, `tx_n2.py`, `rx_n2.py`: N2 with optional payload LDPC and coded-bit interleaver.
- `archive/n1_before_n2_ldpc/`: snapshot of N1 before N2 LDPC work.
- `data/source/`: source payload files.
- `data/n1/`: generated transmit WAVs and sidecars, ignored by Git.
- `runs/n1/`: generated receive/analysis outputs, ignored by Git.
- `docs/code_guide.md`: current command and output guide.
- `archive/legacy/`: old active-tree files moved out of the main layout.
- `archive/experiments/`, `archive/runs/`: earlier archived experiment snapshots and results.

## Documentation / 文档

- [Documentation index / 文档索引](docs/README.md)
- [Code guide / 代码指南](docs/code_guide.md)
