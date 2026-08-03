# AudioModem

AudioModem 是一个面向真实声学信道实验的 OFDM 文件传输项目。当前主线是 Step8：在单次
payload 内加入周期 timing anchors，利用整段录音稳健估计采样时钟 PPM，并同时刷新信道
响应 H。

AudioModem is an OFDM file-transfer project for real acoustic-channel tests.
The current Step8 pipeline inserts periodic timing anchors into one payload,
fits sample-clock PPM across the recording, and refreshes H in-band.

## Current Pipeline / 当前主线

```text
48 kHz, N=512, CP=256
active bins 64-120 and 158-178
BPSK data + rotating QPSK comb pilots
rate-1/2 K=7 soft convolutional FEC
512-byte CRC32 blocks
one payload-start anchor x8, then one payload with timing anchor x8 every 128 logical symbols
```

Step8 real recordings currently recover the 12,936-byte TIFF exactly. The two
measured raw coded BER values are `6.1069%` and `5.5580%`; FEC reduces both to
`0%`, with `26/26` blocks and whole-file CRC passing.

当前两次真实录音都能精确恢复 12,936-byte TIFF。FEC 前 BER 分别为 `6.1069%` 和
`5.5580%`，FEC 后均为 `0%`，`26/26` blocks 和整文件 CRC 全部通过。

## Quick Start / 快速开始

```bash
pip install -r requirements.txt
python tx_step8.py
python rx_step8.py \
  data/step8_clock_anchor/observatory_64_uncompressed_step8.wav \
  --source data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --phase-slope off \
  --out runs/step8_clock_anchor/offline
```

Real recording / 真实录音：

```bash
pw-record --rate 48000 --channels 1 --format s16 --sample-count 3552000 \
  data/step8_clock_anchor/receive_observatory_step8_3.wav
python rx_step8.py \
  data/step8_clock_anchor/receive_observatory_step8_3.wav \
  --source data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --payload-start-anchor-symbols 0 \
  --phase-slope off \
  --out runs/step8_clock_anchor/3
```

## Layout / 目录

- `step8_modem.py`: self-contained PHY, framing, FEC, pilots, timing and H logic.
- `tx_step8.py`: Step8 WAV and metadata generator.
- `rx_step8.py`: synchronization, decoding, metrics and plots.
- `data/step8_clock_anchor/`: current source, transmit WAV and recordings.
- `runs/step8_clock_anchor/`: current generated analysis.
- `archive/experiments/`: runnable Step1-Step7 code snapshots.
- `archive/runs/`: local pre-Step8 results, ignored by Git.
- `docs/`: protocol, code guide, technical route, history and TUI design.

Step8 已不再导入 Step7 或旧版 `audiomodem.py`。历史代码及其依赖保存在
[`archive/README.md`](archive/README.md)。

## Documentation / 文档

- [Documentation index / 文档索引](docs/README.md)
- [Step8 complete protocol / Step8 完整协议](docs/step8_clock_anchor.md)
- [Step8 end-to-end walkthrough / Step8 端到端工程详解](docs/step8_end_to_end_walkthrough.md)
- [Code guide / 代码指南](docs/code_guide.md)
- [Technical route / 技术路线](docs/technical_route.md)
- [Experiment history / 实验历史](docs/experiment_history.md)
- [TUI feasibility / TUI 可行性](docs/tui_feasibility.md)
