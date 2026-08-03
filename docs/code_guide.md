# Step8 Code Guide / Step8 代码指南

This guide covers the current runnable pipeline. Step1-Step7 commands are kept
with their code under `archive/experiments/`.

本文只说明当前可运行的 Step8 主线。Step1-Step7 的命令随代码保存在
`archive/experiments/`。

## 1. Modules / 模块

### `step8_modem.py`

Self-contained protocol implementation / 自包含协议实现：

- mono 16-bit 48 kHz WAV I/O;
- N=512, CP=256 OFDM modulation and demodulation;
- file header, 512-byte blocks and CRC32;
- rate-1/2 K=7 convolutional encoding and soft Viterbi decoding;
- deterministic interleaving, BPSK data and rotating comb pilots;
- sync/preamble generation and H/noise estimation;
- payload timing-anchor framing and robust whole-recording PPM fitting;
- periodic H refresh, per-symbol CPE and optional slow phase slope.

它不再导入 Step7 或旧版 `audiomodem.py`。

### `tx_step8.py`

Reads one file and writes a transmit WAV plus deterministic sidecars. The
payload is sent once; anchors are known training symbols, not repeated file
data.

读取一个源文件并生成发送 WAV 与确定性 sidecars。Payload 只发送一次；anchor 是已知训练
符号，不是文件重复。

### `rx_step8.py`

Accepts one or more WAV recordings, estimates sync/H/clock, produces soft LLRs,
decodes FEC and CRC blocks, and writes metrics, arrays, plots and the recovered
file.

可一次分析一份或多份录音，完成同步、H、时钟、软判决、FEC、CRC 和文件恢复。

## 2. Generate / 生成发送音频

Default TIFF / 默认 TIFF：

```bash
python tx_step8.py
```

Any file / 任意文件：

```bash
python tx_step8.py data/source/example.bin \
  --out data/step8_clock_anchor/example_step8.wav
```

Important transmitter options / 主要发送参数：

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `input` | Step8 TIFF | source file / 源文件 |
| `--noise-seconds` | `0.5` | leading silence used for noise measurement |
| `--sync-symbols` | `64` | initial random-QPSK sync symbols |
| `--preamble-symbols` | `128` | symbols per H-training preamble |
| `--preamble-repeats` | `2` | independent preamble blocks |
| `--block-size` | `512` | CRC-protected file bytes per block |
| `--header-repeats` | `3` | independently interleaved header copies |
| `--timing-anchor-interval` | `128` | logical payload symbols between anchors |
| `--timing-anchor-symbols` | `8` | QPSK symbols in each anchor |
| `--tail-seconds` | `0.25` | trailing silence |
| `--out` | Step8 WAV | transmit WAV path |

Generated sidecars / 生成的 sidecars：

```text
*.meta.json          full profile, frame sizes, seeds and duration
*.sync.npy           known sync symbols
*.preamble.npy       known H-training symbols
*.anchor_starts.npy  physical payload-symbol offsets
*.anchors.npy        all known payload anchor symbols
```

## 3. Record / 录音

The default TIFF WAV is `70.526 s`. A 74-second recording leaves margins before
and after playback:

默认 TIFF WAV 长 `70.526 s`，建议录制 74 秒：

```bash
pw-record --rate 48000 --channels 1 --format s16 --sample-count 3552000 \
  data/step8_clock_anchor/receive_observatory_step8_3.wav
```

Start recording, wait about one second, play the Step8 WAV once, wait another
second, and stop. Avoid touching volume controls while it plays.

开始录音后等待约一秒，只播放一次 Step8 WAV，播完再等待约一秒。播放期间不要调整音量。

## 4. Decode / 解码

Offline / 离线：

```bash
python rx_step8.py data/step8_clock_anchor/observatory_64_uncompressed_step8.wav \
  --source data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --phase-slope off \
  --out runs/step8_clock_anchor/offline
```

Real recording / 真实录音：

```bash
python rx_step8.py data/step8_clock_anchor/receive_observatory_step8_3.wav \
  --source data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --phase-slope off \
  --out runs/step8_clock_anchor/3
```

Batch / 批量：

```bash
python rx_step8.py \
  data/step8_clock_anchor/receive_observatory_step8_1.wav \
  data/step8_clock_anchor/receive_observatory_step8_2.wav \
  --source data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --out runs/step8_clock_anchor/batch
```

Important receiver options / 主要接收参数：

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `--source` | Step8 TIFF | optional truth, used only for BER/file comparison |
| `--channel-alpha` | `0.35` | comb-pilot H update strength |
| `--anchor-h-alpha` | `0.5` | full-band anchor H fusion; `0` disables it |
| `--anchor-min-score` | `0.12` | minimum normalized anchor correlation |
| `--clock-search` | `128` | local anchor search radius in samples |
| `--payload-search` | `16` | payload-start fine search radius |
| `--phase-slope` | `off` | `off` or controlled experiment `slow` |
| `--slope-window` | `64` | robust history length in slow mode |
| `--slope-clip` | `0.05` | maximum applied rad/bin slope |
| `--out` | Step8 runs | output directory |

`--source` never selects synchronization, PPM, H or slope candidates. It is
only used after decoding to report BER and exact file match.

`--source` 不参与同步、PPM、H 或 slope 候选选择，只用于事后 BER 和文件一致性比较。

## 5. Outputs / 输出

| File | Content / 内容 |
|---|---|
| `metrics.json` | authoritative sync, PPM, BER, CRC and mode status |
| `blocks.csv` | CRC result and error text for each file block |
| `summary.csv` | per-bin H, noise and LLR reliability |
| recovered file | exact filename only when whole-file CRC passes |
| `*.partial` | best-effort bytes when the complete file CRC fails |
| `decoded_header.bin` | raw decoded 128-byte header |
| `clock_anchors.npy` | nominal/observed positions, score, residual and acceptance |
| `clock_fit.png` | whole-recording timing fit and residuals |
| `H.npy` | initial sync+preamble channel estimate |
| `H_training_blocks.npy` | phase-aligned training H estimates |
| `H_anchor_track.npy` | periodic full-active-band anchor H estimates |
| `pilot_H_track.npy` | per-logical-symbol tracked channel |
| `rx_llr.npy` | soft coded-bit decisions |
| `cpe.npy` | per-symbol common phase correction |
| `phase_slope*.npy` | measured and applied optional slope |
| `channel_and_noise.png` | initial H and noise powers |
| `phase_tracking.png` | CPE, slope and pilot residual over time |

Accuracy should be judged in this order / 准确性判断顺序：

```text
header_ok -> blocks_ok/blocks_total -> file_crc_ok -> file_match
```

Raw coded BER describes channel difficulty. Post-FEC BER and CRC determine
whether the recovered file is valid.

FEC 前 BER 描述信道难度；FEC 后 BER 和 CRC 才决定恢复文件是否有效。

## 6. Archive / 历史代码

```text
archive/experiments/steps1_to_5/
archive/experiments/step6/
archive/experiments/step7/
archive/runs/
```

Run archived scripts from the repository root. Their recordings remain under
the original `data/step*` directories. See `archive/README.md` for the map.

请从项目根目录运行归档脚本；对应录音仍在原 `data/step*` 目录。映射关系见
`archive/README.md`。
