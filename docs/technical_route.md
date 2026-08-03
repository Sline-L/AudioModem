# Technical Route / 技术路线

## Goal / 目标

The project sends arbitrary files through a real speaker-room-microphone path
using 48 kHz OFDM. The design goal is exact file recovery verified by CRC, not
merely a recognizable image or a low average constellation error.

本项目使用 48 kHz OFDM 通过真实扬声器、房间和麦克风传输任意文件。目标是经 CRC 验证的
精确恢复，而不只是得到一张大致可辨认的图片。

## Evolution / 演进

### Step1-Step3: H and broad sweeps / H 与宽频扫频

Early experiments measured `H=Y/X`, compared BPSK/QPSK/QAM16 and swept broad
frequency bands. They established that BPSK was the realistic modulation for
the current hardware and that H must be measured in the same playback.

早期实验测量 `H=Y/X`，比较多种调制和宽频段，确认当前硬件应优先使用 BPSK，并且 H 应在
同一次播放内测量。

### Step4-Step6: frequency selection / 频点筛选

Bandstep and 1.5-second single-bin sweeps found narrow bad points and produced
non-contiguous masks. Step4 once reached roughly 2%-3% BER, but changing room or
recording position invalidated the mask. Sweeps remain useful for identifying
hardware-wide unusable ranges, not as permanent room calibration.

Bandstep 和逐点 1.5 秒扫频发现了窄带坏点并形成非连续频点。Step4 一度达到约 2%-3%
BER，但换位置后 mask 立即失效。因此扫频适合排除设备长期不可用频段，不适合作为永久房间
标定。

### Step7: coding and in-frame adaptation / 编码与帧内适应

Step7 moved to N=512/CP=256, rotating pilots, same-frame training, soft
rate-1/2 convolutional FEC and independent CRC blocks. It proved that FEC could
correct frequency-selective raw errors, but a biased PPM estimate from the short
opening training accumulated 27.6 samples over a long TIFF payload.

Step7 使用 N=512/CP=256、旋转 pilot、同帧训练、软卷积码和独立 CRC blocks。它证明 FEC
可以修复频率选择性错误，但开头短训练的 PPM 偏差会在长 TIFF 中累计 27.6 samples。

### Step8: whole-recording timing anchors / 全段时钟锚点

Step8 inserts eight known QPSK rows every 128 logical payload symbols. Anchors
provide absolute positions for robust PPM fitting and periodic full-band H
refresh while the file payload remains single-copy.

Step8 每 128 个逻辑 payload symbols 插入 8 个已知 QPSK rows，同时提供全段 PPM 拟合和
周期全频 H 刷新，文件 payload 仍只发送一次。

Two real recordings now recover the TIFF exactly:

```text
recording 1   raw BER 6.1069%, post-FEC 0%, 26/26, exact
recording 2   raw BER 5.5580%, post-FEC 0%, 26/26, exact
```

The second recording also survived intentional late noise. The current default
is therefore:

```text
anchor H alpha 0.5
per-symbol CPE on
phase slope off
soft FEC and CRC on
```

## Current Data Path / 当前数据流

```text
file
 -> CRC header and 512-byte CRC blocks
 -> independent interleaving and K=7 rate-1/2 convolutional FEC
 -> BPSK data + rotating QPSK comb pilots
 -> periodic full-band QPSK timing anchors
 -> real OFDM waveform
 -> acoustic channel
 -> anchor-based resampling and same-frame H
 -> CPE/H tracking and soft LLR
 -> Viterbi, block CRC and whole-file CRC
 -> exact file or explicit partial result
```

## Remaining Priorities / 后续重点

1. Bound payload processing after header decode so trailing silence cannot create
   diagnostic-only fake anchors.
2. Improve multipath-consistent anchor peak tracking without using source bytes.
3. Measure FEC margin with controlled noise, distance and device changes rather
   than deleting bins after each room change.
4. Compare stronger block codes only after establishing repeatable Step8 test
   fixtures and rate/latency targets.
5. Add a TUI for reproducible experiment operation; it should orchestrate the
   modem, not duplicate DSP logic.

1. Header 解码后限制物理 payload 长度，消除尾部静音产生的诊断假 anchor。
2. 在不使用源文件的前提下，提高多径环境中 anchor 峰路径的一致性。
3. 用可控噪声、距离和设备变化评估 FEC 余量，不再每换房间就重新删 bins。
4. 在 Step8 测试夹具和速率目标稳定后，再比较更强 block code。
5. 增加保证实验可复现的 TUI，但界面只编排任务，不复制 DSP。

Full protocol / 完整协议：[`step8_clock_anchor.md`](step8_clock_anchor.md).
