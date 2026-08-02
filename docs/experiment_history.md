# AudioModem Experiment History / AudioModem 实验记录

This document records the acoustic OFDM file-transfer experiments so far. It
keeps the reasoning trail: what was tried, what failed, what improved, and why
the current design changed step by step.

本文记录目前为止的声学 OFDM 文件传输实验路线。重点不是罗列所有文件，而是保留
决策脉络：试了什么、哪里失败、哪里变好、为什么继续换方案。

## Current Conclusion / 当前结论

The best direction so far is selective frequency use, not a full continuous
wide band. The real acoustic channel contains narrow bad frequency points, and a
few bad bins can dominate the final BER.

目前最有希望的方向不是使用整段连续宽频带，而是选择性使用频点。真实声学信道里
存在很窄的坏频点，少数坏 bin 就能把整体 BER 拉高。

Key observations / 关键观察：

- Estimating `H` from a separate probe recording is not reliable enough. Probe
  and file payload should be in the same recording.
- 单独录 probe 再用这个 `H` 去解另一次文件录音不够可靠。probe 和 payload 必须放在
  同一段录音里。
- BPSK is much more realistic than QPSK/QAM16 on the current speaker/microphone
  path.
- 当前扬声器/麦克风链路下，BPSK 明显比 QPSK/QAM16 更现实。
- Per-symbol comb pilots are more useful than sparse whole-symbol pilots for
  tracking phase and frequency-dependent channel drift.
- 每个 payload OFDM symbol 内插 comb pilot，比隔几个 data symbol 放一个整符号 pilot
  更适合跟踪相位和频率选择性变化。
- Payload repetition helps, but it does not solve bad bins by itself.
- payload 重复发送能降 BER，但不能单独解决坏频点问题。
- Step 4 trimmed fband was the first major improvement.
- Step 4 的 trimmed fband 是第一次明显有效的频段优化。
- In the latest tests, `N=512, CP=256` with Step4-mapped bins performed better
  than `N=1024, CP=256` with raw Step4 bins.
- 最新测试中，`N=512, CP=256` 的 Step4 映射频段优于 `N=1024, CP=256` 的原始 Step4
  频段。

Representative results / 代表性结果：

| Experiment / 实验 | Structure / 结构 | Result / 结果 |
| --- | --- | --- |
| Step2 QPSK combo `8-150` | probe + sync32 + QPSK payload | about `49%` BER, failed / 约 `49%` BER，失败 |
| Step2 BPSK pilot4 `8-150` | preamble128, BPSK, pilot every 4 data symbols | best about `4.7%` BER, failed / 最好约 `4.7%` BER，仍失败 |
| Step2 BPSK pilot2 repeat3 `8-150` | pilot length 2, payload repeated 3 times | about `4.0%` BER, header sometimes OK / 约 `4.0%` BER，有时能识别 header |
| Step4 fband trimmed repeat3 | N1024/CP128, non-contiguous bins, comb pilots, repeat3 | `2.6%-3.4%` BER, header OK, file not exact / header 可识别，但文件不完全一致 |
| Step6 overlap N512/CP256 | only overlap band `158-174` | `29%-35%` BER, failed / 失败 |
| Step6 Step4-mapped N512/CP256 | mapped Step4 trimmed bins | `7.92%` BER, failed but better / 失败，但明显变好 |
| Step6 Step4-raw N1024/CP256 | original Step4 trimmed bins | `16.45%` BER, failed / 失败 |

## Step 1: Probe H Measurement / Step 1：测量 Probe H

Initial channel measurement used `ones` probes:

最初用 `ones` probe 测信道：

```text
bins: 8-150
probe symbols: 256
recordings: 5
```

Main outputs / 主要输出：

```text
runs/step1_probe_ones/8_150/H_abs_summary.csv
runs/step1_probe_ones/8_150/H_mean_summary.csv
```

This showed that the channel can be measured and that many low bins have stable
phase coherence. Some early bins in `8-150` had phase coherence near `0.99`.

这一步证明信道可以被测量，而且不少低频 bin 的相位一致性很好，例如 `8-150` 前段
一些 bin 的 phase coherence 接近 `0.99`。

But later file-transfer tests showed the weakness: `H` from one recording does
not match a later file recording well enough. Timing, phase, gain, and device
state can change between recordings.

但后面的文件传输测试说明了一个问题：一次录音测出的 `H` 很难精确匹配下一次文件
录音。两次录音之间的时序、相位、增益、设备状态都会变化。

Conclusion / 结论：

```text
H must be estimated inside the same recording as the file payload.
H 必须在包含文件 payload 的同一段录音中估计。
```

## Step 2: Combo Playback / Step 2：合并播放

The next idea was a single playback WAV:

下一步改成一个连续发送 WAV：

```text
[probe][file preamble][payload]
```

This avoids the mismatch between a separate probe recording and a later file
recording.

这样可以避免“probe H 和文件录音相位不匹配”的问题。

### QPSK, No Pilot / QPSK，无 Pilot

First combo files used:

第一版 combo 使用：

```text
bins: 8-150, 8-200, 8-420
probe: 256 symbols
file preamble: 32 symbols
payload: QPSK
```

Representative results / 代表结果：

```text
8-150: about 49% BER on runs 5-8
8-200: about 44% BER on runs 1-4
8-420: best around 16.8%-17.2%, still failed

8-150：第 5-8 组约 49% BER
8-200：第 1-4 组约 44% BER
8-420：最好约 16.8%-17.2%，仍失败
```

QPSK was too fragile, and the initial `H` from the front of the recording could
not track the payload without pilots.

QPSK 对当前真实声学链路太脆弱；没有 payload pilot 时，只靠前面的 `H` 也无法稳定
跟踪后面的 payload。

Conclusion / 结论：

```text
Switch to BPSK and add pilot tracking.
改用 BPSK，并加入 pilot 跟踪。
```

### BPSK + Pilot Every 4 Symbols / BPSK + 每 4 个 Symbol 插 Pilot

Structure / 结构：

```text
[probe 256][file preamble 128][data x4][pilot][data x4][pilot]...
```

Parameters / 参数：

```text
bins: 8-150
mod: BPSK
file preamble: 128 symbols
pilot interval: 4
pilot kind: random QPSK whole OFDM symbol
```

Best wide-sync results from runs `1-19`:

第 `1-19` 组中，宽同步搜索后的最好结果：

```text
best run: 18
BER: 4.727%
next runs: about 4.9%-6.6%
file_match: false
```

This was a major improvement over QPSK, but not enough for exact file recovery.

这比 QPSK 明显好很多，但仍不足以精确恢复文件。

Practical observation / 实际观察：

Small playback-volume or device changes caused large BER differences. The
likely cause is nonlinear acoustic/device response and automatic processing in
the audio path, not just random noise.

播放音量或设备状态的细小变化会导致 BER 大幅变化。原因很可能不是单纯随机噪声，而是
扬声器、麦克风、系统音频链路的非线性响应或自动处理。

### BPSK + Pilot Averaging + Payload Repeat / BPSK + Pilot 平均 + Payload 重复

Then the payload was repeated inside one audio file:

之后改成一段音频内重复 payload：

```text
[probe 256][file preamble 128]
[payload_with_pilots repeat 1]
[payload_with_pilots repeat 2]
[payload_with_pilots repeat 3]
```

Each repeat used / 每个 repeat 内部：

```text
data x4 + pilot x2
```

Parameters / 参数：

```text
bins: 8-150
mod: BPSK
pilot interval: 4
pilot length: 2
payload repeats: 3
```

Tuned results for recordings `1-3`:

第 `1-3` 组调参后：

```text
BER: about 3.96%-4.03%
one run had header_ok=true
file_match=false
```

This reduced BER slightly, but the raw error rate was still too high for exact
binary file recovery without coding.

这进一步降低了 BER，但裸传文件仍然太高，无法无纠错地精确恢复。

Conclusion / 结论：

```text
Repeating helps, but bad frequency bins are now the main problem.
重复发送有帮助，但主要矛盾已经变成坏频点。
```

## Step 3: Bandstep Sweep / Step 3：Bandstep 扫频

A full-band bandstep probe was generated:

生成全频段 bandstep probe：

```text
kind: bandstep
bins: 1-511
symbols: 4096
bandstep parts: 64
duration: about 98.3 s
```

Recordings / 录音：

```text
data/step3_bandstep/receive_bandstep_1.wav
data/step3_bandstep/receive_bandstep_2.wav
data/step3_bandstep/receive_bandstep_3.wav
```

Main outputs / 主要输出：

```text
runs/step3_bandstep/combined/bandstep_comparison.csv
runs/step3_bandstep/combined/*.png
```

This gave a coarse map of good and bad frequency bands. It was useful, but too
coarse: each bandstep segment covers many bins, so narrow bad points can be
hidden inside an otherwise acceptable band.

这给出了粗略的好/坏频段分布，但仍然太粗。每个 bandstep 段覆盖很多 bin，窄坏点
可能被平均掉，看不出来。

Conclusion / 结论：

```text
Use fband profiles, then do single-bin sweep for finer detail.
先做 fband profile，再用逐点扫频找窄坏点。
```

## Step 4: Fband Optimization / Step 4：Fband 频段优化

Based on bandstep and real payload behavior, non-contiguous active bins were
introduced.

根据 bandstep 和真实 payload 的表现，开始使用非连续频段。

### Conservative Profile / 保守 Profile

First profile / 第一版 profile：

```text
128-160, 164-240, 315-400
```

It used BPSK, per-symbol comb pilots, and payload repeated 3 times.

它使用 BPSK、每个 payload symbol 内 comb pilot，并将 payload 重复 3 次。

### Trimmed Profile / Trimmed Profile

After seeing real payload errors, the profile was trimmed:

根据真实 payload 错误分布，删掉部分坏频点后得到：

```text
128-160, 164-169, 172-240, 315-357
```

Removed regions / 删除区域：

```text
170-171
358-400
```

Structure / 结构：

```text
active bins: 128-160, 164-169, 172-240, 315-357
data bins: 127
comb pilot bins: 24
probe: 256 symbols
file preamble: 128 symbols
payload: 279 symbols x 3 repeats
```

Representative real results / 代表真实录音结果：

```text
run 1: BER 3.368%, header_ok=true
run 2: BER 2.652%, header_ok=true
run 3: BER 2.596%, header_ok=true
file_match=false
```

This was the first profile to make file headers consistently recognizable, but
the body still had too many errors for exact recovery.

这是第一次能比较稳定识别文件 header 的方案，但正文错误仍然太多，无法完全恢复。

Conclusion / 结论：

```text
Fband + comb pilots is the best broad design so far.
Need stronger bin pruning or error correction.

Fband + comb pilot 是目前最有效的总体方向。
下一步需要更强的坏点剔除，或加入纠错。
```

## Step 5: Single-Bin Sweep / Step 5：逐点扫频

Because bandstep was too coarse, a single-bin full sweep was added:

因为 bandstep 太粗，增加逐点扫频：

```text
bins: 1-511
duration per bin: about 1.536 s
total duration: about 13.1 min
```

Recordings / 录音：

```text
normal: receive_probe_singlebin_full_1.wav ... _4.wav
noise:  receive_probe_singlebin_full_noise_1.wav ... _3.wav
```

Main outputs / 主要输出：

```text
runs/step5_singlebin_sweep/summary/singlebin_summary.csv
runs/step5_singlebin_sweep/overlap_step4_step5/overlap_bins.csv
```

Important finding:

重要发现：

```text
Step4 and Step5 good-frequency distributions only partially overlap.
Some apparently good broad bands contain narrow bad points.

Step4 与 Step5 的良好频点分布只部分重合。
一些看似不错的宽频段内部包含很窄的坏点。
```

This motivated Step 6:

这推动了 Step 6：

```text
choose overlap region
insert comb pilots
set other bins to zero
use larger CP
send payload only once while repeating probe/preamble

选择重合区域
插入 comb pilot
其他 bins 置零
增大 CP
payload 只发一次，但 probe/preamble 多发几次来估 H
```

## Step 6: New File Test / Step 6：新文件测试

Step 6 used `data/step6_newexp/file.jpeg` and independent scripts so old
experiments remain reproducible.

Step 6 使用 `data/step6_newexp/file.jpeg`，并新增独立脚本，避免破坏旧实验。

Common structure / 公共结构：

```text
[random probe 256 x3][file preamble 128 x3][payload x1]
```

H estimation / H 估计方式：

```text
1. Estimate H for each probe repeat.
2. Estimate H for each file preamble repeat.
3. Phase-align the 6 H estimates.
4. Average them into initial H.
5. During payload, use per-symbol comb pilots to update/interpolate H.

1. 对每个 probe repeat 单独估 H。
2. 对每个 file preamble repeat 单独估 H。
3. 对 6 个 H 做相位对齐。
4. 求平均得到初始 H。
5. payload 中每个 symbol 用 comb pilot 更新/插值 H。
```

### Step 6a: Overlap Region, N=512, CP=256 / 重合区域

Parameters / 参数：

```text
N: 512
CP: 256
active bins: 158-174
pilot bins: 158, 162, 166, 170, 174
data bins: 12
duration: about 102.048 s
```

Results / 结果：

```text
run 1: BER 29.21%, file_match=false
run 2: BER 34.51%, file_match=false
```

Diagnosis / 诊断：

```text
Some bins were good, but the second half of the selected band was nearly random.
In run 1, bins 167-173 were around 42%-53% BER.

有些 bin 是好的，但所选频段后半段几乎随机。
第 1 组中，167-173 的 BER 约 42%-53%。
```

Conclusion / 结论：

```text
The overlap region was too narrow and too high-frequency-heavy.
这个重合区域太窄，而且高频坏点太多。
```

### Step 6b: Step4 Trimmed Mapped to N=512 / Step4 Trimmed 映射到 N512

Parameters / 参数：

```text
N: 512
CP: 256
active bins: 64-80, 82-84, 86-120, 158-178
pilot bins: 64,72,80,82,84,86,96,106,120,158,166,174,178
data bins: 63
duration: about 34.368 s
```

Result / 结果：

```text
BER: 7.92%
header_ok=false
file_match=false
```

Per-bin diagnosis / 逐 bin 诊断：

```text
worst bins: 88, 77, 66, 76, 89, 78, 65, 119, 68
many other bins were around 0.3%-1% BER

最坏点：88, 77, 66, 76, 89, 78, 65, 119, 68
很多其他 bin 已经在 0.3%-1% BER 左右
```

Estimated effect of pruning / 剔除坏点后的估计：

```text
drop 66,76,77,88,89 -> kept-bin BER about 2.46%
drop 66,76,77,78,88,89 -> kept-bin BER about 1.73%
drop 65,66,68,76,77,78,88,89,119 -> kept-bin BER about 0.93%

删除 66,76,77,88,89 -> 剩余 bin BER 约 2.46%
再删除 78 -> 剩余 bin BER 约 1.73%
删除到 65,66,68,76,77,78,88,89,119 -> 剩余 bin BER 约 0.93%
```

Conclusion / 结论：

```text
This is better than the overlap design. The next N512 attempt should prune the
listed bad bins and avoid using them as pilots.

这比 overlap 设计好。下一版 N512 应该剔除上述坏点，并避免把它们作为 pilot。
```

### Step 6c: Step4 Trimmed Raw, N=1024, CP=256 / 原始 Step4 Trimmed, N1024

Parameters / 参数：

```text
N: 1024
CP: 256
active bins: 128-160, 164-169, 172-240, 315-357
pilot bins: original Step4 trimmed 24 pilots
data bins: 127
duration: about 43.893 s
```

Result / 结果：

```text
BER: 16.45%
header_ok=false
file_match=false
```

By band / 按频段：

```text
128-160: BER 25.87%
164-169: BER 0.56%
172-240: BER 21.28%
315-357: BER 2.96%
```

Conclusion / 结论：

```text
N=1024 did not improve the latest recording. It exposed many narrow bad points.
If N=1024 is kept, only 164-169 and 315-357 are currently attractive.

N=1024 在最新录音中没有变好，反而暴露了更多窄坏点。
如果坚持 N=1024，目前只有 164-169 和 315-357 比较有吸引力。
```

## Step 7: Adaptive Timing and Soft FEC / Step 7：自适应时钟与软判决 FEC

Step4 proved that a frequency profile can work well in one room and fail after
the recording position changes. Step7 therefore treats sweep results only as a
hardware-level candidate-band guide, not as a permanent per-room bin mask.

Step4 证明固定频段可以在一个位置表现很好，却在录音位置改变后失效。因此 Step7 只把
扫频用于确定硬件层面的候选频带，不再把某个房间测出的好点永久写死。

Frame / 帧结构：

```text
[noise-only 0.5 s][sync 64][preamble 128 x2]
[rate-1/2 coded header x3]
[512-byte CRC blocks, convolutionally coded and interleaved, payload x1]
```

Key changes / 关键变化：

```text
N=512, CP=256
candidate bins 64-120 and 158-178
rotating comb pilot with spacing 4
short-template synchronization
multiple timing anchors estimate sample-clock error
per-symbol CPE and phase-slope tracking
soft BPSK LLR + K=7 rate-1/2 convolutional code + Viterbi
CRC32 for the header, every file block, and the whole file

旋转 comb pilot 每 4 个 symbol 覆盖所有候选 bin
用多个 timing anchor 先修正录音采样时钟，再做 payload 相位跟踪
低可靠度 bin 通过较小软 LLR 降权，而不是强行硬判决
header、每个文件块和完整文件均有 CRC32
```

Verification / 验证结果：

```text
offline WAV: coded BER 0%, post-FEC BER 0%, blocks 16/16, file_match=true
synthetic 120 ppm + multipath + 22 dB SNR: file_match=true
synthetic 120 ppm + stronger multipath + 12 dB SNR:
  coded BER 0.0373%, post-FEC BER 0%, blocks 16/16, file_match=true

离线 WAV：编码流 BER 0%，FEC 后 BER 0%，16/16 blocks，文件完全一致
120 ppm + 多径 + 22 dB SNR：文件完全一致
120 ppm + 更强多径 + 12 dB SNR：
  FEC 前 0.0373%，FEC 后 0%，16/16 blocks，文件完全一致
```

The long sync correlation initially failed under 120 ppm clock error. Using an
8-symbol detection template and fitting multiple training anchors recovered an
estimated `-122 ppm`, then restored the file exactly. This confirms that timing
drift was an independent failure mode that frequency sweeping could not reveal.

最初的长 sync 相关在 120 ppm 时钟误差下直接失效。改成 8-symbol 短模板，并用训练段
多个 anchor 拟合后，接收端估计出约 `-122 ppm`，随后完整恢复文件。这说明时钟漂移是
扫频无法发现的一类独立故障。

### Step7 Real JPEG / Step7 真实 JPEG

The first real Step7 JPEG recording initially failed with the default
per-symbol phase-slope tracker:

第一份真实 Step7 JPEG 录音使用默认逐 symbol slope 时失败：

```text
default coded BER: 42.86%, header failed
slope disabled, other defaults: coded BER 10.72%, post-FEC 0.47%, blocks 5/16
```

Using common-phase tracking and slower H updates produced complementary
CRC-valid blocks. A CRC-selected decoder ensemble recovered all 16 blocks and
the whole-file CRC, without using source bytes to choose blocks. This proved
that FEC and block CRC were effective, but also showed that instantaneous slope
tracking was unstable.

使用公共相位和较慢 H 更新后，不同解码候选产生互补的 CRC 正确 blocks。按 CRC 选择后可
恢复 16/16 blocks 和整文件 CRC。这证明 FEC 与 block CRC 有效，但逐 symbol slope 不稳。

### Step7 Uncompressed TIFF / Step7 无压缩 TIFF

An uncompressed `64x64 RGB` TIFF was transmitted once in a 66.814-second WAV.

生成 `64x64 RGB` 无压缩 TIFF，并用约 66.814 秒 WAV 单次发送 payload。

Official no-forced-correction result with slope disabled and FEC retained:

关闭 slope、保留 FEC、无多候选强制修正的正式结果：

```text
sync score:              0.3298
training clock:          +5.12396 ppm
coded BER:               10.6049%
CRC blocks:              15/26 (blocks 0-14)
failed blocks:           15-25
file_match:              false
```

Recording RMS remained approximately `0.078-0.080` from 2 s to 66 s, and local
correlation against the correct WAV stayed `0.34-0.39` through 62 s. Playback
did not stop and the microphone level did not collapse.

录音 2-66 秒 RMS 稳定在约 `0.078-0.080`，正确发送 WAV 的局部相关在 2-62 秒保持
`0.34-0.39`。播放器没有中断，录音电平也没有下降。

Full-record source-aided timing diagnosis measured `-3.64858 ppm`, differing
from the training estimate by `8.77254 ppm` and accumulating `27.58 samples` at
the end. This full-record fit was used only for diagnosis, not as an admissible
receiver method.

全段已知波形诊断得到 `-3.64858 ppm`，与训练估计相差 `8.77254 ppm`，结尾累计
`27.58 samples`。该结果只用于定位原因，不属于正式接收方法。

With only this diagnostic clock correction, slope disabled, and the normal FEC:

```text
coded BER:               1.3859%-1.5735%
CRC blocks:              26/26
post-FEC BER:            0%
file_match:              true
```

Therefore the primary failure was short-baseline clock estimation bias. The
remaining corrected-clock errors were concentrated at bins 99, 114, 69, and
161, but FEC corrected them completely.

因此主要故障是短训练基线导致的时钟比例估偏。校正时钟后剩余错误集中在 bins 99、114、
69 和 161，但 FEC 可以全部修复。

Full Step7 analysis / Step7 完整分析：
[`step7_adaptive_fec.md`](step7_adaptive_fec.md).

## Current Engineering State / 当前代码状态

Important scripts / 重要脚本：

```text
tx_combo.py / rx_combo.py
  General combo sender/receiver for Step2-Step4 experiments.
  Step2-Step4 的通用合并发送/接收脚本。

fband.py
  Defines conservative and trimmed fband profiles.
  定义 conservative 和 trimmed fband profile。

probe.py / analyze.py
  Probe generation and channel analysis. Supports bandstep and singlebin.
  生成 probe 和分析信道，支持 bandstep 和 singlebin。

tx_step6.py / rx_step6.py / step6_modem.py
  Step6 overlap N512/CP256 profile.
  Step6 overlap N512/CP256 profile。

tx_step6_step4.py / rx_step6_step4.py / step6_step4_modem.py
  Step6 Step4-mapped N512/CP256 profile.
  Step6 Step4 映射到 N512/CP256 的 profile。

tx_step6_step4_n1024.py / rx_step6_step4_n1024.py / step6_step4_n1024_modem.py
  Step6 Step4-raw N1024/CP256 profile.
  Step6 原始 Step4 N1024/CP256 profile。

tx_step7.py / rx_step7.py / step7_modem.py
  Step7 rotating-pilot, clock-corrected, soft-FEC profile.
  Step7 旋转 pilot、采样时钟校正和软判决 FEC profile。
```

Important data roots / 重要数据目录：

```text
data/step2_file/
data/step3_bandstep/
data/step4_fband_optimization/
data/step5_singlebin_sweep/
data/step6_newexp/
data/step7_adaptive_fec/
```

Important output roots / 重要输出目录：

```text
runs/step2_file/
runs/step3_bandstep/
runs/step4_fband_optimization/
runs/step5_singlebin_sweep/
runs/step6_newexp/
runs/step7_adaptive_fec/
```

## Recommended Next Experiments / 建议下一步

1. Add known timing anchors across the payload, for example 8 symbols every 256
   payload symbols. These are training overhead, not payload repeats.
2. 在 payload 中周期插入已知 timing anchors，例如每 256 个 payload symbols 插入 8 个
   timing symbols；它们不是 payload 重复。
3. Fit sample scale robustly over the whole recording and reject anchor outliers.
4. 用覆盖整段录音的 anchors 稳健估计 sample scale，并拒绝异常相关峰。
5. Disable direct instantaneous slope application. Use per-symbol common phase
   and only a slowly filtered pilot slope as residual timing evidence.
6. 不直接应用逐 symbol slope；每 symbol 跟踪公共相位，pilot slope 只做慢速残差观测。
7. Keep the current soft convolutional FEC and block CRC. They recovered all
   corrected-clock TIFF errors.
8. 保留当前 soft convolutional FEC 和 block CRC；时钟正确后它们已能恢复全部 TIFF 数据。
9. Keep bins only when they fail across devices and positions; do not prune more
   bins before timing recovery is fixed.
10. 修好 timing 前不继续按单次房间结果删 bins。

Near-term target / 近期目标：

```text
Recover a 60-90 second single-payload recording without source-aided timing and
keep all block and whole-file CRC checks true.
不使用源文件辅助 timing，在 60-90 秒单次 payload 录音中保持全部 block 和整文件 CRC 通过。
```
