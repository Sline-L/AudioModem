# Step7 Adaptive FEC and Timing Recovery / Step7 自适应 FEC 与时钟恢复

## 1. Purpose / 目标

Step7 is the current robust acoustic file-transfer experiment. It was created
after Step4-Step6 showed that a bin mask selected in one room can fail after the
speaker, microphone, or recording position changes.

Step7 是当前的稳健声学文件传输实验。Step4-Step6 证明：在一个位置通过扫频选择出的
固定 bins，换房间、换位置或改变设备状态后可能立即失效。因此 Step7 不再追求一次扫频
得到永久好频点，而是组合以下机制：

- broad hardware-level candidate bands / 较宽的硬件候选频带；
- training inside the same WAV / 同一段 WAV 内训练；
- sample-clock correction / 采样时钟校正；
- rotating comb pilots / 旋转 comb pilot；
- soft BPSK reliability / BPSK 软可信度；
- convolutional FEC and per-block CRC / 卷积码 FEC 和逐块 CRC。

The authoritative Step7 implementation is kept separate from old experiments:

```text
step7_modem.py   frame, OFDM, pilots, coding, interleaving, Viterbi
tx_step7.py      Step7 WAV generator
rx_step7.py      synchronization, timing correction, equalization, FEC, output
```

这些脚本不会修改旧版 `audiomodem.py` 的 `N=1024, CP=128` 默认值。

## 2. Physical and OFDM Parameters / 物理层参数

```text
sample rate:             48000 Hz
FFT size N:              512
cyclic prefix CP:        256 samples
OFDM symbol length:      768 samples = 16 ms
active bins:             64-120, 158-178
frequency ranges:        6.0-11.25 kHz, 14.8125-16.6875 kHz
modulation:              BPSK data, known QPSK pilots
pilot pattern:           rotating comb, spacing 4
payload repeats:         1
```

The bands are candidates, not a claim that every bin is good in every room.
Receiver reliability and FEC are expected to absorb environment-specific bad
bins. A bin should be permanently nulled only after it fails across devices and
positions.

这些频段只是候选范围，不表示每个 bin 在所有环境中都可靠。接收端的软 LLR 和 FEC 用来
吸收当前房间的坏点；只有跨设备、跨位置持续失败的频点才值得永久置零。

## 3. Transmit Frame / 发送帧

```text
[noise-only 0.5 s]
[sync 64 OFDM symbols]
[full-band preamble 128 x2]
[coded header repeat 1]
[coded header repeat 2]
[coded header repeat 3]
[interleaved coded file blocks x1, rotating pilots in every OFDM symbol]
[tail silence 0.25 s]
```

The noise-only interval measures the current recording noise. Only the first
eight sync symbols are used for coarse correlation; the remaining sync symbols
also contribute to H estimation. Two independent preambles provide additional
full-band H estimates.

静音段测量本次录音的底噪。sync 前 8 个 symbols 用于短模板相关，避免长模板被时钟偏差
拉散；完整 sync 和两个 preamble block 都参与 H 估计。

### Header / 文件头

The fixed 128-byte header stores:

```text
magic and version
UTF-8 filename
file size
block size and block count
whole-file CRC32
header CRC32
```

It is convolutionally coded and transmitted three times with independent
interleavers. The receiver deinterleaves the three soft-LLR copies, sums them,
and performs one soft Viterbi decode.

header 固定为 128 bytes，包含文件名、长度、block 参数、整文件 CRC32 和 header CRC32。
编码后发送三份，但文件 payload 仍只发送一次。

### File blocks / 文件块

```text
512 payload bytes
+ 2-byte block index
+ 2-byte valid length
+ 4-byte block CRC32
```

Each block is independently convolutionally encoded and interleaved. This gives
three useful output states:

- exact block: CRC passes / CRC 通过，block 可确认完全正确；
- failed block: Viterbi output exists but is not trusted / 有输出但 CRC 失败；
- exact file: every block and whole-file CRC pass / 所有块及整文件 CRC 通过。

## 4. FEC and Soft Decisions / FEC 与软判决

Step7 uses a rate-1/2 convolutional code:

```text
constraint length K:     7
generators:              171/133 octal
termination:             six zero tail bits
decoder:                 soft-decision Viterbi
interleaver:             deterministic per block
```

BPSK maps bit 0 to `+1` and bit 1 to `-1`. The receiver computes an LLR-like
reliability value from the channel estimate and noise estimate:

```text
LLR[k] proportional to 2 Re(conj(H[k]) Y[k]) / noise_variance[k]
```

A deep-fade bin therefore contributes a low-confidence value instead of an
equally strong hard vote. Interleaving spreads a time/frequency error burst over
the Viterbi trellis.

深衰落频点不会和可靠频点拥有相同权重。交织把连续时间或连续频率错误分散到卷积码序列，
再由 soft Viterbi 恢复。

## 5. Current H Estimation / 当前 H 估计

For each of the sync and two preamble blocks:

```text
H_block[k] = mean(Y[s,k] / X[s,k]) over training symbols s
```

The three H estimates are phase-aligned and averaged into the initial H. The
training residual and noise-only FFT power produce per-bin noise estimates.

对 sync 和两个 preamble 分别求 `Y/X` 平均，得到三组 H。三组 H 做整体相位对齐后平均，
形成 payload 初始 H。训练残差和静音 FFT 功率共同决定每个 bin 的软判决尺度。

During payload, pilot positions rotate with the OFDM symbol index. Every bin is
directly observed as a pilot once per four-symbol cycle. The current code:

1. compares observed pilot H with tracked H;
2. fits a common phase and a phase slope across frequency;
3. applies the phase ramp to all bins;
4. updates current pilot bins with exponential weight `channel_alpha`;
5. computes data LLRs.

payload 中 pilot 每个 symbol 旋转，理论上每四个 symbols 所有 bins 都会直接成为一次
pilot。当前默认 `channel_alpha=0.35`。

## 6. Sample-Clock Recovery / 采样时钟恢复

### Why clock ppm matters / 为什么几 ppm 也重要

The recorder and transmitter do not share a sample clock. If one received OFDM
symbol occupies `scale * 768` samples, a scale error accumulates with time.

发送和录音设备没有共同采样时钟。若接收端估计的时间比例有误，FFT 窗口会相对发送符号
持续移动。时间偏移 `delta_n` 在 bin `k` 上产生近似相位：

```text
phase_error[k] = -2 pi k delta_n / N
```

CP prevents immediate inter-symbol interference while the FFT window remains
inside the cyclic region, but CP does not make this phase error disappear. H
estimated at the preamble becomes progressively stale.

只要窗口仍在循环前缀允许范围内，CP 可以暂时避免 ISI；但它不会消除相对于训练 H 的
逐 bin 相位旋转。

### Current estimator / 当前估计器

Step7 places eight-symbol correlation anchors every 32 symbols across the
64-symbol sync and two 128-symbol preambles. A weighted line fit estimates:

```text
observed_sample = intercept + scale * nominal_sample
clock_ppm = (scale - 1) * 1e6
```

The receiver resamples the synchronized recording to the nominal 768-sample
symbol grid before payload processing.

问题是这些 anchors 只覆盖约 5 秒。相关峰只能定位到整数 sample，并且多径会让峰位置有
几 samples 抖动。短基线上的几 samples 误差会转化为数 ppm 偏差；对一分钟 payload，
最终可累计几十 samples。

## 7. TIFF Experiment / TIFF 实验

### Files / 文件

```text
data/step7_adaptive_fec/observatory_64_uncompressed.tiff
data/step7_adaptive_fec/observatory_64_uncompressed_step7.wav
data/step7_adaptive_fec/receive_observatory_tiff_1.wav
```

The test image is `64x64 RGB`, 8-bit, uncompressed TIFF, about 12.9 KB. The WAV
is about 66.814 seconds and contains one payload transmission.

测试图为 `64x64 RGB`、8-bit、无压缩 TIFF，约 12.9 KB。发送 WAV 约 66.814 秒，
payload 只发送一次。

Offline result / 离线结果：

```text
coded BER:              0%
post-FEC BER:           0%
blocks:                 26/26
file_match:             true
```

### Real recording without forced correction / 真实录音，不强行修正

Default slope tracking:

```text
sync score:             0.3298
coded BER:              10.9343%
CRC blocks:             10/26
file_match:             false
```

Disable phase slope, keep FEC and every other default:

```text
estimated clock:        +5.12396 ppm
payload delta:          -1 sample
coded BER:              10.6049%
CRC blocks:             15/26
passed blocks:          0-14
failed blocks:          15-25
file_match:             false
```

No multi-H selection, alpha sweep, source-byte replacement, or CRC ensemble was
used for this official result.

这个正式结果没有使用多 H 候选、alpha 扫描、原文字节替换或跨解码器 CRC 拼接。

### Best-effort image / 可打开的预览图

The raw TIFF could not open because the IFD directory at the end of the file was
inside failed blocks. For visualization only, the received bytes at the known
uncompressed RGB pixel area were interpreted as `64x64 RGB` and written to:

```text
runs/step7_adaptive_fec/observatory_tiff/1_slope_off/
  observatory_64_best_effort_pixels.png
```

No source pixels were substituted. The PNG preserves received errors and is not
evidence of a correct TIFF file.

原始 TIFF 因末尾 IFD 位于失败 blocks 中而无法打开。分析时只把接收到的未压缩 RGB
像素主体解释为 `64x64` 并导出 PNG，没有用原图像素替换错误。该 PNG 仅供观察花屏，
不代表 TIFF 恢复成功。

## 8. Root-Cause Analysis / 根因分析

### What did not happen / 已排除

- Playback did not stop or switch files. Local correlation against the correct
  transmit WAV remained `0.34-0.39` from 2 s to 62 s.
- 播放没有中断或换文件，2-62 秒始终能匹配正确 TIFF 发送波形。
- Recording level did not fall. RMS stayed approximately `0.078-0.080` from
  2 s to 66 s.
- 录音音量没有下降。
- Timing drift was continuous, not a sudden jump.
- 时序偏移连续变化，没有在 block 15 附近突然跳变。
- Silence noise was about `0.000395`, much lower than the training residual
  `0.304`; background microphone noise alone is not the explanation.
- 静音底噪远低于训练残差，问题不是单纯麦克风白噪声。

There was a large recorder startup transient during the first 0.25 s: RMS
`0.951`, strong negative DC, and 8766 clipped samples. It ended before sync at
about 1.36 s, so it is not the direct cause of late payload failure, but record
startup should still be guarded by silence.

录音开始前 0.25 秒存在明显直流和削波，但在约 1.36 秒 sync 前已经结束，不是后半段失败
的直接原因。保留发送前静音仍然必要。

### Main cause / 主要原因

Training-only timing estimate:

```text
+5.12396 ppm
```

Diagnostic fit using known waveform positions across the entire recording:

```text
-3.64858 ppm
```

Difference and accumulated error:

```text
clock estimate bias:    8.77254 ppm
end-of-payload drift:   27.58 samples
```

This source-aided full-record fit is diagnostic only. A real receiver cannot use
unknown payload data as timing anchors.

全段 `-3.65 ppm` 使用了已知发送 WAV，只用于证明根因，不能作为正式接收算法。

With the diagnostic clock ratio, slope still disabled:

```text
channel alpha 0.05:     coded BER 1.3859%, blocks 26/26, post-FEC 0%
channel alpha 0.35:     coded BER 1.5735%, blocks 26/26, post-FEC 0%
file_match:             true in both diagnostic runs
```

This confirms that clock-estimation bias, not playback interruption or a global
frequency-band collapse, caused the late failure.

诊断性地改正时钟比例后，裸 BER 降到 `1.39%-1.57%`，FEC 完整恢复 26/26 blocks，
证明主要原因是时钟估计偏差。

### Why slope on and slope off both fail / 为什么开关 slope 都不理想

- Slope on: an instantaneous line is fitted from sparse pilots every symbol.
  Multipath and weak pilots contaminate the fit, so false slopes are applied.
- 开 slope：逐 symbol 拟合太敏感，多径和弱 pilot 会产生假 slope。
- Slope off: the biased initial clock estimate is never corrected during the
  long payload, so true residual phase slope accumulates.
- 关 slope：初始时钟偏差在长 payload 中持续累计，固定 H 逐渐失效。

The solution is not a permanent slope-off switch. The receiver needs a slow,
robust timing loop.

正确方向不是永久关闭 slope，而是建立慢速、稳健的 timing loop。

## 9. Remaining Frequency-Selective Errors / 剩余频率坏点

After applying the diagnostic correct clock ratio, raw BER was stable over time
at roughly `1.0%-1.6%`. Remaining errors were frequency selective:

```text
bin 99     9281.25 Hz      BER 20.48%
bin 114   10687.50 Hz      BER 18.13%
bin 69     6468.75 Hz      BER 13.34%
bin 161   15093.75 Hz      BER  9.21%

bins 64-120 overall:       BER 1.72%
bins 158-178 overall:      BER 0.47%
```

FEC corrected these errors completely once timing was correct. Therefore the
next priority is timing recovery, not another one-room bin-pruning pass.

时钟正确后，剩余错误集中在少数频点，且 FEC 已能完全修复。因此下一步优先修 timing，
不继续依据这一次房间结果删 bins。

## 10. Recommended Step7 Revision / 下一版建议

### Primary approach: periodic timing anchors / 首选：周期 timing anchors

Insert a short known full-band timing block during payload, for example eight
OFDM symbols every 256 payload symbols:

```text
[data/pilots x256][timing anchor x8][data/pilots x256][timing anchor x8]...
```

These are training symbols, not payload repeats. The receiver can correlate
them without knowing file data, continually fit observed sample position versus
nominal position, and update the resampling ratio slowly.

建议每 256 个 payload symbols 插入 8 个已知全频 timing symbols。它们不是重复 payload，
而是让接收端在整段音频中持续获得绝对时间位置。

Recommended control rules:

1. keep the short eight-symbol initial sync;
2. estimate initial clock from sync+preamble;
3. add payload anchors spanning the complete recording;
4. robustly fit anchor positions, reject correlation outliers;
5. update sample scale slowly, never from one symbol alone;
6. use pilot common phase every symbol;
7. use pilot phase slope only as a smoothed residual measurement;
8. retain current soft FEC and block CRC.

### Alternative: pilot-based slow SFO loop / 备选：pilot 慢速 SFO 环

Estimate phase slope from pilots over a long window, such as 32-128 symbols,
using robust regression and weak-pilot rejection. Convert the smoothed slope to
a timing offset and adjust the fractional resampler. Do not directly multiply H
by a noisy instantaneous slope every symbol.

可以对 32-128 symbols 的 pilot slope 做稳健平均，拒绝弱 pilot，再把 slope 转换为时间
偏移并缓慢调整重采样器。不要逐 symbol 把噪声 slope 直接乘进 H。

### Output policy / 输出策略

The receiver should preserve both meanings:

```text
strict partial file:     failed CRC blocks are marked/zeroed; never claim exact
best-effort preview:     raw Viterbi bytes retained for visual inspection
exact recovered file:    written only when block and whole-file CRC pass
```

严格文件、best-effort 预览和 CRC 完整文件必须分开命名，避免“能打开”被误解为“传输
正确”。

## 11. Commands / 命令

Generate the TIFF WAV:

```bash
python tx_step7.py data/step7_adaptive_fec/observatory_64_uncompressed.tiff \
  --out data/step7_adaptive_fec/observatory_64_uncompressed_step7.wav
```

Decode a recording:

```bash
python rx_step7.py data/step7_adaptive_fec/receive_observatory_tiff_1.wav \
  --source data/step7_adaptive_fec/observatory_64_uncompressed.tiff \
  --out runs/step7_adaptive_fec/observatory_tiff/1
```

Important outputs:

```text
metrics.json                 synchronization, ppm, BER, block/file status
blocks.csv                   per-block CRC result
summary.csv                  per-bin H, noise, and LLR reliability
H.npy                        initial combined H
H_training_blocks.npy        aligned sync/preamble H estimates
clock_anchors.npy            training timing-anchor positions and scores
rx_llr.npy                   payload soft decisions
cpe.npy / phase_slope.npy    payload tracking values
channel_and_noise.png         H and noise plot
phase_tracking.png            CPE, slope, and pilot-residual plot
```

## 12. Acceptance Criteria for the Next Version / 下一版验收标准

- Use no source payload bytes for synchronization or candidate selection.
- 不使用源文件 payload 做同步或候选选择。
- One payload transmission only; periodic known timing symbols are allowed.
- payload 只发送一次，允许插入已知 timing symbols。
- Estimated clock remains consistent across a 60-90 second recording.
- 在 60-90 秒录音中不发生累计 timing 崩溃。
- Raw BER remains approximately stationary rather than failing after a fixed
  time boundary.
- 裸 BER 随时间基本平稳，不再出现前半成功、后半连续失败。
- All blocks and whole-file CRC pass on the real TIFF recording.
- 真实 TIFF 录音达到全部 block CRC 和整文件 CRC 通过。
- Continue reporting raw coded BER, post-FEC BER, and per-bin reliability.
- 继续输出 FEC 前后 BER 和逐 bin 可靠度。
