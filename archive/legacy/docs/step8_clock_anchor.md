# Step8 Complete Protocol / Step8 完整协议

## 1. Purpose / 目标

Step8 is the current end-to-end acoustic file-transfer protocol. It sends the
file payload once, estimates H from training in the same recording, tracks
per-symbol common phase with comb pilots, and inserts known periodic anchors to
measure sample-clock drift across the complete payload.

Step8 是当前完整的声学文件传输协议。文件 payload 只发送一次；H 在同一份录音中测量；
comb pilot 逐 symbol 跟踪公共相位；周期已知 anchor 覆盖整个 payload，用于估计采样时钟
漂移。

Step7 used only the roughly five-second opening training to estimate PPM. One
66.8-second recording produced `+5.12 ppm` from training but `-3.65 ppm` across
the full waveform. The resulting 27.6-sample drift caused a sharp late BER
failure. Step8 removes that single short-baseline dependency.

Step7 只利用开头约五秒训练估计 PPM；一次 66.8 秒录音的训练估计为 `+5.12 ppm`，全段诊断
值却为 `-3.65 ppm`，最终累计 27.6 samples 并导致后半段突发错误。Step8 用全段 anchor
消除了这一短基线依赖。

## 2. Physical Profile / 物理层参数

```text
sample rate                 48000 Hz
FFT N                       512
cyclic prefix               256 samples
OFDM symbol                 768 samples = 16 ms
active bins                 64-120, 158-178 (78 bins)
active frequencies          6.000-11.250 kHz, 14.8125-16.6875 kHz
data modulation             BPSK
pilot/anchor modulation     deterministic QPSK
payload repeats             1
```

For each OFDM row, active positive-frequency bins are filled and their complex
conjugates are written to `N-k`; the IFFT is therefore real. The final 256 IFFT
samples form the cyclic prefix.

每个 OFDM row 在正频率 active bins 写入符号，并在 `N-k` 写入共轭，因此 IFFT 为实信号；
IFFT 末尾 256 samples 作为 CP。

The active bands came from Step4-Step6 measurements, but Step8 does not assume
that a one-room sweep permanently predicts good bins. FEC, in-frame H, pilots
and anchors handle the current recording.

有效频带来自 Step4-Step6 的经验，但 Step8 不把单次房间扫频当作永久频点结论，而依赖本帧
H、pilot、anchor 和 FEC 适应当前录音。

## 3. Frame Structure / 帧结构

```text
[noise-only 0.5 s]
[sync 64 symbols, seed 7026]
[preamble 128 symbols, seed 8026]
[preamble 128 symbols, seed 8027]
[payload-start anchor 8 symbols, seed 10028]
[logical payload x128][timing anchor x8]
[logical payload x128][timing anchor x8]
...
[remaining logical payload]
[tail silence 0.25 s]
```

The sync and preambles use random QPSK on every active bin. The two preambles
are independent so their H estimates can be phase-aligned and averaged without
relying on one repeated waveform.

Sync 和 preamble 在全部 active bins 使用随机 QPSK。两段 preamble 相互独立，接收端对各段
H 做公共相位对齐后平均，避免只依赖一段重复波形。

For the current 12,936-byte TIFF:

```text
logical payload symbols     3809
timing anchor groups        29
anchor symbols              232
physical payload symbols    4041
periodic anchor overhead    5.741% of physical payload symbols
payload-start anchor        8 symbols / 0.128 s
total WAV duration          70.654 s
```

An anchor does not consume a payload bit or advance the rotating-pilot logical
symbol index. It is physical training inserted between logical payload rows.

Anchor 不消耗 payload bit，也不推进 rotating pilot 的逻辑 symbol 序号；它只是插入逻辑
payload 之间的物理训练段。

## 4. File Framing / 文件封装

### Header / 文件头

The header is exactly 128 bytes. Multi-byte integers are big-endian:

```text
>4sBBHQHHI100s + >I
magic             4 bytes   "AM7F"
version           1 byte    1
filename length   1 byte
reserved          2 bytes
file size         8 bytes
block size        2 bytes   default 512
block count       2 bytes
whole-file CRC32  4 bytes
UTF-8 filename  100 bytes   zero padded
header CRC32      4 bytes   CRC over preceding 124 bytes
```

文件名最终通过 `Path(...).name` 限制为 basename，避免接收文件逃逸输出目录。Header 经
卷积编码后发送三份，每份使用不同 interleaver。

### Data blocks / 数据块

The source is split into 512-byte chunks. Every raw block contains:

```text
block index       uint16
payload length    uint16
payload           512 bytes, zero padded
block CRC32       uint32 over index + length + unpadded payload
```

每个 block 可独立通过 CRC。只有所有 blocks 齐全且 whole-file CRC 正确时，接收端才以原始
文件名输出正式恢复文件；否则输出 `.partial`。

## 5. FEC and Interleaving / 纠错与交织

Step8 uses a terminated rate-1/2 convolutional code:

```text
constraint length K         7
generator polynomials       171 and 133 octal
tail bits                   K-1 zero bits
decoder                     64-state soft Viterbi
LLR clipping                [-24, +24]
```

Positive LLR means coded bit 0; negative LLR means coded bit 1. Each coded frame
is independently permuted with NumPy's deterministic RNG:

```text
header repeat r             seed 7027 + r
data block i                seed 7027 + 1000 + i
```

交织把局部频率或时间错误分散到整个 FEC block；每块独立交织和 CRC，避免一个失败区域破坏
后续全部文件。

## 6. Payload Pilots and LLR / Payload Pilot 与软判决

Pilot spacing is four. Within each of the two disjoint active bands, one of
every four bins is a pilot; the offset rotates with logical symbol index:

```text
offset = logical_symbol_index mod 4
```

Pilot values are deterministic QPSK generated from seed `2027 + symbol_index`.
All non-pilot active bins carry BPSK coded bits until the final row.

Pilot 由 `2027 + symbol_index` 生成确定性 QPSK。其余 active bins 发送 BPSK 编码 bits。

For a data bin, the soft metric is:

```text
LLR = 2 Re(conj(H_effective) Y) / variance
```

The variance is the maximum of the training/noise estimate and the current
median pilot residual. Weak or distorted symbols therefore contribute less to
Viterbi decoding.

方差取训练/静音噪声与当前 pilot residual 中较大者，使弱符号和失真符号在 Viterbi 中权重
降低。

## 7. Initial H and Tracking / 初始 H 与跟踪

Each training block estimates:

```text
H_block = mean(Y / X, axis=time)
```

Sync and both preamble H values are aligned to a common phase and averaged.
Their residual variance is combined with the leading silence spectrum to form
the initial per-bin noise estimate.

Sync 与两段 preamble 的 H 先对齐公共相位再平均；训练 residual 与开头静音频谱共同形成逐
bin 初始噪声估计。

Every data symbol retains CPE correction from its pilots. Pilot H updates use
`channel_alpha=0.35` by default.

逐 symbol CPE 必须保留；comb pilot 默认以 `channel_alpha=0.35` 更新当前 H。

Each timing anchor contains eight full-active-band QPSK rows. Their eight `Y/X`
estimates are common-phase aligned and averaged, then fused as:

```text
H <- (1 - anchor_h_alpha) H + anchor_h_alpha H_anchor
default anchor_h_alpha = 0.5
```

Set `--anchor-h-alpha 0` only for a controlled comparison. In the first real
Step8 recording, disabling anchor H increased raw BER from `6.11%` to `7.02%`
and reduced valid blocks from `26/26` to `19/26`.

第一份真实录音关闭 anchor H 后，FEC 前 BER 从 `6.11%` 上升到 `7.02%`，有效 blocks 从
`26/26` 降到 `19/26`，因此默认必须保留。

The dedicated payload-start anchor is placed immediately before the encoded
header. After global clock correction, the receiver correlates its unique
eight-symbol template within `+-16 samples`, uses that exact OFDM grid to
estimate full-band H, and starts header demodulation after the anchor. This
prevents a weak comb-pilot score from selecting a plausible but phase-ramped
position inside the cyclic prefix.

专用 payload-start anchor 紧贴编码 header 之前。全局校时后，接收端在 `+-16 samples`
内相关其唯一的 8-symbol 模板，在同一 OFDM 采样网格上估计全频 H，然后从 anchor 之后开始
解 header。这避免低信噪比时 comb-pilot 把 CP 内某个带相位斜率的位置误选为 payload 起点。

## 8. Timing Anchors and PPM / Timing Anchor 与 PPM

The start anchor uses seed `10028`; periodic anchor `i` uses random QPSK seed
`9028+i`, so every anchor has a known unique
template. Around each predicted position the receiver performs normalized
correlation within `+-128 samples`; a three-point parabola estimates the
fractional peak position.

起始 anchor 使用 seed `10028`，第 `i` 组周期 anchor 使用 seed `9028+i`。接收端在理论
位置附近 `+-128 samples` 做归一化互相关，并用三点抛物线得到亚采样峰值。

The accepted positions fit:

```text
observed_i = intercept + scale * nominal_i
ppm = (scale - 1) * 1e6
```

The fit uses a Theil-Sen initial line followed by score-squared weighted
Huber/MAD iterations. Candidates below correlation `0.12` and robust timing
outliers are rejected.

拟合使用 Theil-Sen 初值和相关分数平方加权的 Huber/MAD 迭代，拒绝分数低于 `0.12` 及稳健
残差异常点。

A full fit requires at least four accepted payload anchors spanning at least
half the attempted payload duration. Otherwise the receiver continues with the
opening training estimate and reports:

```text
clock_status = training_fallback
```

完整拟合至少需要 4 个有效 payload anchors，且覆盖候选 payload 时间跨度的一半；否则明确
退回 training 估计，不把退化状态伪装成全段校正。

The recording is globally resampled once using the fitted intercept and scale.
The receiver then locates the known payload-start anchor within `+-16 samples`.
Its correlation and H estimate use no source bytes. Set
`--payload-start-anchor-symbols 0` only to decode the older Step8 air format;
that compatibility mode restores the pilot-residual start search.

拟合后先对整段录音做一次全局重采样，再在 `+-16 samples` 内定位已知 payload-start
anchor；相关与 H 估计均不使用源文件。只有解旧 Step8 空中格式时才设置
`--payload-start-anchor-symbols 0`，此兼容模式会恢复旧 pilot-residual 起点搜索。

## 9. Phase Slope Policy / Phase Slope 策略

Per-symbol CPE is always active. `--phase-slope off` is the default and the only
mode used for official real results.

逐 symbol CPE 始终开启；`--phase-slope off` 是默认值，也是正式真实结果使用的模式。

`--phase-slope slow` is an explicit experiment. It uses the median of the last
64 raw slope estimates, clips to `+-0.05 rad/bin`, treats the two active bands
as separate unwrap segments, and resets history at each full-band anchor.

`slow` 只用于对照实验：最近 64 个 slope 取中位数，限制为 `+-0.05 rad/bin`，两段 active
band 分别 unwrap，并在 anchor 后清空历史。

In the first real recording slow slope produced `7.29%` raw BER and only
`18/26` valid blocks. It must not be automatically selected using source-file
results.

第一份真实录音使用 slow slope 时 BER 为 `7.29%`、仅 `18/26` blocks，因此不能借助源文件
自动选用。

## 10. Receiver Data Flow / 接收流程

```text
WAV validation
  -> coarse sync
  -> training and payload-anchor correlation
  -> robust full-recording clock fit or explicit fallback
  -> global fractional resampling
  -> sync + preamble H/noise estimation
  -> payload-start anchor correlation and full-band H refresh
  -> begin header demodulation immediately after the start anchor
  -> remove physical anchors while producing logical-symbol LLRs
  -> per-symbol CPE and H tracking
  -> combine three interleaved header copies
  -> soft Viterbi header decode + header CRC
  -> independent Viterbi block decode + block CRC
  -> whole-file CRC and optional source comparison
```

Source bytes are used only for post-analysis coded BER, post-FEC BER and
`file_match`. They do not influence synchronization, H, PPM, slope, candidate
rejection or recovered bytes.

源文件只用于事后 BER 和 `file_match`，不参与同步、H、PPM、slope、候选筛选或恢复字节。

## 11. Verified Results / 已验证结果

### Deterministic tests / 确定性测试

```text
direct offline           ppm +0.0000, BER 0, 26/26, exact
synthetic +20 ppm        measured +20.0034, BER 0, 26/26, exact
synthetic -20 ppm        measured -20.0034, BER 0, 26/26, exact
anchors severely damaged training_fallback reported explicitly
Step7 offline regression BER 0, 26/26, exact
```

### Real recording 1 / 真实录音 1

```text
sync score               0.338330
initial PPM             -16.1870
full-anchor PPM         -20.1574
raw coded BER             6.106942%
post-FEC BER              0%
CRC blocks               26/26
whole-file CRC            pass
file_match                true
```

Errors remain spread across the payload rather than forming the Step7 late
cliff. Anchor H on, CPE on, slope off is the successful configuration.

错误在 payload 中较均匀分布，不再出现 Step7 的后半段突变。成功配置为 anchor H 开、CPE
开、slope 关。

### Real recording 2 retest / 真实重录 2

```text
sync score               0.270234
initial PPM             -47.9285
full-anchor PPM         -10.8823
accepted anchors         35/41 total, 27 payload anchors
fit residual RMS          1.205 samples
raw coded BER             5.558049%
post-FEC BER              0%
CRC blocks               26/26
whole-file CRC            pass
file_match                true
```

This recording contained a severe startup clipping transient and intentionally
mixed noise near 67.25-68.0 seconds. The last blocks rose to roughly
`5.28%-6.71%` raw BER, but every block was corrected.

该录音开头存在严重削波，并在约 67.25-68.0 秒主动混入噪声。最后 blocks 的 FEC 前 BER
升至约 `5.28%-6.71%`，但全部纠正成功。

The large difference between initial and full PPM is evidence that a short
training estimate is unreliable after startup transients; it is not evidence
that one fixed device has one immutable PPM value.

初始与全段 PPM 的巨大差异说明启动瞬态会污染短训练估计；同一设备并不存在一成不变、可
跨录音硬编码的 PPM 值。

## 12. Outputs / 输出

```text
metrics.json                 authoritative result and configuration
blocks.csv                   per-block CRC status
summary.csv                  per-bin H/noise/LLR summary
clock_anchors.npy            nominal, observed, score, residual, used, kind, index
clock_fit.png                clock line and residuals
H.npy                        initial H
H_training_blocks.npy        aligned training H
H_anchor_track.npy           periodic full-band H
pilot_H_track.npy            logical-symbol H tracking
rx_llr.npy                   coded soft decisions
cpe.npy                      per-symbol CPE
phase_slope.npy              applied slope
phase_slope_raw.npy          measured slope
channel_and_noise.png        initial channel/noise plot
phase_tracking.png           tracking diagnostics
decoded_header.bin           raw decoded header
recovered file or *.partial  CRC-valid file or explicit best effort
```

## 13. Known Limitations / 已知限制

- Unique random anchor templates can choose slightly different multipath peaks;
  robust fitting rejects large shifts, but reported PPM is a modem correction,
  not precision metrology.
- 不同随机 anchor 在多径中可能选择略有差异的峰；稳健拟合会剔除异常，但 PPM 是 modem
  校正量，不是精密仪器标定值。
- The receiver initially processes audio through trailing silence before the
  decoded header reveals the exact payload length. Very low-score tail
  candidates are rejected by clock fitting, but may appear in diagnostic counts.
- 接收端在 header 解出长度前会处理尾部静音，因此诊断中可能出现极低分的尾部假 anchor；
  它们会被时钟拟合拒绝，当前录音的正式数据不受影响。
- Raw BER around 5%-6% leaves useful but finite FEC margin. Stronger noise,
  clipping or a substantially different room can still exceed it.
- 当前约 5%-6% FEC 前 BER 仍有纠错余量，但更强噪声、削波或明显变化的环境可能超过能力。
