# Step8 End-to-End Engineering Walkthrough / Step8 端到端工程详解

本文按照一次真实文件传输发生的顺序，解释当前 Step8 从源文件到恢复文件的完整链路：

```text
源文件 -> 分帧与校验 -> FEC 与交织 -> BPSK/pilot 映射 -> OFDM
       -> WAV -> 扬声器/空气/麦克风 -> 录音 WAV
       -> 同步与 PPM 校正 -> H/相位跟踪 -> 软解调
       -> 去交织与 Viterbi -> CRC -> 接收文件
```

The English text following each major section summarizes the same implementation.
Equations, constants, filenames, and code references apply to both languages.

## 1. 先看完整系统 / The Whole System

你亲手完成的动作只有三类：选择源文件、播放发送 WAV、录制接收 WAV。其余步骤由
`tx_step8.py`、`step8_modem.py` 和 `rx_step8.py` 确定性完成。

当前正式参数是：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| 采样率 | 48,000 Hz | 每秒 48,000 个实数采样 |
| FFT 长度 `N` | 512 | 每个 OFDM symbol 的有效区长度 |
| CP 长度 | 256 | 每个 symbol 前复制的循环前缀 |
| 总 symbol 长度 `L` | 768 samples | `N + CP`，即 16 ms |
| active bins | 64-120, 158-178 | 共 78 个正频率子载波 |
| active frequency | 6-11.25 kHz, 14.8125-16.6875 kHz | `f_k=k Fs/N` |
| 数据调制 | BPSK | coded bit 0/1 映射为 +1/-1 |
| pilot | rotating QPSK comb | 每个逻辑 symbol 约 1/4 active bins |
| FEC | K=7, rate 1/2 convolutional | 生成多项式 171/133（八进制） |
| 文件块 | 512 bytes | 每块单独带 CRC32 |
| payload | 只发送 1 次 | 不依赖 payload 重复投票 |

当前 TIFF 的实际数字为：

```text
source bytes                 12,936
file blocks                  26
coded bits                   222,812
logical payload symbols      3,809
periodic anchors             29 x 8 symbols
physical payload symbols     4,041
complete transmit WAV        70.654 s
```

关键数组的 shape 沿链路变化如下：

```text
source bytes                         (12936,)
coded bit stream                     (222812,)
logical payload, complex             (3809, 78)
physical payload with anchors        (4041, 78)
final real PCM waveform              (3391392,) samples
receiver LLR stream                  one float per recovered coded bit
recovered source                     (12936,) bytes
```

The transmitter turns arbitrary file bytes into protected coded bits, maps them to
OFDM symbols, and writes one deterministic WAV. The acoustic path changes amplitude,
phase, timing, and noise. The receiver estimates those changes from known in-frame
signals, produces soft bit evidence, applies FEC, validates CRCs, and writes the file.

## 2. 第一步：准备源文件 / Prepare the Source File

### 你做了什么

你把待发送文件放入 `data/step8_clock_anchor/`。当前示例是：

```text
data/step8_clock_anchor/observatory_64_uncompressed.tiff
```

它对调制器来说不是“图片”，只是有顺序的 12,936 个字节。TIFF、JPEG、TXT 或 ZIP
进入物理层后没有区别；文件格式只影响应用程序能否在恢复后打开它。

### 程序做了什么

`Path.read_bytes()` 读取完整文件，同时保留：

- UTF-8 文件名；
- 文件长度；
- 完整文件 CRC32；
- 原始字节内容。

### 为什么要保留这些信息

接收端在解调前不知道文件叫什么、多少字节、要解多少块。固定 128-byte header 提供这些
信息。完整文件 CRC32 则回答一个更严格的问题：恢复文件是否逐 bit 完全正确。

At this stage the modem deliberately ignores file semantics. It reads an ordered byte
string and records enough metadata to reconstruct its name, length, and integrity.

## 3. 第二步：构造 Header 和数据块 / Frame the File

这一层把“无结构字节流”变成可独立检查的协议帧。

### 3.1 固定 128-byte header

header 使用 big-endian 编码，布局如下：

| 字段 | 字节数 | 当前示例 |
|---|---:|---|
| magic | 4 | `AM7F` |
| version | 1 | 1 |
| filename length | 1 | UTF-8 名称长度 |
| reserved | 2 | 0 |
| file size | 8 | 12,936 |
| block size | 2 | 512 |
| block count | 2 | 26 |
| whole-file CRC32 | 4 | 源文件 CRC |
| filename storage | 100 | 文件名后补零 |
| header CRC32 | 4 | 前 124 bytes 的 CRC |

数学上，CRC 是对有限域上的多项式做除法并保存余数。它擅长**检测**错误，但不会修正错误。
若 header CRC 不通过，接收端不会相信其中的文件长度和块数，以免错误长度导致后续解析失控。

### 3.2 512-byte 文件块

文件按 512 bytes 切分。每个传输块固定为 520 bytes：

```text
[block index: 2][real length: 2][payload padded to 512][CRC32: 4]
```

对 12,936-byte TIFF：

```text
前 25 块：每块 512 bytes
最后 1 块：136 bytes 有效数据，其余位置补 0
总计：26 块
```

块 CRC 只覆盖 `index + real length + real payload`，不把末块 padding 当作文件内容。
因此一段录音即使局部受损，也能明确指出哪些块可信，而不是只得到一个无法解释的坏文件。

### 实现位置

- `header_bytes()` / `parse_header()`：构造与验证 header；
- `data_block_bytes()` / `parse_data_block()`：构造与验证数据块；
- `coded_frames()`：按 `header1, header2, header3, block0...` 排列帧。

Framing adds machine-readable boundaries and layered integrity checks. CRC does not
repair data; it prevents silently accepting an incorrect header, block, or whole file.

## 4. 第三步：字节变 bit、FEC 编码和交织 / Bits, FEC, and Interleaving

### 4.1 字节变成 bit

每个 byte 按 MSB-first 展开：

```text
0xA6 = 10100110
```

实现是 `np.unpackbits(..., bitorder="big")`。接收时用相同 bit order 做逆变换。

### 4.2 K=7、rate-1/2 卷积编码

编码器保留最近 6 个输入 bit，加当前 bit 后形成 7-bit 寄存器。每输入一个 bit，分别与
两个生成多项式做按位与并计算奇偶校验：

```text
g0 = 171 octal = 1111001 binary
g1 = 133 octal = 1011011 binary
(c0, c1) = (parity(register & g0), parity(register & g1))
```

因此 1 个原始 bit 产生 2 个 coded bits，码率约为 1/2。帧末再输入 `K-1=6` 个 0，把
编码器状态拉回全零，使 Viterbi 回溯有确定终点。长度公式是：

```text
encoded_bits = 2 * (raw_bytes * 8 + 6)
```

所以：

```text
128-byte header -> 2 * (128*8 + 6) = 2,060 coded bits
520-byte block  -> 2 * (520*8 + 6) = 8,332 coded bits
```

header 被发送三份，因此总 coded bits 为：

```text
3*2,060 + 26*8,332 = 222,812 bits
```

三份 header 不是先分别硬判决再投票。接收端会先对三份的软信息求和，再进行一次
Viterbi 解码；高置信度的一份可以弥补另一份的弱置信度。

### 4.3 为什么还要交织

声学干扰常连续影响一段时间，产生 burst errors。若相邻 coded bits 原样连续发送，一次
碰撞可能破坏同一条 Viterbi 路径中的一长串信息。交织用固定 seed 生成伪随机排列：

```text
transmitted = coded_bits[random_permutation]
```

接收端用同一 seed 逆排列。这样空中连续错误回到编码域后被打散，更接近卷积码擅长处理的
分散错误。各帧 seed 不同：

```text
header repeat r: 7027 + r
data block i:    7027 + 1000 + i
```

The convolutional code adds controlled redundancy. Interleaving does not change the
number of errors, but spreads acoustic bursts across the codeword so that the Viterbi
decoder sees a less concentrated error pattern.

## 5. 第四步：映射到 BPSK 数据和旋转 Pilot / Map Bits and Pilots

### 5.1 BPSK 数据映射

每个 coded bit 映射到一个实数星座点：

```text
x = 1 - 2b
b=0 -> x=+1
b=1 -> x=-1
```

BPSK 只有两个相反的点，和 QPSK/QAM16 相比，在同样信噪比下判决距离更大，代价是每个
数据子载波只携带 1 bit。

### 5.2 rotating comb pilot

78 个 active bins 不会全部传数据。每个逻辑 OFDM symbol 约每 4 个 bin 选一个 pilot：

```text
pilot offset = logical_symbol_index mod 4
```

两个不连续频带分别执行这个规则。下一个 symbol 的 offset 向前移动，因此连续 4 个逻辑
symbols 后，每个 active bin 都至少当过一次 pilot。

pilot 值不是全 1，而是由 `seed=2027+symbol_index` 生成的已知 QPSK：

```text
p in {(+1+j), (+1-j), (-1+j), (-1-j)} / sqrt(2)
```

它们不携带文件数据，作用是让接收端在 payload 内持续测量当前相位和局部信道。其余约
3/4 bins 放 BPSK coded bits。最后一个 symbol 没填满的位置保持 0。

### 5.3 逻辑 payload

`payload_symbols()` 按 coded bit 顺序不断填充 data bins，并记录每行实际装入数量。
当前 222,812 bits 共形成 3,809 个逻辑 OFDM symbols。

Pilots spend roughly one quarter of the active subcarriers on channel observations.
Their offset rotates so the receiver periodically observes every active bin rather
than permanently sacrificing one fixed subset.

## 6. 第五步：插入 Payload Anchors / Insert Timing Anchors

pilot 和 anchor 都是已知符号，但用途和尺度不同：

| 已知信号 | 位置 | 覆盖 | 主要用途 |
|---|---|---|---|
| comb pilot | 每个数据 symbol 内 | 约 1/4 bins | 每 symbol CPE、局部 H、LLR 噪声 |
| payload-start anchor | header 之前 8 symbols | 全部 78 bins | 精确 payload 起点、header 前刷新 H |
| periodic anchor | 每 128 个逻辑 symbols 后 8 symbols | 全部 78 bins | 全段 PPM 拟合、周期全频 H 刷新 |

空中 payload 结构是：

```text
[start anchor x8]
[logical data x128][timing anchor x8]
[logical data x128][timing anchor x8]
...
[remaining logical data]
```

起始 anchor 固定 seed 为 10028；第 `i` 个周期 anchor 用 `9028+i`。不同 seed 让相关模板
具有唯一性，降低把相邻重复结构认错的概率。

anchor 不产生 LLR、不消耗 coded bit，也不推进 rotating-pilot 的逻辑序号。当前插入
29 组周期 anchor，共增加 232 个 symbols，物理 payload 从 3,809 增至 4,041 symbols。

The start anchor protects the most consequential boundary: the beginning of the
encoded header. Periodic anchors provide observations distributed across the whole
recording, which is necessary because a short training-only clock estimate can drift
by many samples near the end of a long recording.

## 7. 第六步：OFDM 调制 / OFDM Modulation

### 7.1 建立频域向量

对每个 symbol 建立长度 512 的复数向量 `X[k]`：

- bins 64-120、158-178 放数据、pilot 或 anchor；
- 其他正频率 bins 置 0；
- 负频率位置放共轭镜像 `X[N-k]=conj(X[k])`。

共轭对称保证 IFFT 输出为实数，可以直接由普通声卡播放。

子载波间隔与实际频率为：

```text
delta_f = Fs/N = 48000/512 = 93.75 Hz
f_k = k * 93.75 Hz
```

### 7.2 IFFT

时域有效区由 512 点逆变换得到：

```text
x[n] = (1/N) * sum(X[k] * exp(j*2*pi*k*n/N)), n=0...511
```

不同子载波在 512-sample 窗口内正交。理想同步且信道保持近似不变时，接收端 FFT 可以把
它们重新分离，互不干扰。

### 7.3 循环前缀 CP

将有效区最后 256 samples 复制到 symbol 前面：

```text
[x[256:512]][x[0:512]]
```

CP 时长为 5.333 ms，有效区 10.667 ms，总 symbol 为 16 ms。若主要声学多径延迟短于 CP，
线性卷积在去 CP 后可近似成循环卷积，于是频域每个 bin 满足简单模型：

```text
Y[k] = H[k] X[k] + W[k]
```

这正是后面可以用 `H=Y/X` 均衡的原因。CP 不携带新信息，因此 CP 越长，多径容忍度越高，
但有效吞吐率越低。

### 实现位置

- `ofdm_tx()`：Hermitian 镜像、IFFT、添加 CP；
- `ofdm_rx()`：按 768 samples 分行、去 CP、FFT、取 active bins。

OFDM turns frequency-domain symbols into a real waveform. The cyclic prefix converts
a sufficiently short multipath channel into one complex multiplication per bin,
which makes equalization tractable.

## 8. 第七步：拼接并写成发送 WAV / Assemble the Transmit WAV

完整发送顺序是：

```text
[0.5 s silence]
[sync 64]
[preamble 128, seed 8026]
[preamble 128, seed 8027]
[payload-start anchor 8]
[single payload with periodic anchors]
[0.25 s silence]
```

当前文件的精确时长分解为：

```text
opening silence                  0.500 s
sync: 64 * 0.016                1.024 s
preamble: 256 * 0.016           4.096 s
payload-start anchor: 8 * 0.016 0.128 s
physical payload: 4041 * 0.016 64.656 s
tail silence                     0.250 s
total                           70.654 s
```

### 每段为什么存在

- 开头静音：测量当前录音底噪，而不是假设一个固定噪声值；
- sync：通过互相关找到发送信号大致从哪里开始；
- 两段 preamble：在 payload 前对全部 active bins 测 H 和训练 residual；
- start anchor：锁定 header 起点并在最后一刻刷新 H；
- periodic anchors：跟踪长录音时钟和慢变 H；
- 末尾静音：留出播放/录音停止余量。

所有时域段拼接后，`write_wav()` 统一缩放，使全文件绝对峰值为约 0.95 full scale，再裁剪并
写成 mono、16-bit PCM、48 kHz WAV。这里的统一 gain 不改变各段相对幅度。

生成命令：

```bash
python tx_step8.py \
  data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --out data/step8_clock_anchor/observatory_64_uncompressed_step8_start_anchor.wav
```

同时产生 `.meta.json`、`.sync.npy`、`.preamble.npy`、`.start_anchor.npy`、
`.anchors.npy` 和 `.anchor_starts.npy`。这些 sidecars 是可复现实验的理论真值，不需要播放。

The final WAV is deterministic for fixed input bytes and seeds. Sidecars preserve the
known symbol sequences and frame metadata used for inspection and reproducibility.

## 9. 第八步：播放、空气传播和录音 / Acoustic Transmission

### 你做了什么

1. 先启动 mono、48 kHz、16-bit 录音；
2. 留约 1 秒环境声；
3. 完整播放发送 WAV，期间保持设备位置和系统音量尽量稳定；
4. 播完后再留约 1 秒，停止录音。

推荐录制约 73-74 秒。示例：

```bash
pw-record --rate 48000 --channels 1 --format s16 --sample-count 3552000 \
  data/step8_clock_anchor/receive_observatory_step8_start_anchor_1.wav
```

### 空气信道实际改变了什么

接收信号不是发送信号的简单缩小版。近似模型是：

```text
r(t) = gain * (h(t) * s(t-tau)) + noise(t) + interference(t)
```

其中包括：

- 扬声器、房间和麦克风共同形成的频率选择性 `H[k]`；
- 传播与启动造成的未知起点 `tau`；
- 播放与录音时钟不完全相同造成的采样比例误差 PPM；
- 慢变公共相位 CPE；
- 环境噪声、自动增益、削波和偶发碰撞。

“录法一样”不意味着 `H`、噪声和时钟轨迹完全相同，所以所有关键估计都嵌入同一段录音，
而不是复用之前扫频或之前录音的 H。

The acoustic path is time-varying and frequency-selective. Step8 therefore estimates
timing, noise, channel, and phase from known signals inside the same recording.

## 10. 第九步：读取录音与粗同步 / Read and Coarsely Synchronize

`read_wav()` 首先严格检查录音必须是 mono、16-bit、48 kHz，再把整数 PCM 归一化为浮点数。

接收端重新生成相同 seed 的 sync，并只用前 8 个 sync symbols 形成相关模板。对所有可能
起点 `m` 计算：

```text
C[m] = |sum r[m+n] conj(s[n])| / (||r_m|| ||s||)
```

取最大值位置作为 `coarse_sync_start`，归一化值作为 `sync_score`。短模板降低长模板在尚未
校正 PPM 时因逐渐错位而相关抵消的风险。

粗同步只负责把搜索带到正确附近；最终的小数起点和采样比例由下一步多个 anchors 联合决定。

Normalized correlation is insensitive to overall gain. The short sync template finds
the neighborhood of the frame without requiring clock correction to be known first.

## 11. 第十步：用全段 Anchors 估计 PPM / Estimate Clock PPM

### 11.1 为什么需要 PPM

播放 DAC 和录音 ADC 都标称 48 kHz，但实际晶振有微小误差。假设理论采样位置为 `s_i`，
录音中观察到的位置为 `r_i`：

```text
r_i = a + alpha * s_i
ppm = (alpha - 1) * 1,000,000
```

仅 10 ppm 在 70 秒末端就约累积：

```text
48000 * 70 * 10e-6 = 33.6 samples
```

这足以把 FFT 窗口推到错误位置并造成后半段 BER 突变。

### 11.2 anchor 位置怎样测得

接收端先在 training 内每隔 32 symbols 取 8-symbol 模板，再搜索 start anchor 和各周期
anchor。每个候选只在预计位置 `+-128 samples` 内做局部归一化相关。

离散峰值两边三个点再做抛物线插值，得到亚采样峰位置：

```text
delta = 0.5*(left-right)/(left-2*middle+right)
```

相关分数低于 0.12 的候选先拒绝。

### 11.3 稳健直线拟合

拟合先用 Theil-Sen 得到抗异常初值，再迭代加权最小二乘：

- 基础权重为 `correlation_score^2`；
- 用 MAD 估计 residual 尺度；
- Huber 权重降低异常峰的影响；
- 最后剔除偏离中位 residual 过远的候选并重拟合。

至少要有 4 个有效 payload anchors，且覆盖可分析 payload 时间跨度的一半，才使用全段
`clock_status=full`。否则退回 training-only 拟合，并明确记录
`clock_status=training_fallback`。

### 11.4 重采样校正

得到 `(a, alpha)` 后，对统一时间轴 `n` 从原录音位置 `a+n*alpha` 线性插值：

```text
corrected[n] = interp(rx, a + n*alpha)
```

校正后，sync 理论上从 corrected sample 0 开始，每个 OFDM symbol 再次严格占 768 samples。

Multiple anchors convert clock estimation into a robust line-fitting problem. This
removes accumulated sampling drift before FFT processing rather than asking phase
tracking to hide a timing error it cannot reliably absorb.

## 12. 第十一步：估计底噪和初始 H / Estimate Noise and Initial Channel

### 12.1 底噪

接收端截取 sync 前最多 0.5 秒录音，按 OFDM 窗口 FFT，并对每个 active bin 计算功率中位数：

```text
silence_variance[k] = median(|Y_silence[k]|^2)
```

中位数比均值更不容易被静音段中的单次碰撞拉高。

### 12.2 训练 H

sync 和两段 preamble 的发送符号 `X_b[m,k]` 已知，接收端分别计算：

```text
H_b[k] = mean_m(Y_b[m,k] / X_b[m,k])
```

三个 block 之间可能有不同公共相位，因此先以最后一个 block 为参考做公共相位对齐，再求均值：

```text
H_initial[k] = mean_b(aligned(H_b[k]))
```

同时计算训练 residual：

```text
training_variance_b[k] = mean_m(|Y_b[m,k] - H_b[k]X_b[m,k]|^2)
training_variance[k] = median_b(training_variance_b[k])
```

所以当前 H **不是一次测量**，而是 sync 64 rows、preamble 128x2 rows 分块平均、相位对齐后
再平均的结果。之后还会被 start anchor、comb pilots 和周期 anchors 更新。

The initial channel estimate averages hundreds of known observations per active bin.
Separate block estimates are phase-aligned before averaging so a common phase change
does not cancel otherwise consistent channel measurements.

## 13. 第十二步：锁定 Payload 起点并刷新 H / Start Anchor

PPM 校正后，接收端知道 start anchor 应紧贴 preamble 结束处，但仍在 `+-16 samples` 搜索
其 8-symbol 唯一模板。

- 分数不低于 0.12：使用相关峰，状态为 `detected`；
- 分数过低：使用协议结构边界，状态为 `structural_fallback`。

在选定的同一 OFDM 网格上，对 8 行全频 QPSK 计算 `Y/X`。每行先相对当前 H 做公共相位
对齐，再平均成 `H_start`，最后融合：

```text
H <- 0.5 * H_initial + 0.5 * H_start
payload begins immediately after 8 anchor symbols
```

这一步解决旧格式的危险点：弱信号下只靠 comb-pilot residual 选择 payload 起点，可能在 CP
内部选到一个“看似也不错”但带额外频率相位斜率的位置，header 会首先崩溃。

The dedicated start anchor jointly establishes the exact header boundary and refreshes
all active-bin channel estimates immediately before the header is demodulated.

## 14. 第十三步：逐 Symbol 相位、H 和软解调 / Track and Demodulate

### 14.1 每个数据 symbol 的 pilot 观测

已知 pilot `P[k]` 给出当前观测：

```text
H_observed[k] = Y[k] / P[k]
```

将它与当前 H 比较，估计所有 pilot 共有的相位旋转 CPE：

```text
CPE = angle(weighted sum(H_observed/H_current))
```

CPE 每个数据 symbol 都应用。当前正式模式 `--phase-slope off` 不应用逐频率 phase slope，
因为真实录音中瞬时 slope 噪声曾使结果变差。`slow` 模式仍可用于显式对照实验。

### 14.2 更新 pilot bins 的 H

去掉 CPE 后，pilot 位置使用指数平滑：

```text
H[pilot] <- 0.65 * H[pilot] + 0.35 * H_observed_base[pilot]
```

由于 comb offset 旋转，四个 symbols 后全部 active bins 都获得过一次局部更新。

### 14.3 周期 anchor 更新全频 H

每处理 128 个逻辑 symbols，接下来的 8 个物理 rows 被识别为 anchor，不产生 bit。8 个
`Y/X` 先公共相位对齐并平均，再融合：

```text
H <- 0.5 * H + 0.5 * H_anchor
```

这比 comb pilot 更密集地观察全部 bins，可修复慢变频响；anchor 后清空 slope 历史。

### 14.4 形成 soft LLR

对数据 bin，不先强制判为 0 或 1，而是计算带符号的置信度：

```text
variance[k] = max(training_noise[k], silence_noise[k], current_pilot_residual)
LLR[k] = 2*Re(conj(H_effective[k])*Y[k]) / variance[k]
```

- `LLR > 0` 倾向 coded bit 0；
- `LLR < 0` 倾向 coded bit 1；
- 绝对值越大，置信度越高；
- 最终裁剪到 `[-24, 24]`，避免少数极端值支配 Viterbi。

注意这里等价于 matched-filter 判决，不必显式先算 `Y/H`。乘 `conj(H)` 会把信号旋回实轴，
除以噪声方差则让干净 bin 的证据权重更大。

Per-symbol pilots correct common phase and update a rotating subset of H. Full-band
anchors refresh all bins periodically. The demodulator outputs confidence-valued LLRs,
not hard bits, preserving information needed by the soft Viterbi decoder.

## 15. 第十四步：解 Header / Decode the Header

接收端从 LLR 流开头取三段各 2,060 个值：

1. 每段按自己的 seed 去交织；
2. 三份对齐后的 LLR 逐元素相加；
3. 在 64 状态 trellis 上运行 soft Viterbi；
4. 回溯到已知全零状态，取前 1,024 个原始 bits；
5. 打包成 128 bytes；
6. 检查 header CRC、magic、version 和字段一致性。

Viterbi 的目标不是逐 bit 选最大 LLR，而是在所有合法卷积码路径中选择总 branch metric 最大
的路径。当前实现中某条编码分支期望符号为 `+1/-1`，其 metric 与对应两项 LLR 点积。

header 成功后，接收端才知道恢复文件名、准确长度和 26 个 block。如果 header 失败，程序
仍保存 LLR、H、clock、phase 和 `decoded_header.bin`，但不会猜测文件结构。

Three independently interleaved header copies are combined in the soft domain before
one Viterbi decode. Header CRC and structural checks prevent corrupted metadata from
driving the rest of the decoder.

## 16. 第十五步：解数据块并重组文件 / Decode Blocks and Reassemble

对 header 声明的每个 block：

1. 从 LLR 流取 8,332 项；
2. 用该 block seed 去交织；
3. soft Viterbi 解出 4,160 bits，即 520 bytes；
4. 检查 block index、有效长度和 block CRC32；
5. CRC 正确则保存有效 payload；否则标为坏块。

所有块正确时按 index 拼接，并截断到 header 的精确 `file_size=12,936`。然后检查 header
中保存的完整文件 CRC32。只有完整文件 CRC 通过，输出才使用原文件名：

```text
runs/.../observatory_64_uncompressed.tiff
```

若 header 已恢复但某些块坏，程序仍输出相同长度的 `.partial` 文件；坏块位置补 0：

```text
runs/.../observatory_64_uncompressed.tiff.partial
```

`.partial` 是诊断产物，不代表通过校验。TIFF 可能仍能打开，也可能因关键目录或 strip 受损而
打不开；“能打开”不等于“字节正确”。

Each block is decoded and validated independently. The whole-file CRC is the final
source-independent proof that every reconstructed byte is correct.

## 17. 第十六步：事后准确性分析 / Post-Decode Accuracy Analysis

`--source` 是可选的实验真值，只用于**解码完成后的比较**，不参与同步、PPM、H、相位、
起点选择、FEC 或字节恢复。

有源文件时可以额外得到：

```text
coded_bit_error_rate = FEC 前硬判 coded bits 与理论 coded bits 的 BER
post_fec_bit_error_rate = Viterbi 后文件 payload bits 的 BER
file_match = 恢复 bytes 是否与源文件逐字节相同
```

CRC 不需要源文件，因此没有 `--source` 时仍能判断 header、每块和完整文件是否自洽正确。

推荐解码命令：

```bash
python rx_step8.py \
  data/step8_clock_anchor/receive_observatory_step8_start_anchor_1.wav \
  --source data/step8_clock_anchor/observatory_64_uncompressed.tiff \
  --phase-slope off \
  --out runs/step8_clock_anchor/start_anchor_1
```

Ground-truth comparison measures BER for experiments, but the decoder itself remains
source-independent. CRCs provide integrity decisions in real deployment where the
receiver cannot possess the source file in advance.

## 18. 输出文件怎样读 / How to Read Receiver Outputs

| 输出 | 回答的问题 |
|---|---|
| `metrics.json` | 最终是否成功，BER、CRC、PPM、同步和 anchor 状态如何 |
| `blocks.csv` | 哪个文件块通过或失败 |
| `clock_fit.png` | anchor 是否沿一条稳定时钟直线，哪些点被拒绝 |
| `clock_anchors.npy` | 每个 anchor 的理论/实测位置、分数、residual、采用状态 |
| `channel_and_noise.png` | 每个 active bin 的 `|H|` 与噪声水平 |
| `H.npy` | sync+preamble 得到的初始 H |
| `H_payload_start_anchor.npy` | start anchor 单独测得的 H |
| `H_anchor_track.npy` | 每个周期 anchor 的全频 H |
| `pilot_H_track.npy` | 每个逻辑 payload symbol 使用的有效 H |
| `phase_tracking.png` | CPE、slope 和 pilot residual 随时间变化 |
| `rx_llr.npy` | 完整 coded soft decisions |
| `decoded_header.bin` | Viterbi 后的原始 128-byte header |
| 恢复文件 | CRC 全过时为原名，否则通常为 `.partial` |

最重要的成功条件应同时满足：

```text
header_ok = true
blocks_ok = blocks_total
file_crc_ok = true
file_match = true          # 只有提供 source 时才有此项的完整意义
post_fec_bit_error_rate = 0
```

FEC 前 BER 不必为 0。当前成功的真实录音曾有约 5.56%-6.11% raw coded BER，但 FEC 后为
0，26/26 blocks 和整文件 CRC 均通过。这正是纠错编码存在的意义。

## 19. 一次完整实验中你到底做了什么 / What You Did in One Experiment

从工程责任看，一次实验可以压缩成下面十步：

1. 你选择了一个源文件；程序把它当作 bytes。
2. 程序写入 header、块序号、长度和三级 CRC。
3. 程序用卷积码增加冗余，并用交织打散潜在 burst errors。
4. 程序把 coded bits 映射成 BPSK，并嵌入 comb pilots。
5. 程序插入 payload-start anchor 和周期 anchors。
6. 程序执行 Hermitian OFDM、加 CP，生成可播放的 PCM WAV。
7. 你播放 WAV，并用真实麦克风录下经过声学信道的版本。
8. 程序从同一录音估计起点、PPM、噪声、H 和逐 symbol CPE。
9. 程序产生 soft LLR，去交织并用 Viterbi 纠错，再逐块检查 CRC。
10. 程序按 header 重组文件，并用完整文件 CRC 与可选 source comparison 给出结论。

这条链路中没有使用“以前某次扫频的 H”来强行修正当前录音，也没有用源文件内容选择最好
的同步或相位参数。发送端与接收端共享的是协议参数和伪随机 seed，而不是待恢复 payload。

In operational terms, you supply a file and an acoustic recording. The modem supplies
deterministic framing, waveform generation, in-recording estimation, error correction,
integrity checks, and the recovered file. Shared seeds describe known training signals;
they do not reveal the unknown payload.

## 20. 常见失败对应哪一层 / Mapping Failures to Stages

| 现象 | 首先检查 | 更可能的层 |
|---|---|---|
| `sync_score` 很低 | 录音音量、是否完整播放、频带是否被设备滤掉 | 播放/录音、粗同步 |
| `clock_status=training_fallback` | 后半段 anchors 分数和录音长度 | PPM anchor 检测 |
| clock residual 突然跳变 | 中途噪声、多径峰切换、丢采样 | 时钟拟合/录音链路 |
| start anchor 分数低 | preamble 后噪声、起点附近削波 | payload 起点 |
| header 失败、全段 BER 高 | H、payload delta、CPE、录音 SNR | 均衡/软解调 |
| 前面好、后面突然坏 | PPM 是否正确、anchor 是否覆盖全段 | 时钟漂移 |
| 少数 blocks CRC 失败 | `blocks.csv` 与 pilot residual 时间位置 | 局部 burst error |
| CRC 全过但 `file_match=false` | source 是否真是同一版本 | 实验真值路径 |
| `.partial` 能打开但画面坏 | 坏块是否落在像素区或 TIFF 结构区 | 文件格式容错，不是 modem 成功 |

排查时应顺着接收链路向下走：先确认同步和 clock，再看 start anchor 与 H，然后看 pilot
residual/LLR，最后才看 FEC 和文件格式。这样不会把上游时间错误误判成“FEC 不够强”。

## 21. 代码调用关系 / Implementation Map

```text
tx_step8.py
  coded_frames()
    header_bytes(), data_block_bytes()
    bits_from_bytes(), conv_encode(), interleave()
  payload_symbols()
    pilot_indices(), pilot_values(), BPSK mapping
  frame_payload()
  ofdm_tx(), write_wav()

rx_step8.py
  read_wav(), find_sync()
  detect_clock_anchors(), robust_clock_fit(), correct_clock()
  ofdm_rx(), estimate_training(), noise_variance()
  correlate_near(), estimate_anchor_channel()
  payload_llrs_anchored()
    CPE, H updates, anchor removal, soft LLR
  decode_header()
    deinterleave(), viterbi_decode(), parse_header()
  decode_blocks()
    deinterleave(), viterbi_decode(), parse_data_block()
  whole-file CRC, optional source BER/file_match, plots and metrics
```

核心 DSP 和协议逻辑全部位于 `step8_modem.py`；两个 CLI 负责参数、文件路径、流程编排和
结果落盘。因而未来实现 TUI 时，界面应调用同一套生成/解码逻辑，而不复制算法。

## 22. 最后一句话 / Final Mental Model

Step8 的本质不是“把文件声音化后再猜回来”，而是建立一条可观测、可校正、可验证的数字
通信链路：已知 training/anchor/pilot 负责估计信道，CP+OFDM 把多径问题拆成逐 bin 复数
增益，soft FEC 负责修正剩余 bit errors，分块 CRC 和完整文件 CRC 负责证明恢复结果可信。

Step8 is a measured digital link: known signals estimate the current acoustic path,
OFDM makes that path equalizable, soft FEC repairs residual errors, and layered CRCs
prove whether the reconstructed file is trustworthy.
