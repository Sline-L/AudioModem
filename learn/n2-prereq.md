# N2 预备知识：读懂代码前必须会的 9 件事

三人第 1 周共用。每件事都落到 `n2/` 里的一个函数。实验脚本在 `learn/experiments/`，不改 `n2/` 和 `standard/`。

读代码时先看数组的 `shape`，再看公式。`DESIGN.md` 里的 Wiener、NLMS、Turbo 不在主路径上，本章不讲。

---

## 1. 采样与 FFT bin

连续频率 \(f\) 在长度 \(N\)、采样率 \(f_s\) 的 FFT 里落在下标

\[
k = \frac{f}{f_s} N, \qquad f = k \frac{f_s}{N}.
\]

`params.derived()`：\(f_s=48000\)，\(N=8192\)，数据子载波是 `bin 400 .. 2391`（`phy.active_bins`）。

- bin 400：\(400 \times 48000 / 8192 \approx 2344\,\text{Hz}\)
- bin 2391：\(\approx 14010\,\text{Hz}\)

声学通带大约 2–14 kHz。bin 0 是直流，bin \(N/2=4096\) 是 24 kHz。

发射的是实信号，频谱共轭对称：`X[N-k] = conj(X[k])`。`phy.bytes_to_qpsk_grid` 和 `phy.train_freq_time` 都写了这一行。IFFT 之后取 `.real`，虚部数值上应接近 0。

`np.fft.rfft` 只返回 \(0..N/2\)，长度 \(N/2+1=4097\)。`align.extract_active_sfo` 用 rFFT，再切 `bins`，所以返回形状是 `(n_sym, 1992)`，不再带镜像 bin。

**5 分钟实验**

```python
import numpy as np
fs, N = 48000, 8192
print(400 * fs / N, 2391 * fs / N)
```

**自测**

1. bin 1000 大约多少 Hz？
2. 为什么接收端做信道估计时可以只保留正频率 1992 个 bin？

---

## 2. OFDM 一个符号

一个符号 = 循环前缀 + IFFT 实部。`N=8192`，`CP=2048`，`symbol_len=10240`。

`phy.train_freq_time`：

```python
time_body = np.fft.ifft(freq, axis=1).real
with_cp = np.hstack((time_body[:, -cp:], time_body))  # (16, 10240)
```

最后 2048 点复制到开头，就是循环前缀。多径时延短于 CP 时，去掉 CP 再做 FFT，线性卷积变成圆周卷积，每个 bin 上只剩一个复数增益 \(H[k]\)，不和相邻符号叠在一起。2048 点 / 48000 Hz ≈ 42.7 ms，室内声反射一般落在这个量级以内。

数据符号在 `example_tx.Ofdm_tx.time_domain_convert` 里做同样的拼接，再把整段 OFDM 体按峰值归一化到 0.8。

**自测**

1. 一个符号多少样点？CP 占多少毫秒？
2. CP 比多径还短时，FFT 之后的 bin 还会是「一个 \(H[k]\)」吗？

---

## 3. 这套 QPSK 映射

不是 Gray 码教材里的 45° 星座。发射端 `Ofdm.mapp` 与 `phy.bytes_to_qpsk_grid`：

```python
d_i = i // 4
b_i = 3 - (i % 4)
I = 1 - 2 * ((byte >> (2*b_i    )) & 1)
Q = 1 - 2 * ((byte >> (2*b_i + 1)) & 1)
X[400+i] = I + 1j*Q
```

- 1992 个子载波，每 4 个用掉 1 个字节，一块正好 498 字节。
- 字节内从高位对走到低位对：`i%4=0` 时 `b_i=3`，取 bit7、bit6。
- 每对里 **低位走 I，高位走 Q**。bit 0 → +1，bit 1 → −1。
- 星座点只有 `±1±j`，幅度 \(\sqrt{2}\)。

`phy.slice_qpsk` 硬判决只看实部、虚部的符号。`phy.qpsk_llr_scaled` 先输出 Q 的 LLR 再输出 I 的 LLR，拼回去仍是 bit7、bit6、bit5…bit0。

字节 `0x80`（只有 bit7=1）：第一个符号 Q=−1、I=+1，即 `1-1j`；另外三个符号都是 `1+1j`。

**自测**

1. 字节 `0x01` 的第 4 个符号（`i%4=3`）I、Q 各是多少？
2. 为什么硬判决不用算欧氏距离？

---

## 4. 匹配滤波就是相关

模板 \(h[n]\) 时间反转后与接收信号卷积，等于相关：

```python
corr = fftconvolve(rx, template[::-1], mode="valid")
metric = corr * corr
```

`mode="valid"` 的第 0 个输出对应模板与 `rx[0:L]` 对齐，峰值下标就是模板起点。平方只是为了看能量。

`capture.parabolic_peak` 在整数峰 \(i\) 及左右两点 \(a,b,c\) 上拟合抛物线：

\[
\delta = \mathrm{clip}\left(\frac{a-c}{2(a-2b+c)},\,-1,\,1\right), \qquad \text{亚采样位置}=i+\delta.
\]

r1 的第一训练起点因此是 `212925.047`，不是整数。`capture.matched_chirp_metric` 和 `align.matched_start` 都走这条路。

**自测**

1. `valid` 相关的长度和 `rx`、模板长度是什么关系？
2. 抛物线修正量为什么被限制在 ±1 个样点？

---

## 5. 线性调频为什么用来同步

`phy.linear_chirp` 复现发射端相位（函数名叫 `log_chirp_gen`，注释掉的对数扫频不要用）：

```python
t = np.linspace(0, T, int(fs*T), endpoint=False)   # T=3.0, 144000 点
phi = 2*pi*100*t + pi*(20000-100)*t*t/T
```

频率从 100 Hz 扫到 20 kHz。它的自相关很尖：只有对齐时正频率分量才同相相加，偏开几个样点能量就散掉。所以相关峰可以定到亚采样，并且比单频正弦抗多径。幅度系数 `amp=0.8`。OFDM 体另外做峰值归一化，扫频本身用这个幅度。

**自测**

1. Chirp 多少样点？静音多少样点？
2. 本地模板的相位差一个符号，峰会怎样？

---

## 6. 采样钟偏 SFO

录音设备的采样率和 48000 Hz 有相对偏差 \(s\)，单位是无量纲比例，ppm 是 \(s\times 10^6\)。r1 约 −11.65 ppm，即 \(s\approx -1.165\times 10^{-5}\)。

偏差随时间累积。一个符号 10240 点，r1 的 OFDM 体是 \(8+42+8=58\) 个符号、593920 点：

\[
|s|\times 593920 \approx 6.9 \text{ 个样点}.
\]

只锁开头、后面仍按 10240 等间隔切，尾部会滑出 FFT 窗。主路径不整段重采样（r1 的 `resampled=false`），而是在 `align.extract_active_sfo` 里把钟偏放进取样位置：

```python
pos = start + (1+sfo) * (m*10240 + 2048 + n)
```

`+2048` 先跳过 CP，再对原始 WAV 线性插值，rFFT，只留 bin 400–2391。越界抛 `ValueError`，调用方把这次候选当成失败。

**自测**

1. +20 ppm、100 个符号，累积大约多少样点？
2. `resampled=false` 是不是表示没有补偿钟偏？

---

## 7. 最小二乘信道

训练符号的频域已知，记为 \(X\)，接收为 \(Y\)。每个有效 bin：

\[
H = \mathrm{mean}(Y/X), \qquad \text{每音残差}=\frac{|Y-HX|^2}{|H|^2}.
\]

`channel.h_and_llr_scale` 对多行训练做这个平均。有效音是 \(|H|\) 大于最大幅度 \(10^{-8}\) 倍的 bin。噪声方差取有效音残差的中位数，并且不小于 \(10^{-3}\)。

训练 QPSK 幅度是 \(\sqrt{2}\approx 1.41\)。r1 的 `mean_abs_h≈2.34`，所以平均 \(|Y|\) 大约是 \(|H|\sqrt{2}\)。这个数包含发射归一化和录音音量，用来比较不同录音，不是一个固定常数。

**自测**

1. 8 个训练、1992 个 bin，`Y/X` 的 shape 是什么？`H` 的 shape 是什么？
2. 残差除以 \(|H|^2\) 之后，深衰落的 bin 会被放大还是缩小？

---

## 8. LLR

\[
\mathrm{LLR}=\log\frac{P(\mathrm{bit}=0)}{P(\mathrm{bit}=1)}.
\]

正值更像 0。映射是 \(I=1-2b\)，所以实部为正时 I 比特更像 0。`phy.qpsk_llr_scaled`：

```python
llr[0::2] = z.imag * scale   # Q
llr[1::2] = z.real * scale   # I
return np.clip(llr, -30, 30)
```

`channel.h_and_llr_scale` 里的尺度只改幅度：

```python
scale = 3 * clip(sqrt(noise_var / 每音残差), 0.25, 4)
```

残差小的子载波尺度接近上限 12，深衰落或很吵的子载波被压到 0.75 附近。差的子载波少影响 LDPC，但不翻转比特。裁剪到 ±30 是为了不让个别极大 LLR 把置信传播里的 `tanh` 打满。

`phy.descramble_llr`：扰码比特为 1 时把该 LLR 乘 −1。这是「比特异或 1 会交换 0 和 1 的似然」。先硬判再异或会丢掉软信息。

**自测**

1. `Z=0.2+0.9j`，`scale=3`，两条 LLR 是多少？两个比特更像 0 还是 1？
2. 为什么尺度函数不能改 LLR 的符号？

---

## 9. LDPC 与扰码

一个数据符号：1992 个 QPSK × 2 bit = 3984 bit = 498 字节，这是一个码字。码率 1/2，信息 1992 bit = 249 字节。

`decode_path.make_coder`：

```python
ldpc.code(standard="802.16", rate="1/2", z=166)  # z = 498/3
app, it = coder.decode(llr, dectype="sumprod2")
```

`coder.N=24z=3984`，`coder.K=12z=1992`。校验矩阵来自 802.16 原型图，移位取 `proto % z`，必须和发射端用同一个 `LDPC/new_ldpc`。`app<0` 判为 1。`it<200` 视为收敛，200 是库的迭代上限，表示没收敛。

译码看的是每个比特「更像 0 还是更像 1」的程度，所以输入必须是 LLR，不是 0/1。

扰码是 15 位 LFSR，种子 `0x5A4D`，输出 bit14，反馈 bit14 XOR bit13。`phy.lfsr_sequence`。**每个码字从种子重新开始。** 发射端先 LDPC 再与扰码异或，字节内 MSB 先入。接收端在 LLR 上解扰，见第 8 节。

Windows 上需要 `LDPC/new_ldpc/bin/c_ldpc.dll`。没有这个文件时 `LdpcBank()` 失败，主路径在译码前返回阶段 `ldpc`。

**自测**

1. 249、498、1992、3984 之间的乘法关系是什么？
2. 迭代次数等于 200 表示什么？

---

## 读数值代码的习惯

- 先打印 `shape` 和 `dtype`。`np.asarray(..., dtype=np.float64).ravel()` 把输入收成一维双精度。
- `fftconvolve(..., mode="valid")` 的长度是 `len(rx)-len(template)+1`。
- `rfft` 输出长度是 `N/2+1`，下标就是 bin 号。
- 复数除法 `Y/X` 在 `X` 接近 0 时会爆，代码里写成 `X+1e-12`。
- 随机数：训练序列用 `random.seed` / `random.randint`。NumPy 的随机序列和它不同，换了之后训练符号对不上，相关峰和信道估计一起错。

---

## 答案

1. bin 1000 ≈ \(1000\times48000/8192 \approx 5859\,\text{Hz}\)。实信号镜像与正频率共轭，主路径的 rFFT 只保留正频率；镜像在发射端已经由共轭对称确定。
2. 10240 点；\(2048/48000\approx 42.7\,\text{ms}\)。CP 短于多径时，符号间干扰还在，一个 bin 不再是单个 \(H[k]\)。
3. `i%4=3` 时 `b_i=0`，取 bit1、bit0。`0x01` 只有 bit0=1，所以 I=−1，Q=+1。四个点都在对角线上，符号判决就是最大似然。
4. `len(rx)-len(template)+1`。三点只能确定峰在相邻整数之间，修正量超过 1 说明整数峰选错了。
5. 144000 与 24000。相位反号时模板和信号不正交叠加，主峰消失或变成旁瓣。
6. \(20\times10^{-6}\times100\times10240\approx 20.5\) 个样点。`resampled=false` 只表示没有整段重采样；钟偏在 `extract_active_sfo` 的取样网格里。
7. `(8, 1992)` 与 `(1992,)`。除以 \(|H|^2\) 把深衰落的残差放大，这些 bin 的 `scale` 会被压低。
8. LLR_Q=2.7，LLR_I=0.6，都为正，两个比特都更像 0。尺度只表示可信度。
9. \(498\times8=3984\)，\(3984/2=1992\)，\(1992/8=249\)，\(1992/4=498\)。200 表示到达迭代上限，校验没有全部满足。
