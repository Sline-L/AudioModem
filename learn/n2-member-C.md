# 成员 C：从频谱到 249 字节

在 `learn/n2-course.html` 第 3 章之上加深。给定 B 的起点和钟偏，说明一个数据符号怎样变成 249 字节，以及失败录音里未收敛的码字落在哪里。

实验：

```text
python learn/experiments/c1_symbol_walk_r1.py
python learn/experiments/c2_diagnose.py --run-dir run/n2/r11
python learn/experiments/c2_diagnose.py --run-dir run/n2/r4
```

`recover_array` 把 `turbo` 写成 `False`。Wiener、NLMS、Hermitian 门控不在主路径上。

---

## 1. 逐函数

### channel.h_and_llr_scale

`y, x` 的形状是 `(n_train, 1992)`。

```python
h = mean(y / (x + 1e-12), axis=0)          # (1992,)
carrier_noise = mean(|y - x*h|^2) / |h|^2  # 每音
noise_var = max(median(carrier_noise[valid]), 1e-3)
scale = 3 * clip(sqrt(noise_var / carrier_noise), 0.25, 4)
```

`valid` 是 `|H|` 大于最大幅度 \(10^{-8}\) 倍的 bin。`scale` 因此落在 `[0.75, 12]`。残差小的子载波尺度接近 12，差的子载波接近 0.75。这个尺度只乘 LLR 的幅度。

r1 的 `median_training_noise≈3.000`。字段名叫 noise，`report.py` 里却是 `art["training_noise"] = scale` 的中位数。3.0 表示典型尺度正好在 `3 * 1` 附近，不是噪声方差本身。真正的 `noise_var` 在 `diag["sigma2"]`，没有写进 `metrics.json`。

`mean_abs_h≈2.34` 是联合信道 `art["H"]` 的平均幅度。训练符号幅度 \(\sqrt{2}\approx 1.41\)，平均 `|Y|≈|H|√2`。用来比较录音，不是常数。

### channel.lerp_h

`H = (1-α) H_front + α H_tail`，`α` 限制在 `[0, 1]`。

### phy.qpsk_llr_scaled / descramble_llr / bits_to_bytes / slice_qpsk

`Z` 长度 1992，LLR 长度 3984，先 Q 后 I，裁剪 ±30。解扰是扰码比特为 1 时 LLR 乘 −1，每个码字单独调用。`bits_to_bytes` 按大端打包。`slice_qpsk` 是 `sign(real)+1j*sign(imag)`，给公共相位当参考，不直接当输出比特。

### receiver._equalize_once

1. `Z = Y/H`，只除 `valid` 的 bin。
2. `X̂ = slice_qpsk(Z)`。
3. `θ = angle(Σ |H|² Z conj(X̂))`。权重大的子载波主导相位。
4. `Z ← Z * exp(-jθ)`，整符号一起转。
5. 用同一套 `scale` 出 LLR 并译码。

### receiver._decode_payload_symbol

先试插值：`α = (i+1) / max(K-1, 1)`，`H` 和 `scale` 都按 `α` 在前后训练之间插。r1 的 `K=42`，第一个载荷符号 `α=1/41`，最后一个 `α=1`。

迭代 `<200` 就接受。否则按顺序改试联合 `H`、仅前训练、仅后训练。比较的是 `(迭代次数, -中位|LLR|)`，迭代更少优先，其次软信息更强。

### receiver._payload_sfo_search 交给 C 的量

这一步已经用前后 16 个训练做过联合最小二乘，得到后面失败时要回退的 `h_joint` 和 `scale_joint`。代价是中位训练残差。

### recover_array 后半（约第 349 行起）

1. 用 `payload_sfo` 取 `8+K+8` 行。尾部越界就退成只取 `8+K`，`h_tail` 退化成联合 `H`。
2. 数据行是 `body[8:8+K]`，载荷循环用 `data[1:]`，所以统计上是 `K-1` 个码字。
3. 信息流 = 搜头时留下的 249 字节 `header_raw` + 载荷信息字节。
4. `extract_payload` 从 `offset+249` 取 `file_size` 字节。主路径 `offset=0`。
5. `payload_complete`：有载荷，且每个迭代次数 `<200`。全收敛则 `ok=True`、阶段 `ok`。有文件但有码字未收敛：阶段 `ldpc`，`reason` 写明已按帧头长度写出。`write_run_outputs` 在头 CRC 通过且长度对得上时仍用原文件名，不因为部分码字未收敛就改成 `.partial`。`.partial` 用于没有合法头、或长度对不上的情况。

### decode_path.LdpcBank / extract_payload

见成员 A 第 4 节。`extract_payload` 长度不够就返回 `None`，`recover_array` 记阶段 `ldpc`、reason「载荷长度与 file_size 不一致」。

### report.py 写出的文件

| 文件 | 内容 |
|---|---|
| `metrics.json` | `stage`、`header_ok`、`K`、`blocks_ok/blocks_total`、`mean_abs_h`、`median_abs_llr`、`file_match` |
| `ldpc_symbols.csv` | 列 `symbol, iters, converged, median_abs_llr`。`converged` 是 `iters<200` |
| `summary.csv` | 每 bin 的 `abs_h`、`training_noise`（其实是 scale）、`mean_abs_llr` |
| `H.npy` / `H_end.npy` | 联合信道 / 尾训练信道 |
| `cpe.npy` | 每个载荷符号的公共相位 |
| `decoded_header.bin` | 249 字节头 |
| `clock_fit.png` | 锚点的钟偏拟合，锚点不够时没有这张图 |
| `channel_and_noise.png` / `phase_tracking.png` | 有对应数组时才画 |

`file_match`：`source/` 里有与帧头同名的文件才逐字节比。没有同名文件时是 `n/a`，不是失败。r1 对 `source/cute.jpg`，结果 `true`。

---

## 2. 在 r1 上走一个符号

`c1_symbol_walk_r1.py` 用 `sync_start` 和 `clock_error_ppm` 调用 `extract_active_sfo`，再：

- 画 `|H_front|` 和 `|H_tail|`
- 画载荷符号 0、20、最后一个的 Y、`Z=Y/H`、去 CPE 之后的星座
- 译载荷第 0 块，和 `source/cute.jpg` 的前 249 字节比较

信息流的布局是 `[249 字节头 | 文件字节 | 补零]`。载荷第 `i` 块（从 0 起）对应文件字节 `[i*249 : (i+1)*249]`，直到 `file_size`。

`cpe.npy` 若随符号近似一条斜线，说明还有没被 `payload_sfo` 吃掉的残余时延或残余钟偏，公共相位在把这个斜率吸走。r1 全部收敛，`median_abs_llr≈3`，斜线即使存在也还在 LDPC 能纠正的范围内。

---

## 3. 失败录音

未收敛下标来自各目录的 `ldpc_symbols.csv`，`c2_diagnose.py` 会再画一遍。

### r11　234/235，ppm=−7.4，文件 cute.m4a，58339 字节

只有符号 **175** 未收敛。`K=236`，`α=(175+1)/(236-1)=176/235≈0.75`。前后都收敛，`median_abs_llr` 仍然约 3.0。这是孤立的一个符号，不像钟偏把尾部整段拖出窗外。先看这个符号的中位 `|LLR|` 和 `cpe.npy` 有没有单点跳跃；不要改 B 的 ppm。

### r12　206/235，ppm=−2.3，同一文件的另一次录音

29 个未收敛，全部落在前半（下标 < 117）。最长的一段是 **87–99**（13 个）。尾部是好的，所以不是「越往后钟偏越糟」。更像前半段信道或干扰的一小段。`median_abs_llr` 仍约 3.0，说明多数子载波的软信息还在，是局部符号失败。

### r4　67/141，ppm=+28.0，snr2 只有约 135

74 个未收敛。最长一段是 **42–87**（46 个），横在中部，不是从某个符号开始单调坏到结尾。`n_sym_raw=150.0042`，训练间距是齐的，B 的几何锁没有 r2 那种 0.15 符号的错位。第二 Chirp 的 SNR 比第一 Chirp 小两个数量级，值得让 B 看一眼 `peak2` 是不是假峰，但 LDPC 的分布本身不像残余钟偏斜坡。责任先留在 C：看这段符号的 `|H|` 和 CPE，而不是先把 +28 ppm 改成别的常数。

### r6　114/155，ppm=−4.7

41 个未收敛。最长一段 **60–80**（21 个），另有 **25–31**。后半只有 3 个。ppm 很小，尾部反而好。和 r12 同类：中前部的一段失败，不是采样钟把帧尾拉飞。

---

## 4. 库函数（只记一句）

| 函数 | 它动的是 | DESIGN.md |
|---|---|---|
| `wiener_smooth` | 沿频率平滑 H | I3，主路径用 `lerp_h` |
| `cpe_and_nlms` / `equalize_symbol` | 相位之后还按 NLMS 改 H | I4，主路径只去公共相位，不更新 H |
| `hermitian_gate` | 镜像不一致时缩小 LLR 幅度 | I5，主路径不乘这个门 |
| `unused_bin_noise` / `tone_reliability` | 保护音噪声、每音 ρ | I6，主路径的尺度来自训练残差 |
| `turbo_refine` / `decode_frame` | CRC 失败后再估 H | I7，`diag["turbo"]=False` |
| `search_ph_in_stream` | 在信息流里滑动找 PH | I8，主路径头固定在第 1 个数据符号 |

---

## 5. 接口

**给 A：** 信息流 = `header_raw`（249 字节）+ 载荷各块。`extract_payload` 切 `[249 : 249+file_size]`。A 的 CRC 范围和文件名布局决定这块 249 字节能不能被 `parse_ph_header` 接受。

**给 B：** 如果 `median_abs_llr` 接近 0，而且大部分迭代等于 200，先回到起点和 `payload_sfo`，再看 `|H|` 是不是在通带里塌了。r4/r6/r11/r12 的中位 `|LLR|` 仍在 3 左右，不属于这种情况。

## 6. 15 分钟讲解提纲

1. 一个 bin：`Y=HX+噪声`，`Z=Y/H` 之后应靠近 `±1±j`。
2. `scale` 怎样让差的子载波少影响 LDPC；`median_training_noise≈3` 是尺度不是方差。
3. r1：42 个数据符号，第 1 个是头，后 41 个进载荷循环，`α` 从 `1/41` 到 1。
4. r11 只有符号 175 失败，r4 是中部连续 46 个，两者不要用同一种解释。

## 7. 自测与答案

1. `Z=0.2+0.9j`，`scale=3`，LLR 顺序和数值？  
   先 Q 后 I：2.7、0.6。都为正，两个比特都更像 0。

2. `K=42` 时最后一个载荷符号的 `α`？  
   1。载荷下标 `i` 从 0 到 40，`α=(i+1)/41`。

3. 迭代 200 之后还会试哪几套 H？  
   联合、仅前、仅后。插值那套已经试过。

4. `median_training_noise=3` 是什么？  
   LLR 尺度数组的中位数，不是 `σ²`。

5. 头 CRC 通过但 1 个码字未收敛，输出文件名带不带 `.partial`？  
   不带。`write_named` 只要求头合法且长度等于 `file_size`。阶段仍是 `ldpc`，`ok=false`。

6. r11 未收敛的符号下标？  
   175。

7. 为什么 r4 不要先判成钟偏斜坡？  
   失败集中在 42–87，尾部并没有单调变差；`n_sym_raw` 离整数只有 0.004。

8. `file_match=n/a` 是失败吗？  
   不是。`source/` 里没有同名文件。

9. 换一套 H 再译，QPSK 和 LDPC 变了吗？  
   没有。变的是均衡后的 `Z` 和 LLR。

10. 载荷第 0 块对应文件的哪些字节？  
    `[0:249]`，如果文件比 249 短则到 `file_size` 为止。
