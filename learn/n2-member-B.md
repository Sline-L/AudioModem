# 成员 B：从 WAV 到起点和钟偏

在 `learn/n2-course.html` 第 2 章之上加深。交付物是 r1 的时间轴，外加 r2、r9、r8、r10 的诊断。C 拿到的是起点和钟偏已经选好之后的频谱。

实验：

```text
python learn/experiments/b1_timeline_r1.py
python learn/experiments/b2_diagnose.py --wav data/r1.wav
python learn/experiments/b2_diagnose.py --wav data/r2.wav
python learn/experiments/b2_diagnose.py --wav data/r8.wav
```

主路径取符号走 `extract_active_sfo`。r1 上 `resampled=false`，`cfo_hz=0`。

---

## 1. 逐函数

### capture.parabolic_peak

三点 \(a,b,c\)，\(\delta=0.5(a-c)/(a-2b+c)\)，限制在 ±1。峰在数组两端时不插值。r1 的 `lock.frac≈0.047` 就是这个小数。

### capture.matched_chirp_metric / find_two_chirps

模板 `linear_chirp`，长度 144000。`fftconvolve(..., "valid")` 再平方。先取全局最大，挖掉左右各半个 Chirp（72000 点），再取第二峰，按时间排序。噪声用 `median(metric)`，`snr1>8` 才算 `ok`。

r1：`peak1≈44926.07`，`peak2≈830837.04`，两个 SNR 都在 \(10^5\) 量级。

### capture.coarse_body_start

`第一峰 + 144000 + 24000`。这是 OFDM 体粗起点，允许偏几百个样点。r1 上粗起点是 `44926+168000=212926`，最终锁点 `212925.047`，只差大约 1 个样点。

### align.matched_start

模板是前 2 个训练符号接起来，长 `2*10240`。搜索窗是粗起点前后 `search_syms=2.5` 个符号（左右各约 25600 点）。返回的 `snr` 在 `_lock_frame` 里要 ≥ 4。r1 的 `lock.snr≈7271`。

### align.find_end_preamble

模板是后 8 个训练符号。相关峰出现在这 8 个符号的起点，所以

\[
K = \mathrm{round}(n\_sym\_raw) - 8.
\]

r1：`n_sym_raw≈49.9994`，`K=42`。`_lock_frame` 还要求 `1 ≤ K ≤ 20000`。

### align.extract_active_sfo

```python
pos = start + (1+sfo) * (m*10240 + 2048 + n)
```

线性插值，`rfft`，切 bin 400–2391。返回 `(n_sym, 1992)` 复数。`pos` 越出 `[0, len(rx)-1]` 就抛 `ValueError`。

### align.training_peaks / best_sfo_near

Chirp 锁失败时才用。全段找前 2 个训练符号的相关峰，最多 4 个，相邻峰至少隔半个符号，后续峰 SNR 低于 3 就停。每个峰在 ±200 ppm、步长 25 ppm 上，用 8 个前训练的中位残差选钟偏。

### receiver._lock_frame

`find_two_chirps` 成功 → `coarse_body_start` → `matched_start` 的 snr≥4 → `find_end_preamble` 的 K 合法。任一步失败返回 `None`。

### receiver._trust_sfo

记 `aligned = |n_sym_raw - round(n_sym_raw)| < 0.08`，`end_ok = |sfo_end| < 5e-3`，`chirp_ok = |sfo_chirp| < 5e-3`。

| 条件 | 返回 |
|---|---|
| aligned 且 end_ok，并且 chirp 也可靠、两者差 < 80 ppm | 两者平均 |
| aligned 且 end_ok，但不满足上一条 | `sfo_end` |
| 上面不成立，但 chirp_ok | `sfo_chirp` |
| 只剩 end_ok | `sfo_end` |
| 都不可靠 | 0 |

两个差很远的估计不能取平均，也不能取中位数（两个数的中位数就是平均数）。r1 的 `n_sym_raw` 离 50 只有约 0.0006，走训练估计。

数值例子：

- 训练已对齐，`sfo_end=-12 ppm`，`sfo_chirp=-20 ppm`：差 8 ppm < 80，返回平均 −16 ppm。
- 训练已对齐，两者差 200 ppm：返回 `sfo_end`。
- `n_sym_raw` 离整数 0.15，`sfo_chirp=-720 ppm`：`aligned` 为假，返回 `sfo_chirp`。这就是 r2 的分支。
- 两个估计都是 `nan` 或绝对值 ≥ 5000 ppm：返回 0。

### receiver._header_jobs

锁点候选的起点偏移顺序是 `0, -128, 128, -256, 256, -64, 64`。钟偏是中心再加上 `0, ±10, ±20, ±5, ±30` ppm。`|sfo_main| > 40 ppm` 时，中心列表里再加一个 0。`abs(sfo) > 5e-3`（5000 ppm）的候选直接丢掉。

`recover_array` 里一旦有一个候选 CRC 通过就 `break`，所以列表里靠前的锁点优先。全失败才第二次调用，带上 `training_peaks`。

### receiver._try_header（定时部分）

用该起点和钟偏取 9 个符号（8 训练 + 第 1 个数据符号）。C 负责后面的 H 和译码。CRC 通过后还要求 `declared_symbols == (file_size+248)//249` 且文件名非空。

### receiver._payload_sfo_search

头已经通过之后，在入选钟偏的 ±50 ppm 内按 1 ppm 步进。每个假设分别取前 8 个和后 8 个训练，拼成 16 行，做一次最小二乘，用中位残差当代价。后面所有数据符号用这个 `payload_sfo`。

`metrics.json` 里的 `clock_error_ppm` 来自搜头阶段的 `clock["ppm"]`，不是这一步的 `payload_ppm`。`payload_ppm` 写在诊断字典里，没有抄进 `metrics.json`。

---

## 2. r1 时间轴

`b1_timeline_r1.py` 会复算下面这些数，并和 `run/n2/r1/metrics.json` 比较。

| 标记 | 样点 |
|---|---|
| peak1 | 44926.07 |
| coarse = peak1+168000 | 212926.07 |
| lock.start / sync_start | 212925.047 |
| 粗起点和锁点之差 | ≈ 1 个样点 |
| end_offset | 511993.96 |
| end_offset / 10240 | 49.9994 = 8+K |
| K | 42 |

−11.65 ppm 在 58×10240 = 593920 个样点上累积约 **6.9 个样点**。所以尾部必须用 `(1+sfo)` 拉伸取样位置。脚本把 `sfo` 改成 0 再取同一段：前训练残差约 0.094，正确钟偏下的尾训练残差约 0.70，`sfo=0` 时尾训练残差约 4.5。

---

## 3. 三种钟偏

记

\[
s_{end}=\frac{end\_offset}{(8+K)\times 10240}-1, \qquad
s_{chirp}=\frac{peak2-peak1}{\text{标称间距}}-1.
\]

标称间距 = 一条扫频 + 两段静音 + `(16+K)` 个符号。`_header_jobs` 里就是这么算的。

| 录音 | n_sym_raw 离整数 | clock_error_ppm（搜头） | 走哪条 |
|---|---|---|---|
| r1 | 0.0006 | −11.65 | 训练已对齐，用 `sfo_end`（或与 Chirp 平均，如果差 < 80 ppm） |
| r5 | 0.0032 | +21.43 | 同上 |
| r7 | 0.0006 | +3.44 | 同上 |
| r14 | 0.0006 | −11.50 | 同上 |
| r2 | 0.156 | −719.6 | 没对齐，改用 Chirp |
| r9 | 0.141 | −551.0 | 没对齐，改用 Chirp |

r2 上如果硬用后训练间距：`189.8443/190 - 1 ≈ -819 ppm`，和记录里的 −719.6 不是同一个数，所以记录的是 Chirp 那一支。最终发给 C 的是 `_payload_sfo_search` 的 1 ppm 网格结果；头都没通过时，这一步不会发生。

---

## 4. 失败录音

### r2　stage=header，ppm=−719.6，K=182

`lock.snr≈3021`，Chirp SNR 也在 \(10^5\)，几何上「找到了」。`n_sym_raw=189.8443`，离 190 差 0.156 个符号，约 **1594 个样点**，大于搜头用的 ±256。`aligned` 为假，`_trust_sfo` 返回 Chirp 的 −719.6 ppm。因为绝对值大于 40 ppm，候选里包含 0 ppm 中心，再 ±30 ppm，CRC 仍然全失败。

结论：不能只怪这一个 ppm。0 ppm 附近也没解出头，说明起点本身偏离超过偏移网格，或者真实钟偏在两个中心的 ±30 ppm 之外。下一步用 `b2_diagnose.py` 看训练相关曲线的主峰是否唯一，不要先改 LDPC。

### r9　stage=header，ppm=−551，K=236

同一分支。`n_sym_raw=243.8587`，离 244 差 0.141，约 1447 个样点。后训练间距对应约 −579 ppm，记录的 −551 ppm 更接近 Chirp 估计。长帧上 500 ppm 的斜率到最后一个符号会积累 `5e-4 * (16+236) * 10240 ≈ 1290` 个样点，远远滑出 FFT 窗。救援如果要做，是扩大起点搜索或重看尾训练峰，不是把 −551 和 0 再平均一次。

### r8　stage=align，ppm=0，reason 仍是「PH 头 CRC 未通过」

`lock` 和 `end` 都是 null，Chirp SNR 却很高（\(10^6\) 级）。`_lock_frame` 返回了 `None`：训练匹配 snr<4，或尾训练的 K 不合法。`recover_array` 在一个头都没找到时，只要第一份 `jobs`（锁点候选）是空的，stage 就是 `align` 而不是 `header`；reason 字符串在这之前已经写成 CRC 未通过。看 stage 和 `lock is None`，不要被 reason 的字面带去改译码。

### r10　stage=align

同样没有 lock/end。Chirp 的 snr1、snr2 都极大。`error` 字段直接是 `align`。和 r8 同一类：扫频在，训练锁不上。

---

## 5. 库函数（只记一句）

| 函数 | 它动的是 | DESIGN.md |
|---|---|---|
| `dechirp_sfo` | 用扫频二次相位估钟偏 | I1，主路径不用 |
| `clock_scale_from_anchors` / `resample_clock` / `apply_sfo` | 整段重采样 | I1，r1 的 `resampled=false` |
| `cfo_search` | 频偏 Hz | I2，主路径 `cfo_hz=0` |
| `sto_phase` / `extract_ffts` | 整数切窗 + 频域补偿分数时延 | I2，主路径走插值网格 |
| `apply_symbol_delays` / `delay_from_h` | 由 H 的相位斜率估额外时延 | 未接入 `recover_array` |

---

## 6. 给 C 的接口

`extract_active_sfo` 返回 `(n_sym, 1992)` 复数，行的顺序是调用者给定的符号顺序。主路径取 `8+K+8` 行时：`[0:8]` 前训练，`[8:8+K]` 数据，`[8+K:]` 后训练。数据第 0 行是帧头符号，C 的载荷循环从第 1 行开始。

## 7. 15 分钟讲解提纲

1. r1 上 peak1、粗起点、sync_start 三者只差约 1 个样点。
2. `end_offset/10240≈50`，减 8 得 K。
3. −11.65 ppm 累积约 6.9 个样点，所以尾部要用 `(1+sfo)`。
4. r2 为什么 0 ppm 也进了候选、为什么还是失败。

## 8. 自测与答案

1. Chirp 模板多长？前 2 训练模板多长？  
   144000；20480。

2. `n_sym_raw=50.02` 时 K 是多少？算 aligned 吗？  
   `round=50`，K=42。`|0.02|<0.08`，aligned 为真。

3. 训练已对齐且与 Chirp 相差 200 ppm，`_trust_sfo` 返回哪个？  
   `sfo_end`。200 ppm > 80 ppm，不平均。

4. 为什么两个数不能用 median？  
   两个数的中位数等于平均数，会把正常 ppm 和错误 Chirp 间距捏成一个无效值。

5. `|sfo_main|=50 ppm` 时，0 ppm 会不会进中心列表？  
   会。阈值是 40 ppm。

6. 搜头候选的最大起点偏移是多少样点？  
   256。

7. `clock_error_ppm` 和 `payload_sfo` 谁被写进 `metrics.json`？  
   只有搜头阶段的 ppm。`payload_ppm` 不在这份 JSON 里。

8. stage=align 且 lock 为 null，第一件事看什么？  
   Chirp SNR 和训练相关曲线。reason 写着 CRC，不代表已经锁上了帧。

9. r1 粗起点和锁点差大约多少？  
   约 1 个样点。

10. `extract_active_sfo` 越界时谁负责接住异常？  
    `_try_header`、`_payload_sfo_search`、`recover_array` 里的 `try/except ValueError`，记成这次候选失败或阶段 `align`。
