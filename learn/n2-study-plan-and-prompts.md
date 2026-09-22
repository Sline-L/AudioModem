# N2 接收机三人学习规划 + 交给其他 AI 的提示词

材料已经按本规划写好，直接用下面这些文件，不必再把提示词交给别的 AI。

| 文件 | 对应 |
|---|---|
| `learn/n2-prereq.md` | P1 预备知识 |
| `learn/n2-member-A.md` | P2 |
| `learn/n2-member-B.md` | P3 |
| `learn/n2-member-C.md` | P4 |
| `learn/n2-failure-playbook.md` | P5 |
| `learn/experiments/` | P6，六个脚本已对 r1 / r11 跑通 |
| `learn/n2-quiz.md` | P7 |
| `learn/n2-final-session.md`、`learn/n2-final-record.md` | P8 |

> 第一部分是给三个人看的分工与时间表。第二部分是当时用来生成上述材料的提示词，留作以后要改某一章时重跑。已有的 `learn/n2-course.html` 是骨架，新材料是补充和加深。

---

## 第一部分　分工规划

### 1. 目标与结业标准

学完后三个人都应能做到：

1. 独立跑 `python selftest.py` 和 `python run_rx.py --wav ..\data\rX.wav --out-dir ..\run\n2\study_rX`，并解释 `metrics.json` 里每个字段来自哪个函数。
2. 纸上重算 r1 的 40 / 42 / 41（`declared_symbols` / `K` / 载荷码字数）。
3. 拿到一份 `ok=false` 的录音，只看 `stage` 与 `reason` 就能判断问题该交给谁，并说出下一步要看哪个数组或哪张图。
4. 各自负责的模块：能对着代码逐函数讲 15 分钟，能回答另外两人的接口问题。

### 2. 三人角色

| 成员 | 主题 | 拥有的文件 | 在 `receiver.py` 里负责的函数 | 交付物 |
|---|---|---|---|---|
| A　空中接口 | 发射端到底发了什么；字节↔比特↔QPSK↔子载波；训练/扰码/LDPC/PH 头 | `standard/example_tx.py`、`n2/params.py`、`n2/phy.py`、`n2/selftest.py`、`LDPC/new_ldpc/README.md` | `coded_symbol_count`、`recover_wav`、`recover_from_channels` | ① 「一个字节的旅程」对照表 ② `test_phy_against_tx` 逐条解释 ③ PH 头字节布局图 |
| B　捕获与对准 | Chirp 匹配滤波、训练锁定、K 的推算、采样钟偏 SFO、定时/钟偏候选、头 CRC 判决 | `n2/capture.py`、`n2/align.py` | `_lock_frame`、`_trust_sfo`、`_header_jobs`、`_try_header`（定时部分）、`_payload_sfo_search` | ① r1 时间轴图（两个 Chirp 峰、粗起点、锁点、尾训练、K）② SFO 三种来源对比表 ③ 失败录音 r2/r8/r9/r10 的诊断报告 |
| C　信道与译码 | LS 信道、LLR 尺度、前后训练插值 H、公共相位、软 QPSK、解扰、LDPC、写文件、诊断图 | `n2/channel.py`、`n2/decode_path.py`、`n2/report.py` | `_equalize_once`、`_decode_payload_symbol`、`_decode_symbol`、`recover_array` 第 349 行以后 | ① 一个数据符号从频谱到 249 字节的流程图 ② `summary.csv`/`ldpc_symbols.csv` 读图指南 ③ 失败录音 r4/r6/r11/r12 的诊断报告 |

共同拥有：`n2/receiver.py::recover_array` 前半（第 247–348 行）三人一起读；`n2/DESIGN.md` 三人一起读并标注哪些创新点落地了。

### 3. 案例录音分配

`run/n2/` 里已有 13 份结果，按阶段分给对应成员当练习材料（`r13.wav` 还没跑过，作为合练时的盲测）：

| 录音 | 结果 | ppm | 归谁 | 学习价值 |
|---|---|---|---|---|
| r1 | ok, K=42, 41/41, file_match=true | −11.6 | 全员 | 基线，贯穿全书 |
| r3 | ok, K=182, 182/182, file_match=true | ≈0 | A | 钟偏几乎为 0，疑似数字回环，对比声学录音 |
| r5 / r7 / r14 | ok | +21.4 / +3.4 / −11.5 | A | 用不同 file_size 练 K 的计算；正负 ppm 都有 |
| r2 | header 失败, K=182 | −719.6 | B | 钟偏估计离谱导致头 CRC 全失败，`_trust_sfo` 的边界 |
| r9 | header 失败, K=236 | −551.0 | B | 同上，长帧 |
| r8 | align 失败 | 0.0 | B | Chirp 或训练没锁上，看 `capture.snr1` 和 `lock.snr` |
| r10 | align 失败 | 无 | B | 同上 |
| r11 | ldpc, 234/235 | −7.4 | C | 只差 1 个码字，看是哪一个符号、`α` 多少、`|LLR|` 多少 |
| r12 | ldpc, 206/235 | −2.3 | C | 中等损失，看未收敛码字在时间上是否连续 |
| r4 | ldpc, 67/141 | +28.0 | C（B 协助） | 一半以上未收敛，判断是钟偏漂移还是信道变化 |
| r6 | ldpc, 114/155 | −4.7 | C | 同上 |

### 4. 四周时间表（每人每周约 4–5 小时，含 40 分钟碰头）

| 周 | 全员共同 | 成员 A | 成员 B | 成员 C | 碰头内容 |
|---|---|---|---|---|---|
| 第 1 周　预备知识 + 跑通 | 装环境、编译 LDPC dll、跑 `selftest.py` 和 r1；读 `learn/n2-course.html` 第 0 章；读 AI 生成的预备知识讲义（P1） | 读 `example_tx.py` 全文，画帧结构 | 读预备知识里的匹配滤波、SFO 两节 | 读预备知识里的 LS、LLR、LDPC 三节 | 每人讲 5 分钟「我这周最没想通的一个点」 |
| 第 2 周　各自精读 | 一起读 `recover_array` 第 247–348 行 | `params.py` → `phy.py` → `selftest.py::test_phy_against_tx`，做 P2 的实验 | `capture.py` → `align.py` → `_lock_frame`/`_trust_sfo`，做 P3 的实验 | `channel.h_and_llr_scale` → `_equalize_once` → `_decode_payload_symbol`，做 P4 的实验 | A 讲 15 分钟（比特账 + PH 头），B、C 只问接口 |
| 第 3 周　案例 + 库函数 | 一起读 `DESIGN.md` 第 4 节，对照附录 A 标注 I1–I8 哪些在主路径上 | 对比 r3 与 r1 的 `metrics.json`，写「数字回环 vs 声学录音」差异 | r2/r9/r8/r10 诊断报告（P5） | r11/r12/r4/r6 诊断报告（P5） | B 讲 15 分钟（r1 时间轴 + 一份失败录音），C 讲 15 分钟（一个符号到 249 字节 + 一份失败录音） |
| 第 4 周　合练 + 结业 | 盲测 `r13.wav`；三人合填「对照表」；做 P7 测验 | 出测验题给 B、C | 出测验题给 A、C | 出测验题给 A、B | 每人回答另外两人的题；按结业标准互评 |

### 5. 碰头会流程（40 分钟）

1. 5 分钟：当周负责人跑一次 `run_rx.py`，大家看终端那一行摘要。
2. 15 分钟：负责人按「讲解提纲」讲，只讲主路径，库函数一句话带过。
3. 15 分钟：另外两人只问接口问题：「你给我的数组形状是什么」「这个数错了我会看到什么」。
4. 5 分钟：更新 `learn/` 里的材料，记录没解决的问题，下周由 AI 补讲。

### 6. 规则

- 不改 `n2/` 和 `standard/` 里的代码。实验一律在 `learn/experiments/` 下写脚本，`import` n2 的函数来调用。
- 所有实验输出写到 `run/n2/study_*`，不覆盖已有的 `run/n2/r1` 等目录。
- 读到 `DESIGN.md` 提到但 `recover_array` 没调用的函数（Wiener、NLMS、Turbo、Hermitian 门控、dechirp_sfo、cfo_search 等），一律先标「库函数」，不花时间深挖。

---

## 第二部分　交给其他 AI 的提示词

### 使用说明

- 每次对话：先贴 **P0**，再贴一条 P1–P8。
- 让 AI 先用工具读文件再写，不允许凭记忆编造函数名和数字。P0 末尾有自检清单，要求 AI 交付前逐条过。
- 产物统一放在 `learn/` 目录，格式和已有 `learn/n2-course.html` 一致（或 Markdown），中文。
- 如果 AI 说「无法读取文件」，把对应文件内容直接粘给它。

---

### P0　通用背景（每次必贴）

```
你是一位数字通信课程的助教，帮助一个三人小组学习一个声学 OFDM 接收机的 Python 实现。
仓库根目录：g:\OFDM_gerson。先用工具读取下面列出的文件，再开始写；不要凭记忆编造任何函数名、行号或数值。

【仓库结构】
standard/example_tx.py        发射端，空中接口唯一真源，不得修改
n2/params.py                  常量与 derived()
n2/phy.py                     linear_chirp, lfsr_sequence, train_freq_time, active_bins, bytes_to_qpsk_grid,
                              qpsk_llr_scaled, descramble_llr, bits_to_bytes, parse_ph_header, read_wav_channels 等
n2/capture.py                 parabolic_peak, matched_chirp_metric, find_two_chirps, coarse_body_start;
                              库函数: dechirp_sfo, clock_scale_from_anchors, resample_clock, apply_sfo
n2/align.py                   matched_start, find_end_preamble, extract_active_sfo, training_peaks, best_sfo_near;
                              库函数: cfo_search, sto_phase, extract_ffts, apply_symbol_delays, delay_from_h
n2/channel.py                 h_and_llr_scale, lerp_h;
                              库函数: unused_bin_noise, ls_from_train, wiener_smooth, tone_reliability,
                              estimate_from_known, hermitian_gate, cpe_and_nlms, equalize_symbol, turbo_refine
n2/decode_path.py             LdpcBank(decode_codeword), extract_payload; 库函数: llr_from_symbol, decode_frame,
                              scramble_bits, rebuild_qpsk_from_coded
n2/receiver.py                coded_symbol_count, _lock_frame, _decode_symbol, _try_header, _trust_sfo, _header_jobs,
                              _equalize_once, _lerp_scale, _decode_payload_symbol, _payload_sfo_search,
                              recover_array(主编排, 第247行起), recover_from_channels, recover_wav
n2/report.py                  write_run_outputs, format_console_line, _try_plots
n2/run_rx.py  n2/selftest.py  n2/paths.py  n2/README.md  n2/DESIGN.md
LDPC/new_ldpc/                Jossy 2018 的 802.16 LDPC，py/ldpc.py + bin/c_ldpc.dll
data/r1.wav ... r14.wav       14 份录音
source/cute.jpg cute.png      用于字节对照的原文件
run/n2/r1 ... r14             已有结果（每个目录有 metrics.json, summary.csv, ldpc_symbols.csv, *.npy, *.png）
learn/n2-course.html          已有的学习课本（三章分工 + 第 0 章共同课 + 附录），你的产物是对它的补充和加深

【关键数字，全部来自 n2/params.py::derived()】
fs=48000, N=8192, CP=2048, symbol_len=10240, start_index=400, n_active=1992, end_index=2392,
chunk_size=498(码字字节), info_bytes=249(信息字节), ldpc_z=166, preamble_count=8(前后各8个训练),
chirp_len=144000(3.0 s), silence_len=24000(0.5 s), 训练种子 random.seed(80) 用 CPython random,
扰码 15 位 LFSR 种子 0x5A4D 每码字复位, 帧头魔术字 b'PH', 帧结构:
[线性扫频 3 s][静音 0.5 s][8 训练][K 个数据符号][8 训练][静音 0.5 s][线性扫频 3 s]

【主路径（recover_array 实际调用的顺序），务必与 DESIGN.md 区分】
1 _lock_frame: find_two_chirps → coarse_body_start → matched_start(前 2 个训练相关) → find_end_preamble(后 8 训练) 得 K
2 _header_jobs + _try_header: 在起点偏移 {0,±64,±128,±256} × 钟偏 {中心,±5,±10,±20,±30 ppm} 上取 9 个符号,
  估 H, 解第 1 个数据符号, parse_ph_header CRC 通过者入选; 全失败则 training_peaks + best_sfo_near
3 _payload_sfo_search: ±50 ppm 步长 1 ppm, 用前后 16 训练的最小二乘残差选 payload_sfo
4 extract_active_sfo(rx, start, 8+K+8, payload_sfo): 取样点 start+(1+sfo)*(m*10240+2048+n), 线性插值, rFFT, 取 bin 400..2391
5 h_front/h_tail = h_and_llr_scale(前 8 训练 / 后 8 训练); 对第 i 个载荷符号 alpha=(i+1)/(K-1), H=lerp_h
6 _equalize_once: Z=Y/H → 硬切片 → |H|² 加权公共相位 θ → Z*=exp(-jθ) → qpsk_llr_scaled → descramble_llr → LDPC sumprod2
  迭代 <200 视为收敛; 否则依次换联合 H / 仅前 / 仅后再译
7 info_bytes = header_raw + payload_bytes; extract_payload 取 [249:249+file_size]; write_run_outputs 写文件与诊断
DESIGN.md 里的 I1 去斜、I2 二维 STO-CFO、I3 Wiener、I4 NLMS、I5 Hermitian 门控、I7 Turbo 目前都不在主路径上,
receiver.py 里 diag["turbo"]=False, diag["cfo_hz"]=0.0, diag["resampled"]=False。

【基线录音 r1（run/n2/r1/metrics.json）】
file_size=9829 (cute.jpg), declared_symbols=40, K=42, blocks_ok=41/41, file_match=true
capture.peak1≈44926.07, peak2≈830837.04, snr1≈2.5e5
lock.start=sync_start≈212925.047, lock.snr≈7271
end.end_offset≈511993.96, n_sym_raw≈49.9994, K=42
clock_error_ppm≈-11.65, mean_abs_h≈2.34, median_abs_llr≈3.02

【全部录音状态】
r1 ok K=42 41/41 -11.6ppm | r3 ok K=182 182/182 ≈0ppm | r5 ok K=142 +21.4ppm | r7 ok K=156 +3.4ppm | r14 ok K=40 -11.5ppm
r2 header失败 K=182 -719.6ppm | r9 header失败 K=236 -551.0ppm | r8 align失败 0ppm | r10 align失败
r11 ldpc 234/235 -7.4ppm | r12 ldpc 206/235 -2.3ppm | r4 ldpc 67/141 +28.0ppm | r6 ldpc 114/155 -4.7ppm
r13 尚未运行

【三人分工】
A 空中接口: example_tx.py, params.py, phy.py, selftest.py, LDPC README
B 捕获与对准: capture.py, align.py, receiver 的 _lock_frame/_trust_sfo/_header_jobs/_payload_sfo_search
C 信道与译码: channel.py, decode_path.py, report.py, receiver 的 _equalize_once/_decode_payload_symbol 及 recover_array 后半

【写作要求】
- 中文，面向本科高年级/研一，学过信号与系统、概率，但没学过 OFDM 和 LDPC。
- 每个结论都要能落到「文件:函数」或 metrics.json 的字段上；引用代码时给出文件名和函数名，可给行号范围。
- 严格区分「主路径」和「库函数」，库函数只在附录里出现。
- 数学只写到能解释代码为止，先写公式再写对应的那一两行代码。
- 实验脚本放 learn/experiments/，只 import n2 的函数，不修改 n2/ 和 standard/；输出目录一律 run/n2/study_*。
- 每一节末尾给「自测题」，答案放文末附录。
- 不要写空话和鼓励性语句，不要总结「本章我们学习了」。

【交付前自检】
1 每个函数名都在上面的列表里或你已读过的文件里出现过。
2 每个数字都能在 params.py、metrics.json 或代码里找到出处。
3 没有把 DESIGN.md 的 I1–I8 当成主路径来讲。
4 K 的来源是 coded_symbol_count(file_size)（头通过后）与 find_end_preamble（锁帧时的 K 提示），不是 declared_symbols。
5 QPSK 比特顺序：每字节从高位对到低位对，每对里低位走 I、高位走 Q；LLR 输出顺序先 Q 后 I。
6 CRC 覆盖 ba[:11+len(filename)]，不含结束 0x00 和 CRC 本身。
```

---

### P1　预备知识讲义（第 1 周，全员）

```
任务：写 learn/n2-prereq.md（或同风格 HTML），题为「N2 预备知识：读懂代码前必须会的 9 件事」。
每件事的结构固定为：① 一句话是什么 ② 最少的公式 ③ 它在 n2 的哪个函数里出现（文件:函数）④ 一个 5 分钟以内能跑的 numpy 小实验 ⑤ 2 道自测题。
9 件事：
1 采样与 FFT bin：f = k·fs/N；为什么 bin 400..2391 对应约 2.3–14 kHz；实信号谱的共轭对称与 X[N-k]=conj(X[k])。→ phy.active_bins, mirror_bins, bytes_to_qpsk_grid
2 OFDM 一个符号：IFFT 取实部、循环前缀把线性卷积变圆周卷积、为什么 CP=2048 能吃掉多径。→ train_freq_time 的时域拼接
3 QPSK 与 Gray 无关的这套映射：I=1-2b，星座 ±1±j，硬判决只看符号。→ phy.slice_qpsk, bytes_to_qpsk_grid
4 匹配滤波 = 相关：fftconvolve(x, template[::-1], 'valid') 为什么等价相关；峰值位置的含义；三点抛物线亚采样插值公式。→ capture.matched_chirp_metric, parabolic_peak, align.matched_start
5 线性调频 Chirp 为什么适合同步：自相关尖锐、抗多径。→ phy.linear_chirp
6 采样钟偏 SFO：ppm 定义、10240 点一个符号累积多少样点、为什么要用 (1+sfo) 拉伸取样位置而不是整段重采样。→ align.extract_active_sfo
7 最小二乘信道估计：Y=HX+n，H=mean(Y/X)，残差 |Y-HX|²/|H|² 的物理含义。→ channel.h_and_llr_scale
8 LLR：定义 log P(0)/P(1)，QPSK 下 LLR ∝ Re(Z)、Im(Z)，为什么噪声大的子载波要缩小 LLR 幅度而不是翻符号，裁剪到 ±30 的原因。→ phy.qpsk_llr_scaled, descramble_llr
9 LDPC 直觉：校验矩阵、置信传播、迭代次数为什么能当收敛标志、码率 1/2 与 3984/1992 的关系；扰码为什么要在 LLR 上做。→ decode_path.LdpcBank, LDPC/new_ldpc/README.md
最后加一节「读 Python 数值代码的习惯」：先看 shape，np.asarray/ravel/astype 在做什么，fftconvolve 的 mode，rfft 输出长度 N/2+1。
篇幅：每件事 300–500 字加代码，共不超过 6000 字。
```

---

### P2　成员 A 详细教程（空中接口）

```
任务：写 learn/n2-member-A.md，题为「成员 A：一个字节的旅程」。在 learn/n2-course.html 第 1 章基础上加深，不重复已有内容，重点补：

第一部分　逐函数精读 standard/example_tx.py
按类的方法顺序（构造函数、log_chirp_gen、train_symbol_gen、sequence/LFSR、header_gen、file_process、ldpc_encoding、mapp、时域拼接与归一化、写 WAV），每个方法：输入输出的类型与 shape、关键行逐行注释、对应到 n2/phy.py 的哪个逆函数、如果这里差一个比特接收端会在哪个阶段报错。

第二部分　用 selftest.py::test_phy_against_tx 做对照实验
读 n2/selftest.py 的 _tx_class / _construct / test_phy_against_tx，解释它比对了哪些数组。然后写 learn/experiments/a1_compare_tx.py：
- 用 example_tx 生成 Chirp、16 个训练符号、扰码序列，与 phy.linear_chirp / train_freq_time / lfsr_sequence 比较，打印最大绝对差。
- 故意把训练种子换成 numpy RNG，展示差异。
- 构造一个 20 字节文件，调用 header_gen，用 hexdump 逐字节标注 PH/版本/file_size/declared_symbols/文件名/0x00/CRC，再用 phy.parse_ph_header 解析，并展示改动 CRC 覆盖范围内外各 1 字节的结果。

第三部分　字节账
- 给定 file_size 计算 declared_symbols、K、载荷码字数 K-1 的通用公式和 5 个例子（含 file_size 恰为 249 的倍数、恰为 498 倍数减 249 等边界）。
- 用 run/n2/r1、r3、r5、r7、r14 的 metrics.json 验证：从 file_size 推 K 是否与记录一致。
- 末尾全零码字：什么时候出现，接收端怎么容忍。

第四部分　LDPC 库
读 LDPC/new_ldpc/README.md 和 py/ldpc.py 的 code 类的构造与 decode 签名，说明 z=166、standard="802.16"、rate="1/2"、dectype="sumprod2" 各是什么、返回值 app 与 it 的含义、为什么 app<0 判 1。写 learn/experiments/a2_ldpc_roundtrip.py：随机 1992 bit → 编码 → 扰码 → 加噪声 → 解扰 LLR → 译码，打印迭代次数与误码，噪声从小到大扫 5 档。

第五部分　给 B 和 C 的接口卡片（各一张表）
给 B：train_freq_time 返回的两个数组 shape 与含义，B 做相关用哪个；linear_chirp 的长度与幅度。
给 C：known=freq[:, bins] 的 shape、qpsk_llr_scaled 输出的长度与顺序、LdpcBank.decode_codeword 的输入输出、bits_to_bytes 的打包顺序。

附：15 分钟讲解提纲 + 10 道自测题 + 答案。
```

---

### P3　成员 B 详细教程（捕获与对准）

```
任务：写 learn/n2-member-B.md，题为「成员 B：从 WAV 到起点和钟偏」。在 learn/n2-course.html 第 2 章基础上加深，重点补：

第一部分　逐函数精读（每个函数：数学 → 代码逐行 → 输入输出 shape → r1 上的实际数值 → 出错时的表现）
capture.py: parabolic_peak, matched_chirp_metric, find_two_chirps(阈值 snr>8 的含义), coarse_body_start
align.py: matched_start(search_syms=2.5 的窗口、snr≥4), find_end_preamble(为什么峰在后 8 训练起点、K=round(n_sym_raw)-8), extract_active_sfo(取样位置公式、线性插值、越界 ValueError), training_peaks, best_sfo_near
receiver.py: _lock_frame, _trust_sfo(四个分支，逐分支给一个数值例子), _header_jobs(候选列表的生成顺序，为什么先加入的优先), _try_header 里定时相关的部分, _payload_sfo_search(为什么用 16 行训练的中位残差当代价)

第二部分　r1 时间轴实验 learn/experiments/b1_timeline_r1.py
只 import n2 的函数，对 data/r1.wav：
- 画 Chirp 相关曲线全图和两个峰的局部放大（含抛物线插值点）。
- 打印 peak1、peak1+144000+24000、matched_start 返回的 start，三者差多少样点。
- 画前 2 训练相关曲线的局部图，标出 lock.start。
- 打印 end_offset、n_sym_raw、K，与 metrics.json 对比。
- 计算 −11.65 ppm 在 (8+42+8)×10240 个样点上累积多少样点。
- 把 sfo 改成 0 调 extract_active_sfo，用 h_and_llr_scale 算前、后训练残差，与正确 sfo 时对比，说明尾部训练为什么变差。

第三部分　SFO 的三种来源对比表
sfo_end（后训练间距）、sfo_chirp（双 Chirp 间距）、payload_sfo（1 ppm 网格）在 r1、r5、r7、r14 上各是多少（从 metrics.json 的 clock 字段和 end 字段算），彼此差多少 ppm，为什么最终用 payload_sfo。

第四部分　失败案例诊断（写成 4 份简短报告，每份 ≤300 字 + 一张图）
r2（header 失败，ppm −719.6，K=182）：读 metrics.json 的 capture/lock/end/clock 字段，判断 −719 ppm 来自哪一支；n_sym_raw 离整数多远；_trust_sfo 走了哪个分支；_header_jobs 有没有把 0 ppm 放进候选；如果要救，应该改候选范围还是改 _trust_sfo 的阈值。
r9（header 失败，ppm −551，K=236）：同上，并比较长帧下 ppm 误差对最后一个符号的样点偏移。
r8（align 失败，ppm 0）与 r10（align 失败）：读 capture.snr1/snr2、lock.snr，判断是 Chirp 没找到、训练没锁上、还是尾训练越界；stage 为 align 的判定逻辑在 recover_array 的哪一行。
写 learn/experiments/b2_diagnose.py：给一个 wav 路径，打印 _lock_frame 的全部中间量并画 Chirp 与训练相关曲线。

第五部分　库函数一览（只列不讲）：dechirp_sfo、clock_scale_from_anchors、resample_clock、cfo_search、sto_phase、extract_ffts、apply_symbol_delays、delay_from_h，各一句话说明它改变的是定时、钟偏还是频偏，以及 DESIGN.md 中对应的编号 I1/I2。

附：15 分钟讲解提纲 + 10 道自测题 + 答案。给 C 的接口卡片：extract_active_sfo 返回数组的 shape、行的含义（前 8 训练 / K 数据 / 后 8 训练）、复数的尺度。
```

---

### P4　成员 C 详细教程（信道与译码）

```
任务：写 learn/n2-member-C.md，题为「成员 C：从频谱到 249 字节」。在 learn/n2-course.html 第 3 章基础上加深，重点补：

第一部分　逐函数精读（数学 → 代码逐行 → shape → r1 数值 → 出错表现）
channel.py: h_and_llr_scale（H=mean(Y/X)、valid 掩膜、noise_var 下限 1e-3、scale=3*clip(sqrt(noise_var/残差),0.25,4) 每一步的物理含义）, lerp_h
phy.py: qpsk_llr_scaled（输出顺序先 Q 后 I，裁剪 ±30）, descramble_llr, bits_to_bytes, parse_ph_header, slice_qpsk
receiver.py: _decode_symbol, _equalize_once（公共相位 θ 的加权估计为什么用 |H|²、为什么用硬判决当参考）, _lerp_scale, _decode_payload_symbol（四套 H 的尝试顺序、优劣判据「迭代更少、其次中位 |LLR| 更大」）, _payload_sfo_search 里 h 与 scale 的来源, recover_array 第 349 行到末尾（has_tail 分支、payload_complete 判定、三种返回）
decode_path.py: LdpcBank, extract_payload
report.py: write_run_outputs 写出了哪些文件、.partial 的条件、file_match 怎么算、_try_plots 三张图各画什么

第二部分　r1 实验 learn/experiments/c1_symbol_walk_r1.py
只 import n2 的函数，用 metrics.json 里的 sync_start 和 payload_sfo（若无则用 clock_error_ppm）调 extract_active_sfo 取整段体，然后：
- 画 |H_front| 与 |H_tail| 对 bin 的曲线，标出通带中心和带边；打印 mean_abs_h 与 metrics.json 对比。
- 取第 1、21、41 个载荷符号，画均衡前 Y、Z=Y/H、去 CPE 后的 Z 三张星座图。
- 对同一符号打印 LLR 直方图与中位 |LLR|。
- 手动走一遍：LLR → descramble_llr → LdpcBank.decode_codeword → bits_to_bytes，与 run/n2/r1 写出的文件对应字节段比较。
- 载入 cpe.npy，画公共相位随符号变化的曲线，解释斜率与残余钟偏的关系。

第三部分　失败案例诊断（4 份简短报告，每份 ≤300 字 + 一张图）
r11（234/235）：读 ldpc_symbols.csv 找出未收敛的那个码字下标，它的 α 是多少；看 summary.csv 该位置的 |LLR|；判断是孤立噪声突发还是尾部 H 插值不够。
r12（206/235）：未收敛码字在时间上的分布（连续段还是散点）；结合 cpe.npy 看相位是否在那些符号上跳变。
r4（67/141，+28 ppm）：未收敛是否集中在后半段；若是，说明 payload_sfo 仍有残差，应交给 B；给出判断依据。
r6（114/155）：同上流程，写结论。
写 learn/experiments/c2_diagnose.py：给一个 run/n2/rX 目录，画未收敛码字位置、每符号中位 |LLR|、cpe 曲线三合一图。

第四部分　库函数一览（只列不讲）：unused_bin_noise、ls_from_train、phase_aligned_mean、wiener_smooth、tone_reliability、estimate_from_known、hermitian_gate、cpe_and_nlms、equalize_symbol、turbo_refine、llr_from_symbol、decode_frame、search_ph_in_stream，各一句话说明它改变的是 H、LLR 幅度还是流程，以及 DESIGN.md 中对应的编号 I3–I8。

附：15 分钟讲解提纲 + 10 道自测题 + 答案。给 A 的接口卡片：info_bytes 的拼接顺序、extract_payload 的切片位置；给 B 的接口卡片：「若我这边 median_abs_llr 接近 0 且大部分迭代=200，请你先检查什么」。
```

---

### P5　失败录音诊断手册（第 3 周，B 与 C 合用）

```
任务：写 learn/n2-failure-playbook.md，题为「N2 失败录音诊断手册」。
第一节　决策树：从 metrics.json 的 stage 出发（capture / align / header / ldpc / ok），每个分支列出：该看的字段（capture.snr1, lock.snr, end.n_sym_raw, clock.ppm, header_attempts, blocks_ok, median_abs_llr…）、该看的文件（clock_fit.png, channel_and_noise.png, phase_tracking.png, ldpc_symbols.csv, summary.csv）、责任成员、下一步动作。决策树的每个判定条件都要能对应到 recover_array 里的具体 if 语句。
第二节　把 run/n2 里 8 份失败录音（r2 r4 r6 r8 r9 r10 r11 r12）逐份走一遍决策树，每份写：观察到的数值 → 走到哪个叶子 → 归谁 → 一句话结论。
第三节　「可以尝试的救援」清单（只描述思路与需要改动的参数名，不改代码）：候选偏移范围、±ppm 范围、_trust_sfo 阈值 0.08 与 5e-3、payload 网格 ±50 ppm、noise_var 下限、scale 的 clip 范围。每项写「改它能救哪类 stage，风险是什么」。
第四节　r13.wav 尚未运行：写出盲测步骤（命令、要记录的字段、三人各自负责看什么）。
```

---

### P6　实验脚本生成（按需，配合 P2–P4）

```
任务：在 learn/experiments/ 下生成 P2–P4 里列出的脚本（a1_compare_tx.py, a2_ldpc_roundtrip.py, b1_timeline_r1.py, b2_diagnose.py, c1_symbol_walk_r1.py, c2_diagnose.py）。
要求：
- 每个脚本开头用 sys.path.insert 加入仓库根和 n2 目录，import n2 的函数与 standard.example_tx，不复制、不修改 n2 的代码。
- 命令行参数用 argparse：--wav 或 --run-dir，--out-dir 默认 run/n2/study_<脚本名>。
- 每个脚本运行结束打印一段「期望看到什么」与「实际得到什么」的对照，用 assert 或明确的 OK/FAIL 标记。
- matplotlib 保存 png 到 out-dir，不弹窗。
- 在 learn/experiments/README.md 里写每个脚本的用途、运行命令、预期输出与截图位置。
- 先读 n2 对应函数的签名再写调用，不要猜参数。写完后自己跑一遍 r1，把实际输出贴进 README。
```

---

### P7　测验题库与评分标准（第 4 周）

```
任务：写 learn/n2-quiz.md。
分三卷（A/B/C 各 15 题）+ 一卷全员（10 题）。题型：数值计算（给 file_size 算 K；给 ppm 与符号数算累积样点；给 Z 与 scale 算 LLR）、代码定位（「这个字段由哪个函数写出」）、故障推理（给一段 metrics.json 片段问归谁）、接口题（「B 给 C 的数组 shape」）。
每题标注：考点、对应文件:函数、难度 1–3。答案与评分点单独成节。
全员卷必须包含：r1 的 40/42/41 重算；CRC 覆盖范围；QPSK 比特顺序；_trust_sfo 四分支各一例；stage 五个取值的判定。
末尾给「结业互评表」：三条标准，每条 0/1/2 分，如何打分。
```

---

### P8　合练脚本与结业评估（第 4 周）

```
任务：写 learn/n2-final-session.md，一份 90 分钟合练的逐分钟脚本。
0–10 分：跑 r13.wav（未跑过），三人各盯自己的字段，现场填对照表（表格模板给出：问题 / A 的数 / B 的数 / C 的数 / 由哪个函数写出 / 错了下一环看到什么）。
10–40 分：若 r13 不是 ok，按 n2-failure-playbook 走决策树；若 ok，改用 r4 复现一次决策树。
40–70 分：互相出题（用 n2-quiz 的题库随机抽），每人回答另外两人的 5 题。
70–90 分：三人一起对照 DESIGN.md 第 4 节 I1–I8 与附录 A，填「已实现 / 未实现 / 库函数已写但未接入」三列表，并各自提一条「如果要接入某个库函数，我负责的模块要改哪个接口」。
输出：learn/n2-final-record.md 的模板（留空待填）。
```

---

## 附　已有材料与新材料的对应

| 已有 `learn/n2-course.html` | 新增（由 AI 生成） |
|---|---|
| 第 0 章 共同课 | P1 预备知识讲义（补基础） |
| 第 1 章 成员 A | P2 逐函数精读 + 实验 + 接口卡片 |
| 第 2 章 成员 B | P3 逐函数精读 + r1 时间轴实验 + 4 份失败报告 |
| 第 3 章 成员 C | P4 逐函数精读 + 符号走查实验 + 4 份失败报告 |
| 第 4 章 合练 | P5 诊断手册 + P8 合练脚本 |
| 附录 B 检查题 | P7 三卷题库 + 互评表 |
| — | P6 六个可运行实验脚本 |
