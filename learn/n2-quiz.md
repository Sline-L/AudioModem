# N2 测验

卷 A / B / C 各 15 题，全员卷 10 题。难度 1 是直接计算或定位，2 要联两处代码，3 要拿一份录音推理。答案在文末。

## 全员

1. （难度 1，`coded_symbol_count` / `header_gen`）`file_size=9829` 时 declared、K、载荷码字数。
2. （难度 1，`parse_ph_header`）CRC 覆盖到哪里，结束 `0x00` 在不在里面。
3. （难度 2，`bytes_to_qpsk_grid` / `qpsk_llr_scaled`）一个字节的 8 个比特怎样落到 4 个 QPSK，LLR 先输出哪一路。
4. （难度 2，`_trust_sfo`）训练已对齐且与 Chirp 差 200 ppm，返回哪个。
5. （难度 2，`_trust_sfo`）`n_sym_raw` 离整数 0.15，Chirp 估计 −700 ppm 且绝对值小于 5000 ppm，返回哪个。
6. （难度 1，`recover_array`）`stage` 五个取值各表示什么。
7. （难度 1，预备知识）−11.65 ppm 在 58 个符号上累积大约多少样点。
8. （难度 2，`h_and_llr_scale`）`scale` 为什么不能翻转 LLR 符号。
9. （难度 1，`report.py`）`file_match=n/a` 是什么意思。
10. （难度 2，成员分工）`median_abs_llr≈0` 且几乎全部 `iters=200`，先找谁。

## 卷 A　空中接口

1. （1，`derived`）`ldpc_z` 为什么是 166。
2. （1，`linear_chirp`）Chirp 多少样点，相位是线性调频还是对数扫频。
3. （2，`mapp`）字节 `0x80` 的第一个符号。
4. （2，`mapp`）字节 `0x01` 的第四个符号的 I、Q。
5. （1，`train_freq_time`）训练随机数必须用哪个生成器。
6. （2，`lfsr_sequence`）每个码字要不要复位，接收端作用在比特上还是 LLR 上。
7. （2，`header_gen`）20 字节文件名时，`0x00` 和 CRC 的下标。
8. （1，`LdpcBank`）`app<0` 判成什么；迭代 200 是什么。
9. （2，字节账）`file_size=498` 的 K，以及会不会多出一个全零信息块。
10. （3，r3）为什么不要用 r3 的 `blocks_total=182` 证明当前代码把帧头符号也算进载荷循环。
11. （1，`selftest.py`）`test_phy_against_tx` 比了哪四样。
12. （2，`ldpc_encoding`）发射顺序是先扰码还是先 LDPC。
13. （1，`save_sound`）int16 的缩放系数，接收端用什么除回来。
14. （2，接口）`train_freq_time` 给 B 的时域数组形状。
15. （3，`file_process`）`declared_symbols` 少算了哪一段，r1 上少算了几个符号。

## 卷 B　捕获与对准

1. （1，`find_two_chirps`）`snr1` 的阈值。
2. （1，`matched_start`）`_lock_frame` 要求的 snr 阈值，模板是几个训练符号。
3. （1，`coarse_body_start`）粗起点的公式。
4. （2，`find_end_preamble`）峰在后训练的哪里，K 怎样从 `n_sym_raw` 得到。
5. （2，`extract_active_sfo`）第 m 个符号、FFT 内第 n 点的取样位置。
6. （1，`parabolic_peak`）修正量限制在多少。
7. （2，`_header_jobs`）起点偏移有哪些，什么时候额外试 0 ppm。
8. （2，`_trust_sfo`）为什么不能对两个估计取 median。
9. （3，r2）−719.6 ppm 来自 Chirp 还是后训练间距，依据是什么。
10. （3，r8）stage=align 且 lock 为 null，reason 却写 CRC，以哪个为准。
11. （1，`_payload_sfo_search`）网格范围和步长。
12. （2，`metrics.json`）`clock_error_ppm` 是不是 payload 细化后的值。
13. （2，r1）粗起点和 `sync_start` 差大约多少样点。
14. （3，r9）0.14 个符号的偏差大约是多少样点，和 ±256 的搜索比怎样。
15. （1，库函数）主路径的 `cfo_hz` 和 `resampled` 在 r1 上各是多少。

## 卷 C　信道与译码

1. （1，`h_and_llr_scale`）H 的估计公式，`noise_var` 的下限。
2. （2，`report.py`）`median_training_noise≈3` 实际是什么数组的中位数。
3. （1，`qpsk_llr_scaled`）`Z=0.2+0.9j`、`scale=3` 的两条 LLR。
4. （2，`_decode_payload_symbol`）`K=42` 时第一个和最后一个载荷符号的 α。
5. （2，`_decode_payload_symbol`）不收敛时另外三套 H 的顺序，优劣怎么比。
6. （1，`extract_payload`）切哪一段。
7. （2，`write_run_outputs`）头通过、长度对、但有码字未收敛，文件名带不带 `.partial`。
8. （3，r11）未收敛下标和对应的大致 α。
9. （3，r4）为什么失败分布不像钟偏把尾部拖出窗外。
10. （2，`lerp_h`）插值是沿频率还是沿符号。DESIGN.md 的 I3 主张哪一种，主路径用哪一种。
11. （1，`slice_qpsk`）公共相位的参考符号从哪来。
12. （2，r12）未收敛集中在前半还是后半。
13. （1，`LdpcBank.decode_codeword`）输入 LLR 是加扰前还是加扰后。
14. （2，给 B 的接口）什么时候你应该把问题退回 B。
15. （3，`recover_array`）载荷循环为什么是 K−1 而不是 K。

---

## 答案与评分点

每题 1 分。数值题允许四舍五入到整数样点或 0.1 ppm。代码定位题必须点到函数名。

### 全员

1. 40、42、41。`ceil(2*9829/498)=40`，`2*ceil(10078/498)=42`，载荷 `data[1:]` 共 41。
2. `ba[:11+len(filename)]`，不含结束 `0x00`，不含 CRC。
3. 高位对到低位对；每对低位走 I、高位走 Q。LLR 先 Q 后 I。
4. `sfo_end`。差 200 ppm > 80 ppm，不平均。
5. `sfo_chirp`。`aligned` 为假，Chirp 仍在 5000 ppm 以内。
6. `capture` 波形太短或 Chirp 没有且无候选；`align` 锁失败或取符号越界；`header` 候选都没过 CRC；`ldpc` 译码器或载荷码字问题；`ok` 头过且载荷全部收敛。
7. 约 6.9。`11.65e-6 * 58 * 10240`。
8. 尺度表示可信度。翻符号会把 0/1 说反。
9. `source/` 没有同名原文件，不是解码失败。
10. B。先看起点和钟偏。C 的 `|H|` 是第二步。

### 卷 A

1. `498/3`。`chunk_size % 6 == 0` 保证它是整数。
2. 144000。线性调频（二次相位）。
3. `1-1j`。
4. I=−1，Q=+1。`b_i=0`，bit0 走 I，bit1 走 Q，只有 bit0 为 1。
5. CPython `random.seed` / `random.randint`，种子 80。
6. 每个码字复位。接收端乘在 LLR 上。
7. 文件名 `[11,31)`，`0x00` 在 31，CRC 在 `[32,36)`。
8. 判 1。200 是没收敛（迭代上限）。
9. K=4。747 补到 996，多 249 个 0，是一个全零信息块。
10. 当前代码 `payload_specs = data[1:]`，r1/r5/r7/r14 的 `blocks_total` 都是 K−1。r3 等于 K，和这份源码的统计口径不一致。
11. Chirp、64 位扰码、16 个训练的频域和时域、一个 PH 头。
12. 先 LDPC，再扰码。
13. 乘 32767 写成 int16；`_wav_to_float` 对 int16 除以 32767。
14. `(16, 10240)`。
15. 不含 249 字节帧头。r1 上 declared=40，K=42，少 2 个符号的量级（头本身会再占一个数据符号，补零规则使 K 比 declared 大，这里大 2）。评分点：说出「不含帧头」并且用 40 和 42 这两个数。

### 卷 B

1. 8。
2. 4。前 2 个训练符号。
3. 第一峰 + 144000 + 24000。
4. 峰在后 8 个训练的起点。`K = round(n_sym_raw) - 8`。
5. `start + (1+sfo) * (m*10240 + 2048 + n)`。
6. ±1 个样点。
7. `0, ±128, ±256, ±64`。`|sfo_main|>40 ppm` 时加 0 ppm 中心。
8. 两个数的中位数等于平均数。
9. Chirp。后训练间距约 −819 ppm，和 −719.6 对不上；`n_sym_raw` 离整数 0.156，`aligned` 为假。
10. 以 stage 和 `lock is None` 为准。reason 在 `found is None` 时就会写成 CRC 失败。
11. 入选钟偏 ±50 ppm，步长 1 ppm。
12. 不是。它是搜头阶段的值。
13. 约 1 个样点（212926 对 212925.047）。
14. `0.141*10240≈1447` 样点，远大于 256。
15. `cfo_hz=0`，`resampled=false`。

### 卷 C

1. `H=mean(Y/X)`。下限 `1e-3`。
2. LLR 的 `scale`，不是 σ²。
3. 2.7 和 0.6，先 Q 后 I，两个比特都更像 0。
4. `1/41` 和 1。
5. 联合、仅前、仅后。比 `(iters, -median|LLR|)`。
6. `[offset+249 : offset+249+file_size]`，主路径 offset=0。
7. 不带。`ok` 仍是 false，阶段 `ldpc`。
8. 175。`α=176/235≈0.75`。
9. 最长失败在 42–87，尾部没有单调变差；`n_sym_raw` 离整数只有 0.004。
10. 主路径 `lerp_h` 沿符号。I3 主张沿频率 Wiener，不在主路径上。
11. `slice_qpsk(Z)` 的硬切片。
12. 前半。最长 87–99，下标 117 之后没有。
13. 加扰后的 LLR。函数内部再解扰。
14. 中位 `|LLR|` 接近 0 且大部分迭代是 200。r11 这种孤立失败不退回。
15. 第 1 个数据符号在 `_try_header` 里译过，`data[1:]` 才进载荷循环。

## 结业互评

每人被另外两人打分，取平均。每条 0/1/2。

| 条 | 2 分 | 1 分 | 0 分 |
|---|---|---|---|
| 能重算 r1 的 40/42/41，并说清三个数的来源 | 三个数和三个函数都对 | 数对，来源只对一个 | 数错 |
| 抽一个 `metrics.json` 字段，指出函数 | 函数和它在流水线的位置都对 | 只说出模块（capture/align/channel） | 归错模块 |
| 拿一份失败录音，说出交给谁、下一步看哪个文件 | stage、责任人、文件三者一致 | 责任人对，文件错 | 责任人错 |
