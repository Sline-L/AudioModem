# N2 失败录音诊断手册

只看 `metrics.json`、`ldpc_symbols.csv` 和三张图，决定问题交给谁。判定条件对应 `n2/receiver.py` 的 `recover_array`。

---

## 1. 决策树

先读 `stage`，再读 `lock` 是不是 null。`error` / `reason` 在头没找到时一律写成「PH 头 CRC 未通过」，r8 就是这样，不能单看这句话。

```text
stage == ok
  结束。核对 file_match、blocks_ok==blocks_total、header_ok。

stage == capture
  波形短于两个 Chirp，或 find_two_chirps 失败且没有任何头候选。
  看 capture.snr1。交给 B。

stage == align
  若 lock 为 null：Chirp 可能在，训练没锁上。
    recover_array 里：jobs 为空（_lock_frame 返回 None）且 first 不是 None，stage 记 align。
    看 capture.snr1 / snr2，跑 b2_diagnose.py 看训练相关。交给 B。
  若 lock 有值、reason 是「尾训练越界」或 fft_error：
    起点有了，extract_active_sfo 越界。交给 B，看 K 和 ppm。

stage == header
  试过候选，parse_ph_header 都没过。
  看 n_sym_raw 离整数有多远、clock_error_ppm、lock.snr。交给 B。
  不要把 declared_symbols 或 LDPC 迭代拿来解释，这一步还没有载荷迭代。

stage == ldpc
  若 error 像「译码器初始化」：缺 c_ldpc.dll。环境问题，三人一起看。
  若 header_ok 且 blocks_total>0：头已经过了。
    median_abs_llr 接近 0 且大部分 iters=200 → 先退回 B 的起点和钟偏。
    median_abs_llr 仍在 3 左右、未收敛是一段或一个点 → C 看 ldpc_symbols.csv 和 cpe.npy。
```

| 要看的字段 | 谁写的 |
|---|---|
| `capture.snr1/snr2/peak1/peak2` | `find_two_chirps` |
| `lock.snr/start` | `matched_start`；null 表示 `_lock_frame` 失败 |
| `end.n_sym_raw/K` | `find_end_preamble` |
| `clock_error_ppm` | 搜头阶段的 `_trust_sfo`，不是 payload 1 ppm 网格 |
| `blocks_ok/blocks_total` | 载荷循环，当前代码是 K−1 |
| `median_abs_llr` | 载荷 LLR 的中位数 |
| `mean_abs_h` | 联合 H 的平均幅度 |

图：`clock_fit.png` 给 B，`channel_and_noise.png` 和 `phase_tracking.png` 给 C，`ldpc_symbols.csv` 给 C。

---

## 2. 八份失败录音

| 录音 | 观察到的数 | 叶子 | 交给 | 一句话 |
|---|---|---|---|---|
| r2 | header，ppm −719.6，`n_sym_raw=189.8443`（离整数 0.156），lock.snr≈3021 | header，且 0 ppm 中心也试过 | B | 尾训练不在符号网格上，±256 样点的头搜索不够 |
| r9 | header，ppm −551，`n_sym_raw=243.8587`（离整数 0.141） | 与 r2 相同 | B | 长帧上这个 ppm 会把帧尾推出 FFT 窗；先重看尾峰 |
| r8 | align，lock 为 null，ppm 0，reason 仍写 CRC 未通过，Chirp SNR 很高 | align 且没锁上 | B | 扫频在，训练匹配没过 snr≥4 |
| r10 | align，没有 lock/end，Chirp SNR 极高 | 与 r8 相同 | B | 同一类，不要去看 LDPC |
| r11 | ldpc 234/235，ppm −7.4，median \|LLR\|≈3.03，只有符号 175 | 孤立一个码字 | C | 不是钟偏斜坡 |
| r12 | ldpc 206/235，ppm −2.3，未收敛全在前半，最长 87–99 | 前半一段 | C | 尾部是好的 |
| r4 | ldpc 67/141，ppm +28，最长失败 42–87；snr2≈135；`n_sym_raw` 离整数 0.004 | 中部连续失败 | C，B 看一眼第二 Chirp | 失败分布不是尾部单调变差 |
| r6 | ldpc 114/155，ppm −4.7，最长失败 60–80，后半只有 3 个 | 中前部一段 | C | ppm 已经很小 |

---

## 3. 可以试的救援（只改参数，先写在实验脚本里）

| 参数 | 位置 | 可能救的 stage | 风险 |
|---|---|---|---|
| `offsets_lock` 现在最大 ±256 | `_header_jobs` | r2/r9 这种离整数 >0.08 的 header | 候选变多，错误的头更可能撞上 CRC |
| `ppm_local` 现在 ±30 | 同上 | 真实钟偏落在中心 ±30 以外 | 同上 |
| 0 ppm 中心的阈值 `40e-6` | `_header_jobs` | 主估计很离谱但 0 附近才对 | r2 已经包含 0 ppm 仍失败，单改这个救不了 r2 |
| `aligned` 的 0.08 和 `5e-3` | `_trust_sfo` | 把错误 Chirp 间距挡在外面 | 放宽后再平均，会回到「两个数取平均」那个坑 |
| payload 网格 ±50 ppm、步长 1 | `_payload_sfo_search` | 头已通过、尾部码字成片失败 | 每档都要两次 `extract_active_sfo`，更慢；救不了头都没过的录音 |
| `noise_var` 下限 `1e-3`，`scale` 的 clip `[0.25,4]` 再乘 3 | `h_and_llr_scale` | 个别子载波 LLR 过大或过小 | 整体平移尺度，孤立码字（r11）不一定动 |

r8/r10 不在这张表里：训练峰都没锁上时，先看波形和相关曲线，改 ppm 网格没有输入。

---

## 4. r13 盲测

`data/r13.wav` 还没有 `run/n2/r13`。合练时：

```text
cd n2
python run_rx.py --wav ..\data\r13.wav --out-dir ..\run\n2\study_r13
```

| 谁 | 当场看 |
|---|---|
| A | `header.file_size`、`declared_symbols`、`K` 能否互相推出来 |
| B | `capture`、`lock`、`end.n_sym_raw`、`clock_error_ppm` |
| C | `stage`、`blocks_ok/blocks_total`、`median_abs_llr`、恢复出的文件 |

不是 `ok` 就回到第 1 节，只重跑对应成员的诊断脚本。
