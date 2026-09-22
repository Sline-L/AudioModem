# 实验脚本

在仓库根目录运行。脚本只 `import` `n2` 和 `standard`，不改它们。图和文本写到 `run/n2/study_*`。

| 脚本 | 命令 | 预期 |
|---|---|---|
| `a1_compare_tx.py` | `python learn/experiments/a1_compare_tx.py` | Chirp / 训练 / 扰码差值为 0；CPython 与 NumPy 的前 8 个随机数不同；改文件名内字节后 PH 解析失败，改 CRC 之后的补零仍成功 |
| `a2_ldpc_roundtrip.py` | `python learn/experiments/a2_ldpc_roundtrip.py` | `K=1992`，`N=3984`；`sigma=0.05` 时误码为 0 且迭代小于 200 |
| `b1_timeline_r1.py` | `python learn/experiments/b1_timeline_r1.py` | peak、lock、K 与 `run/n2/r1/metrics.json` 一致；`sfo=0` 时尾训练残差更大 |
| `b2_diagnose.py` | `python learn/experiments/b2_diagnose.py --wav data/r1.wav` | 打印 `_lock_frame`，保存 `chirp_metric.png`、`train_metric.png` |
| `c1_symbol_walk_r1.py` | `python learn/experiments/c1_symbol_walk_r1.py` | 载荷第 0 块与 `source/cute.jpg` 前 249 字节相同 |
| `c2_diagnose.py` | `python learn/experiments/c2_diagnose.py --run-dir run/n2/r11` | 未收敛下标与 `ldpc_symbols.csv` 一致 |

## 已跑过的结果

`a1`：差值为 0。`random.seed(80)` 的前 8 个是 `[2, 3, 3, 2, 2, 3, 2, 0]`，NumPy `default_rng(80)` 是 `[3, 2, 0, 0, 2, 0, 1, 2]`。20 字节文件名的 `0x00` 在下标 31，CRC 在 `[32, 36)`。

`a2`：`sigma=0.05` 迭代 0 次、误码 0。`sigma=0.4` 起迭代到 200。

`b1`：`peak1=44926.070`，粗起点 `212926.070`，`lock.start=212925.047`（差 1.023 样点），`n_sym_raw=49.999410`，`K=42`。−11.65 ppm 在 58 个符号上累积 6.92 样点。尾训练残差：正确钟偏 0.70，`sfo=0` 时 4.52。图在 `run/n2/study_b1/chirp_peaks.png`。

`b2`（r1）：`lock.snr=7270.6`，离整数 0.00059。图在 `run/n2/study_b2_r1/`。

`c1`：`|H_front|` 平均 2.59，`metrics` 里联合 H 的 `mean_abs_h=2.34`。载荷第 0 块迭代 6 次，中位 `|LLR|=2.89`，与 `cute.jpg` 前 249 字节一致。图在 `run/n2/study_c1/`。

`c2`（r11）：未收敛 1 个，下标 175。图在 `run/n2/study_c2_r11/ldpc_llr_cpe.png`。
