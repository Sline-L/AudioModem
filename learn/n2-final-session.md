# 第 4 周合练（90 分钟）

材料：`learn/n2-failure-playbook.md`、`learn/n2-quiz.md`、`learn/n2-final-record.md`。输出写到 `run/n2/study_r13`，不要覆盖已有的 `run/n2/r*`。

## 0–10 分钟　盲测 r13

```text
cd g:\OFDM_gerson\n2
python run_rx.py --wav ..\data\r13.wav --out-dir ..\run\n2\study_r13
```

当场把 `metrics.json` 填进记录表。每人只填自己的列，不替别人算。

| 问题 | A | B | C | 函数 | 错了下一环看见什么 |
|---|---|---|---|---|---|
| file_size / declared / K |  |  |  | `parse_ph_header`，`coded_symbol_count` | 训练对不上或 CRC 范围错；循环上界错 |
| sync_start、ppm、n_sym_raw |  |  |  | `_lock_frame`，`_trust_sfo` | 取符号越界或头 CRC 失败 |
| blocks、mean_abs_h、median_abs_llr、file_match |  |  |  | `_decode_payload_symbol`，`extract_payload`，`write_run_outputs` | LDPC 不收敛或文件字节对不上 |

A 用 file_size 重算 declared 和 K，看是否和 JSON 一致。B 看 `n_sym_raw` 离整数是否小于 0.08。C 看 `blocks_total` 是否等于 K−1。

## 10–40 分钟　决策树

- r13 不是 `ok`：按 `n2-failure-playbook.md` 第 1 节走到叶子，把责任人和要看的文件写进记录。
- r13 是 `ok`：改用 `run/n2/r4`。C 指出未收敛最长段是符号 42–87，B 指出 `n_sym_raw` 离整数只有 0.004、snr2≈135。结论应是「先留在 C，B 只核对第二 Chirp」，不是把 +28 ppm 再平均一次。

这一段只开 `b2_diagnose.py` 或 `c2_diagnose.py`，不改 `n2/`。

## 40–70 分钟　互问

从 `n2-quiz.md` 抽题，不看答案。A 答 B 卷 5 题，B 答 C 卷 5 题，C 答 A 卷 5 题。每题 2 分钟。错题记下函数名，不当场翻代码争论超过 1 分钟。

## 70–90 分钟　对照 DESIGN.md

打开 `n2/DESIGN.md` 第 4 节 I1–I8 和 `learn/n2-course.html` 附录 A。填三列：

| 编号 | 主路径在用 | 库函数已写、未接入 | 没写 |
|---|---|---|---|
| I1 去斜 / 整段重采样 | 抛物线峰、不用双 Chirp 当唯一 SFO | `dechirp_sfo`，`resample_clock` |  |
| I2 二维 STO–CFO | 训练匹配 + 头 CRC 选候选 | `cfo_search`，`extract_ffts` |  |
| I3 Wiener |  | `wiener_smooth`。主路径是 `lerp_h` |  |
| I4 NLMS | 只去公共相位 | `cpe_and_nlms` |  |
| I5 Hermitian 门控 |  | `hermitian_gate` |  |
| I6 保护音噪声 | 训练残差做尺度 | `unused_bin_noise` |  |
| I7 Turbo | `turbo=false` | `turbo_refine` |  |
| I8 滑动搜头 | CRC 范围与发射端一致；K 来自同步 | `search_ph_in_stream` 未进主路径 |  |

每人写一条「如果要把某个库函数接进 `recover_array`，我这边的接口要改什么」：

- A：接入后训练序列、LLR 顺序、CRC 范围都不许变。
- B：接入 `resample_clock` 会改变 `extract_active_sfo` 看到的时间轴，ppm 不能补偿两次。
- C：接入 `wiener_smooth` 替换的是 `lerp_h` 的输入 H，不是 LLR 的符号。

## 结业

用 `n2-quiz.md` 末尾的互评表打分，写进 `learn/n2-final-record.md`。2 分的标准是能指出函数，不是能背 ppm。
