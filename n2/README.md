# OFDM n2 接收端

第二版声学 OFDM 接收机，空中接口以 [`standard/example_tx.py`](../standard/example_tx.py) 为准。LDPC 直接使用仓库内 [`LDPC/new_ldpc`](../LDPC/new_ldpc/README.md)，不另写译码器。

设计说明见 [`DESIGN.md`](DESIGN.md)。

## 目录约定

相对仓库根目录：

| 目录 | 内容 |
|------|------|
| `data/` | 接收到的音频（WAV） |
| `source/` | 未做 OFDM 的原始文件，用于对照 |
| `run/n2/` | 本接收端恢复结果（可再分子文件夹） |

## 依赖

```bash
pip install -r n2/requirements.txt
```

译码前需按 LDPC 库说明编译 C 扩展（在 `LDPC/new_ldpc` 下）：

```bash
gcc -lm -shared -fPIC -o bin/c_ldpc.so src/c_ldpc.c
```

Windows 将输出改为 `bin/c_ldpc.dll`。

## 用法

先进入 `n2` 目录，路径按仓库根的 `data` / `run` 写（与 n1 相同形式）：

```bash
cd g:\OFDM_gerson\n2

python run_rx.py --wav ..\data\recv.wav --out-dir ..\run\n2\trial1
python run_rx.py --wav ..\data\recv.wav --out-dir ..\run\n2
python selftest.py
```

窗口版（同一套解调，结果仍写到 `run/n2/`）：

```bash
cd g:\OFDM_gerson\n2
python app.py
```

也可以双击 `n2/启动N2接收机.bat`。在窗口里选 `data/` 中的录音，点 Decode，即可看摘要、波形、诊断图和恢复文件。界面英文在前，中文附在后面。

`data/` 里只有一个 WAV 时可省略 `--wav`。默认输出目录是 `run\n2\`。`--out-dir` 可写成 `..\run\n2\自定义名`。若 `source/` 中有与帧头同名的原文件，会自动做字节对照。

终端打一行摘要（风格对齐 [AudioModem](https://github.com/Sline-L/AudioModem)），详细结果在输出目录：

| 文件 | 内容 |
|------|------|
| `metrics.json` | 同步、钟偏 ppm、头 CRC、LDPC 收敛、file_match |
| `ldpc_symbols.csv` | 每个码字迭代次数 |
| `summary.csv` | 每子载波 \|H\|、训练残差、平均 \|LLR\| |
| 恢复文件 | CRC 通过时用原文件名；否则 `*.partial` |
| `decoded_header.bin` | 译出的 249 字节信息头 |
| `H.npy` / `H_end.npy` / `cpe.npy` / `rx_llr.npy` | 信道与软信息 |
| `clock_fit.png` / `channel_and_noise.png` / `phase_tracking.png` | 诊断图 |

准确性按下面看，**有恢复文件且 `header_ok=True` 就是解出来了**。`file_match=n/a` 只表示 `source/` 里没有同名原文件，不是失败。

1. `recovered=文件名`：已经按帧头写出恢复文件  
2. `header_ok`：PH 头 CRC 通过  
3. `ldpc=收敛数/总数`、`complete`：载荷码字是否全部收敛  
4. `file_match`：`True` / `False` / `n/a`（无对照源）

采样钟用频域线性相位：符号仍按 `N+CP` 等间隔切窗，第 g 个符号乘 `exp(-j2πkτ/N)`，`τ = frac(start) + sfo·(g·(N+CP)+CP)`。PH 头 CRC 在多个定时候选里做最终判决，再用前后训练 1 ppm 网格细化 payload 钟偏。

对较难录音额外做了：

1. **不要把两个钟偏取中位数**（两个数的 median 等于均值，会把正常 ppm 和错误 Chirp 间距平均成几十万 ppm）。
2. **头搜索同时试 0 ppm 和训练峰**，Chirp 锁失败时仍能找帧。
3. **立体声逐路试**；载荷用前后训练插值 H + 公共相位，失败码字换一套 H 再译。

## 功能通过标准

1. 理想回环：恢复字节与发射文件一致。
2. 帧头 CRC 通过，文件名与 `file_size` 正确。
3. 轻度损伤（延迟 / AWGN）下恢复成功，或给出 `capture / align / channel / ldpc / header` 之一的失败阶段。
