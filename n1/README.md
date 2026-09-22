# OFDM n1 接收端设计说明

与下列标准互通的声学 OFDM 接收端（版本 **n1**）：

- 发射参考代码：[`standard/example_tx.py`](../standard/example_tx.py)
- 物理层标准表：[`standard/Standard document.docx`](../standard/Standard%20document.docx)
- 标准化会议纪要：[`standard/Standardisation Meeting 10.docx`](../standard/Standardisation%20Meeting%2010.docx)
- **LDPC（必须使用）**：[`LDPC/new_ldpc`](../LDPC/new_ldpc)（Jossy 课程包）

完整链路：chirp 同步 → 定时/CFO → 信道估计/均衡 → 软 QPSK → 解扰/LDPC → PH 文件头 → 还原文件。

## 兼容参数（对照标准文档）

| 参数 | 取值 | 说明 |
|------|------|------|
| `sample_rate` | 48000 | 采样率 |
| `N` | 8192 | FFT 点数 |
| `CP` | 2048 | 循环前缀 |
| 频带 | 约 2–15 kHz | 有效子载波 `k=400..2391` |
| 调制 | QPSK | 逆时针 Gray，`00→1+j` |
| 字节序 | 大端 | Big-endian |
| 训练 | 前 8 + 后 8 | 固定种子 QPSK |
| Chirp | 线性，0.1–20 kHz，3 s | 前后各一条；功率归一化不入标准（会议 #10） |
| LDPC | 802.16，1/2，`Z=166` | 信息 1992 bit=249 字节，码字 498 字节 |
| 加扰 | 15 位 LFSR，`0x5A4D` | LDPC 之后；每次调用复位 |
| 文件头 | `PH` + 版本 + 长度 + 文件名 + `0x00` + CRC32 | 首个载荷符号信息块开头 |

### 发射端 / 标准陷阱

1. 头字段 `Payload Symbol Count` 不可当作真实 `K`；以 `file_size` + CRC 与 preamble 间距为准。
2. 未编码流按 498 字节对齐补零 → 可能多一个全零码字。
3. 训练序列用 CPython `random`。
4. Chirp 为线性调频；与 OFDM 主体相对幅度由各组自定。
5. CRC32 覆盖到文件名最后一字节（不含 `0x00`）。

## 帧结构

```
[chirp 3s] [静音 0.5s] [8 训练] [K 数据] [8 训练] [静音 0.5s] [chirp 3s]
```

**`K` 主路径**：前后 preamble / 双 chirp 间距估计；再用 `file_size` 截断载荷。

## 目录约定

相对仓库根目录：

| 目录 | 用途 |
|------|------|
| [`data/`](../data/) | 接收音频（WAV），例如 `data/r1.wav` |
| [`source/`](../source/) | 原始文件（对照）；若无则用 `sourse/` |
| [`run/n1/`](../run/n1/) | 本接收端输出（可再分子目录） |

## 使用方法

在**仓库根**执行：

```bash
pip install -r n1/requirements.txt

# 解析 data/r1.wav → 结果写入 run/n1/r1/
python n1/run_rx.py --wav data/r1.wav --out-dir run/n1/r1

# 默认：data/ 里唯一 wav，或显式指定后写到 run/n1/<名字>/
python n1/run_rx.py --wav data/r3.wav

# 多个文件：父目录 run/n1/，各自子目录 r1/、r5/
python n1/run_rx.py --wav data/r1.wav data/r5.wav --out-dir run/n1

# 环回自测
python n1/selftest.py
```

`--out-dir run/n1/r1` 就是最终输出目录，**不会**再套一层 `r1/`。目录内含还原文件、`metrics.json`、诊断图等。

## 结果输出（对齐 AudioModem 风格）

每次解调在 `--out-dir` 写出：

| 文件 | 含义 |
|------|------|
| `metrics.json` | 同步/CFO/PPM、train_mse、header_ok、turbo、error 等 |
| `channel.png` | \|H_start\| / \|H_end\| |
| `constellation.png` | 均衡后星座（前几个符号） |
| `H_start.npy` / `H_end.npy` | 信道估计 |
| `info_stream.partial.bin` | LDPC 信息字节流（头失败时也保留） |
| 还原文件 或 `recovered.partial.bin` | 成功写文件名；失败写 partial |

控制台一行摘要示例：

```text
r1.wav: stage=done K=42 ppm=-11.4955 cfo_hz=-0.012 train_mse=7.447e-01 header_ok=True turbo=False recovered=cute.jpg file_match=True
recovered=...\run\n1\r1\cute.jpg
out_dir=...\run\n1\r1
metrics=...\run\n1\r1\metrics.json
```

`file_match=n/a` 只表示 `source/` 没有同名原文件，不是失败。

批量：

```bash
python n1/run_rx.py --wav data/r1.wav data/r5.wav --out-dir run/n1
# → run/n1/r1/、run/n1/r5/，以及 run/n1/batch_summary.csv
```

可选 `--source` 对比原始文件，填充 `file_match`。

## 处理流水线与创新点

1. 双 chirp 相关 + 抛物线峰值 + SFO 估计/重采样
2. 训练精定时（亚采样）+ 匹配网格 CFO + 相邻前导相位差精修 CFO
3. **多候选 body 起点**（CP 邻域 ±128/256…），**PH CRC 裁决**（对齐 [AudioModem n3_2](https://github.com/Sline-L/AudioModem/tree/codex/latest-unsuccessful-version)）
4. **线性插值采样钟校正**（`corrected[n]=rx[n·scale]`，可表示十几 ppm；避免 `resample_poly` 量化掉钟偏）
5. 多训练相位对齐 LS + Wiener；首尾 H 相位斜率诊断；按残余时钟做符号间时延；条件插值尾 H
6. MMSE + Hermitian 镜像 MRC；数据符号 CPE + NLMS 跟踪
7. **逐子载波 LLR 门控**（训练残余可靠性）→ 课程包 `ldpc.code.decode`
8. 首轮头失败时 Turbo 决策反馈再估 H
9. PH 滑动搜索 + `file_size` 截断

方法参考：[AudioModem n3_2](https://github.com/Sline-L/AudioModem/tree/codex/latest-unsuccessful-version)（多候选 + 头 CRC 权威 + 诊断输出）、声学 OFDM 软 LLR 加权与精确时钟重采样。

## 模块划分

| 文件 | 作用 |
|------|------|
| `course_ldpc.py` | 将 `LDPC/new_ldpc/py` 注入为 `import ldpc` |
| `ofdm_rx.py` | 接收编排 |
| `sync.py` / `equalizer.py` / `demod.py` | 同步 / 均衡 / 解调与头解析 |
| `run_rx.py` / `selftest.py` | CLI 与环回自测 |

## LDPC 使用说明

接收端与自测**只使用**仓库内 [`LDPC/new_ldpc`](../LDPC/new_ldpc)，通过 `course_ldpc.install_as_ldpc_module()` 让 `import ldpc` 指向该包。

首次使用前需编译 C 解码库（课程 README）：

```bash
cd LDPC/new_ldpc
# Linux / macOS:
gcc -lm -shared -fPIC -o bin/c_ldpc.so src/c_ldpc.c
# Windows（MinGW 示例）:
gcc -shared -o bin/c_ldpc.dll src/c_ldpc.c -lm
```

`ldpc.py` 已改为按包目录定位 `bin/`，不依赖当前工作目录。

## 功能通过标准

1. 理想环回字节一致。
2. 文件头 CRC 通过；文件名与 `file_size` 正确。
3. 轻度损伤（AWGN 或时延）可还原或给出明确失败阶段。
