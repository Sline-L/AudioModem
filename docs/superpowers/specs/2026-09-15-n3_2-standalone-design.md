# N3 2 独立标准声学调制解调器设计

## 目标

在 `n3_2/` 中从零实现独立的 OFDM 文件发射端和接收端。N3.2 的发射线格式严格复现
`Standardization/example_tx.py`，接收端保留 AudioModem 的鲁棒同步、SFO 校正、信道
估计、LDPC 软判决和诊断能力。

N3.2 不得导入、调用、复制运行时对象或依赖 `n3/`、`n3_1/`。它只可使用 Python、
NumPy、SciPy 和仓库已有的 `lib/ldpc` 库。

已有 `n3/`、`n3_1/`、N1、N2 和归档数据保留在磁盘上，但不构成 N3.2 的运行依赖。

## 标准来源与优先级

`Standardization` 文件夹中的可执行参考优先级最高：

1. `example_tx.py` 定义实际发射的帧格式与处理顺序。
2. `header_gen.py`、`training_symbol_gen.py` 与 `scrambler.py` 分别定义 Header、训练和
   扰码算法。
3. `Standard document.docx` 仅补充不与可执行参考冲突的参数。

因此 N3.2 发送一个 Header 信息块，不实现 Word 文档中与示例冲突的 Header 三副本组合。

## 目录与边界

```text
n3_2/
  __init__.py
  modem.py
  ldpc_codec.py
  sync.py
  tx.py
  rx.py
  ber_test.py
  README.md
tests/
  test_n3_2_protocol.py
  test_n3_2_ldpc.py
  test_n3_2_tx.py
  test_n3_2_sync.py
  test_n3_2_rx.py
  test_n3_2_independence.py
```

所有 N3.2 模块仅使用 `n3_2` 内部相对导入。`test_n3_2_independence.py` 会扫描 N3.2
源码，拒绝 `import n3`、`from n3`、`import n3_1` 和 `from n3_1`。

## 固定物理层

| 参数 | 值 |
|---|---:|
| 采样率 | 48000 Hz |
| FFT 长度 | 8192 |
| 循环前缀 | 2048 samples |
| OFDM symbol 长度 | 10240 samples |
| 有效正频子载波 | 400 至 2391 |
| 有效子载波数 | 1992 |
| 调制 | 参考 QPSK，未归一化 `±1 ± 1j` |
| 位序 | 大端；第一位控制虚部，第二位控制实部 |
| 前置训练 | 8 个 OFDM symbols |
| 后置训练 | 8 个 OFDM symbols |
| chirp | 3 秒，100 Hz 至 20000 Hz 线性扫频 |
| 静音保护 | OFDM 前后各 0.5 秒 |

训练使用 Python `random.seed(80)` 生成 16 个 QPSK 频域 symbols，前 8 个置于数据前、后
8 个置于数据后。正频子载波的镜像位置必须填入共轭值，使 IFFT 输出为实数。

## Header、LDPC 与扰码

第一个 249-byte 信息块为 Header：`PH`、版本 0、4-byte 大端文件长度、4-byte 大端
payload 信息 symbol 数、UTF-8 文件名、零终止符和 4-byte 大端 CRC32。CRC 仅覆盖文件名
最后一个字节之前的内容，不包含零终止符。

Header 与文件内容拼接后补零到 498-byte 的整数倍，再切为 249-byte 信息块。每个信息块：

```text
249 bytes -> 1992 big-endian bits -> IEEE 802.16 LDPC rate 1/2 Z=166
-> 3984 coded bits -> XOR 0x5A4D scrambler -> 1992 QPSK carriers
```

扰码对每个码字重置。接收端先在软 LLR 域解除扰码，再执行 LDPC 解码。

文件的 Header `payload_symbols` 为 `ceil(file_size / 249)`；总编码 OFDM symbol 数可能
更大，因为参考发射端会补齐到 498-byte 边界。接收端只恢复 Header 指定数量的 payload
信息块，并忽略末尾填充块。

## 独立发射端

`python -m n3_2.tx INPUT --out OUTPUT` 是唯一标准发射命令。它不提供非标准调制、非
LDPC、替代 training 或替代 chirp 参数。帧结构必须为：

```text
3 s chirp -> 0.5 s silence -> 8 training -> coded header and payload
-> 8 training -> 0.5 s silence -> same 3 s chirp
```

发射端写入 WAV、`.training.npy`、`.header.bin` 和 `.meta.json`。生成文件置于
`data/n3_2/`。

## 独立接收端

`python -m n3_2.rx INPUT --out DIR [--source FILE]` 只解码上述标准帧。它执行：

```text
WAV 格式校验 -> 多 chirp 候选 -> 合法前后 chirp pair
-> training 精确时序 -> 粗细 SFO 搜索 -> 信道估计与均衡
-> 第一个码字的 Header LDPC 解码 -> Header CRC/字段校验
-> 指定 payload 块解码 -> 恢复文件与诊断
```

接收端可对双声道 48 kHz 录音分别评估左右声道的 chirp/training 分数，选择更可靠的一路；
单声道输入保持原样。它拒绝不是 16-bit 48 kHz 的 WAV。该声道选择只影响接收端，不改变
标准发射线格式。

接收端总写入 `metrics.json`、`header_debug.json`、`decoded_header.bin`、
`decoded_payload.bin`、`H.npy`、`training_phase_fit.npy`、`payload_symbols.npy` 和
`summary.csv`。提供 `--source` 时，`header_debug.json` 还记录 Header 位错误数与 BER。

## 错误处理与验证

Header 未能解码时，诊断必须区分 WAV 格式、声道选择、chirp 检测、training 时序、SFO、
LDPC 与 Header CRC/字段失败，不能笼统称为 Header 错误。

测试必须验证：

1. Header、训练和扰码与 `Standardization` 黄金向量逐字节或逐数组一致。
2. QPSK 位序、共轭镜像、CP、chirp、帧样本数与参考一致。
3. IEEE 802.16 rate 1/2 Z=166 LDPC 无噪声编解码与校验矩阵 syndrome 为零。
4. 发射—接收离线回环、增益/延迟、轻噪声和双声道重复录音均可恢复。
5. 非法 WAV、截断帧、Header 损坏与无效字段产生明确诊断且不写错误恢复文件。
6. N3.2 源码不依赖 N3/N3.1；Python 编译检查与所有 N3.2 测试通过。

现有 N1/N2/N3/N3.1 文件不得被删除或修改。
