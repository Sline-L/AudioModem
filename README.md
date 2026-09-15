# AudioModem

AudioModem 是一个面向真实声学信道实验的 OFDM 文件传输项目。仓库保留了
N1、N2 实验线，并新增了严格遵循 `Standardization/example_tx.py` 的 N3 标准线。

## N1 实验线

N1 使用首尾扫频信号（chirp）、固定训练/帧头和变长负载数据，作为无导频（pilot）、无前向纠错（FEC）的
实验基线。

帧结构如下：

```text
500 ms 前置扫频信号，1k-9k
30 ms 静音保护
8 个训练 OFDM 符号
9 个帧头 OFDM 符号，3 份排列副本，每份 3 个符号
根据文件长度确定数量的负载数据 OFDM 符号
50 ms 帧间间隔
500 ms 后置扫频信号，1k-9k
```

N1 的 OFDM 参数为：

```text
48 kHz，N=8192，CP=2048
有效数据频带 2 kHz 到 7 kHz，子载波索引 342-1194
负载数据调制：bpsk、qpsk 或 qam16
```

接收端会枚举首尾扫频信号对，用已知的 8 个训练符号细化 OFDM 起点，
再在粗略 SFO 附近以 0.25 ppm 精度搜索。N1 帧头使用三份 BPSK 副本，并结合
训练一致性、帧头可靠度和文件 CRC 选择最终候选。

N1 快速开始：

```powershell
pip install -r requirements.txt
python tx_n1.py data/source/file16_test.txt --out data/n1/n1.wav
python rx_n1.py data/n1/n1.wav `
  --source data/source/file16_test.txt `
  --out runs/n1/offline
Compare-Object (Get-Content -Encoding Byte data/source/file16_test.txt) (Get-Content -Encoding Byte runs/n1/offline/file16_test.txt)
```

## 目录说明

- `n3/`：经过黄金向量验证的标准物理层（PHY）核心。
- `n3_1/`：N3.1 标准发射端、接收端和帧头诊断入口。
- `modem_n1.py`、`tx_n1.py`、`rx_n1.py`：N1 WAV、同步、OFDM、帧头、调制和恢复逻辑。
- `modem_n2.py`、`tx_n2.py`、`rx_n2.py`：N2 实验线，保留用于历史录音和对比。
- `data/source/`：待发送的源文件。
- `data/n1/`、`data/n3/`、`data/n3_1/`：生成的发射 WAV 和附属文件。
- `runs/n1/`、`runs/n3/`、`runs/n3_1/`：接收和分析输出。
- `docs/code_guide.md`：命令、协议和输出说明。
- `archive/`：历史实验数据，不应删除。

## 文档

- [文档索引](docs/README.md)
- [代码指南](docs/code_guide.md)
- [N3.1 使用说明](n3_1/README.md)

## 录音工具

使用 `record_audio.py` 可通过麦克风录制用于接收端解码的 WAV 文件。录音格式固定为
48 kHz、双声道、16-bit PCM。

先查看可用录音设备：

```powershell
python record_audio.py --list-devices
```

录制 10 秒音频：

```powershell
python record_audio.py data/rx/receive.wav --seconds 10
```

如果需要指定录音设备，使用设备编号；如果文件已存在，添加 `--force`：

```powershell
python record_audio.py data/rx/receive.wav --seconds 10 --device 2 --force
```

录音完成后，将 WAV 文件交给对应版本的接收器，例如 N3.2：

```powershell
python -m n3_2.rx data/rx/receive.wav --out runs/n3_2/receive
```

## N3.1 标准兼容线

N3.1 以 `Standardization/example_tx.py` 的可执行行为为准，固定使用：

- 48 kHz，`N=8192`，`CP=2048`，有效子载波索引 `400-2391`；
- 仅使用大端位序 QPSK；
- 前后各 8 个训练符号；
- 3 秒、100 Hz-20 kHz 线性扫频信号，前后各 0.5 秒静音；
- 一个 `PH`、版本 0 的 249 字节帧头信息块；
- IEEE 802.16、码率 1/2、`Z=166` 的 LDPC；
- 每个码字重置为 `0x5A4D` 的扰码序列。

N3.1 帧头只发送一次，因为这是标准示例代码的实际行为。N3.1 同时保留
AudioModem 自己的多候选同步、SFO 搜索、信道估计和诊断输出，但不会改变标准线格式。

生成并恢复标准 WAV：

```powershell
python -m n3_1.tx data/source/file16_test.txt --out data/n3_1/file16_standard.wav
python -m n3_1.rx data/n3_1/file16_standard.wav `
  --source data/source/file16_test.txt `
  --out runs/n3_1/offline
```

发射端会写入 WAV、`.training.npy`、`.header.bin` 和 `.meta.json`。接收端会写入恢复
文件以及 `metrics.json`、`summary.csv`、`H.npy`、`training_phase_fit.npy`、
`payload_symbols.npy`、`decoded_header.bin` 和 `decoded_payload.bin`。

N3.1 的 BER 计算与 N2 使用相同口径。运行接收端时加入原始文件参数，程序结束会打印
`ber_header`、`ber_payload`、`ber_overall`、`ber_ldpc_off` 和 `ber_ldpc_on`：

```powershell
python -m n3_1.rx data/n3_1/r1.wav `
  --source data/source/cute.jpg `
  --out runs/n3_1/recorded
```

如果不提供 `--source`，接收文件仍会生成，但无法计算 BER，并会打印
`ber_unavailable`。

N3.1 接收端还会根据训练符号估计整体噪声置信度，并按有效子载波的信道增益
和训练误差调整 LLR。该增强只影响接收端软判决，不改变标准发射格式；本次权重会
记录在 `metrics.json` 的 `llr_scale`、`llr_scale_min` 和 `llr_scale_max` 字段中。它还会
写入 `header_debug.json`，在提供 `--source` 时给出帧头的精确位误码数和误码率。
