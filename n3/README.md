# N3 标准调制解调器

N3 是根据 `Standardization/example_tx.py` 重写的标准兼容实现。现有 N1 和 N2
实现继续保留，用于历史实验和录音对比。

## 固定协议参数

- 48 kHz、单声道、有符号 16-bit PCM
- FFT 8192，CP 2048，OFDM symbol 长度 10240
- 有效正频率 bins 为 400 到 2391
- 仅使用大端位序 QPSK
- 前置和后置各 8 个 training symbols，Python 随机种子为 80
- 前后使用相同的 3 秒线性 chirp，频率范围 100 Hz 到 20 kHz
- OFDM 区域前后各有 0.5 秒静音
- IEEE 802.16 LDPC，码率 1/2，Z=166
- 每个 3984-bit 码字都使用 `0x5A4D` 重新开始的扰码序列

第一个 249 字节信息块是标准示例生成的 `PH`、版本 0 Header。Header 与文件内容
拼接后补齐为 498 字节码字对，再逐个按 249 字节信息块进行编码。Header 只发送一次，
因为 N3 以可执行示例代码的实际行为为准。

## 使用命令

```powershell
python -m n3.tx data/source/file16_test.txt --out data/n3/file16_standard.wav
python -m n3.rx data/n3/file16_standard.wav `
  --source data/source/file16_test.txt `
  --out runs/n3/offline
```

发射端会写入 WAV、`.training.npy`、`.header.bin` 和 `.meta.json`。接收端会写入恢复
文件、`metrics.json`、`summary.csv`、`H.npy`、`training_phase_fit.npy`、
`payload_symbols.npy`、`decoded_header.bin` 和 `decoded_payload.bin`。

提供 `--source` 时，接收端会额外报告 `file_match`、payload BER、字节错误数、SFO、
时序、chirp 分数和 LDPC 迭代次数。只有 Header 和完整 payload 都成功恢复时才返回
退出码 0；非法输入或恢复失败会保留诊断信息并返回非零退出码。

BER 的计算口径与 N2 一致。要在运行结束时打印 BER，必须提供原始文件：

```powershell
python -m n3.rx data/n3/r1.wav `
  --source data/source/cute.jpg `
  --out runs/n3/recorded
```

终端会打印 `ber_header`、`ber_payload`、`ber_overall`、`ber_ldpc_off` 和
`ber_ldpc_on`。不提供 `--source` 时仍会恢复文件，但无法计算 BER，终端会提示
`ber_unavailable`。

接收端的 LDPC 软判决会根据前后 training symbols 的残差调整整体 LLR 置信度，
并根据每个有效子载波的信道增益与 training error 进行逐载波加权。该处理只存在于
接收端，不改变 `example_tx.py` 的发射格式；`metrics.json` 中的 `llr_scale`、
`llr_scale_min` 和 `llr_scale_max` 可用于检查本次录音采用的软判决权重。

SFO 搜索采用先粗搜 1 ppm、再在最优点附近以 0.25 ppm 精搜的两阶段策略；每个候选帧的
FFT 结果只提取一次，搜索点只重新应用相位校正。它只减少接收端重复计算，不改变候选范围、
最终精度或任何标准帧字段。

如需构造已知的 10% Payload 编码比特误码文件，可运行：

```powershell
python -m n3.ber_test data/source/file16_test.txt `
  --ber 0.10 --seed 20260913 `
  --out data/n3/file16_ber10.wav
```

该工具只翻转 LDPC 编码后的 Payload QPSK 比特，不修改 Header 或 training，
并写入同名 `.ber.json` 注入记录。它用于测试 LDPC 对“高置信度硬翻转”的能力；
这种测试与真实声学信道中的软误码不能直接等同。
