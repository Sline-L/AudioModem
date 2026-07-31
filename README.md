# AudioModem

面向真实声学信道测试的 OFDM 音频 modem。新主流程固定使用 48 kHz、1024 点 FFT、128 点 CP，payload 支持 BPSK、QPSK 和 16-QAM。

英文说明见 [docs/README.md](docs/README.md)。
代码详解见 [docs/code_guide.md](docs/code_guide.md)。
技术路线见 [docs/technical_route.md](docs/technical_route.md)。

## 结构

- `audiomodem.py`：核心函数，含 WAV、OFDM、调制、同步头和 probe 生成。
- `tx.py`：把 `data/source/` 里的文件调制成发送 WAV。
- `rx.py`：用估计出的 `H.npy` 或理想信道恢复文件。
- `probe.py`：生成 `ones`、`chirp`、`step`、`bandstep`、`random` 训练信号。
- `analyze.py`：从录音估计 `H`，输出 `Y`、理论 `Y`、频谱图和汇总。
- `data/`：`source/` 原文件，`tx/` 发送 WAV，`rx/` 录音。
- `runs/`：实验输出。
- `archive/week2_challenge/`：旧 Week2 challenge 代码和结果归档。

## 常用命令

```bash
pip install -r requirements.txt
python tx.py data/source/file16_test.txt --bins 8 150 --out data/tx/file16.wav
python rx.py data/tx/file16.wav --bins 8 150 --out runs/recovered
pw-record --rate 48000 --channels 1 --format s16 --sample-count 【336000】 data/rx/receive.wav
(choose) "wpctl status" and then add --target bluez_input.0C:9A:E6:EE:D3:3D
```

`tx.py` 默认在 payload 前加入 32 个 OFDM 符号的随机 QPSK 同步头；`rx.py` 默认先互相关同步，再解 payload。旧格式可加 `--no-sync`。

真实录音恢复示例：

```bash
python rx.py data/rx/receive_file16.wav --bins 8 150 --h runs/probe_ones_8_150/H.npy --out runs/recovered_real
```

切换调制方式：

```bash
python tx.py data/source/file16_test.txt --mod qam16 --bins 8 150 --out data/tx/file16_qam16.wav
python rx.py data/tx/file16_qam16.wav --mod qam16 --bins 8 150 --out runs/recovered_qam16
```

生成 probe 并分析录音：

```bash
python probe.py --kind ones --bins 8 150 --symbols 256 --out data/tx/probe.wav
python analyze.py data/rx/receive.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_ones_8_150
```

后续重点测试 probe：`ones`、`chirp`、`step`、`bandstep`；三组频段：`8 420`、`8 200`、`8 150`。
