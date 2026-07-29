# 代码功能与用法详解

这份文档说明当前新结构下每个脚本的作用。主流程固定为：采样率 `48000 Hz`，FFT 点数 `1024`，循环前缀 `128`。

## 1. `audiomodem.py`

这是唯一的核心库，其他脚本都从这里导入函数。它不直接运行实验，只提供基础能力。

### 关键常量

- `FS = 48000`：WAV 采样率。
- `N = 1024`：OFDM FFT 点数。
- `CP = 128`：循环前缀长度。
- `L = N + CP = 1152`：一个 OFDM 符号的总采样点数。
- `I16 = 32768.0`：16-bit PCM 归一化尺度。

### 主要函数

- `bins(a=8, b=420)`  
  生成可用子载波编号，例如 `bins(8, 150)` 得到 `8..150`。范围必须满足 `1 <= a <= b <= 511`，因为实数 OFDM 只使用正频率一半。

- `read_wav(p)`  
  读取单声道、16-bit、48 kHz WAV，并归一化到浮点数组，大约在 `[-1, 1]`。

- `write_wav(p, x)`  
  把浮点音频写成单声道 16-bit 48 kHz WAV。写入前会自动缩放到峰值约 `0.95`，避免削波。

- `wav_gain(x)`  
  返回 `write_wav()` 使用的缩放系数。`analyze.py` 用它保证理论 probe 和实际写出的 WAV 尺度一致。

- `qpsk(data)`  
  把 bytes 转成 QPSK 星座点。QPSK 有 4 个星座点，所以每个复数符号承载 2 bit，不是 4 bit 或 16 bit。

- `bytes_from_qpsk(z)`  
  把均衡后的 QPSK 星座点按象限判决，恢复成 bytes。

- `mod_symbols(data, mod="qpsk")` / `bytes_from_mod(z, mod="qpsk")`  
  通用调制和判决函数，支持：
  - `bpsk`：2 个星座点，1 bit/符号，最稳但速率最低。
  - `qpsk`：4 个星座点，2 bit/符号，当前默认。
  - `qam16`：16 个星座点，4 bit/符号，速率更高但对噪声和 H 估计更敏感。

- `pack(path)` / `unpack(data)`  
  文件打包格式是：

  ```text
  filename\0size\0payload
  ```

  `rx.py` 恢复时依靠这个头部知道输出文件名和 payload 长度。

- `ofdm_tx(s, k)`  
  把频域符号 `s` 放进子载波 `k`，构造共轭对称频谱，IFFT 得到实数音频，再加入 CP。

- `ofdm_rx(x, k)`  
  把接收音频按 `1152` 点切成 OFDM 符号，去掉 `128` 点 CP，FFT 后取出子载波 `k`。

- `preamble_symbols(k, n=32, seed=2026)` / `preamble_wave(k, n=32, seed=2026)`  
  生成文件传输用的物理层同步头。同步头是固定随机 QPSK OFDM 符号，默认 32 个 OFDM 符号，时长约 `0.768 s`。它不是 `filename\0size\0payload` 文件头。

- `find_sync(rx, preamble)`  
  用互相关在录音中寻找同步头位置，返回 `sync_start` 和 `sync_score`。`rx.py` 用它自动跳过录音前面的静音和设备延迟。

- `file_symbols(path, k)`  
  文件发送专用：读文件、打包头部、按 `--mod` 映射，并补零成完整 OFDM 符号矩阵。

- `probe_symbols(kind, k, n, seed=2026)`  
  生成训练频域符号，支持：
  - `ones`：所有目标频点始终为 `1+0j`，整段时间同时激励全部 bins。
  - `chirp`：所有目标频点幅度为 1，相位随时间和频点平滑变化，像连续线性扫频。
  - `step`：所有目标频点幅度为 1，但相位按时间分成 16 档跳变，像阶梯状全频扫频。
  - `bandstep`：每个时间段只激励一段连续 bins，下一时间段跳到下一段；这是部分区间、阶梯状扫频。
  - `random`：随机 QPSK，主要用于调试和对比。

## 2. `tx.py`

`tx.py` 负责把一个源文件调制成可播放的发送 WAV。

### 数据流

```text
[随机QPSK同步头] + [文件 -> filename\0size\0payload -> BPSK/QPSK/16-QAM -> OFDM] -> WAV
```

如果传入 `--h H.npy`，发送端会先把频域符号乘以 `H`。一般真实硬件测试不需要这么做；默认就是理想发送信道。

默认会在 payload 前加入同步头。同步头用于真实录音起点同步，和文件头是两回事：

```text
[physical preamble][payload bytes header + payload data]
```

### 用法

```bash
python tx.py data/source/file16_test.txt --bins 8 150 --out data/tx/file16_8_150.wav
```

切换调制方式：

```bash
python tx.py data/source/file16_test.txt --mod bpsk --bins 8 150 --out data/tx/file16_bpsk.wav
python tx.py data/source/file16_test.txt --mod qpsk --bins 8 150 --out data/tx/file16_qpsk.wav
python tx.py data/source/file16_test.txt --mod qam16 --bins 8 150 --out data/tx/file16_qam16.wav
```

### 参数

- `input`：要发送的源文件。默认是 `data/source/file16_test.txt`。
- `--bins START END`：使用的子载波范围。推荐测试 `8 420`、`8 200`、`8 150`。
- `--mod bpsk|qpsk|qam16`：文件调制方式，默认 `qpsk`。
- `--sync-symbols N`：同步头 OFDM 符号数，默认 `32`。
- `--sync-seed N`：同步头随机种子，默认 `2026`。
- `--no-sync`：关闭同步头，输出旧式纯 payload WAV。
- `--h path/to/H.npy`：可选，发送端预乘一个频域信道。
- `--out path.wav`：输出 WAV 路径。

### 输出

- 一个可播放的 WAV，例如 `data/tx/file16_8_150.wav`。
- 命令行会打印 OFDM 符号数和使用的 bins。

## 3. `rx.py`

`rx.py` 负责从接收 WAV 中恢复文件。

### 数据流

```text
WAV -> 互相关找同步头 -> 去 CP -> FFT -> 除以 H -> BPSK/QPSK/16-QAM 判决 -> bytes -> 文件
```

如果没有传 `--h`，默认使用理想信道 `H=1`，适合离线闭环测试。真实录音要传 `analyze.py` 得到的 `H.npy`。

### 用法

离线测试：

```bash
python rx.py data/tx/file16_8_150.wav --bins 8 150 --out runs/recovered
```

如果发送端用了 `--mod qam16`，接收端必须同样传 `--mod qam16`：

```bash
python rx.py data/tx/file16_qam16.wav --mod qam16 --bins 8 150 --out runs/recovered_qam16
```

真实信道测试：

```bash
python rx.py data/rx/receive.wav --bins 8 150 --h runs/probe_ones_8_150/H.npy --out runs/recovered_real
```

### 参数

- `input`：接收 WAV。默认是 `data/tx/file.wav`。
- `--bins START END`：必须和发送、估计 H 时一致。
- `--mod bpsk|qpsk|qam16`：必须和 `tx.py` 一致。
- `--sync-symbols N`：必须和发送端一致。
- `--sync-seed N`：必须和发送端一致。
- `--no-sync`：接收旧式纯 payload WAV，不做互相关同步。
- `--h path/to/H.npy`：频域信道响应，长度必须等于 bins 数量。
- `--out dir`：恢复文件输出目录。

### 输出

- 恢复出的原文件。
- `rx_symbols.npy`：均衡后的 QPSK 星座点，后续可以用它计算 BER 或画星座图。
- 命令行会打印 `sync_start`、`sync_score`、`payload_start`。

### 注意

默认同步只处理起点偏移，不处理采样率漂移。真实录音如果 `sync_score` 很低，优先检查播放音量、录音音量、bins 和 `--sync-seed` 是否一致。

## 4. `probe.py`

`probe.py` 负责生成信道估计训练 WAV。

### 数据流

```text
训练频域符号 X -> OFDM -> probe.wav
```

同时保存一份 `.symbols.npy`，记录理论频域训练符号。

### 用法

```bash
python probe.py --kind ones --bins 8 150 --symbols 256 --out data/tx/probe_ones_8_150.wav
python probe.py --kind chirp --bins 8 200 --symbols 256 --out data/tx/probe_chirp_8_200.wav
python probe.py --kind step --bins 8 420 --symbols 256 --out data/tx/probe_step_8_420.wav
python probe.py --kind bandstep --bins 8 150 --symbols 256 --out data/tx/probe_bandstep_8_150.wav
```

### 参数

- `--kind ones|chirp|step|bandstep|random`：训练信号类型。
- `--bins START END`：训练覆盖的子载波范围。
- `--symbols N`：训练 OFDM 符号数。默认 `256`，时长约 `256 * 1152 / 48000 = 6.144 s`。
- `--seed N`：`random` 训练使用的随机种子。
- `--out path.wav`：输出 probe WAV。

### 输出

- `probe.wav`：播放用训练音频。
- `probe.symbols.npy`：理论频域训练符号矩阵，形状是 `(symbols, bins_count)`。

## 5. `analyze.py`

`analyze.py` 负责从录到的 probe 中同步训练段，计算真实信道响应 `H = Y / X`，并输出图和数据。

### 数据流

```text
重新生成理论 probe -> 与录音互相关同步 -> 得到接收频域 Y -> H = Y / X -> 保存数据和图
```

### 用法

录到 `data/rx/receive.wav` 后：

```bash
python analyze.py data/rx/receive.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_ones_8_150
```

离线自检可以把发送 probe 当作接收输入：

```bash
python analyze.py data/tx/probe_ones_8_150.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_loopback
```

理想情况下会看到：

```text
sync_start=0
sync_score=1.000000
mean_abs_h≈1
```

### 参数

- `receive`：录音 WAV。
- `--kind`：必须和 `probe.py` 生成时一致。
- `--bins START END`：必须和 `probe.py` 一致。
- `--symbols N`：必须和 `probe.py` 一致。
- `--seed N`：如果是 `random` probe，必须和 `probe.py` 一致。
- `--out dir`：输出目录。

### 输出

输出目录中会有：

- `Y.npy`：接收频域符号，形状是 `(symbols, bins_count)`。
- `Y_theory.npy`：理论发送频域符号 `X`，形状同上。
- `H.npy`：估计出的频域信道响应，长度是 `bins_count`。
- `summary.csv`：每个 bin 的频率、`|Y|`、理论 `|Y|`、`H` 实部/虚部/幅度/相位。
- `Y_spectrum.png`：接收 `Y` 与理论 `Y` 的频谱幅度对比。
- `H.png`：估计信道幅度图。

命令行指标：

- `sync_start`：在录音中找到的 probe 起点采样点。
- `sync_score`：同步相关分数，越接近 1 越好。真实录音明显低于 1 是正常的，但太低说明录音弱、噪声大或参数不匹配。
- `mean_abs_h`：平均信道幅度。它会包含扬声器音量、麦克风增益和空气信道。

## 推荐实验顺序

1. 离线文件闭环：

   ```bash
   python tx.py data/source/file16_test.txt --bins 8 150 --out data/tx/file16_8_150.wav
   python rx.py data/tx/file16_8_150.wav --bins 8 150 --out runs/recovered_test
   cmp data/source/file16_test.txt runs/recovered_test/file16_test.txt
   ```

2. 离线 H 闭环：

   ```bash
   python probe.py --kind ones --bins 8 150 --symbols 256 --out data/tx/probe_ones_8_150.wav
   python analyze.py data/tx/probe_ones_8_150.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_loopback
   ```

3. 真实信道估计：

   播放 `data/tx/probe_ones_8_150.wav`，录成 `data/rx/receive.wav`，然后运行：

   ```bash
   python analyze.py data/rx/receive.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_ones_8_150
   ```

4. 真实文件传输：

   ```bash
   python tx.py data/source/file16_test.txt --bins 8 150 --out data/tx/file16_8_150.wav
   python rx.py data/rx/receive_file16.wav --bins 8 150 --h runs/probe_ones_8_150/H.npy --out runs/recovered_real
   ```

5. 批量组合测试：

   依次测试：

   ```text
   kind: ones, chirp, step, bandstep
   bins: 8-420, 8-200, 8-150
   ```

   每次保持 `probe.py` 和 `analyze.py` 的 `kind/bins/symbols/seed` 完全一致。
