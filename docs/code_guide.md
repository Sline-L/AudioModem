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
python probe.py --kind bandstep --bins 1 511 --symbols 4096 --bandstep-parts 64 --out data/tx/probe_bandstep_full_64parts_long.wav
```

### 参数

- `--kind ones|chirp|step|bandstep|random`：训练信号类型。
- `--bins START END`：训练覆盖的子载波范围。
- `--symbols N`：训练 OFDM 符号数。默认 `256`，时长约 `256 * 1152 / 48000 = 6.144 s`。
- `--seed N`：`random` 训练使用的随机种子。
- `--bandstep-parts N`：`bandstep` 划分的连续频带数，默认 `16`；发送和分析时必须一致。
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
python analyze.py data/step3_bandstep/receive_bandstep_1.wav --kind bandstep --bins 1 511 --symbols 4096 --bandstep-parts 64 --out runs/step3_bandstep/1
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
- `--bandstep-parts N`：如果是 `bandstep` probe，必须和 `probe.py` 一致。
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

## 6. `tx_combo.py`

`tx_combo.py` 负责生成“一次播放”的合并 WAV，把信道训练 probe 和文件发送放在同一个音频里。

### 数据流

```text
[probe训练段][文件同步头][文件payload] -> WAV
```

默认用于 `exp2.txt` 的 `8..150` 子载波实验：

```bash
python tx_combo.py
```

这会生成：

- `data/step2_file/exp2_combo_8_150.wav`：你需要播放的唯一音频。
- `data/step2_file/exp2_combo_8_150.probe.npy`：写出 WAV 后带缩放系数的理论 probe 符号。
- `data/step2_file/exp2_combo_8_150.meta.json`：记录 probe、同步头、payload 的长度和起点。

默认结构是：

```text
256 个 probe OFDM symbols
32 个 file preamble OFDM symbols
124 个 exp2.txt payload OFDM symbols
```

当前 `exp2.txt` 的 `8_150` 合并音频时长约 `9.888 s`。

### 参数

- `input`：要发送的源文件，默认 `data/step2_file/exp2.txt`。
- `--bins START END`：默认 `8 150`。
- `--fband-profile conservative|trimmed`：使用 bandstep 和真实 payload 实测后挑出的非连续频段。`conservative` 为第一版 `128-160,164-240,315-400`；`trimmed` 会删掉真实 payload 中错误集中的 `170-171` 和 `358-400`，保留 `128-160,164-169,172-240,315-357`。每个 profile 都使用每个 OFDM symbol 内的 comb pilot，没有列入 active 的 bins 全部置零。
- `--mod bpsk|qpsk|qam16`：默认 `qpsk`。
- `--probe-kind ones|chirp|step|bandstep|random`：默认 `ones`。
- `--probe-symbols N`：默认 `256`。
- `--probe-seed N`：默认 `2026`。
- `--sync-symbols N`：文件同步头长度，默认 `32`。
- `--sync-seed N`：文件同步头随机种子，默认 `2026`。
- `--pilot-interval N`：每隔多少个 data OFDM symbols 插入 1 个整符号 pilot。默认 `0`，表示不插 pilot。
- `--pilot-len N`：每次插入连续多少个 pilot OFDM symbols，默认 `1`。多个 pilot 会在接收端平均估计 H。
- `--pilot-kind ones|chirp|step|bandstep|random`：pilot 类型，默认 `random`。
- `--pilot-seed N`：pilot 随机种子，默认 `2027`。
- `--payload-repeats N`：同一个 framed payload 在一段音频中重复几遍，默认 `1`。接收端会对多遍 bit 做多数投票。
- `--out path.wav`：输出 WAV 路径。

### BPSK + pilot 推荐实验

当前真实声学链路优先测试更稳的 BPSK、128 个 file preamble symbols、每 4 个 data symbols 插入 1 个 pilot：

```bash
python tx_combo.py data/step2_file/exp2.txt --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --out data/step2_file/exp2_combo_bpsk_pilot4_8_150.wav
```

当前 `exp2.txt` 会生成：

```text
256 个 probe OFDM symbols
128 个 file preamble OFDM symbols
248 个 BPSK data OFDM symbols
62 个 pilot OFDM symbols
```

音频时长约 `16.656 s`。真实录音建议录 `19 s` 或更长，给播放前后各留约 1 秒余量。

更稳的单段音频实验使用 BPSK、128 个 file preamble symbols、每 4 个 data symbols 插入 2 个 pilot，并把 payload 重复 3 遍：

```bash
python tx_combo.py data/step2_file/exp2.txt --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --pilot-len 2 --payload-repeats 3 --out data/step2_file/exp2_combo_bpsk_pilot2_repeat3_8_150.wav
```

当前 `exp2.txt` 会生成：

```text
256 个 probe OFDM symbols
128 个 file preamble OFDM symbols
248 个 BPSK data OFDM symbols
124 个 pilot OFDM symbols
单次 framed payload 372 个 OFDM symbols
payload 重复 3 遍
```

音频时长约 `36.000 s`。真实录音建议录 `39-40 s`。

### Step 4: fband + comb pilot 实验

bandstep 和第一轮真实 payload 实测后，优先测试删掉坏频点的非连续频段和每符号 comb pilot：

```bash
python tx_combo.py data/step4_fband_optimization/exp2.txt --fband-profile trimmed --mod bpsk --sync-symbols 128 --payload-repeats 3 --out data/step4_fband_optimization/exp2_combo_fband_trimmed_bpsk_repeat3.wav
```

当前 `exp2.txt` 会生成：

```text
active bins: 128-160, 164-169, 172-240, 315-357
data bins: 127 个
comb pilot bins: 24 个，每个 payload OFDM symbol 都发送
256 个 probe OFDM symbols
128 个 file preamble OFDM symbols
279 个 payload OFDM symbols x 3 repeats
```

音频时长约 `29.304 s`。真实录音建议录 `32-33 s`，保存到：

```text
data/step4_fband_optimization/receive_exp2_fband_trimmed_1.wav
```

## 7. `rx_combo.py`

`rx_combo.py` 负责从一次播放的录音中先估计当前 `H`，再恢复后面的文件。

### 数据流

```text
录音 WAV -> probe 同步 -> H = Y_probe / X_probe
         -> 文件同步头二次同步/估 H -> payload FFT -> pilot 跟踪 H -> 判决 -> unpack -> 文件
```

离线自检：

```bash
python rx_combo.py data/step2_file/exp2_combo_8_150.wav --out runs/step2_file/combo_8_150/offline
```

真实实验时，你只需要播放：

```text
data/step2_file/exp2_combo_8_150.wav
```

然后把录音保存成：

```text
data/step2_file/receive_exp2_combo_8_150_1.wav
```

再运行：

```bash
python rx_combo.py data/step2_file/receive_exp2_combo_8_150_1.wav --out runs/step2_file/combo_8_150/1
```

批量分析 5 组录音：

```bash
python rx_combo.py data/step2_file/receive_exp2_combo_8_150_1.wav data/step2_file/receive_exp2_combo_8_150_2.wav data/step2_file/receive_exp2_combo_8_150_3.wav data/step2_file/receive_exp2_combo_8_150_4.wav data/step2_file/receive_exp2_combo_8_150_5.wav --out runs/step2_file/combo_8_150
```

BPSK + pilot 接收：

```bash
python rx_combo.py data/step2_file/receive_exp2_combo_bpsk_pilot4_8_150_1.wav --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --out runs/step2_file/combo_bpsk_pilot4_8_150/1
```

BPSK + pilot 平均 + payload 重复投票接收：

```bash
python rx_combo.py data/step2_file/receive_exp2_combo_bpsk_pilot2_repeat3_8_150_1.wav --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --pilot-len 2 --payload-repeats 3 --out runs/step2_file/combo_bpsk_pilot2_repeat3_8_150/1
```

Step 4 fband + comb pilot 接收：

```bash
python rx_combo.py data/step4_fband_optimization/receive_exp2_fband_trimmed_1.wav --source data/step4_fband_optimization/exp2.txt --fband-profile trimmed --mod bpsk --sync-symbols 128 --payload-repeats 3 --repeat-combine hard --pilot-smooth 12 --out runs/step4_fband_optimization/fband_trimmed_bpsk_repeat3/1
```

接收端默认会在 probe 后面 `±8192` samples 范围内重新寻找 file preamble。这个范围用于处理 `ones` probe 带来的整 OFDM symbol 起点歧义。

调试真实录音时可以加：

```bash
--file-sync-mode best --pilot-smooth 8 --repeat-combine soft
```

- `--file-sync-mode best`：同时做 probe 附近搜索和全局 file preamble 搜索，自动选同步分数更高的峰；当 `ones` probe 把起点带偏时，这个选项能救回 file sync。
- `--pilot-smooth N`：对连续 pilot 估计出的 `H` 做时间平滑，真实录音中 `N=8` 当前表现比逐个 pilot 直接使用更稳。
- `--repeat-combine soft`：BPSK + payload repeats 时用均衡后实部软合并，而不是每遍硬判决后多数投票。

### 输出

每组输出目录中会有：

- `H.npy` / `H_sync.npy`：file preamble 估计出的复数信道响应。
- `H_probe.npy`：最前面 probe 估计出的复数信道响应。
- `pilot_H.npy`：启用 pilot 时，每个 pilot 更新后的 H。
- `Y_probe.npy` / `Y_probe_theory.npy`：接收和理论 probe 频域符号。
- `Y_sync.npy` / `Y_sync_theory.npy`：接收和理论 file preamble 频域符号。
- `rx_symbols.npy`：均衡后的 payload 星座点。
- `rx_symbols_repeatN.npy`：启用 `--payload-repeats` 时，每一遍 payload 独立均衡后的星座点。
- `decoded_raw.bin`：判决出的原始 byte 流，即使解包失败也会保存。
- `decoded_raw_repeatN.bin`：启用 `--payload-repeats` 时，每一遍 payload 独立判决出的 byte 流。
- `summary.csv`：每个 bin 的 `H` 幅度和相位。
- `metrics.json`：同步分数、payload 起点、pilot residual、每遍 repeat BER、投票后 BER、`mean_abs_h`、是否成功识别文件头、是否和源文件完全一致等指标。
- 恢复出的文件，例如 `exp2.txt`。

批量模式额外输出：

- `batch_summary.csv`：每组录音的同步分数、BER、是否完全恢复等汇总。

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

5. 一次播放 combo 实验：

   ```bash
   python tx_combo.py
   python rx_combo.py data/step2_file/exp2_combo_8_150.wav --out runs/step2_file/combo_8_150/offline
   ```

   离线通过后，播放 `data/step2_file/exp2_combo_8_150.wav`，录成 `data/step2_file/receive_exp2_combo_8_150_1.wav`，再运行：

   ```bash
   python rx_combo.py data/step2_file/receive_exp2_combo_8_150_1.wav --out runs/step2_file/combo_8_150/1
   ```

6. BPSK + pilot 稳定恢复实验：

   ```bash
   python tx_combo.py data/step2_file/exp2.txt --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --out data/step2_file/exp2_combo_bpsk_pilot4_8_150.wav
   python rx_combo.py data/step2_file/exp2_combo_bpsk_pilot4_8_150.wav --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --out runs/step2_file/combo_bpsk_pilot4_8_150/offline
   ```

   离线通过后，播放 `data/step2_file/exp2_combo_bpsk_pilot4_8_150.wav`，录成 `data/step2_file/receive_exp2_combo_bpsk_pilot4_8_150_1.wav`，再运行：

   ```bash
   python rx_combo.py data/step2_file/receive_exp2_combo_bpsk_pilot4_8_150_1.wav --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --out runs/step2_file/combo_bpsk_pilot4_8_150/1
   ```

7. BPSK + pilot 平均 + payload 重复投票实验：

   ```bash
   python tx_combo.py data/step2_file/exp2.txt --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --pilot-len 2 --payload-repeats 3 --out data/step2_file/exp2_combo_bpsk_pilot2_repeat3_8_150.wav
   python rx_combo.py data/step2_file/exp2_combo_bpsk_pilot2_repeat3_8_150.wav --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --pilot-len 2 --payload-repeats 3 --out runs/step2_file/combo_bpsk_pilot2_repeat3_8_150/offline
   ```

   离线通过后，播放 `data/step2_file/exp2_combo_bpsk_pilot2_repeat3_8_150.wav`，录成 `data/step2_file/receive_exp2_combo_bpsk_pilot2_repeat3_8_150_1.wav`，再运行：

   ```bash
   python rx_combo.py data/step2_file/receive_exp2_combo_bpsk_pilot2_repeat3_8_150_1.wav --bins 8 150 --mod bpsk --sync-symbols 128 --pilot-interval 4 --pilot-len 2 --payload-repeats 3 --out runs/step2_file/combo_bpsk_pilot2_repeat3_8_150/1
   ```

8. 批量组合测试：

   依次测试：

   ```text
   kind: ones, chirp, step, bandstep
   bins: 8-420, 8-200, 8-150
   ```

   每次保持 `probe.py` 和 `analyze.py` 的 `kind/bins/symbols/seed` 完全一致。
