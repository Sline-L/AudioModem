# 技术路线简要总结

本项目的目标是用 48 kHz 声音信号完成真实声学信道下的 OFDM 文件传输。整体路线分为三步：先估计信道 `H`，再利用 `H` 恢复真实 payload，最后通过分段扫频调整实际可用频段。

## 1. 信道估计

先分别发射 `ones` 和 `random` 两类 probe，并在扬声器和麦克风位置固定的情况下多次录音。

- `ones`：所有目标频点同时发射 `1+0j`，便于直观看每个频点的幅度和相位响应。
- `random`：每个频点发射随机 QPSK 符号，更接近真实数据传输状态，适合做稳健的平均估计。

每次录音后用 `analyze.py` 得到一次 `H.npy`。对多次测试得到的 `H` 做复数平均，得到更稳定的平均信道响应：

```text
H_avg[k] = mean(H_1[k], H_2[k], ..., H_n[k])
```

后续文件解调统一使用这个 `H_avg`，而不是只依赖单次录音结果。

## 2. 文件传输与误码率测试

保持扬声器、麦克风、音量和录音增益不变，选择一个文本或图片文件进行编码发送。

发送端流程：

```text
source file -> payload bits -> BPSK/QPSK/16-QAM -> OFDM -> speaker
```

接收端流程：

```text
microphone recording -> sync -> FFT 得到 Y -> X_hat = Y / H_avg -> 判决 -> recovered file
```

恢复后把接收文件和源文件逐 bit 或逐 byte 比较，得到误码率：

```text
BER = error_bits / total_bits
```

同时保存接收频域符号 `Y`、均衡后的符号 `X_hat` 和恢复文件，方便对比不同调制方式、不同频段下的稳定性。

## 3. 频段选择

最后发射 `bandstep` probe 做分段扫频。`bandstep` 每次只激励一小段连续频率，然后跳到下一段，因此适合观察哪些频段在真实扬声器和麦克风链路中更稳定。

重点比较三组频段：

- `8-150`：保守范围，通常更稳，速率较低。
- `8-200`：折中范围，适合优先尝试。
- `8-420`：频带最宽，速率最高，但更容易受高频衰减和设备响应影响。

根据 `bandstep` 的 `Y_spectrum.png`、`H.png` 和 BER 结果，选择实际可用的 bins 范围。最终目标是在稳定 BER 和传输速率之间取得平衡。

## 推荐实验顺序

```bash
python probe.py --kind ones --bins 8 200 --symbols 256 --out data/tx/probe_ones_8_200.wav
python analyze.py data/rx/receive_ones.wav --kind ones --bins 8 200 --symbols 256 --out runs/probe_ones_8_200

python probe.py --kind random --bins 8 200 --symbols 256 --out data/tx/probe_random_8_200.wav
python analyze.py data/rx/receive_random.wav --kind random --bins 8 200 --symbols 256 --out runs/probe_random_8_200

python tx.py data/source/file16_test.txt --mod qpsk --bins 8 200 --h runs/H_avg.npy --out data/tx/file16_8_200.wav
python rx.py data/rx/receive_file16.wav --mod qpsk --bins 8 200 --h runs/H_avg.npy --out runs/recovered_file16

python probe.py --kind bandstep --bins 8 420 --symbols 512 --out data/tx/probe_bandstep_8_420.wav
python analyze.py data/rx/receive_bandstep.wav --kind bandstep --bins 8 420 --symbols 512 --out runs/probe_bandstep_8_420
```

