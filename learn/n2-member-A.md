# 成员 A：一个字节的旅程

在 `learn/n2-course.html` 第 1 章之上加深。交付物是一张对照表：字节怎样变成子载波，又怎样变回来。B、C 的数组形状以这张表为准。

阅读顺序：`n2/params.py` → `n2/phy.py` → `standard/example_tx.py` → `LDPC/new_ldpc/README.md` 前 100 行 → `n2/selftest.py`。

实验：`python learn/experiments/a1_compare_tx.py`，`python learn/experiments/a2_ldpc_roundtrip.py`。

---

## 1. 发射端逐段

`Ofdm.__init__` 把默认参数写成属性，并 `assert chunk_size % 6 == 0`（498 可以被 6 整除，这样 `z=chunk_size/3=166` 是整数，同时每字节 4 个 QPSK 也能整除）。然后 `ldpc.code(standard="802.16", rate="1/2", z=166)`。`n2/params.py` 的 `TX_DEFAULTS` 与这些默认值对齐，`derived()` 只做推导。

### log_chirp_gen → phy.linear_chirp

输入无，输出长度 `int(48000*3)=144000` 的 float。相位是二次的，注释掉的对数扫频不要用。差一个符号，B 的匹配峰就消失。幅度 `0.8`。OFDM 体在 `tx()` 里另外 `0.8 * x / max(x)`，扫频不参与这次归一化。

### train_symbol_gen → phy.train_freq_time

`random.seed(80)` 之后每个数据 bin `random.randint(0, 3)`：

```python
I = 1 - 2*(rand_num & 1)
Q = 1 - 2*((rand_num >> 1) & 1)
```

返回频域 `(16, 8192)` 和带 CP 的时域 `(16, 10240)`。前 8 个放数据前，后 8 个放数据后。B 做相关用时域，C 做信道估计用频域的 bin 400–2391。换成 NumPy RNG，序列立刻不同，`a1_compare_tx.py` 会把两组前 8 个数打印出来。

### sequence → phy.lfsr_sequence

15 位，种子 `0x5A4D`。输出 bit14，反馈 bit14 XOR bit13，左移，新的最低位放反馈，状态限制在 15 位（`& 0x7FFF`）。每个码字单独从种子开始。发射端在 LDPC 之后异或；接收端在 LLR 上乘 `(1-2*scram)`。

### header_gen → phy.parse_ph_header

249 字节，`bytearray` 初值全 0。

| 字节 | 内容 |
|---|---|
| 0:2 | `b'PH'` |
| 2 | 版本 `0x00` |
| 3:7 | `file_size`，4 字节大端 |
| 7:11 | `declared_symbols`，4 字节大端 |
| 11 起 | UTF-8 文件名 |
| 文件名后 1 字节 | 保持 `0x00`（没有另写） |
| 再 4 字节 | CRC32 大端 |

CRC 是 `zlib.crc32(ba[:11+len(filename)])`，写到下标 `12+len`。覆盖范围含魔术字到文件名最后一字节，不含结束 `0x00`，不含 CRC 自身。`parse_ph_header` 重算，不一致返回 `None`。文件内容里也可能出现 `PH`，所以只有 CRC 通过才算找到头。

`declared_symbols` 来自 `ceil(2*file_size/498)`，只按文件长度估算，不含 249 字节帧头，也不能当数据符号数 `K`。

### file_process

读文件，拼 `header_bytes + file_byte`。头里的 symbol_count 就是上面的 `declared_symbols`。

### ldpc_encoding

先把「头 + 文件」补零到 498 的整数倍，再按 249 字节切块。每块 249 字节拆成 MSB 先的比特，`coder.encode`，再与 `sequence(3984)` 异或，按 MSB 打回 498 字节。块数 `n_chunks = 2 * padded_len / 498`，所以数据符号数 `K` 一定是偶数。末尾可以多出一个全零信息块。

这里差一个比特：扰码种子错、MSB 顺序反了、或者换了别的 802.16 库，头 CRC 会失败（阶段 `header`）或载荷对不上（阶段 `ldpc` / `file_match=false`）。

### mapp → phy.bytes_to_qpsk_grid

见预备知识第 3 节。一个 498 字节块 → 长度 8192 的复频谱，含共轭镜像。

### time_domain_convert 与 tx

IFFT 取实部，CP 接到前面，拼成一条时域。训练 8 + 数据 K + 训练 8，整段峰值归一化到 0.8。最后拼接：

```text
chirp + 0.5 s 静音 + OFDM 体 + 0.5 s 静音 + chirp
```

`save_sound` 把 float 乘 32767 写成 int16。`phy._wav_to_float` 对 int16 除以 32767 读回来。

---

## 2. selftest 在比什么

`n2/selftest.py` 的 `test_phy_against_tx`：

- `linear_chirp` 对 `log_chirp_gen`
- `lfsr_sequence(64)` 对 `sequence(64)`
- `train_freq_time` 的频域和带 CP 时域，对 `train_symbol_gen(..., symbol_count=16)`
- `header_gen("测.bin", 12, 1, 249)` 能被 `parse_ph_header` 解出同样的文件名和长度

`test_ldpc_noiseless` 用 `y = 10*(0.5-x)` 这条无噪声 LLR，信息比特应一次译回。`test_loopback` 用发射端写 WAV，再 `recover_wav`，要求字节相同。

`a1_compare_tx.py` 把差值打印出来，并演示：改文件名里的 1 个字节，解析变 `None`；改 CRC 之后的补零，解析仍然成功。

---

## 3. 字节账

记 \(S\) 为 `file_size`。

\[
\mathrm{declared}=\left\lceil\frac{2S}{498}\right\rceil=\left\lceil\frac{S}{249}\right\rceil
\]

\[
K=2\left\lceil\frac{249+S}{498}\right\rceil
\]

当前 `recover_array` 里第 1 个数据符号在搜头时译完，载荷循环从第 2 个开始，所以 `blocks_total = K-1`。`declared_symbols` 只用来在 `_try_header` 里核对 `(file_size+248)//249`，循环上界用 `K`。

| file_size | declared | 未编码字节 | K | 载荷码字 K−1 | 备注 |
|---|---|---|---|---|---|
| 1000 | 5 | 1249 | 6 | 5 | 课程练习 |
| 9829 | 40 | 10078 | 42 | 41 | r1，`cute.jpg` |
| 249 | 1 | 498 | 2 | 1 | 249+249 刚好一块 498 |
| 498 | 2 | 747 | 4 | 3 | 补零到 996，多 249 个 0 |
| 747 | 3 | 996 | 4 | 3 | 刚好两块 498，没有额外补零 |
| 9693 | 39 | 9942 | 40 | 39 | r14，`picture_1.jpg` |
| 34685 | 140 | 34934 | 142 | 141 | r5 |
| 38583 | 155 | 38832 | 156 | 155 | r7 |
| 44833 | 181 | 45082 | 182 | 181 | r3 的文件长度 |

r1、r5、r7、r14 的 `metrics.json` 里 `K` 与上表一致，`blocks_total = K-1`。

r3 的 `blocks_total=182`，等于 `K` 而不是 `K-1`。当前代码是 `payload_specs = data[1:]`（`receiver.py` 里取数据符号时丢掉第 0 个）。这份结果的统计口径和现在的代码不一致，学习时以当前代码和 r1 为准。r3 仍然是一份有用的对照：`clock_error_ppm≈0`，`median_abs_llr≈12`，Chirp SNR 在百万级，像数字回环而不是声学录音。

末尾全零码字：`249+S` 不是 498 的整数倍时，`ljust` 补出的 0 会单独占一个或两个信息块。写出文件时 `extract_payload` 只取 `file_size` 字节，这些 0 留在信息流里，不进恢复文件。

---

## 4. LDPC 库

`LDPC/new_ldpc/py/ldpc.py` 的 `code(standard, rate, z)`。本仓库固定 `802.16`、`1/2`、`z=166`。802.16e 文档里的 `z` 通常不超过 96，这里故意用 166，移位按 `proto % z`，换一个「标准 802.16」库会解不开。

`encode(u)` 输入长度 `c.K=1992`。`decode(y, dectype="sumprod2")` 返回 `app` 和 `it`。`app` 与 LLR 同号约定：负值判 1。`it` 最大 200，提前满足校验就停。`sumprod2` 是对数域、数值更稳的和积算法。

`decode_path.LdpcBank.decode_codeword` 先 `descramble_llr`，再译码，返回信息比特 `hard[:K]`、整码字 `hard[:N]`、迭代次数。

`a2_ldpc_roundtrip.py` 用五个噪声档走一遍。`sigma=0.05` 时应 0 误码且 `it<200`。

---

## 5. 给 B 和 C 的接口

**给 B**

| 函数 | 形状 | 用途 |
|---|---|---|
| `train_freq_time` 的时域 | `(16, 10240)` | 前 2 个拼起来做锁帧相关；后 8 个拼起来找尾训练 |
| `train_freq_time` 的频域 | `(16, 8192)` | B 的 `best_sfo_near` 要用前 8 行的有效 bin |
| `linear_chirp` | `(144000,)`，幅度 0.8 | Chirp 匹配模板 |

**给 C**

| 函数 | 形状 / 约定 |
|---|---|
| `known = freq[:, active_bins]` | `(16, 1992)`，前 8 行对前训练，后 8 行对后训练 |
| `qpsk_llr_scaled` | 输入 `(1992,)` 复数，输出 `(3984,)`，偶数下标是 Q，奇数下标是 I，裁剪 ±30 |
| `LdpcBank.decode_codeword` | 输入加扰后的 3984 LLR；输出 1992 信息比特、3984 码字比特、迭代次数 |
| `bits_to_bytes` | `np.packbits(..., bitorder="big")`，bit7 在前 |

---

## 6. 15 分钟讲解提纲

1. 498、249、1992、3984 的换算，以及 `K` 为什么是偶数。
2. 字节 `0x80` 的 8 个比特落到 4 个 QPSK：先 Q 后 I。
3. 训练必须 `random.seed(80)`，给 B、C 看 `a1` 里两组随机数。
4. r1 的 40、42、41 各是什么；r3 的 `blocks_total` 为什么不要拿来当当前代码的例子。

## 7. 自测与答案

1. `file_size=1000` 的 declared、K、载荷码字数？  
   5、6、5。

2. 字节 `0x80` 的第一个符号？  
   `1-1j`。其余三个是 `1+1j`。

3. 20 字节文件名时，CRC 覆盖的最后一字节下标、`0x00` 下标、CRC 起点？  
   文件名在 `[11, 31)`，CRC 覆盖到下标 30，`0x00` 在 31，CRC 在 `[32, 36)`。

4. 为什么 `declared_symbols` 不能当循环上界？  
   它不含 249 字节帧头。r1 上它是 40，真正的数据符号是 42。

5. 扰码在发射端和接收端各作用在什么量上？  
   发射端异或在 LDPC 之后的比特上；接收端乘在 LLR 上，每个码字复位。

6. `app<0` 表示什么？  
   比特判为 1。

7. 换 NumPy 生成训练序列，先坏哪一步？  
   B 的训练相关峰和 C 的 `Y/X` 同时错。头 CRC 不会过。

8. r14 的 `file_size=9693`，K 是多少？  
   `2*ceil(9942/498)=2*20=40`，与 metrics 一致。

9. 末尾补零会不会写进恢复文件？  
   不会。`extract_payload` 只取 `file_size` 字节。

10. `test_phy_against_tx` 没有比较哪一项？  
    没有比较 `mapp` / `bytes_to_qpsk_grid`。QPSK 比特顺序要自己对着 `mapp` 和 `bytes_to_qpsk_grid` 看，两段的 `I + 1j*Q` 公式相同。
