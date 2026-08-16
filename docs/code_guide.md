# N1 Code Guide / N1 代码指南

N1 uses a single-file frame with two chirps and no payload pilots:

```text
150 ms front chirp, 1k-9k
30 ms silence guard
8 training OFDM symbols
9 header OFDM symbols, 3 repeated copies x 3 symbols
variable payload OFDM symbols
50 ms inter-frame gap
150 ms tail chirp, 1k-9k
```

The tail chirp lets the receiver estimate sampling drift for one-shot file
transfer. Payload length is determined by input file size.

## 1. Modules / 模块

### `modem_n1.py`

Self-contained N1 modem implementation:

- mono 16-bit 48 kHz WAV I/O;
- OFDM with `N=4096`, `CP=2048`, symbol length `6144`;
- active data band from 2 kHz to 7 kHz, bins `171..597`;
- 150 ms linear chirp sync from 1 kHz to 9 kHz;
- 8 known QPSK training OFDM symbols for channel estimation;
- fixed 9-symbol BPSK header: 3 repeated 64-byte header copies with bit-majority vote;
- variable BPSK/QPSK/QAM16 payload with no pilot and no FEC;
- chirp-to-chirp SFO estimate and open-loop linear phase correction.

### `tx_n1.py`

Reads one file and writes a transmit WAV plus deterministic sidecars:

```text
*.training.npy
*.header.npy
*.meta.json
```

### `rx_n1.py`

Finds front/tail chirps, estimates sampling drift, estimates `H` from training,
decodes header, then decodes the exact number of payload OFDM symbols from the
header.

Main outputs:

```text
metrics.json
summary.csv
H.npy
training_phase_fit.npy
payload_symbols.npy
decoded_header.bin
recovered file, when CRC passes
decoded_payload.bin, when decode fails
```

## 2. Generate / 生成发送音频

Default small text file:

```bash
python tx_n1.py
```

Any file:

```bash
python tx_n1.py data/source/file15.txt --out data/n1/file15.wav
```

Important transmitter options:

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `input` | `data/source/file16_test.txt` | source file / 源文件 |
| `--mod` | `qpsk` | payload modulation: `bpsk`, `qpsk`, or `qam16` |
| `--training-seed` | `3026` | deterministic training symbols |
| `--out` | `data/n1/n1.wav` | transmit WAV path |

## 3. Decode / 解码

Offline loopback:

```bash
python rx_n1.py data/n1/n1.wav \
  --source data/source/file16_test.txt \
  --out runs/n1/offline
```

Recorded WAV:

```bash
python rx_n1.py data/rx/receive.wav \
  --source data/source/file16_test.txt \
  --out runs/n1/recording
```

Important receiver options:

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `--source` | none | optional truth file for exact match reporting |
| `--training-seed` | `3026` | must match transmitter |
| `--tail-search-seconds` | `0.5` | search radius around expected tail chirp |
| `--out` | `runs/n1` | output directory |

## 4. Checks / 检查

Compile-check:

```bash
python -m py_compile modem_n1.py tx_n1.py rx_n1.py
```

Offline file loopback:

```bash
python tx_n1.py data/source/file16_test.txt --out data/n1/n1.wav
python rx_n1.py data/n1/n1.wav \
  --source data/source/file16_test.txt \
  --out runs/n1/offline
cmp data/source/file16_test.txt runs/n1/offline/file16_test.txt
```

Expected metadata constants:

```text
chirp_samples = 7200
guard_samples = 1440
training_symbols = 8
header_symbols = 9
header_copies = 3
header_copy_symbols = 3
header_size = 64
inter_frame_gap_samples = 2400
```

Success is judged by:

```text
front_chirp_score -> tail_chirp_score -> header_ok -> file_crc_ok -> file_match
```
