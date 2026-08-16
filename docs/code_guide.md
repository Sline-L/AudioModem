# N1 Code Guide / N1 代码指南

N1 uses a single-file frame with two chirps and no payload pilots:

```text
150 ms front chirp, 1k-9k
30 ms silence guard
8 training OFDM symbols
9 header OFDM symbols, 3 permuted copies x 3 symbols
variable payload OFDM symbols
50 ms inter-frame gap
150 ms tail chirp, 1k-9k
```

The tail chirp lets the receiver estimate sampling drift for one-shot file
transfer. The receiver first estimates payload OFDM symbol count from the
front-to-tail chirp distance, then decodes the header with SFO correction.

## 1. Modules / 模块

### `modem_n1.py`

Self-contained N1 modem implementation:

- mono 16-bit 48 kHz WAV I/O;
- OFDM with `N=4096`, `CP=2048`, symbol length `6144`;
- active data band from 2 kHz to 7 kHz, bins `171..597`;
- 150 ms linear chirp sync from 1 kHz to 9 kHz;
- 8 known QPSK training OFDM symbols for channel estimation;
- fixed 9-symbol BPSK header: 3 permuted 64-byte header copies with bit-majority vote;
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

Finds front/tail chirps, estimates payload symbol count and sampling drift,
estimates `H` from training, decodes the permuted-copy header, then decodes the
exact number of payload OFDM symbols from the header.

Receiver data flow:

```text
front chirp -> tail chirp -> estimate payload_symbols/SFO -> training H -> permuted header vote -> payload
```

Header copy mapping:

```text
copy 1 identity
copy 2 permutation A
copy 3 permutation B
```

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

When `--source` is provided, `metrics.json` also includes BER diagnostics even
if header CRC or file CRC fails:

```text
ber_header_vote_ber
ber_payload_ber
ber_overall_useful_ber
ber_payload_byte_error_rate
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
| `--tail-search-seconds` | `0.5` | fallback search radius around expected tail chirp |
| `--out` | `runs/n1` | output directory |

## 4. Record / 录音

Install the recording dependency with `requirements.txt`, then record a WAV at
the fixed format of 48 kHz, mono, signed 16-bit PCM:

```bash
python record_audio.py data/n1/rec_n1_4.wav --sample-count 400000
```

Set the actual recording duration directly in seconds:

```bash
python record_audio.py data/n1/rec_n1_4.wav --seconds 10
```

Use `--list-devices` to list audio devices, `--device ID` to select an input,
and `--force` to overwrite an existing WAV.

## 5. Checks / 检查

Compile-check:

```bash
python -m py_compile modem_n1.py tx_n1.py rx_n1.py record_audio.py
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
header_permutation_used = true
inter_frame_gap_samples = 2400
```

Success is judged by:

```text
front_chirp_score -> tail_chirp_score -> payload_symbols_guess -> sfo_ppm -> header_ok -> file_crc_ok -> file_match -> ber_payload_ber
```
