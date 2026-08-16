# N1 Code Guide / N1 代码指南

N1 uses a single-file frame with two chirps and no payload pilots:

```text
500 ms front chirp, 1k-9k
30 ms silence guard
8 training OFDM symbols
9 header OFDM symbols, 3 permuted copies x 3 symbols
variable payload OFDM symbols
50 ms inter-frame gap
500 ms tail chirp, 1k-9k
```

The two chirps provide coarse frame boundaries, payload length and SFO. The
known training waveform then refines the OFDM timing, while training/header
quality selects the final pair and SFO used for open-loop correction.

## 1. Modules / 模块

### `modem_n1.py`

Self-contained N1 modem implementation:

- mono 16-bit 48 kHz WAV I/O;
- OFDM with `N=8192`, `CP=2048`, symbol length `10240`;
- active data band from 2 kHz to 7 kHz, bins `342..1194`;
- 500 ms linear chirp sync from 1 kHz to 9 kHz;
- 8 known QPSK training OFDM symbols for channel estimation;
- fixed 9-symbol BPSK header: 3 permuted 64-byte header copies with bit-majority vote;
- variable BPSK/QPSK/QAM16 payload with no pilot and no FEC;
- dynamic profile metadata derived from the current `N`, `CP`, and band;
- chirp coarse SFO plus training/header-guided fine SFO search.

### `tx_n1.py`

Reads one file and writes a transmit WAV plus deterministic sidecars:

```text
*.training.npy
*.header.npy
*.meta.json
```

### `rx_n1.py`

Enumerates legal front/tail chirp pairs, estimates payload symbol count and
coarse sampling drift, and correlates the complete known training block within
`+/-CP` of each predicted start. Significant paths are those above both half
the main peak and `median + 6*MAD`; the earliest is selected with a `CP/64`
safety margin. The best eight pairs enter the header/SFO search.

Receiver data flow:

```text
chirp candidates -> legal pairs -> training timing -> top 8 pairs
-> local/global ppm search -> training/header joint score -> payload
```

For an in-range chirp estimate, the local search covers `chirp_ppm +/- 5 ppm`
in `0.25 ppm` steps. An estimate outside `[-80, +80] ppm` first triggers a
5 ppm coarse search over that range, followed by the same local search. Header
CRC wins first; otherwise candidates are ranked by 50% normalized training
residual, 30% header soft error, and 20% copy disagreement.

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

The receiver stores both the raw chirp-derived SFO and the selected SFO from
header search:

```text
chirp_sfo_ppm
selected_sfo_ppm
chirp_front_start
ofdm_frame_start
fft_timing_offset_samples
fft_timing_peak_score
fft_timing_confident
sync_pair_candidates_evaluated
sync_pair_shortlist
local_ppm_scores
global_ppm_scores
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
chirp_samples = 24000
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
