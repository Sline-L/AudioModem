# N1/N2 Code Guide / N1/N2 代码指南

N2 keeps the N1 frame and adds optional payload LDPC. LDPC coded bits are
interleaved by default before modulation. N1 remains available as the no-FEC
baseline.

```text
500 ms front chirp, 1k-9k
30 ms silence guard
12 front training OFDM symbols
9 header OFDM symbols, 3 permuted copies x 3 symbols
variable payload OFDM symbols, uncoded or LDPC-coded/interleaved
optional 12 tail training OFDM symbols, enabled by --tail-training
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
- OFDM with `N=4096`, `CP=2048`, symbol length `6144`;
- active data band from 2 kHz to 7 kHz, bins `171..597`;
- 500 ms linear chirp sync from 1 kHz to 9 kHz;
- 12 known QPSK front training OFDM symbols for channel estimation;
- optional 12-symbol tail training block after payload when `--tail-training` is enabled;
- with `--tail-training`, RX assumes time-invariant `H` and averages all 24 front+tail training symbols for one global channel estimate;
- fixed 9-symbol BPSK header: 3 permuted 64-byte header copies with bit-majority vote;
- variable BPSK/QPSK/QAM16 payload with no pilot;
- N2 optional payload LDPC with deterministic coded-bit interleaving via `--ldpc`;
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

### `modem_n2.py`, `tx_n2.py`, `rx_n2.py`

N2 is the current LDPC experiment line. It copies the N1 synchronization,
training, header voting, SFO search and optional tail-training behavior. With
`--ldpc`, payload coded bits are interleaved before modulation and received LLRs
are deinterleaved before LDPC decoding:

```bash
python tx_n2.py data/source/file16_test.txt --ldpc --out data/n2/n2_ldpc_interleaved.wav
python rx_n2.py data/n2/n2_ldpc_interleaved.wav \
  --ldpc \
  --source data/source/file16_test.txt \
  --out runs/n2/ldpc_interleaved
```

LDPC is controlled by global variables in `modem_n2.py`:

```text
LDPC_ENABLED_DEFAULT = False
LDPC_STANDARD = "802.11n"
LDPC_RATE = "1/2"
LDPC_Z = 27
LDPC_PTYPE = "A"
LDPC_DECODER = "sumprod2"
LDPC_CORR_FACTOR = 0.7
LDPC_LLR_SCALE = 1.0
LDPC_INTERLEAVER_ENABLED = True
LDPC_INTERLEAVER_SEED = 20260816
```

N2 header magic is `AMN2`. Header flags bit 0 records whether payload LDPC is
enabled. TX/RX must use matching LDPC and interleaver globals; the header does
not carry the full LDPC/interleaver configuration. Old non-interleaved N2 LDPC
recordings require `LDPC_INTERLEAVER_ENABLED=False` to decode correctly.

When `--source` is provided, `metrics.json` also includes BER diagnostics even
if header CRC or file CRC fails:

```text
ber_header_vote_ber
ber_payload_ber
ber_overall_useful_ber
ber_payload_byte_error_rate
ber_ldpc_off
ber_ldpc_off_interleaved
ber_ldpc_off_deinterleaved
ber_ldpc_on
ber_payload_raw_coded_ber
ldpc_decode_iterations_mean
ldpc_decode_iterations_max
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

Enable the front+tail training layout:

```bash
python tx_n1.py data/source/file16_test.txt \
  --tail-training \
  --out data/n1/n1_tail_training.wav
```

Important transmitter options:

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `input` | `data/source/file16_test.txt` | source file / 源文件 |
| `--mod` | `qpsk` | payload modulation: `bpsk`, `qpsk`, or `qam16` |
| `--training-seed` | `3026` | deterministic training symbols |
| `--tail-training` | off | insert the same training block after payload |
| `--ldpc` | off | N2 only: LDPC-encode payload bits |
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

Decode a frame generated with tail training:

```bash
python rx_n1.py data/n1/n1_tail_training.wav \
  --tail-training \
  --source data/source/file16_test.txt \
  --out runs/n1/tail_training
```

Important receiver options:

| Option | Default | Meaning / 含义 |
|---|---:|---|
| `--source` | none | optional truth file for exact match reporting |
| `--training-seed` | `3026` | must match transmitter |
| `--tail-training` | off | expect the same training block after payload and average front+tail training for `H` |
| `--ldpc` | off | N2 only: LDPC-decode payload bits |
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
python -m py_compile lib/ldpc/py/ldpc.py modem_n2.py tx_n2.py rx_n2.py
```

Offline file loopback:

```bash
python tx_n1.py data/source/file16_test.txt --out data/n1/n1.wav
python rx_n1.py data/n1/n1.wav \
  --source data/source/file16_test.txt \
  --out runs/n1/offline
cmp data/source/file16_test.txt runs/n1/offline/file16_test.txt
```

N2 LDPC interleaved loopback:

```bash
python tx_n2.py data/source/file16_test.txt --ldpc --out data/n2/n2_ldpc_interleaved.wav
python rx_n2.py data/n2/n2_ldpc_interleaved.wav \
  --ldpc \
  --source data/source/file16_test.txt \
  --out runs/n2/ldpc_interleaved
cmp data/source/file16_test.txt runs/n2/ldpc_interleaved/file16_test.txt
```

Expected metadata constants:

```text
chirp_samples = 24000
guard_samples = 1440
training_symbols = 12
tail_training_symbols = 0, or 12 with --tail-training
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
