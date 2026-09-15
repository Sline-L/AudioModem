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

## 6. N3.1 standard-compatible implementation

N3.1 is isolated under `n3_1/` and follows the executable reference in
`Standardization/example_tx.py`. It uses mono 48 kHz PCM, FFT `8192`, CP
`2048`, active bins `400..2391`, QPSK only, 8 leading plus 8 trailing training
symbols generated with seed `80`, identical 3-second linear chirps from 100 Hz
to 20 kHz, 0.5-second silence guards, one 249-byte `PH` version-0 header block,
IEEE 802.16 rate-1/2 LDPC with `Z=166`, and the per-codeword `0x5A4D`
scrambler. Header copies, `AMN2`, 802.11n LDPC, and N2's extra interleaver are
not part of N3.1.

Generate a standard frame and recover it offline:

```powershell
python -m n3_1.tx data/source/file16_test.txt --out data/n3_1/file16_standard.wav
python -m n3_1.rx data/n3_1/file16_standard.wav `
  --source data/source/file16_test.txt `
  --out runs/n3_1/offline
```

`n3_1.tx` writes the WAV plus deterministic training, header, and metadata
sidecars. `n3_1.rx` writes the recovered file and diagnostics (`metrics.json`,
`summary.csv`, `H.npy`, `training_phase_fit.npy`, `payload_symbols.npy`,
`decoded_header.bin`, and `decoded_payload.bin`). It retains N2's candidate
chirp pairing, training timing refinement, SFO search, channel estimation, and
BER reporting without changing the N3.1 wire format. N3.1 also writes
`header_debug.json`; with `--source`, it records the decoded Header's exact
bit-error count and rate against the expected 249-byte Header. SFO trials use a coarse
1-ppm pass followed by a 0.25-ppm local refinement; a candidate's FFT blocks
are cached so only phase correction is repeated during the search.

For a controlled payload-error experiment, `python -m n3_1.ber_test` creates a
standard frame and flips a deterministic fraction of post-LDPC QPSK bits while
leaving Header and training unchanged. The generated `.ber.json` records the
requested and actual injected BER. This measures hard, high-confidence bit
flips and is intentionally distinct from a noisy acoustic recording with soft
LLR reliability.
# N3.2 独立标准版本

`n3_2/` 是独立的标准帧实现，不依赖 N3/N3_1。使用 `python -m n3_2.tx INPUT --out WAV` 发射，使用 `python -m n3_2.rx WAV --out DIR [--source INPUT]` 恢复；接收目录含恢复文件和同步、Header、信道诊断。
