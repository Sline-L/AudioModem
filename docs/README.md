# AudioModem

AudioModem is now organized around real acoustic channel experiments. The main pipeline uses 48 kHz audio, a 1024-point FFT, and a 128-sample cyclic prefix.

Detailed Chinese code guide: [code_guide.md](code_guide.md).

## Layout

- `audiomodem.py`: compact shared WAV, OFDM, QPSK, and probe helpers.
- `tx.py`: creates OFDM/QPSK transmit WAV files.
- `rx.py`: recovers files with either ideal channel bins or an estimated `H.npy`.
- `probe.py`: creates `ones`, `chirp`, `step`, `bandstep`, or `random` probe WAV files.
- `analyze.py`: estimates `H`, saves received `Y`, theoretical `Y`, CSV summaries, and plots.
- `data/`: source files, transmit WAVs, and recorded receive WAVs.
- `runs/`: experiment outputs.
- `archive/week2_challenge/`: old Week 2 challenge code and generated results.

## Quick Run

```bash
pip install -r requirements.txt
python tx.py data/source/file16_test.txt --mod qpsk --bins 8 150 --out data/tx/file16.wav
python rx.py data/tx/file16.wav --mod qpsk --bins 8 150 --out runs/recovered
```

Probe workflow:

```bash
python probe.py --kind ones --bins 8 150 --symbols 256 --out data/tx/probe.wav
python analyze.py data/rx/receive.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_ones_8_150
```

Planned measurement grid: probe kinds `ones`, `chirp`, `step`, `bandstep`; bin ranges `8-420`, `8-200`, and `8-150`.
