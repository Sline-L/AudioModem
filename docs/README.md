# AudioModem

AudioModem is organized around real acoustic channel experiments. The main pipeline uses 48 kHz audio, a 1024-point FFT, a 128-sample cyclic prefix, and a random-QPSK sync preamble for file transfers.

Detailed Chinese code guide: [code_guide.md](code_guide.md).

Experiment history: [experiment_history.md](experiment_history.md).

Step7 frame design, soft FEC, timing recovery, TIFF experiment, and the current
sample-clock problem: [step7_adaptive_fec.md](step7_adaptive_fec.md).

## Layout

- `audiomodem.py`: compact shared WAV, OFDM, modulation, sync, and probe helpers.
- `tx.py`: creates OFDM transmit WAV files with a default sync preamble.
- `rx.py`: recovers files with either ideal channel bins or an estimated `H.npy`.
- `probe.py`: creates `ones`, `chirp`, `step`, `bandstep`, or `random` probe WAV files.
- `analyze.py`: estimates `H`, saves received `Y`, theoretical `Y`, CSV summaries, and plots.
- `data/`: source files, transmit WAVs, and recorded receive WAVs.
- `runs/`: experiment outputs.
- `archive/week2_challenge/`: old Week 2 challenge code and generated results.
- `step7_modem.py`, `tx_step7.py`, `rx_step7.py`: independent `N=512, CP=256`
  rotating-pilot and soft-FEC experiment pipeline.

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
