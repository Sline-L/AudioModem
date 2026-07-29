# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact Python OFDM audio modem for real acoustic channel experiments. File transfer supports `bpsk`, `qpsk`, and `qam16`.

- `audiomodem.py`: shared core functions for WAV I/O, modulation, OFDM, bins, and probe symbols.
- `tx.py`, `rx.py`, `probe.py`, `analyze.py`: command-line entry points for transmit, recover, probe generation, and channel analysis.
- `data/source/`: input payload files. `data/tx/`: generated transmit WAVs. `data/rx/`: recorded WAVs.
- `runs/`: experiment outputs such as `H.npy`, `Y.npy`, CSV summaries, and plots.
- `docs/`: user-facing documentation. `archive/`: legacy Week 2/Week 4 code and results.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate and recover a file offline:

```bash
python tx.py data/source/file16_test.txt --bins 8 150 --out data/tx/file16_8_150.wav
python rx.py data/tx/file16_8_150.wav --bins 8 150 --out runs/recovered_test
cmp data/source/file16_test.txt runs/recovered_test/file16_test.txt
```

Generate and analyze a probe:

```bash
python probe.py --kind ones --bins 8 150 --symbols 256 --out data/tx/probe_ones_8_150.wav
python analyze.py data/tx/probe_ones_8_150.wav --kind ones --bins 8 150 --symbols 256 --out runs/probe_loopback
```

Compile-check Python files:

```bash
python -m py_compile audiomodem.py tx.py rx.py probe.py analyze.py
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation and concise, direct functions. Keep the main signal path readable, similar to `mainv2.py`: small helpers, short variables for math (`k`, `x`, `y`, `h`), and minimal comments. Prefer `Path` for paths and `numpy` vector operations for DSP logic. Keep constants uppercase in `audiomodem.py` (`FS`, `N`, `CP`, `L`).

## Testing Guidelines

There is no formal test framework yet. Use deterministic command-line checks. At minimum, run `py_compile`, an offline file loopback with `cmp`, and a probe loopback expecting `sync_score=1.000000` and `mean_abs_h` near `1`. Test the planned bin ranges: `8 420`, `8 200`, and `8 150`.

## Commit & Pull Request Guidelines

Existing commits use short, descriptive messages such as `week3 challenge v1.2: change demodulator and readme`. Follow that style: include the experiment area and a clear action. Pull requests should describe the signal-processing change, list commands run, mention generated artifacts, and include plots or summary metrics when `analyze.py` output changes.

## Agent-Specific Instructions

Do not delete archived experiment data unless explicitly requested. Keep generated files under `data/` or `runs/`, and update `docs/code_guide.md` when changing script interfaces.
