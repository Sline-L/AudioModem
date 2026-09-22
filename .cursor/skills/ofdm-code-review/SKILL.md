---
name: ofdm-code-review
description: >-
  Reviews OFDM n1 receiver implementation for bit-exact compatibility with
  standard/example_tx.py: shapes, QPSK bit order, scrambler, training RNG,
  header CRC bounds, and numerical stability. Use when reviewing n1/*.py code
  or after implementing Ofdm_rx.
---

# OFDM Code Review

## When to use

After n1 receiver code exists. Compare implementation to `standard/example_tx.py`.

## Inputs

- `standard/example_tx.py`
- `n1/ofdm_rx.py`, `n1/sync.py`, `n1/equalizer.py`, `n1/demod.py`, `n1/run_rx.py`, `n1/selftest.py`
- `n1/README.md`

## Checklist

1. **Import / params** — Defaults match TX; TX file not edited.
2. **Chirp template** — Same phase as TX `log_chirp_gen` linear formula.
3. **Training regeneration** — `random.seed(80)` + same QPSK packing as `train_symbol_gen`.
4. **QPSK bit order** — Inverse of TX: 4 symbols per byte, MSB nibble first (`b_i = 3..0`).
5. **Scrambler** — Identical LFSR (`sequence`, seed `0x5A4D`); XOR before LDPC decode; reset each codeword.
6. **LDPC** — `ldpc.code(standard="802.16", rate="1/2", z=chunk_size//3)`; info 249 B / coded 498 B.
7. **Header CRC** — zlib.crc32 over bytes through last filename byte (exclude null); big-endian sizes.
8. **K / framing** — Do not use header `symbol_count` as sole loop bound; use preamble gap and/or `file_size`.
9. **EQ path** — MMSE + mirror MRC present; start/end H interpolation across data symbols.
10. **Shapes** — Symbol length `N+CP`; active bins length `4*chunk_size`.
11. **Selftest** — Ideal loopback + header check + at least one impairment case.
12. **Stability** — Guard divide-by-zero in MMSE; handle mono/stereo WAV; int16→float scaling.

## Output format

```markdown
## Code review verdict: PASS | FAIL

### Blocking issues
- file:line — issue

### Non-blocking suggestions
- ...

### Bit-exact compatibility notes
- ...
```

Fail on wrong bit packing, wrong scrambler/training seed, broken CRC bounds, or missing required innovations from the design.
