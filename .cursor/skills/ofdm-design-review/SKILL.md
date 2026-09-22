---
name: ofdm-design-review
description: >-
  Reviews OFDM n1 receiver design against standard/example_tx.py for frame
  structure, parameters, header semantics, and verifiable innovations. Use when
  reviewing n1/README.md, OFDM RX design docs, or before implementing the
  acoustic OFDM receiver.
---

# OFDM Design Review

## When to use

Before implementing or changing the n1 receiver. Read this skill and review the design documents against the TX standard.

## Inputs

- `standard/example_tx.py` (source of truth for air interface)
- `n1/README.md` (receiver design)
- Any attached plan describing n1 scope

## Checklist

1. **Parameters** — Confirm `fs=48000`, `N=8192`, `CP=2048`, `start_index=400`, `chunk_size=498`, `preamble_count=8`, chirp 100→20000 Hz / 3 s, silence 0.5 s.
2. **Chirp math** — Design must use TX **linear** chirp phase, not the commented log formula.
3. **Training** — Same bin range, `random.seed(80)`, 8+8 symbols; CPython `random` not NumPy.
4. **Modulation / Hermitian** — QPSK `{±1±j}` and conjugate mirrors as in TX `mapp`.
5. **Coding** — 802.16 LDPC 1/2 `z=166`, LFSR scramble `0x5A4D` per codeword.
6. **Header traps** — Design must NOT trust header `symbol_count` as `K`; must use `file_size` + CRC; tolerate pad-to-498 extra zero codeword.
7. **Full chain** — Sync → timing/CFO → CE/EQ → demod → LDPC → header → file must all be present for n1.
8. **Innovations** — Dual-chirp SFO, start/end CE interpolation, MMSE+MRC, soft LLR LDPC, header semantics fix — each must be testable.
9. **Non-goals** — Must not modify `standard/example_tx.py`.

## Output format

```markdown
## Design review verdict: PASS | FAIL

### Blocking issues
- ...

### Non-blocking suggestions
- ...

### Compatibility risks
- ...
```

Fail if any blocking issue remains (wrong chirp model, trusting `symbol_count` as K, missing pipeline stage, wrong training seed/RNG).
