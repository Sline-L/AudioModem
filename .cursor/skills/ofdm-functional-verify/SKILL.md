---
name: ofdm-functional-verify
description: >-
  Runs and interprets OFDM n1 functional verification: TX-to-RX loopback,
  header CRC, and mild impairment tests via n1/selftest.py. Use when validating
  receiver recovery, WAV interoperability, or OFDM selftest results.
---

# OFDM Functional Verify

## When to use

After n1 code is implemented. Execute or interpret `n1/selftest.py` and CLI recovery.

## Required steps

1. Ensure dependencies from `n1/requirements.txt` are installed (`numpy`, `scipy`, `ldpc`; `sounddevice` optional for play mode).
2. Run from repo root:
   ```bash
   python n1/selftest.py
   ```
3. Confirm tests cover:
   - **Ideal loopback**: TX saves WAV → RX recovers → payload bytes identical.
   - **Header**: CRC OK; filename and `file_size` match.
   - **Mild impairment**: AWGN and/or small delay; recover or report clear stage failure.
4. Optionally:
   ```bash
   python n1/run_rx.py --wav <path> --out-dir n1/recovered/
   ```

## Pass criteria

| Case | Pass |
|------|------|
| Ideal loopback | Byte-identical payload |
| Header | CRC pass + correct name/size |
| Impairment | Recovered OR explicit fail stage in `{sync, CE, LDPC, header}` |

## Failure triage

Map failures to stage:

- No chirp peaks / bad body start → **sync**
- Preamble correlation weak / absurd H → **CE**
- CRC fail / garbage after decode → **LDPC** or bit-order bug
- CRC ok but wrong length/name → **header**

## Output format

```markdown
## Functional verify verdict: PASS | FAIL

### Commands run
- ...

### Results
- Ideal: PASS/FAIL — detail
- Header: PASS/FAIL — detail
- Impairment: PASS/FAIL — detail

### Failed stage (if any)
- sync | CE | LDPC | header | other

### Logs / hashes
- ...
```
