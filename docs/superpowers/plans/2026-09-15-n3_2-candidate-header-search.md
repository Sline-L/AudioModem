# N3.2 recorded-frame candidate search

1. Preserve the standalone N3.2 modem and its standard frame format.
2. Generate a small ranked set of chirp/training/SFO frame candidates instead of
   committing to one low-confidence synchronizer result.
3. Decode the Header for each candidate and select only a CRC-valid Header.
4. Use the selected candidate for the existing tail-training payload decoder.
5. Regress against the supplied `r4.wav` and `r5.wav` recordings as well as `r1.wav`.
