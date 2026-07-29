## structure

f04
1. f01
2. t01: find payload files
3. t02: read channel and calculate frequency response
4. f02
    1. build filename\0size\0payload
    2. t03: bytes to QPSK
    3. pad into OFDM symbols
    4. multiply by channel response
    5. build conjugate-symmetric FFT frames
    6. IFFT
    7. add CP
5. t04: write WAV
6. f03

## note

Source payload files live in `source/` by default. This is the inverse order of
the demodulator. It creates mono 16-bit 48000 Hz WAV files that can be decoded
by `Week2_Challenge_Demodulator/main.py`.
