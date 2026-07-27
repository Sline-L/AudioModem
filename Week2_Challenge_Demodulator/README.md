## structure

f04
1. f01
2. t01: find files
3. f02
    1. t03: read .wav
    2. split samples, remove CP
    3. FFT
    4. devided by channel
    5. t04: QPSK
    6. t06: parse header
    7. return result
4. f03

## note

there is something wrong with the "fmt chunk size" of "recovered/file14.wav"
it should be "10 00 00 00"(16) instead of "10 00 00 02"(33554448)