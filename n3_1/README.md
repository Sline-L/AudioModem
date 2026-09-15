# N3 1 标准声学调制解调器

`n3_1` 使用经过验证的 N3 物理层，并增加帧头诊断报告。它与
`Standardization/example_tx.py` 保持兼容：48 kHz、FFT 8192、CP 2048、
有效子载波索引 400 至 2391、参考 QPSK、前后各 8 个训练符号、3 秒扫频信号、
IEEE 802.16 码率 1/2 且 Z=166 的 LDPC，以及每个码字均重置的
`0x5A4D` 扰码序列。

运行标准离线回环：

```powershell
python -m n3_1.tx data/source/file16_test.txt --out data/n3_1/file16.wav
python -m n3_1.rx data/n3_1/file16.wav --source data/source/file16_test.txt --out runs/n3_1/offline
```

接收端总会在 `metrics.json` 同级目录写入 `header_debug.json`。提供
`--source` 时，报告会给出已解码帧头相对于发射端应生成帧头的位错误数和位误码率，
从而区分问题发生在帧头解析之前，还是发生在帧头字段本身。
