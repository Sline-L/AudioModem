# N3.2 独立标准声学调制解调器

N3.2 完全独立，不导入 N3 或 N3_1；发射格式以 `Standardization/example_tx.py` 为准。

发射：

```powershell
python -m n3_2.tx data/source/file16_test.txt --out data/n3_2/file16_test.wav
```

接收：

```powershell
python -m n3_2.rx data/n3_2/file16_test.wav --out runs/n3_2_file16 --source data/source/file16_test.txt
```

接收器支持单声道或双声道 48 kHz/16-bit PCM WAV；双声道会选择同步分数更高的一路。输出目录保存恢复文件和 `metrics.json`、`header_debug.json`、`H.npy` 等诊断。`stage` 可定位为 `wav_format`、`channel`、`chirp`、`training`、`sfo`、`ldpc`、`header_crc`、`header_fields` 或 `ldpc_payload`。

清洁回环 BER：

```powershell
python -m n3_2.ber_test data/source/file16_test.txt --out runs/n3_2_ber
```
