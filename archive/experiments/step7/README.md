# Step7 Baseline / Step7 基线归档

Step7 introduced the N=512/CP=256 profile, rotating comb pilots, soft
convolutional FEC, CRC-protected 512-byte blocks, and training-only clock
estimation. Step8 preserves its wire format and adds periodic timing anchors.

Step7 引入 N=512/CP=256、旋转 comb pilot、软判决卷积码、带 CRC 的 512-byte blocks 和
仅训练段时钟估计。Step8 保持其空中格式并增加周期 timing anchor。

Example / 示例：

```bash
python archive/experiments/step7/tx_step7.py \
  data/step7_adaptive_fec/observatory_64_uncompressed.tiff \
  --out data/step7_adaptive_fec/observatory_64_uncompressed_step7.wav
python archive/experiments/step7/rx_step7.py \
  data/step7_adaptive_fec/observatory_64_uncompressed_step7.wav \
  --source data/step7_adaptive_fec/observatory_64_uncompressed.tiff \
  --out runs/legacy_step7_offline
```

Full analysis / 完整分析：`docs/step7_adaptive_fec.md`.
