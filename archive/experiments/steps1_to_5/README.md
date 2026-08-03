# Steps 1-5 Legacy Pipeline / Step1-5 历史流程

This snapshot contains the original N=1024, CP=128 modem, probe analysis,
band/single-bin sweeps, and combo file-transfer experiments.

本快照包含原始 N=1024、CP=128 modem、probe 分析、分段/逐点扫频以及 combo 文件传输。

Run commands from the repository root, for example:

```bash
python archive/experiments/steps1_to_5/probe.py --kind ones --bins 8 150 \
  --symbols 256 --out data/tx/probe.wav
python archive/experiments/steps1_to_5/analyze.py data/rx/receive.wav \
  --kind ones --bins 8 150 --symbols 256 --out runs/legacy_probe
```

The detailed chronology is in `docs/experiment_history.md`.

完整实验演进记录见 `docs/experiment_history.md`。
