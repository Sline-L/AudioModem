# AudioModem Archive / AudioModem 归档

This directory preserves code and generated results from experiments before
Step8. The current implementation remains in the repository root.

本目录保存 Step8 以前的实验代码和生成结果。当前正式实现保留在项目根目录。

## Layout / 目录

```text
experiments/steps1_to_5/   original modem, probes, sweeps, combo transfer
experiments/step6/         N512/N1024 and Step4-bin comparison profiles
experiments/step7/         adaptive pilot, FEC, CRC and training-clock baseline
runs/                      local generated analysis results, ignored by Git
week2_challenge/           earlier Week2 challenge snapshot
week4_channel_estimation_legacy/
tmp/                       existing large local backup, ignored by Git
```

`data/step1...step7` has intentionally not moved. Run archived scripts from the
repository root so their historical relative data paths continue to resolve.

`data/step1...step7` 有意保持原位。请从项目根目录运行归档脚本，使旧版相对数据路径继续
有效。

The three experiment snapshots each include the version of `audiomodem.py`
they depended on. This avoids coupling a historical experiment to Step8.

三个实验归档各自保留所依赖的 `audiomodem.py` 快照，避免历史实验反向依赖 Step8。

`archive/runs/` is about 5.7 GB and is excluded by `.gitignore`. It is an
experiment record on this machine, not a Git-distributed artifact.

`archive/runs/` 约 5.7 GB，已由 `.gitignore` 排除。它是本机实验记录，不通过 Git 分发。
