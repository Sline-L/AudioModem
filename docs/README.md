# AudioModem Documentation / AudioModem 文档

Step8 is the current implementation. Earlier experiments remain reproducible
under `archive/experiments/`, while their original recordings stay in `data/`.

Step8 是当前正式实现。旧实验代码保存在 `archive/experiments/`，原始录音仍保留在
`data/`，历史实验可以继续复现。

## Current Documents / 当前文档

- [Step8 complete protocol / Step8 完整协议](step8_clock_anchor.md): PHY,
  framing, FEC, pilots, timing, H, receiver flow, outputs and measured results.
- [Step8 end-to-end walkthrough / Step8 端到端工程详解](step8_end_to_end_walkthrough.md):
  follows one file through framing, coding, audio transmission, synchronization,
  channel estimation, soft decoding, CRC validation, and file recovery, with the
  implementation mathematics at every stage.
- [Code guide / 代码指南](code_guide.md): current CLI parameters, commands,
  metadata and result files.
- [Technical route / 技术路线](technical_route.md): why the design evolved from
  sweeps to in-frame adaptation and what remains to improve.
- [Experiment history / 实验历史](experiment_history.md): Step1-Step8 evidence
  and representative BER results.
- [Step7 baseline / Step7 基线](step7_adaptive_fec.md): historical diagnosis that
  motivated periodic timing anchors.
- [TUI feasibility / TUI 可行性](tui_feasibility.md): recommended Textual-based
  architecture and staged implementation.

## Code Layout / 代码结构

```text
step8_modem.py      current self-contained modem and protocol implementation
tx_step8.py         transmitter CLI
rx_step8.py         receiver and analysis CLI
data/step8_clock_anchor/
runs/step8_clock_anchor/
archive/experiments/steps1_to_5/
archive/experiments/step6/
archive/experiments/step7/
```

Generated pre-Step8 analysis is stored locally under `archive/runs/` and is not
tracked by Git because it is approximately 5.7 GB.

Step8 以前的分析结果位于本机 `archive/runs/`，总量约 5.7 GB，因此不进入 Git。
