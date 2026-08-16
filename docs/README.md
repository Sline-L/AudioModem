# AudioModem Documentation / AudioModem 文档

The current n1 version is a minimal OFDM file-transfer baseline with
only sync, preamble and data.

当前 `n1` 版本只保留同步头、preamble 和 data，是最小 OFDM 文件传输基线。

## Current Documents / 当前文档

- [Code guide / 代码指南](code_guide.md): current CLI parameters, commands,
  generated files and offline loopback checks.

## Archived Documents / 归档文档

Old Step7/Step8 protocol notes, experiment history, technical route notes, TUI
notes and structure drafts were moved to:

```text
archive/legacy/docs/
```

旧 Step7/Step8 协议说明、实验历史、技术路线、TUI 说明和结构草稿已移动到
`archive/legacy/docs/`。

## Code Layout / 代码结构

```text
modem_n1.py
tx_n1.py
rx_n1.py
data/source/
data/n1/
runs/n1/
archive/legacy/
```
