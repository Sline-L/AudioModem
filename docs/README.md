# AudioModem Documentation / AudioModem 文档

The current branch is `simple-main`: a minimal OFDM file-transfer baseline with
only sync, preamble and data.

当前分支是 `simple-main`：只保留同步头、preamble 和 data 的最小 OFDM 文件传输基线。

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
step8_modem.py
tx_step8.py
rx_step8.py
data/source/
data/simple_main/
runs/simple_main/
archive/legacy/
```
