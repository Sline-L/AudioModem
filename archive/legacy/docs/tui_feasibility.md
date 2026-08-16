# TUI Feasibility / TUI 可行性

## Verdict / 结论

A local terminal UI is feasible and useful. It can reduce wrong-file playback,
parameter mismatch and output-directory mistakes, but it will not directly
lower BER. DSP must remain in the Step8 implementation.

本地终端界面可行且有价值，可以减少播错文件、参数不一致和输出目录错误，但不会直接降低
BER；DSP 必须继续由 Step8 核心实现。

The recommended framework is [Textual](https://github.com/Textualize/textual).
It provides forms, tables, tabs, command handling and an asynchronous
[Worker API](https://textual.textualize.io/guide/workers/) suitable for decoding
and subprocess logging without freezing the interface.

推荐框架为 Textual。它具备表单、表格、tabs、命令处理和异步 Worker，适合在界面不中断
的情况下运行长时间解码或读取 subprocess 日志。

This document is design-only. The current `requirements.txt` is unchanged.

本文只完成设计，本轮不修改 `requirements.txt`，也不增加 TUI 代码。

## Proposed Architecture / 推荐架构

```text
Textual screens
  -> experiment controller
      -> Step8 generate/decode Python API
      -> PipeWire adapter (phase 2)
  -> structured progress events
  -> metrics.json and artifact browser
```

The TUI must not parse human console strings as its primary interface. Before
implementing it, transmitter and receiver entry points should expose callable
functions returning structured result objects and emitting progress events.
The existing CLIs should call the same functions.

TUI 不应把人类可读终端输出当作主要 API。实施前应把发送和接收流程暴露为返回结构化结果、
可发送进度事件的 Python 函数，现有 CLI 与 TUI 共用这些函数。

## Phase 1: Experiment TUI / 第一阶段：实验 TUI

Recommended screens / 推荐页面：

1. **Dashboard / 总览**: recent recordings, latest file status, BER, CRC blocks,
   clock status and PPM.
2. **Generate / 生成**: source picker, Step8 preset, duration and output preview,
   then WAV generation.
3. **Decode / 解码**: one or multiple recording paths, optional source, output
   path, slope mode and controlled advanced parameters.
4. **Jobs / 任务**: live logs, elapsed time, cancellation and exit status.
5. **Results / 结果**: metrics table, per-block CRC, worst bins, accepted anchors
   and links to generated plots/files.

The default view should expose only the proven profile. Advanced parameters
belong behind a separate panel so an ordinary recording cannot accidentally use
different transmitter and receiver seeds.

默认界面只显示已验证 profile；高级参数放在独立面板，避免普通实验意外让收发 seed 或帧参数
不一致。

### Phase 1 acceptance / 第一阶段验收

- Generate the same deterministic WAV and metadata as `tx_step8.py`.
- Decode offline and real WAVs with the same metrics as `rx_step8.py`.
- Keep the UI responsive while NumPy/SciPy decoding runs in a worker process.
- Show exact/partial/failed as distinct states; never equate “image opens” with
  file integrity.
- Persist an experiment manifest containing source, TX WAV, RX WAV, arguments,
  output directory and timestamps.

## Phase 2: PipeWire Recording / 第二阶段：PipeWire 录放

Linux/Arch is the first supported environment. A small adapter should use:

```text
wpctl status              device discovery
pw-record                 mono s16 48 kHz capture
pw-play or selected player playback
```

Required controls / 必要控制：

- explicit microphone/output device selection;
- calculated recording duration from transmit WAV plus margins;
- record, wait, play once, wait, stop state machine;
- live peak, clipping count and elapsed-time indicators;
- abort that terminates child processes cleanly;
- command and device IDs saved in the experiment manifest.

录放必须采用明确状态机，不能依赖用户在多个终端手工估时。设备 ID、命令、录音长度和削波
统计都应写入实验 manifest。

## Risks / 风险

- PipeWire node names and Bluetooth profiles can change between sessions.
- Terminal plots are suitable for summaries, not detailed constellation or
  clock figures; the TUI should show metrics and open/save existing PNG files.
- Long NumPy work should run in a process, because a thread can still make UI
  cancellation and resource ownership unclear.
- A TUI adds operational repeatability but also another state layer; all
  generated commands and parameters must remain visible and exportable.

- PipeWire node 名称和蓝牙 profile 可能变化。
- 终端适合展示摘要，不适合替代详细星座图和 clock 图；TUI 应展示指标并引用现有 PNG。
- 长时间 NumPy 任务建议使用独立进程，保证取消和资源回收明确。
- TUI 增加可复现性，也增加状态层，因此所有命令和参数必须可见、可导出。

## Recommended Next Step / 推荐下一步

Implement only Phase 1 first. Refactor Step8 into callable generate/decode APIs,
add a machine-readable progress event schema, then build the five Textual
screens. Integrate PipeWire only after CLI/TUI result equivalence is tested.

先实施第一阶段：提取 Step8 可调用 API 和机器可读进度事件，再构建五个 Textual 页面；只有
CLI/TUI 结果一致性验证完成后，才接入 PipeWire。
