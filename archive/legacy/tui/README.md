# AudioModem TUI

这是一个完全位于 `tui/` 下的独立实验界面。它不导入或修改项目根目录的 Step8
实现，新生成的数据只写入 `tui/data/` 和 `tui/run/`。

This is an independent Textual experiment frontend. It keeps the root Step8
pipeline untouched and stores all new artifacts below `tui/`.

## 安装与启动 / Install and launch

从项目根目录运行：

```bash
.venv/bin/pip install -r tui/requirements.txt
.venv/bin/python -m tui
```

快捷键：`Ctrl+R` 刷新，`Ctrl+C` 取消当前任务，`q` 退出。界面路径输入允许引用
项目内外已有源文件或录音，但不会修改这些输入。

## 页面 / Screens

- **总览**：读取 `tui/run/*/manifest.json`，显示最近任务和核心指标。
- **编码**：选择源文件与 profile，生成 WAV 和 sidecar。
- **录放**：独立录音、独立播放，或运行编码到解码的完整串联流程。
- **解码**：从 TX meta 自动载入空中参数，调整 slope 与 H 刷新后恢复文件。
- **结果**：区分完全一致、仅 CRC 通过、部分恢复和解码失败。
- **Profiles / 任务**：复制和编辑 profile，查看结构化任务日志并取消任务。

## 数据目录 / Data layout

```text
tui/data/tx/<experiment-id>/    generated WAV and sidecars
tui/data/rx/<experiment-id>/    PipeWire recordings
tui/run/<experiment-id>/        profile snapshot, manifest, metrics and recovered files
tui/profiles/                    reusable named profiles
```

实验 ID 为 `YYYYMMDD_HHMMSS_mmm_label`。每个 manifest 保存 profile 快照、命令历史、
设备 target、录音指标、任务阶段、BER/CRC/PPM 和产物路径。输出默认不覆盖。

## Profiles 与协议 / Profiles and protocols

### Step8 Compatible

不可变兼容 profile。默认发送样本与根目录 Step8 逐样本一致，使用：

```text
48 kHz, N=512, CP=256
bins 64-120,158-178
BPSK, pilot spacing 4
K=7 rate-1/2 convolutional FEC
AM7F header and current Step8 anchors/seeds
```

兼容录音应选择这个 profile。接收端 slope、H alpha 和搜索参数可以调整，这些参数不改变
空中 profile ID。

### TUI v1

独立的新协议，支持：

- FFT `256..4096` 的二次幂、`1 <= CP <= N`；
- 以 bin 为主的多个 active ranges，并实时显示对应 Hz；
- BPSK 或 QPSK 软 LLR；
- `none`、卷积码、RS(255,223)、RS+卷积码；
- rotating comb pilots、payload-start anchor 和周期 timing anchors。

串联 FEC 顺序为：

```text
file block + CRC -> RS outer code -> convolutional inner code -> interleaver
```

接收端执行逆序。TUI v1 使用 `AMT1` header 和 profile tag，选择错误 profile 时会明确
报告不匹配。旧 Step8 与 TUI v1 WAV 不应交叉解码。

## PipeWire

点击“刷新设备”调用 `wpctl status`。系统默认路由始终可用，也可以选择明确的 source/sink
target。录音固定 mono、s16、48 kHz，并实时计算短窗 Peak/RMS 与累计削波率。

完整串联流程使用手填的总录音时长和播放前等待时间。总时长不足以覆盖 WAV 时拒绝运行；
播放后余量低于 0.25 秒时给出警告。取消任务时 TUI 会终止整个子进程组并关闭 WAV。

## Worker CLI

界面通过 JSON Lines worker 运行长任务，也可直接诊断：

```bash
.venv/bin/python -m tui.worker encode SOURCE \
  --profile tui/profiles/tui_v1_default.json --out /tmp/tx.wav

.venv/bin/python -m tui.worker decode RECORDING \
  --profile tui/profiles/tui_v1_default.json --source SOURCE --out /tmp/decode

.venv/bin/python -m tui.worker devices
```

## 验证 / Verification

```bash
python -m py_compile tui/*.py
.venv/bin/python -m unittest discover -s tui/tests -v
```

当前测试覆盖 profile 校验、四种 FEC、RS 错误修复、PipeWire 设备解析、模拟 PCM
录音统计、Textual 六页 smoke test，以及 QPSK+RS+卷积码离线闭环。
