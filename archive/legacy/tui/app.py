from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import traceback

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, RichLog, Select, Static,
    Switch, TabbedContent, TabPane,
)

from .controller import ExperimentController, Job
from .manifest import ROOT, recent_manifests
from .media import discover_devices, wav_seconds
from .profiles import ModemProfile, ensure_builtin_profiles, parse_ranges


def field(label: str, widget) -> Horizontal:
    return Horizontal(Label(label, classes="field-label"), widget, classes="field")


class AudioModemTUI(App):
    TITLE = "AudioModem 实验台"
    SUB_TITLE = "独立 TUI v1"
    CSS = """
    Screen { background: #101419; color: #e7ebef; }
    Header { background: #17324d; }
    TabPane { padding: 1 2; }
    .section { margin: 0 0 1 0; padding: 1; border: solid #38546d; height: auto; }
    .field { height: 3; }
    .field-label { width: 24; padding: 1 1 0 0; color: #a9bfd0; }
    Input, Select { width: 1fr; }
    Switch { margin: 0 1; }
    Button { margin: 1 1 0 0; }
    .metric { width: 1fr; padding: 1; border: solid #38546d; min-height: 4; }
    #global-status { dock: bottom; height: 3; padding: 1 2; background: #18222c; }
    #job-log { height: 18; border: solid #38546d; }
    DataTable { height: 18; }
    .title { text-style: bold; color: #7fc8ff; margin-bottom: 1; }
    .warning { color: #ffcc66; }
    """
    BINDINGS = [("ctrl+r", "refresh", "刷新"), ("ctrl+c", "cancel_job", "取消任务"), ("q", "quit", "退出")]

    def __init__(self) -> None:
        super().__init__()
        ensure_builtin_profiles()
        self.controller = ExperimentController()
        self.current_job: Job | None = None
        self.latest_metrics: dict = {}

    def profile_options(self) -> list[tuple[str, str]]:
        options = []
        for path in sorted((ROOT / "profiles").glob("*.json")):
            try:
                profile = ModemProfile.load(path)
                options.append((profile.name, str(path)))
            except Exception:
                continue
        return options

    def compose(self) -> ComposeResult:
        profiles = self.profile_options()
        default_profile = str(ROOT / "profiles" / "step8_compatible.json")
        device_options = [("系统默认", "default")]
        yield Header()
        with TabbedContent(initial="dashboard"):
            with TabPane("总览", id="dashboard"):
                yield Static("最近实验", classes="title")
                yield DataTable(id="recent-table", cursor_type="row")
                yield Button("刷新", id="refresh-dashboard", variant="primary")
            with TabPane("编码", id="encode"):
                with VerticalScroll():
                    with Container(classes="section"):
                        yield Static("源文件编码为发送 WAV", classes="title")
                        yield field("源文件", Input(placeholder="/path/to/file", id="encode-source"))
                        yield field("Profile", Select(profiles, value=default_profile, allow_blank=False, id="encode-profile"))
                        yield field("输出文件名", Input(value="transmit.wav", id="encode-name"))
                        yield Static("选择文件后将显示预计时长", id="encode-preview")
                        yield Button("生成 WAV", id="encode-start", variant="success")
            with TabPane("录放", id="media"):
                with VerticalScroll():
                    with Container(classes="section"):
                        yield Static("独立录音", classes="title")
                        yield field("录音秒数", Input(value="75", type="number", id="record-seconds"))
                        yield field("录音文件名", Input(value="recording.wav", id="record-name"))
                        yield field("输入设备", Select(device_options, value="default", allow_blank=False, id="record-target"))
                        yield field("录音流音量", Input(value="1.0", type="number", id="record-volume"))
                        yield Static("Peak --  RMS --  Clipping --", id="meter")
                        yield Button("开始录音", id="record-start", variant="error")
                        yield Button("停止当前任务", id="media-stop")
                    with Container(classes="section"):
                        yield Static("独立播放", classes="title")
                        yield field("发送 WAV", Input(placeholder="/path/to/transmit.wav", id="play-wav"))
                        yield field("输出设备", Select(device_options, value="default", allow_blank=False, id="play-target"))
                        yield field("播放流音量", Input(value="1.0", type="number", id="play-volume"))
                        yield Button("播放一次", id="play-start", variant="primary")
                        yield Button("刷新设备", id="devices-refresh")
                    with Container(classes="section"):
                        yield Static("编码 → 录音 → 播放 → 解码", classes="title")
                        yield field("源文件", Input(placeholder="/path/to/file", id="pipe-source"))
                        yield field("Profile", Select(profiles, value=default_profile, allow_blank=False, id="pipe-profile"))
                        yield field("TX 文件名", Input(value="transmit.wav", id="pipe-tx-name"))
                        yield field("RX 文件名", Input(value="recording.wav", id="pipe-rx-name"))
                        yield field("总录音秒数", Input(value="75", type="number", id="pipe-duration"))
                        yield field("播放前等待秒数", Input(value="1", type="number", id="pipe-preroll"))
                        yield Static("TUI 会校验总时长能覆盖发送 WAV", id="pipe-preview")
                        yield Button("运行完整实验", id="pipeline-start", variant="success")
            with TabPane("解码", id="decode"):
                with VerticalScroll():
                    with Container(classes="section"):
                        yield Static("录音恢复文件", classes="title")
                        yield field("录音 WAV", Input(placeholder="/path/to/recording.wav", id="decode-recording"))
                        yield field("TX meta.json", Input(placeholder="可选；优先自动加载参数", id="decode-meta"))
                        yield field("Profile", Select(profiles, value=default_profile, allow_blank=False, id="decode-profile"))
                        yield field("源文件", Input(placeholder="可选；仅用于 BER/file_match", id="decode-source"))
                        yield field("源文件比较", Switch(value=True, id="decode-compare"))
                        yield field("Phase slope", Select([("关闭", "off"), ("慢速", "slow")], value="off", allow_blank=False, id="decode-slope"))
                        yield field("起始 Anchor H 刷新", Switch(value=True, id="decode-start-h"))
                        yield field("周期 Anchor H 刷新", Switch(value=True, id="decode-anchor-h"))
                        yield Button("开始解码", id="decode-start", variant="success")
                    with Container(classes="section"):
                        yield Static("高级接收参数", classes="title")
                        yield field("Channel alpha", Input(value="0.35", type="number", id="rx-channel-alpha"))
                        yield field("Start H alpha", Input(value="0.5", type="number", id="rx-start-alpha"))
                        yield field("Anchor H alpha", Input(value="0.5", type="number", id="rx-anchor-alpha"))
                        yield field("Anchor min score", Input(value="0.12", type="number", id="rx-min-score"))
                        yield field("Clock search(samples)", Input(value="128", type="integer", id="rx-clock-search"))
                        yield field("Payload search(samples)", Input(value="16", type="integer", id="rx-payload-search"))
                        yield field("Slope window", Input(value="64", type="integer", id="rx-slope-window"))
                        yield field("Slope clip(rad/bin)", Input(value="0.05", type="number", id="rx-slope-clip"))
            with TabPane("结果", id="results"):
                with Horizontal():
                    yield Static("尚无结果", id="result-state", classes="metric")
                    yield Static("BER --", id="result-ber", classes="metric")
                    yield Static("CRC --", id="result-crc", classes="metric")
                    yield Static("Clock --", id="result-clock", classes="metric")
                yield DataTable(id="artifact-table", cursor_type="row")
                yield Button("刷新结果", id="results-refresh", variant="primary")
            with TabPane("Profiles / 任务", id="profiles"):
                with VerticalScroll():
                    with Container(classes="section"):
                        yield Static("Profile 编辑器", classes="title")
                        yield field("基于", Select(profiles, value=str(ROOT / "profiles" / "tui_v1_default.json"), allow_blank=False, id="profile-base"))
                        yield field("新名称", Input(value="My TUI Profile", id="profile-name"))
                        yield field("FFT N", Input(value="512", type="integer", id="profile-n"))
                        yield field("CP samples", Input(value="256", type="integer", id="profile-cp"))
                        yield field("Active bins", Input(value="64-120,158-178", id="profile-bins"))
                        yield Static("6000-11250 Hz, 14812.5-16687.5 Hz", id="profile-hz")
                        yield field("调制", Select([("BPSK", "bpsk"), ("QPSK", "qpsk")], value="bpsk", allow_blank=False, id="profile-mod"))
                        yield field("FEC", Select([("None", "none"), ("Conv K7", "conv"), ("RS(255,223)", "rs"), ("RS + Conv", "rs_conv")], value="conv", allow_blank=False, id="profile-fec"))
                        yield field("Pilot spacing", Input(value="4", type="integer", id="profile-pilot-spacing"))
                        yield field("前置静音秒数", Input(value="0.5", type="number", id="profile-noise"))
                        yield field("尾部静音秒数", Input(value="0.25", type="number", id="profile-tail"))
                        yield field("Sync symbols", Input(value="64", type="integer", id="profile-sync"))
                        yield field("Sync correlation", Input(value="8", type="integer", id="profile-sync-corr"))
                        yield field("Sync seed", Input(value="7026", type="integer", id="profile-sync-seed"))
                        yield field("Preamble symbols", Input(value="128", type="integer", id="profile-preamble"))
                        yield field("Preamble repeats", Input(value="2", type="integer", id="profile-preamble-repeats"))
                        yield field("Preamble seed", Input(value="8026", type="integer", id="profile-preamble-seed"))
                        yield field("Block bytes", Input(value="512", type="integer", id="profile-block"))
                        yield field("Header repeats", Input(value="3", type="integer", id="profile-header-repeats"))
                        yield field("Pilot seed", Input(value="2027", type="integer", id="profile-pilot-seed"))
                        yield field("FEC/interleave seed", Input(value="7027", type="integer", id="profile-fec-seed"))
                        yield field("Timing anchor interval", Input(value="128", type="integer", id="profile-anchor-interval"))
                        yield field("Timing anchor symbols", Input(value="8", type="integer", id="profile-anchor-symbols"))
                        yield field("Timing anchor seed", Input(value="9028", type="integer", id="profile-anchor-seed"))
                        yield field("Start anchor symbols", Input(value="8", type="integer", id="profile-start-symbols"))
                        yield field("Start anchor seed", Input(value="10028", type="integer", id="profile-start-seed"))
                        yield field("Training anchor symbols", Input(value="8", type="integer", id="profile-training-symbols"))
                        yield field("Training anchor step", Input(value="32", type="integer", id="profile-training-step"))
                        yield Button("载入基础 Profile", id="profile-load")
                        yield Button("另存 Profile", id="profile-save", variant="success")
                    with Container(classes="section"):
                        yield Static("任务日志", classes="title")
                        yield RichLog(id="job-log", wrap=True, markup=True)
                        yield Button("取消当前任务", id="job-cancel", variant="error")
        yield Static("就绪", id="global-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#recent-table", DataTable)
        table.add_columns("时间/ID", "任务", "状态", "BER", "CRC", "PPM")
        artifacts = self.query_one("#artifact-table", DataTable)
        artifacts.add_columns("产物", "路径")
        self.refresh_dashboard()

    def selected_profile(self, select_id: str) -> ModemProfile:
        value = self.query_one(select_id, Select).value
        if not isinstance(value, str):
            raise ValueError("请选择 Profile")
        return ModemProfile.load(Path(value))

    def event(self, event: dict) -> None:
        kind = event.get("event", "log")
        log = self.query_one("#job-log", RichLog)
        if kind == "meter":
            self.query_one("#meter", Static).update(
                f"Peak {event['peak']:.3f}  RMS {event['rms']:.3f}  "
                f"Clipping {event['clipping']} ({event['clipping_fraction']:.3%})  "
                f"剩余 {event['remaining']:.1f}s"
            )
        elif kind == "progress":
            self.query_one("#global-status", Static).update(
                f"{event.get('stage', '运行中')}  {event.get('progress', 0) * 100:.0f}%"
            )
        elif kind == "command":
            log.write("[cyan]$ " + " ".join(map(str, event.get("command", []))) + "[/cyan]")
        elif kind == "warning":
            log.write("[yellow]警告: " + event.get("text", "") + "[/yellow]")
        elif kind == "error":
            log.write("[red]" + event.get("error", "unknown error") + "[/red]")
        elif kind == "log":
            log.write(event.get("text", ""))
        elif kind == "result":
            log.write(f"[green]完成: {event.get('status', 'success')}[/green]")

    def show_error(self, exc: Exception) -> None:
        self.query_one("#global-status", Static).update(f"失败: {exc}")
        self.query_one("#job-log", RichLog).write(f"[red]{type(exc).__name__}: {exc}[/red]")

    def refresh_dashboard(self) -> None:
        table = self.query_one("#recent-table", DataTable)
        table.clear()
        for item in recent_manifests():
            metrics = item.get("metrics") or {}
            ber = metrics.get("coded_bit_error_rate")
            crc = f"{metrics.get('blocks_ok', '--')}/{metrics.get('blocks_total', '--')}"
            ppm = metrics.get("clock_error_ppm")
            table.add_row(
                item.get("experiment_id", ""), item.get("kind", ""), item.get("status", ""),
                f"{ber:.3%}" if isinstance(ber, (int, float)) else "--", crc,
                f"{ppm:+.2f}" if isinstance(ppm, (int, float)) else "--",
            )

    def refresh_results(self, metrics: dict | None = None) -> None:
        metrics = metrics or self.latest_metrics
        if not metrics:
            return
        exact = metrics.get("file_match")
        crc_ok = metrics.get("file_crc_ok")
        header = metrics.get("header_ok")
        state = "完全一致" if exact else "CRC 通过，未比对源文件" if crc_ok else "部分恢复" if header else "解码失败"
        self.query_one("#result-state", Static).update(f"状态\n{state}")
        ber = metrics.get("coded_bit_error_rate")
        post = metrics.get("post_fec_bit_error_rate")
        self.query_one("#result-ber", Static).update(
            f"BER\nRaw {ber:.4%}\nPost-FEC {post:.4%}" if isinstance(ber, (int, float)) and isinstance(post, (int, float))
            else f"BER\nRaw {ber:.4%}" if isinstance(ber, (int, float)) else "BER\nN/A"
        )
        self.query_one("#result-crc", Static).update(f"CRC blocks\n{metrics.get('blocks_ok', 0)}/{metrics.get('blocks_total', 0)}")
        self.query_one("#result-clock", Static).update(
            f"Clock\n{metrics.get('clock_status', '--')}\n{metrics.get('clock_error_ppm', 0):+.3f} ppm"
        )
        table = self.query_one("#artifact-table", DataTable); table.clear()
        out = Path(metrics.get("out", ""))
        if out.exists():
            for path in sorted(out.iterdir()): table.add_row(path.name, str(path))

    def decode_profile(self) -> ModemProfile:
        meta_text = self.query_one("#decode-meta", Input).value.strip()
        if meta_text:
            raw = json.loads(Path(meta_text).read_text(encoding="utf-8"))
            if isinstance(raw.get("profile"), dict):
                profile = ModemProfile.from_dict(raw["profile"])
            else:
                profile = ModemProfile.load(ROOT / "profiles" / "step8_compatible.json")
        else:
            profile = self.selected_profile("#decode-profile")
        profile = replace(
            profile, immutable=False,
            phase_slope=str(self.query_one("#decode-slope", Select).value),
            channel_alpha=float(self.query_one("#rx-channel-alpha", Input).value),
            payload_start_anchor_h_alpha=(float(self.query_one("#rx-start-alpha", Input).value)
                if self.query_one("#decode-start-h", Switch).value else 0.0),
            anchor_h_alpha=(float(self.query_one("#rx-anchor-alpha", Input).value)
                if self.query_one("#decode-anchor-h", Switch).value else 0.0),
            anchor_min_score=float(self.query_one("#rx-min-score", Input).value),
            clock_search=int(self.query_one("#rx-clock-search", Input).value),
            payload_search=int(self.query_one("#rx-payload-search", Input).value),
            slope_window=int(self.query_one("#rx-slope-window", Input).value),
            slope_clip=float(self.query_one("#rx-slope-clip", Input).value),
        )
        profile.validate(); return profile

    @work(group="main-jobs", exclusive=True)
    async def run_encode(self) -> None:
        try:
            source = Path(self.query_one("#encode-source", Input).value)
            profile = self.selected_profile("#encode-profile")
            job = self.controller.create("encode", source.stem, profile); self.current_job = job
            out, _ = await self.controller.encode(job, source, self.query_one("#encode-name", Input).value, self.event)
            self.query_one("#play-wav", Input).value = str(out)
            self.query_one("#decode-meta", Input).value = str(out.with_suffix(".meta.json"))
            self.query_one("#global-status", Static).update(f"编码完成: {out}")
            self.refresh_dashboard()
        except Exception as exc: self.show_error(exc)

    @work(group="main-jobs", exclusive=True)
    async def run_record(self) -> None:
        try:
            job = self.controller.create("record", "recording"); self.current_job = job
            target = str(self.query_one("#record-target", Select).value); target = "" if target == "default" else target
            out, _ = await self.controller.record(
                job, float(self.query_one("#record-seconds", Input).value), self.query_one("#record-name", Input).value,
                target, float(self.query_one("#record-volume", Input).value), self.event,
            )
            self.query_one("#decode-recording", Input).value = str(out)
            self.query_one("#global-status", Static).update(f"录音完成: {out}")
            self.refresh_dashboard()
        except Exception as exc: self.show_error(exc)

    @work(group="main-jobs", exclusive=True)
    async def run_play(self) -> None:
        try:
            wav = Path(self.query_one("#play-wav", Input).value)
            job = self.controller.create("play", wav.stem); self.current_job = job
            target = str(self.query_one("#play-target", Select).value); target = "" if target == "default" else target
            await self.controller.play(job, wav, target, float(self.query_one("#play-volume", Input).value), self.event)
            self.query_one("#global-status", Static).update("播放完成")
            self.refresh_dashboard()
        except Exception as exc: self.show_error(exc)

    @work(group="main-jobs", exclusive=True)
    async def run_decode(self) -> None:
        try:
            recording = Path(self.query_one("#decode-recording", Input).value)
            profile = self.decode_profile()
            source_text = self.query_one("#decode-source", Input).value.strip()
            source = Path(source_text) if source_text and self.query_one("#decode-compare", Switch).value else None
            job = self.controller.create("decode", recording.stem, profile); self.current_job = job
            self.latest_metrics = await self.controller.decode(job, recording, source, self.event)
            self.refresh_results(); self.refresh_dashboard()
            self.query_one("#global-status", Static).update("解码完成")
        except Exception as exc: self.show_error(exc)

    @work(group="main-jobs", exclusive=True)
    async def run_pipeline(self) -> None:
        try:
            source = Path(self.query_one("#pipe-source", Input).value)
            profile = self.selected_profile("#pipe-profile")
            duration = float(self.query_one("#pipe-duration", Input).value)
            pre = float(self.query_one("#pipe-preroll", Input).value)
            input_target = str(self.query_one("#record-target", Select).value); input_target = "" if input_target == "default" else input_target
            output_target = str(self.query_one("#play-target", Select).value); output_target = "" if output_target == "default" else output_target
            job = self.controller.create("pipeline", source.stem, profile); self.current_job = job
            self.latest_metrics = await self.controller.pipeline(
                job, source, self.query_one("#pipe-tx-name", Input).value,
                self.query_one("#pipe-rx-name", Input).value, duration, pre,
                input_target, output_target, float(self.query_one("#record-volume", Input).value),
                float(self.query_one("#play-volume", Input).value), self.event,
            )
            self.refresh_results(); self.refresh_dashboard()
            self.query_one("#global-status", Static).update("完整实验完成")
        except Exception as exc: self.show_error(exc)

    @work(group="devices", exclusive=True)
    async def refresh_devices(self) -> None:
        devices = await __import__("asyncio").to_thread(discover_devices)
        sources = [("系统默认", "default")] + [(item["name"], item["id"]) for item in devices["sources"]]
        sinks = [("系统默认", "default")] + [(item["name"], item["id"]) for item in devices["sinks"]]
        self.query_one("#record-target", Select).set_options(sources)
        self.query_one("#play-target", Select).set_options(sinks)
        self.query_one("#global-status", Static).update(
            f"发现 {len(sources)-1} 个输入、{len(sinks)-1} 个输出" + (f"; {devices['error']}" if devices.get("error") else "")
        )

    def load_profile_editor(self) -> None:
        profile = self.selected_profile("#profile-base")
        values = {
            "#profile-name": profile.name + " Copy", "#profile-n": profile.fft_size,
            "#profile-cp": profile.cp_samples,
            "#profile-bins": ",".join(f"{a}-{b}" for a, b in profile.active_ranges),
            "#profile-pilot-spacing": profile.pilot_spacing, "#profile-sync": profile.sync_symbols,
            "#profile-noise": profile.noise_seconds, "#profile-tail": profile.tail_seconds,
            "#profile-sync-corr": profile.sync_correlation_symbols, "#profile-sync-seed": profile.sync_seed,
            "#profile-preamble": profile.preamble_symbols, "#profile-preamble-repeats": profile.preamble_repeats,
            "#profile-preamble-seed": profile.preamble_seed,
            "#profile-block": profile.block_size, "#profile-header-repeats": profile.header_repeats,
            "#profile-pilot-seed": profile.pilot_seed, "#profile-fec-seed": profile.fec_seed,
            "#profile-anchor-interval": profile.timing_anchor_interval,
            "#profile-anchor-symbols": profile.timing_anchor_symbols,
            "#profile-anchor-seed": profile.timing_anchor_seed,
            "#profile-start-symbols": profile.payload_start_anchor_symbols,
            "#profile-start-seed": profile.payload_start_anchor_seed,
            "#profile-training-symbols": profile.training_anchor_symbols,
            "#profile-training-step": profile.training_anchor_step,
        }
        for selector, value in values.items(): self.query_one(selector, Input).value = str(value)
        self.query_one("#profile-mod", Select).value = profile.modulation
        self.query_one("#profile-fec", Select).value = profile.fec
        self.update_hz_preview()

    def editor_profile(self) -> ModemProfile:
        base = self.selected_profile("#profile-base")
        profile = replace(
            base, name=self.query_one("#profile-name", Input).value.strip(), protocol="tui_v1", immutable=False,
            fft_size=int(self.query_one("#profile-n", Input).value), cp_samples=int(self.query_one("#profile-cp", Input).value),
            active_ranges=parse_ranges(self.query_one("#profile-bins", Input).value),
            modulation=str(self.query_one("#profile-mod", Select).value), fec=str(self.query_one("#profile-fec", Select).value),
            pilot_spacing=int(self.query_one("#profile-pilot-spacing", Input).value),
            noise_seconds=float(self.query_one("#profile-noise", Input).value),
            tail_seconds=float(self.query_one("#profile-tail", Input).value),
            sync_symbols=int(self.query_one("#profile-sync", Input).value),
            sync_correlation_symbols=int(self.query_one("#profile-sync-corr", Input).value),
            sync_seed=int(self.query_one("#profile-sync-seed", Input).value),
            preamble_symbols=int(self.query_one("#profile-preamble", Input).value),
            preamble_repeats=int(self.query_one("#profile-preamble-repeats", Input).value),
            preamble_seed=int(self.query_one("#profile-preamble-seed", Input).value),
            block_size=int(self.query_one("#profile-block", Input).value),
            header_repeats=int(self.query_one("#profile-header-repeats", Input).value),
            pilot_seed=int(self.query_one("#profile-pilot-seed", Input).value),
            fec_seed=int(self.query_one("#profile-fec-seed", Input).value),
            timing_anchor_interval=int(self.query_one("#profile-anchor-interval", Input).value),
            timing_anchor_symbols=int(self.query_one("#profile-anchor-symbols", Input).value),
            timing_anchor_seed=int(self.query_one("#profile-anchor-seed", Input).value),
            payload_start_anchor_symbols=int(self.query_one("#profile-start-symbols", Input).value),
            payload_start_anchor_seed=int(self.query_one("#profile-start-seed", Input).value),
            training_anchor_symbols=int(self.query_one("#profile-training-symbols", Input).value),
            training_anchor_step=int(self.query_one("#profile-training-step", Input).value),
        )
        profile.validate(); return profile

    def update_hz_preview(self) -> None:
        try:
            n = int(self.query_one("#profile-n", Input).value)
            ranges = parse_ranges(self.query_one("#profile-bins", Input).value)
            text = ", ".join(f"{a * 48000 / n:.1f}-{b * 48000 / n:.1f} Hz" for a, b in ranges)
        except Exception as exc: text = f"无效频段: {exc}"
        self.query_one("#profile-hz", Static).update(text)

    def save_profile(self) -> None:
        profile = self.editor_profile()
        filename = "_".join(profile.name.lower().split()) + ".json"
        path = ROOT / "profiles" / filename
        if path.exists(): raise FileExistsError(f"Profile 已存在: {path.name}")
        profile.save(path)
        options = self.profile_options()
        for selector in ("#encode-profile", "#pipe-profile", "#decode-profile", "#profile-base"):
            self.query_one(selector, Select).set_options(options)
        self.query_one("#global-status", Static).update(f"已保存 Profile: {path}")

    def action_refresh(self) -> None:
        self.refresh_dashboard(); self.refresh_results()

    def action_cancel_job(self) -> None:
        if self.current_job: self.current_job.cancel(); self.query_one("#global-status", Static).update("已请求取消任务")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id in ("profile-n", "profile-bins"): self.update_hz_preview()
        if event.input.id == "play-wav":
            try: self.query_one("#pipe-preview", Static).update(f"WAV 时长 {wav_seconds(Path(event.value)):.3f}s")
            except Exception: pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "refresh-dashboard": self.refresh_dashboard, "results-refresh": self.refresh_results,
            "encode-start": self.run_encode, "record-start": self.run_record,
            "play-start": self.run_play, "decode-start": self.run_decode,
            "pipeline-start": self.run_pipeline, "devices-refresh": self.refresh_devices,
            "media-stop": self.action_cancel_job, "job-cancel": self.action_cancel_job,
            "profile-load": self.load_profile_editor, "profile-save": self.save_profile,
        }
        action = actions.get(event.button.id)
        if action:
            try: action()
            except Exception as exc: self.show_error(exc)


def run() -> None:
    AudioModemTUI().run()
