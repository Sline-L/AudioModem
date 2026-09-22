"""n2 接收端窗口。在 n2/ 下运行：python app.py

解调仍调用 run_rx.py，输出目录与命令行相同。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

N2 = Path(__file__).resolve().parent
if str(N2) not in sys.path:
    sys.path.insert(0, str(N2))

import numpy as np

from paths import DATA_DIR, N2_RUN_DIR, _after_marker
from phy import read_wav_channels

BG = "#0f1115"
PANEL = "#161a21"
PANEL2 = "#1c222b"
LINE = "#2a323d"
FG = "#e6e9ef"
DIM = "#9aa4b2"
ACCENT = "#4da3ff"
OK = "#5ad1a0"
WARN = "#ffb454"
BAD = "#f07178"
FONT = "Microsoft YaHei UI"
MONO = "Cascadia Mono"

STAGE_NAME = {
    "ok": "done 完成",
    "capture": "capture 捕获",
    "align": "align 对准",
    "channel": "channel 信道",
    "ldpc": "LDPC 译码",
    "header": "header 帧头",
}


def _dpi():
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _fatal(message):
    log = N2 / "app-error.log"
    try:
        log.write_text(message, encoding="utf-8")
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message[:1500], "N2 Receiver 接收机", 0x10)
    except Exception:
        pass


def _nat_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def expected_out_dir(name: str | None) -> Path:
    """与命令行 --out-dir 同一规则，但不提前创建目录。"""
    text = (name or "").strip()
    if not text:
        return N2_RUN_DIR
    path = Path(text)
    if path.is_absolute():
        return path
    base, rest = _after_marker(text)
    if base is not None:
        return base / rest if rest.parts else base
    return N2_RUN_DIR / text


def _envelope(samples, buckets=1600):
    x = np.asarray(samples, dtype=np.float32)
    n = int(x.size)
    if n <= 1:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    buckets = int(min(buckets, n))
    edges = np.linspace(0, n, buckets + 1, dtype=np.int64)
    lo = np.empty(buckets, dtype=np.float32)
    hi = np.empty(buckets, dtype=np.float32)
    t = np.empty(buckets, dtype=np.float64)
    for i in range(buckets):
        a = int(edges[i])
        b = int(edges[i + 1])
        if b <= a:
            b = min(n, a + 1)
        seg = x[a:b]
        lo[i] = float(seg.min())
        hi[i] = float(seg.max())
        t[i] = (a + b) * 0.5
    return t, lo, hi


def _one_line(text, limit=16):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _human_size(n):
    n = float(n)
    if n < 1024:
        return f"{int(n)} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def _summary(metrics: dict) -> tuple[str, str]:
    recovered = metrics.get("recovered_name") or "—"
    header = "Header OK 帧头通过" if metrics.get("header_ok") else "Header failed 帧头未过"
    blocks = f"{metrics.get('blocks_ok')}/{metrics.get('blocks_total')}"
    match = metrics.get("file_match")
    if match is True:
        match_s = "Matches source 与原文件一致"
    elif match is False:
        match_s = "Differs from source 与原文件不一致"
    else:
        match_s = "No source file 无对照原文件"
    ppm = metrics.get("clock_error_ppm")
    ppm_s = f"{ppm:+.2f} ppm 钟偏" if isinstance(ppm, (int, float)) else "Clock n/a 钟偏 n/a"
    stage = STAGE_NAME.get(str(metrics.get("stage") or ""), str(metrics.get("stage") or ""))
    if metrics.get("header_ok"):
        color = OK
        text = f"Recovered {recovered} 已恢复 · {header} · LDPC {blocks} · {match_s} · {ppm_s}"
    else:
        color = BAD if metrics.get("error") else WARN
        reason = metrics.get("error") or stage or "not decoded 未解出"
        text = f"Incomplete 未完整恢复 · Stopped at {stage} 停在此阶段 · {reason}"
    return text, color


class N2App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.ttk = ttk
        self.root = root
        self.root.title("N2 Receiver 接收机")
        self.root.geometry("1180x760")
        self.root.minsize(960, 640)
        self.root.configure(bg=BG)

        self.wav_path: Path | None = None
        self.wave = None
        self.metrics = None
        self.running = False

        self._build()
        self.refresh_wavs()

    def _build(self):
        tk = self.tk
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)
        body.grid_columnconfigure(0, weight=0, minsize=320)
        body.grid_columnconfigure(2, weight=1)
        body.grid_rowconfigure(0, weight=1)

        side = tk.Frame(body, bg=PANEL, padx=16, pady=16)
        side.grid(row=0, column=0, sticky="nsew")
        tk.Frame(body, bg=LINE, width=1).grid(row=0, column=1, sticky="ns")
        main = tk.Frame(body, bg=BG, padx=18, pady=16)
        main.grid(row=0, column=2, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)

        title = tk.Frame(side, bg=PANEL)
        title.pack(anchor="w")
        tk.Label(title, text="N2 Receiver", bg=PANEL, fg=FG, font=(FONT, 18, "bold")).pack(side=tk.LEFT)
        tk.Label(title, text="接收机", bg=PANEL, fg=DIM, font=(FONT, 9)).pack(side=tk.LEFT, padx=(8, 0), pady=(8, 0))
        tk.Label(
            side,
            text="Pick a WAV and decode 选择录音并解调",
            bg=PANEL,
            fg=DIM,
            font=(FONT, 9),
        ).pack(anchor="w", pady=(4, 12))

        tk.Label(side, text="Recordings 录音", bg=PANEL, fg=DIM, font=(FONT, 9)).pack(anchor="w")
        list_wrap = tk.Frame(side, bg=LINE, padx=1, pady=1)
        list_wrap.pack(fill=tk.BOTH, expand=True, pady=(6, 8))
        inner = tk.Frame(list_wrap, bg=PANEL2)
        inner.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(
            inner,
            bg=PANEL2,
            fg=FG,
            selectbackground=ACCENT,
            selectforeground="#081018",
            highlightthickness=0,
            relief=tk.FLAT,
            activestyle="none",
            font=(FONT, 10),
            width=24,
            height=2,
            exportselection=False,
        )
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "N2.Vertical.TScrollbar",
            background=LINE,
            troughcolor=PANEL2,
            bordercolor=PANEL2,
            arrowcolor=DIM,
            lightcolor=LINE,
            darkcolor=LINE,
        )
        scroll = self.ttk.Scrollbar(inner, command=self.listbox.yview, style="N2.Vertical.TScrollbar")
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.start())

        row = tk.Frame(side, bg=PANEL)
        row.pack(fill=tk.X, pady=(0, 8))
        self._button(row, "Browse… 浏览", self.browse).pack(side=tk.LEFT)
        self._button(row, "Refresh 刷新", self.refresh_wavs).pack(side=tk.LEFT, padx=(8, 0))

        self.meta = tk.Label(side, text=" ", bg=PANEL, fg=DIM, font=(FONT, 9), anchor="w", justify=tk.LEFT)
        self.meta.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            side,
            text="Output folder 输出子目录  ·  run/n2/",
            bg=PANEL,
            fg=DIM,
            font=(FONT, 9),
            justify=tk.LEFT,
        ).pack(anchor="w")
        self.out_var = tk.StringVar()
        entry = tk.Entry(
            side,
            textvariable=self.out_var,
            bg=PANEL2,
            fg=FG,
            insertbackground=FG,
            relief=tk.FLAT,
            font=(FONT, 11),
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=ACCENT,
        )
        entry.pack(fill=tk.X, ipady=6, pady=(6, 12))

        self.run_btn = tk.Button(
            side,
            text="Decode 开始解调",
            command=self.start,
            bg=ACCENT,
            fg="#081018",
            activebackground="#79b8ff",
            activeforeground="#081018",
            relief=tk.FLAT,
            font=(FONT, 12, "bold"),
            cursor="hand2",
            padx=8,
            pady=8,
        )
        self.run_btn.pack(fill=tk.X)

        style.configure(
            "N2.Horizontal.TProgressbar",
            troughcolor=PANEL2,
            background=ACCENT,
            borderwidth=0,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
        )
        self.bar = self.ttk.Progressbar(side, mode="indeterminate", style="N2.Horizontal.TProgressbar")

        self.status = tk.Label(
            side,
            text="Select a recording on the left. 从左侧选择一条录音。",
            bg=PANEL,
            fg=DIM,
            font=(FONT, 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=300,
        )
        self.status.pack(fill=tk.X, pady=(0, 8))

        self.log = tk.Text(
            side,
            height=4,
            bg=BG,
            fg=DIM,
            insertbackground=FG,
            relief=tk.FLAT,
            font=(MONO, 9),
            highlightthickness=1,
            highlightbackground=LINE,
            wrap=tk.WORD,
            width=28,
            padx=8,
            pady=6,
        )
        self.log.pack(fill=tk.BOTH, expand=False)
        self.log.configure(state=tk.DISABLED)

        self.summary = tk.Label(
            main,
            text="No result yet 还没有解调结果",
            bg=BG,
            fg=FG,
            font=(FONT, 13, "bold"),
            anchor="w",
            justify=tk.LEFT,
            wraplength=760,
        )
        self.summary.grid(row=0, column=0, sticky="ew")

        actions = tk.Frame(main, bg=BG)
        actions.grid(row=1, column=0, sticky="w", pady=(8, 12))
        self.open_dir_btn = self._button(actions, "Open folder 打开目录", self.open_out_dir)
        self.open_dir_btn.pack(side=tk.LEFT)
        self.open_file_btn = self._button(actions, "Open file 打开文件", self.open_recovered)
        self.open_file_btn.pack(side=tk.LEFT, padx=(8, 0))

        cards = tk.Frame(main, bg=BG)
        cards.grid(row=2, column=0, sticky="ew")
        self.cards = {}
        specs = (
            ("recovered", "Recovered", "恢复文件"),
            ("header", "Header", "帧头"),
            ("ldpc", "LDPC", "译码"),
            ("match", "Match", "对照"),
            ("ppm", "Clock", "钟偏"),
            ("sync", "Sync", "同步"),
        )
        cards.grid_rowconfigure(0, weight=1)
        for col, (key, en, zh) in enumerate(specs):
            cards.grid_columnconfigure(col, weight=1, uniform="cards")
            self.cards[key] = self._card(cards, en, zh)
            self.cards[key].grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))

        pages_host = tk.Frame(main, bg=BG)
        pages_host.grid(row=3, column=0, sticky="nsew", pady=(14, 0))

        tab_bar = tk.Frame(pages_host, bg=BG)
        tab_bar.pack(fill=tk.X)
        self.pages = {}
        self.tab_btns = {}
        tabs = (
            ("wave", "Wave 波形"),
            ("channel", "Channel 信道"),
            ("phase", "Phase 相位"),
            ("clock", "Clock 钟偏"),
            ("file", "File 恢复文件"),
        )
        deck = tk.Frame(pages_host, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        deck.pack(fill=tk.BOTH, expand=True)
        for key, label in tabs:
            btn = tk.Button(
                tab_bar,
                text=label,
                command=lambda k=key: self.show_tab(k),
                bg=BG,
                fg=DIM,
                activebackground=BG,
                activeforeground=FG,
                relief=tk.FLAT,
                font=(FONT, 10),
                padx=12,
                pady=6,
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT)
            self.tab_btns[key] = btn
            page = tk.Frame(deck, bg=PANEL)
            self.pages[key] = page

        self._build_wave(self.pages["wave"])
        self.views = {
            "channel": ImageView(self.pages["channel"], "Channel and noise appear after decoding. 解调完成后显示信道与噪声。"),
            "phase": ImageView(self.pages["phase"], "Phase tracking appears after decoding. 解调完成后显示相位跟踪。"),
            "clock": ImageView(self.pages["clock"], "Clock fit appears after decoding. 解调完成后显示钟偏拟合。"),
            "file": ImageView(self.pages["file"], "A recovered image appears here after the header passes. 帧头通过后，图片类恢复文件会显示在这里。"),
        }
        for view in self.views.values():
            view.pack(fill=tk.BOTH, expand=True)
        self.file_caption = tk.Label(self.pages["file"], text="", bg=PANEL, fg=DIM, font=(FONT, 9))
        self.file_caption.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.show_tab("wave")
        self._set_cards_empty()

    def _button(self, parent, text, command):
        return self.tk.Button(
            parent,
            text=text,
            command=command,
            bg=PANEL2,
            fg=FG,
            activebackground="#243041",
            activeforeground=FG,
            relief=self.tk.FLAT,
            font=(FONT, 9),
            padx=10,
            pady=4,
            cursor="hand2",
        )

    def _card(self, parent, caption_en, caption_zh):
        tk = self.tk
        shell = tk.Frame(parent, bg=LINE, padx=1, pady=1)
        inner = tk.Frame(shell, bg=PANEL, padx=10, pady=8)
        inner.pack(fill=tk.BOTH, expand=True)
        value = tk.Label(
            inner,
            text="—",
            bg=PANEL,
            fg=FG,
            font=(FONT, 12, "bold"),
            anchor="w",
            height=1,
        )
        value.pack(fill=tk.X)
        value_zh = tk.Label(
            inner,
            text=" ",
            bg=PANEL,
            fg=DIM,
            font=(FONT, 9),
            anchor="w",
            height=1,
        )
        value_zh.pack(fill=tk.X)
        tk.Label(
            inner,
            text=caption_en,
            bg=PANEL,
            fg=DIM,
            font=(FONT, 8),
            anchor="w",
            height=1,
        ).pack(fill=tk.X, pady=(6, 0))
        tk.Label(
            inner,
            text=caption_zh,
            bg=PANEL,
            fg=DIM,
            font=(FONT, 8),
            anchor="w",
            height=1,
        ).pack(fill=tk.X)
        shell.value = value
        shell.value_zh = value_zh
        return shell

    def _build_wave(self, parent):
        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib import font_manager
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        names = {font.name for font in font_manager.fontManager.ttflist}
        for name in ("Microsoft YaHei", "Microsoft YaHei UI", "SimHei"):
            if name in names:
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break

        self.fig = Figure(figsize=(7.4, 3.6), dpi=100, facecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.16)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        widget = self.canvas.get_tk_widget()
        widget.configure(bg=PANEL, highlightthickness=0)
        widget.pack(fill=self.tk.BOTH, expand=True)
        self.ax.set_title("Select a WAV to show the waveform 选择录音后显示波形", color=DIM, fontsize=11, loc="left")
        self.canvas.draw_idle()

    def _style_ax(self):
        ax = self.ax
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        ax.grid(True, color=LINE, alpha=0.85, linewidth=0.6)
        ax.set_xlabel("Time (s) 时间", color=DIM, fontsize=9)
        ax.set_ylabel("Amplitude 幅度", color=DIM, fontsize=9)

    def show_tab(self, key):
        for name, page in self.pages.items():
            if name == key:
                page.pack(fill=self.tk.BOTH, expand=True)
            else:
                page.pack_forget()
        for name, btn in self.tab_btns.items():
            on = name == key
            btn.configure(
                fg=ACCENT if on else DIM,
                bg=PANEL if on else BG,
                activebackground=PANEL if on else BG,
                font=(FONT, 10, "bold" if on else "normal"),
            )

    def refresh_wavs(self):
        wavs = sorted(DATA_DIR.glob("*.wav"), key=_nat_key) + sorted(DATA_DIR.glob("*.WAV"), key=_nat_key)
        seen = set()
        self._wavs = []
        for path in wavs:
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            self._wavs.append(path)
        self.listbox.delete(0, self.tk.END)
        for path in self._wavs:
            self.listbox.insert(self.tk.END, path.name)
        if self.wav_path and self.wav_path.resolve() not in {p.resolve() for p in self._wavs}:
            self.listbox.insert(0, self.wav_path.name)
            self._wavs.insert(0, self.wav_path)
        current = self.wav_path if self.wav_path is not None else (self._wavs[0] if self._wavs else None)
        if current is None:
            return
        already = self.wav_path is not None and current.resolve() == self.wav_path.resolve()
        self._select_in_list(current)
        if not already:
            self._use_wav(current)

    def _select_in_list(self, path: Path):
        target = path.resolve()
        self._suspend = True
        try:
            for i, item in enumerate(self._wavs):
                if item.resolve() == target:
                    self.listbox.selection_clear(0, self.tk.END)
                    self.listbox.selection_set(i)
                    self.listbox.see(i)
                    return
        finally:
            self._suspend = False

    def _on_select(self, _event=None):
        if self.running or getattr(self, "_suspend", False):
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        path = self._wavs[sel[0]]
        if self.wav_path and path.resolve() == self.wav_path.resolve():
            return
        self._use_wav(path)

    def browse(self):
        picked = self.filedialog.askopenfilename(
            title="Select received WAV 选择接收 WAV",
            initialdir=str(DATA_DIR),
            filetypes=[("WAV", "*.wav"), ("All files 全部文件", "*.*")],
        )
        if not picked:
            return
        path = Path(picked)
        if path.resolve() not in {p.resolve() for p in self._wavs}:
            self._wavs.insert(0, path)
            self.listbox.insert(0, path.name)
        self._select_in_list(path)
        self._use_wav(path)

    def _use_wav(self, path: Path):
        self.wav_path = path
        self.out_var.set(path.stem)
        self._load_wave(path)
        cached = self._read_metrics(expected_out_dir(path.stem))
        if cached:
            self._apply_metrics(cached, cached_note=True)
            self._set_status("Last result loaded. Press Decode to run again. 已载入上次结果，可点 Decode 重解。")
            if cached.get("header_ok") and _is_image(cached.get("out_path")):
                self.show_tab("file")
        else:
            self.metrics = None
            self._set_cards_empty()
            self.summary.configure(text="No result yet 还没有解调结果", fg=FG)
            self._clear_views()
            self._set_status("Ready to decode. 可以开始解调。")

    def _load_wave(self, path: Path):
        try:
            fs, channels = read_wav_channels(path)
        except Exception as exc:
            self.wave = None
            self.meta.configure(text=f"Cannot read 无法读取：{exc}")
            self._draw_wave()
            return
        self.wave = (fs, channels, path)
        dur = channels[0].size / float(fs) if channels and fs else 0.0
        ch = "mono 单声道" if len(channels) == 1 else f"{len(channels)} ch 声道"
        self.meta.configure(
            text=f"{fs / 1000:.1f} kHz · {ch} · {dur:.2f} s · {_human_size(path.stat().st_size)}"
        )
        self._draw_wave()

    def _draw_wave(self):
        ax = self.ax
        ax.clear()
        self._style_ax()
        if not self.wave:
            ax.set_title("No waveform 没有波形", color=DIM, fontsize=11, loc="left")
            self.canvas.draw_idle()
            return
        fs, channels, path = self.wave
        colors = (ACCENT, "#c08cff", OK, WARN)
        for i, ch in enumerate(channels[:4]):
            t, lo, hi = _envelope(ch)
            seconds = t / float(fs)
            ax.fill_between(seconds, lo, hi, color=colors[i % len(colors)], alpha=0.85, linewidth=0, label=f"ch {i + 1} 声道")
        title = path.name
        metrics = self.metrics or {}
        start = metrics.get("sync_start")
        end = (metrics.get("end") or {}) if isinstance(metrics.get("end"), dict) else {}
        if isinstance(start, (int, float)) and start > 0 and fs:
            t0 = float(start) / float(fs)
            span = end.get("end_offset")
            if isinstance(span, (int, float)) and span > 0:
                t1 = t0 + float(span) / float(fs)
                ax.axvspan(t0, t1, color=ACCENT, alpha=0.16, label="Frame 帧")
            ax.axvline(t0, color=WARN, linewidth=1.0)
            title = f"{path.name}  ·  frame start {t0:.3f} s 帧起点"
        if len(channels) > 1 or (isinstance(start, (int, float)) and start > 0):
            ax.legend(facecolor=PANEL, edgecolor=LINE, labelcolor=FG, fontsize=8, loc="upper right")
        ax.set_title(title, color=FG, fontsize=11, loc="left")
        self.canvas.draw_idle()

    def _set_card(self, card, value, zh, color):
        card.value.configure(text=value, fg=color)
        card.value_zh.configure(text=zh if zh else " ", fg=color)

    def _set_cards_empty(self):
        for card in self.cards.values():
            self._set_card(card, "—", " ", FG)

    def _clear_views(self):
        notes = {
            "channel": "Channel and noise appear after decoding. 解调完成后显示信道与噪声。",
            "phase": "Phase tracking appears after decoding. 解调完成后显示相位跟踪。",
            "clock": "Clock fit appears after decoding. 解调完成后显示钟偏拟合。",
            "file": "A recovered image appears here after the header passes. 帧头通过后，图片类恢复文件会显示在这里。",
        }
        for key, view in self.views.items():
            view.show_message(notes[key])
        self.file_caption.configure(text="")

    def _set_status(self, text, color=DIM):
        self.status.configure(text=text, fg=color)

    def _log(self, text):
        self.log.configure(state=self.tk.NORMAL)
        self.log.insert(self.tk.END, text.rstrip() + "\n")
        self.log.see(self.tk.END)
        self.log.configure(state=self.tk.DISABLED)

    def start(self):
        if self.running or self.wav_path is None:
            if self.wav_path is None:
                self._set_status("Choose a WAV first. 先选择一条 WAV。", WARN)
            return
        wav = self.wav_path
        out_name = self.out_var.get().strip() or wav.stem
        self.running = True
        self.run_btn.configure(state=self.tk.DISABLED, bg="#2a323d", fg=DIM)
        self.bar.pack(fill=self.tk.X, pady=(10, 6), before=self.status)
        self.bar.start(12)
        self._set_status("Decoding… sync, channel, and LDPC can take a while. 正在解调… 同步、信道和 LDPC 会花一些时间。", ACCENT)
        self._log(f"Start 开始 {wav.name} → {expected_out_dir(out_name)}")

        def work():
            try:
                metrics, output = _decode(wav, out_name)
                self.root.after(0, lambda m=metrics, o=output: self._on_done(m, o))
            except Exception as exc:
                detail = traceback.format_exc()
                self.root.after(0, lambda e=exc, d=detail: self._on_fail(e, d))

        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, metrics, output):
        self.running = False
        self.bar.stop()
        self.bar.pack_forget()
        self.run_btn.configure(state=self.tk.NORMAL, bg=ACCENT, fg="#081018")
        for line in output.splitlines():
            if line.strip():
                self._log(line)
        if metrics:
            self._apply_metrics(metrics, cached_note=False)
            self._set_status("Decode finished. 解调结束。", OK if metrics.get("header_ok") else WARN)
            if metrics.get("header_ok") and _is_image(metrics.get("out_path")):
                self.show_tab("file")
            else:
                self.show_tab("channel")
        else:
            self._set_status("Decode finished, but metrics.json was not written. 解调结束，但没有写出 metrics.json。", BAD)

    def _on_fail(self, exc, detail):
        self.running = False
        self.bar.stop()
        self.bar.pack_forget()
        self.run_btn.configure(state=self.tk.NORMAL, bg=ACCENT, fg="#081018")
        self._set_status(str(exc), BAD)
        self._log(detail)

    def _apply_metrics(self, metrics, cached_note=False):
        self.metrics = metrics
        text, color = _summary(metrics)
        if cached_note:
            text = "Last result 上次结果 · " + text
        self.summary.configure(text=text, fg=color)
        if cached_note:
            folder = Path(str(metrics.get("out") or "")).name
            self._log(f"Loaded 载入 run/n2/{folder}")
        recovered = str(metrics.get("recovered_name") or "—")
        self._set_card(self.cards["recovered"], _one_line(recovered, 16), " ", FG)
        header_ok = bool(metrics.get("header_ok"))
        self._set_card(
            self.cards["header"],
            "Pass" if header_ok else "Fail",
            "通过" if header_ok else "未过",
            OK if header_ok else BAD,
        )
        blocks_ok = metrics.get("blocks_ok")
        blocks_total = metrics.get("blocks_total")
        ldpc_color = OK if blocks_total and blocks_ok == blocks_total else WARN
        self._set_card(self.cards["ldpc"], f"{blocks_ok}/{blocks_total}", " ", ldpc_color)
        match = metrics.get("file_match")
        if match is True:
            self._set_card(self.cards["match"], "Match", "一致", OK)
        elif match is False:
            self._set_card(self.cards["match"], "Differ", "不一致", BAD)
        else:
            self._set_card(self.cards["match"], "None", "无原文件", DIM)
        ppm = metrics.get("clock_error_ppm")
        ppm_text = f"{ppm:+.2f}" if isinstance(ppm, (int, float)) else "—"
        self._set_card(self.cards["ppm"], ppm_text, " ", FG)
        sync = metrics.get("sync_score")
        if isinstance(sync, (int, float)):
            sync_text = f"{sync:.0f}" if abs(sync) >= 100 else f"{sync:.3f}"
        else:
            sync_text = "—"
        self._set_card(self.cards["sync"], sync_text, " ", FG)

        out = Path(str(metrics.get("out") or ""))
        self.views["channel"].show_file(out / "channel_and_noise.png", "No channel plot for this run. 这次没有信道图。")
        self.views["phase"].show_file(out / "phase_tracking.png", "No phase plot for this run. 这次没有相位图。")
        self.views["clock"].show_file(out / "clock_fit.png", "No clock plot for this run. 这次没有钟偏图。")
        recovered_path = metrics.get("out_path")
        if _is_image(recovered_path):
            self.views["file"].show_file(recovered_path, "Cannot display this image. 无法显示这张图片。")
        else:
            self.views["file"].show_message("Recovered file is not an image. Use Open file to view it. 恢复文件不是图片。可以用「打开文件」查看。")
        if recovered_path:
            path = Path(str(recovered_path))
            size = _human_size(path.stat().st_size) if path.is_file() else ""
            self.file_caption.configure(text=f"{path.name}  {size}".strip())
        else:
            self.file_caption.configure(text="")
        self._draw_wave()

    def _read_metrics(self, out_dir: Path):
        path = out_dir / "metrics.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def open_out_dir(self):
        metrics = self.metrics or {}
        out = Path(str(metrics.get("out") or expected_out_dir(self.out_var.get())))
        if not out.exists():
            self._set_status("Output folder does not exist yet. 输出目录还不存在。", WARN)
            return
        try:
            os.startfile(out)  # noqa: S606 - 打开用户自己的结果目录
        except OSError as exc:
            self._set_status(str(exc), BAD)

    def open_recovered(self):
        path = (self.metrics or {}).get("out_path")
        if not path or not Path(str(path)).is_file():
            self._set_status("No recovered file yet. 还没有恢复文件。", WARN)
            return
        try:
            os.startfile(path)  # noqa: S606 - 打开本次解调写出的文件
        except OSError as exc:
            self._set_status(str(exc), BAD)


def _is_image(path):
    if not path:
        return False
    return Path(str(path)).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


def _decode(wav: Path, out_name: str):
    """在子进程里跑命令行接收端，避免和窗口里的 matplotlib 抢后端。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(
        [sys.executable, str(N2 / "run_rx.py"), "--wav", str(wav), "--out-dir", out_name],
        cwd=str(N2),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=flags,
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    metrics_path = expected_out_dir(out_name) / "metrics.json"
    metrics = None
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            output += f"\nCannot read 无法读取 {metrics_path}: {exc}"
    if metrics is None and proc.returncode != 0:
        tail = output.strip() or f"Receiver exit code 接收端退出码 {proc.returncode}"
        raise RuntimeError(tail[-800:])
    return metrics, output.strip()


class ImageView(object):
    def __init__(self, parent, empty):
        import tkinter as tk

        self.tk = tk
        self.empty = empty
        self.path = None
        self.missing = empty
        self._photo = None
        self._job = None
        self.frame = tk.Frame(parent, bg=PANEL)
        self.label = tk.Label(self.frame, text=empty, bg=PANEL, fg=DIM, font=(FONT, 10))
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        self.frame.bind("<Configure>", self._schedule)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def show_message(self, text):
        self.path = None
        self._photo = None
        self.label.configure(image="", text=text)

    def show_file(self, path, missing):
        path = Path(path)
        self.missing = missing
        if not path.is_file():
            self.show_message(missing)
            return
        self.path = path
        self._render()

    def _schedule(self, _event=None):
        if self._job is not None:
            self.frame.after_cancel(self._job)
        self._job = self.frame.after(120, self._render)

    def _render(self):
        self._job = None
        if self.path is None:
            return
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self.label.configure(image="", text="Image preview needs pillow: pip install pillow 预览图片需要 pillow")
            return
        try:
            image = Image.open(self.path)
            width = max(self.frame.winfo_width() - 24, 200)
            height = max(self.frame.winfo_height() - 24, 120)
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            scale = min(width / image.width, height / image.height, 2.0)
            size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(size, resample)
            self._photo = ImageTk.PhotoImage(image)
            self.label.configure(image=self._photo, text="")
        except Exception as exc:
            self.label.configure(image="", text=f"{self.missing}\n{exc}")


def main():
    _dpi()
    try:
        import tkinter as tk
    except Exception as exc:
        _fatal(f"Cannot load the window toolkit 无法加载窗口组件：{exc}")
        return 1
    try:
        root = tk.Tk()
        N2App(root)
        root.mainloop()
    except Exception:
        _fatal(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
