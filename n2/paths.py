"""项目目录约定：接收音频 data/，原始文件 source/，恢复结果 run/。

命令在 n2/ 下运行时，路径按仓库根解析，例如：

    cd g:\\OFDM_gerson\\n2
    python run_rx.py --wav ..\\data\\recv.wav --out-dir ..\\run\\n2\\trial1
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DIR = ROOT / "source"
if not SOURCE_DIR.is_dir() and (ROOT / "sourse").is_dir():
    SOURCE_DIR = ROOT / "sourse"
RUN_DIR = ROOT / "run"
N2_RUN_DIR = RUN_DIR / "n2"

_ROOT_MARKERS = {
    "data": DATA_DIR,
    "source": ROOT / "source",
    "sourse": ROOT / "sourse",
    "run": RUN_DIR,
}


def ensure_dirs():
    for d in (DATA_DIR, SOURCE_DIR, RUN_DIR, N2_RUN_DIR):
        d.mkdir(parents=True, exist_ok=True)
    return DATA_DIR, SOURCE_DIR, RUN_DIR


def _tokens(raw):
    """拆路径并丢掉盘符、UNC 前缀和 ..，便于识别 data/run/source。"""
    s = str(raw).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = s[2:]
    s = s.lstrip("/")
    return [p for p in s.split("/") if p and p not in (".", "..")]


def _after_marker(raw):
    """把 \\data\\a.wav、..\\data\\a.wav、G:\\...\\data\\a.wav 收到仓库根约定目录。"""
    parts = _tokens(raw)
    names = [p.lower() for p in parts]
    for i, name in enumerate(names):
        if name in _ROOT_MARKERS:
            rest = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path()
            return _ROOT_MARKERS[name], rest
    return None, None


def resolve_wav(wav_arg):
    """解析 --wav，始终落到仓库根 data/。"""
    ensure_dirs()
    if wav_arg is None:
        wavs = sorted(DATA_DIR.glob("*.wav")) + sorted(DATA_DIR.glob("*.WAV"))
        if len(wavs) == 1:
            return wavs[0]
        if not wavs:
            raise FileNotFoundError(f"找不到 WAV，请放到 {DATA_DIR} 或指定 --wav")
        names = ", ".join(p.name for p in wavs)
        raise FileNotFoundError(f"{DATA_DIR} 中有多个 WAV，请用 --wav 指定：{names}")

    raw = str(wav_arg)
    base, rest = _after_marker(raw)
    if base is not None:
        path = base / rest if rest.parts else base
        if path.is_file():
            return path
        raise FileNotFoundError(f"找不到接收音频：{path}")

    path = Path(raw)
    if path.is_file():
        return path
    if (DATA_DIR / path.name).is_file():
        return DATA_DIR / path.name
    if (ROOT / path).is_file():
        return ROOT / path
    raise FileNotFoundError(f"找不到接收音频：{wav_arg}（已查 {DATA_DIR}）")


def resolve_run_dir(name=None):
    """解析 --out-dir。默认 run/n2/；\\run\\n2\\trial1 落到仓库根 run/n2/trial1。"""
    ensure_dirs()
    if not name:
        N2_RUN_DIR.mkdir(parents=True, exist_ok=True)
        return N2_RUN_DIR

    base, rest = _after_marker(name)
    if base is not None:
        dest = base / rest if rest.parts else base
        dest.mkdir(parents=True, exist_ok=True)
        return dest.resolve()

    path = Path(name)
    if path.is_absolute():
        path.mkdir(parents=True, exist_ok=True)
        return path

    dest = (N2_RUN_DIR / path).resolve()
    run_root = RUN_DIR.resolve()
    try:
        dest.relative_to(run_root)
    except ValueError as exc:
        raise ValueError(f"输出目录必须在 {RUN_DIR} 下：{name}") from exc
    dest.mkdir(parents=True, exist_ok=True)
    return dest
