"""实验脚本的路径引导。n2 里的模块用同目录导入，所以把 n2/ 放进 sys.path。"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N2 = ROOT / "n2"
STANDARD = ROOT / "standard"
LDPC_PY = ROOT / "LDPC" / "new_ldpc" / "py"
LDPC_ROOT = ROOT / "LDPC" / "new_ldpc"

for p in (str(N2), str(STANDARD), str(LDPC_PY)):
    if p not in sys.path:
        sys.path.insert(0, p)

if "sounddevice" not in sys.modules:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        stub = types.ModuleType("sounddevice")
        stub.play = lambda *a, **k: None
        stub.wait = lambda: None
        sys.modules["sounddevice"] = stub
