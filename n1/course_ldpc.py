# -*- coding: utf-8 -*-
"""将仓库内 ``LDPC/new_ldpc/py`` 注入为 ``import ldpc``（直接使用课程包，无自研回退）。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LDPC_PY_DIR = _REPO / "LDPC" / "new_ldpc" / "py"


def install_as_ldpc_module() -> None:
    """把课程包目录放到 sys.path 最前，并清除可能冲突的已加载 ldpc。"""
    if not (_LDPC_PY_DIR / "ldpc.py").is_file():
        raise FileNotFoundError(f"未找到课程 LDPC：{_LDPC_PY_DIR / 'ldpc.py'}")
    path = str(_LDPC_PY_DIR)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    # 去掉错误的 PyPI ldpc 等已加载模块
    for name in list(sys.modules):
        if name == "ldpc" or name.startswith("ldpc."):
            del sys.modules[name]
