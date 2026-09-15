import re
from pathlib import Path


def test_n3_2_never_imports_old_implementations():
    banned = re.compile(r"^\s*(?:from|import)\s+n3(?:_1)?(?:\s|\.|$)", re.M)
    for source in Path("n3_2").glob("*.py"):
        assert not banned.search(source.read_text(encoding="utf-8")), source
