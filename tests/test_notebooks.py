from __future__ import annotations

import json
from pathlib import Path


def test_only_two_thin_notebooks_remain() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle"
    notebooks = sorted(root.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["flat.ipynb", "p3.ipynb"]
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["cells"]) == 2
        source = "".join(payload["cells"][1]["source"])
        assert '"-m", "rogii", "run"' in source
        assert '"-m", "rogii", "submit"' in source
        assert "protocol" not in source.lower()
