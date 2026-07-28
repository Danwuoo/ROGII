from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_noise_floor_notebook import audit_noise_floor_notebook


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "results" / "measure-your-noise-floor-before-believing-a-lever.ipynb"


def test_noise_floor_notebook_reaudit_recomputes_locked_measurement(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reaudit.json"
    report = audit_noise_floor_notebook(NOTEBOOK, output)

    assert report["passed"] is True
    assert all(report["checks"].values())
    assert report["calculations"]["a_sample_sd"] == pytest.approx(0.03668332954717531)
    assert report["calculations"]["b_sample_sd"] == pytest.approx(0.02687005768508871)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_noise_floor_notebook_reaudit_fails_closed_on_changed_rerun(
    tmp_path: Path,
) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code" and "A = np.array" in "".join(
            cell.get("source", [])
        ):
            source = cell.get("source", [])
            if isinstance(source, str):
                cell["source"] = source.replace("6.700", "6.701", 1)
            else:
                cell["source"] = [line.replace("6.700", "6.701", 1) for line in source]
            break
    changed = tmp_path / "changed.ipynb"
    changed.write_text(json.dumps(notebook), encoding="utf-8")
    output = tmp_path / "should_not_exist.json"

    with pytest.raises(ValueError, match="reruns"):
        audit_noise_floor_notebook(changed, output)
    assert not output.exists()


def test_noise_floor_notebook_reaudit_fails_closed_on_changed_table(
    tmp_path: Path,
) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        joined = "".join(source)
        if (
            cell.get("cell_type") == "markdown"
            and "| branch hedge 0.00 vs 0.60 |" in joined
        ):
            if isinstance(source, str):
                cell["source"] = source.replace("| 0.061 |", "| 0.062 |", 1)
            else:
                cell["source"] = [
                    line.replace("| 0.061 |", "| 0.062 |", 1) for line in source
                ]
            break
    changed = tmp_path / "changed-table.ipynb"
    changed.write_text(json.dumps(notebook), encoding="utf-8")

    with pytest.raises(ValueError, match="audit failed"):
        audit_noise_floor_notebook(changed, tmp_path / "should_not_exist.json")
