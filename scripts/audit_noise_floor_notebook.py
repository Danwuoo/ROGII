from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

from scipy.stats import f as f_distribution


EXPECTED_RERUNS = {
    "A": (6.700, 6.726, 6.641, 6.671),
    "B": (7.219, 7.240, 7.259, 7.282),
    "CC": (7.216, 7.305),
    "DD": (7.246, 7.280),
    "EE": (7.535, 7.647),
}
SIGMA_USED_BY_NOTEBOOK = 0.037
COMPARISONS = (
    ("branch hedge 0.00 vs 0.60", 0.061, "single_vs_four_mean"),
    ("branch hedge 0.60 vs 1.40", 0.009, "single_vs_four_mean"),
    ("GR likelihood width 1.3 vs 1.5", 0.036, "single_vs_single"),
    ("branch hedge 1.40 vs 2.40", 0.125, "single_vs_single"),
    ("branch hedge 0.00 vs 2.40", 0.194, "single_vs_single"),
    ("GR likelihood width 1.0 vs 1.3", 0.176, "single_vs_four_mean"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_reruns(notebook: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    extracted: dict[str, tuple[float, ...]] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                not isinstance(node, ast.Assign)
                or len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
                or node.targets[0].id not in EXPECTED_RERUNS
                or not isinstance(node.value, ast.Call)
                or not node.value.args
            ):
                continue
            name = node.targets[0].id
            try:
                values = ast.literal_eval(node.value.args[0])
            except (ValueError, TypeError, SyntaxError) as error:
                raise ValueError(
                    f"Cannot safely parse notebook reruns for {name}"
                ) from error
            parsed = tuple(float(value) for value in values)
            if name in extracted and extracted[name] != parsed:
                raise ValueError(
                    f"Notebook contains inconsistent repeated reruns for {name}"
                )
            extracted[name] = parsed
    return extracted


def _extract_comparison_table(
    notebook: dict[str, Any],
) -> tuple[tuple[str, float, float, float, str], ...]:
    rows: list[tuple[str, float, float, float, str]] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if "| branch hedge 0.00 vs 0.60 |" not in source:
            continue
        for line in source.splitlines():
            if not line.startswith("|") or " vs " not in line:
                continue
            fields = [field.strip() for field in line.strip().strip("|").split("|")]
            if len(fields) != 5:
                raise ValueError("Noise-floor comparison table schema changed")
            try:
                rows.append(
                    (
                        fields[0],
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                        fields[4],
                    )
                )
            except ValueError as error:
                raise ValueError(
                    "Noise-floor comparison table contains nonnumeric values"
                ) from error
    if len(rows) != len(COMPARISONS):
        raise ValueError("Noise-floor comparison table is missing or incomplete")
    return tuple(rows)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def audit_noise_floor_notebook(
    notebook_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    source = Path(notebook_path).resolve()
    output = Path(output_path).resolve()
    notebook = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict):
        raise ValueError("Notebook root must be a JSON object")
    reruns = _extract_reruns(notebook)
    if reruns != EXPECTED_RERUNS:
        raise ValueError("Notebook rerun arrays differ from the locked measurement")
    displayed_comparisons = _extract_comparison_table(notebook)

    a = reruns["A"]
    b = reruns["B"]
    a_sd = statistics.stdev(a)
    b_sd = statistics.stdev(b)
    variance_ratio = max(a_sd * a_sd, b_sd * b_sd) / min(a_sd * a_sd, b_sd * b_sd)
    f_critical_one_sided = float(f_distribution.ppf(0.95, 3, 3))
    f_critical_two_sided_upper = float(f_distribution.ppf(0.975, 3, 3))
    expected_two_draw_range = 2.0 * SIGMA_USED_BY_NOTEBOOK / math.sqrt(math.pi)
    se = {
        "single_vs_single": SIGMA_USED_BY_NOTEBOOK * math.sqrt(2.0),
        "single_vs_four_mean": SIGMA_USED_BY_NOTEBOOK * math.sqrt(1.0 + 1.0 / 4.0),
    }
    comparison_rows = [
        {
            "comparison": name,
            "difference_ft": difference,
            "standard_error_exact": se[comparison_type],
            "standard_error_notebook_rounded": round(se[comparison_type], 3),
            "z_exact": difference / se[comparison_type],
            "z_notebook_rounded": round(difference / se[comparison_type], 1),
            "clears_two_se": bool(difference / se[comparison_type] >= 2.0),
        }
        for name, difference, comparison_type in COMPARISONS
    ]
    expected_display = tuple(
        (
            row["comparison"],
            row["difference_ft"],
            row["standard_error_notebook_rounded"],
            row["z_notebook_rounded"],
            "real" if row["clears_two_se"] else "nothing",
        )
        for row in comparison_rows
    )
    checks = {
        "rerun_arrays_match_notebook": True,
        "comparison_table_matches_recomputation": (
            displayed_comparisons == expected_display
        ),
        "a_sample_sd_rounds_to_0.037": round(a_sd, 3) == 0.037,
        "b_sample_sd_rounds_to_0.027": round(b_sd, 3) == 0.027,
        "a_range_rounds_to_0.085": round(max(a) - min(a), 3) == 0.085,
        "b_range_rounds_to_0.063": round(max(b) - min(b), 3) == 0.063,
        "variance_ratio_rounds_to_1.86": round(variance_ratio, 2) == 1.86,
        "one_sided_f_critical_rounds_to_9.3": (round(f_critical_one_sided, 1) == 9.3),
        "variance_ratio_below_one_sided_critical": (
            variance_ratio < f_critical_one_sided
        ),
        "expected_two_draw_range_rounds_to_0.042": (
            round(expected_two_draw_range, 3) == 0.042
        ),
        "single_vs_single_se_rounds_to_0.052": (
            round(se["single_vs_single"], 3) == 0.052
        ),
        "single_vs_four_mean_se_rounds_to_0.041": (
            round(se["single_vs_four_mean"], 3) == 0.041
        ),
        "comparison_z_rounding_matches_notebook": (
            [row["z_notebook_rounded"] for row in comparison_rows]
            == [1.5, 0.2, 0.7, 2.4, 3.7, 4.3]
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Noise-floor notebook audit failed: {failed}")

    report = {
        "schema_version": "rogii_noise_floor_reaudit_v1",
        "passed": True,
        "scope": (
            "Independent recomputation of the unchanged-rerun measurement, "
            "F comparison, standard errors, and z values in notebook cells 1-5. "
            "The train-label oracle sections were not executed or used as model evidence."
        ),
        "notebook": str(source),
        "notebook_sha256": _sha256(source),
        "reruns": {name: list(values) for name, values in reruns.items()},
        "calculations": {
            "a_mean": statistics.mean(a),
            "a_range": max(a) - min(a),
            "a_sample_sd": a_sd,
            "b_mean": statistics.mean(b),
            "b_range": max(b) - min(b),
            "b_sample_sd": b_sd,
            "variance_ratio": variance_ratio,
            "f_critical_one_sided_95_df3_df3": f_critical_one_sided,
            "f_critical_two_sided_upper_975_df3_df3": (f_critical_two_sided_upper),
            "expected_two_draw_range_at_sigma_0.037": expected_two_draw_range,
            "standard_errors": se,
            "comparisons": comparison_rows,
        },
        "checks": checks,
        "interpretation": {
            "supported": (
                "A and B are consistent with a rerun standard deviation of order "
                "0.03 ft; single-score differences near this scale are not reliable."
            ),
            "not_supported": (
                "The notebook noise estimate is not an OOF confidence interval and "
                "does not establish hidden-test domain stability."
            ),
            "f_test_note": (
                "The notebook's 9.3 threshold is the one-sided 95% F(3,3) "
                "critical value. The two-sided 5% upper critical value is also "
                "reported; either threshold leaves the qualitative conclusion unchanged."
            ),
        },
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("results/measure-your-noise-floor-before-believing-a-lever.ipynb"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/noise_floor_reaudit_20260729.json"),
    )
    args = parser.parse_args()
    report = audit_noise_floor_notebook(args.notebook, args.output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "notebook_sha256": report["notebook_sha256"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
