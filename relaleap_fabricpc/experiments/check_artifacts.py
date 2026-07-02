"""Inspect completed Phase 0 comparison artifacts without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relaleap_fabricpc.experiments.compare import compare_comparison_to_baseline


def check_comparison_artifacts(
    comparison_dir: Path,
    *,
    baseline_reference: Path | None = None,
    require_baseline_comparison: bool = False,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    summary_path = comparison_dir / "summary.json"
    metrics_path = comparison_dir / "metrics.csv"
    notes_path = comparison_dir / "notes.md"

    for label, path in [
        ("comparison.summary.json", summary_path),
        ("comparison.metrics.csv", metrics_path),
        ("comparison.notes.md", notes_path),
    ]:
        if not path.is_file():
            failures.append(
                {
                    "field": label,
                    "expected": "file exists",
                    "actual": "missing",
                    "path": str(path),
                }
            )

    summary = _read_json(summary_path)
    if summary is None:
        failures.append(
            {
                "field": "comparison.summary",
                "expected": "valid JSON object",
                "actual": "missing or invalid",
            }
        )
        verdict = {}
    else:
        verdict = summary.get("verdict") or {}

    if verdict.get("status") != "pass":
        failures.append(
            {
                "field": "comparison.verdict.status",
                "expected": "pass",
                "actual": verdict.get("status"),
            }
        )
    if verdict.get("invariants_passed") is not True:
        failures.append(
            {
                "field": "comparison.verdict.invariants_passed",
                "expected": True,
                "actual": verdict.get("invariants_passed"),
            }
        )
    if verdict.get("artifact_invariants_passed") is not True:
        failures.append(
            {
                "field": "comparison.verdict.artifact_invariants_passed",
                "expected": True,
                "actual": verdict.get("artifact_invariants_passed"),
            }
        )

    baseline_path = comparison_dir / "baseline_comparison.json"
    baseline_comparison: dict[str, Any] = {"present": baseline_path.is_file()}
    if require_baseline_comparison and not baseline_path.is_file():
        failures.append(
            {
                "field": "comparison.baseline_comparison.json",
                "expected": "file exists",
                "actual": "missing",
                "path": str(baseline_path),
            }
        )
    if baseline_path.is_file():
        baseline_comparison = _read_json(baseline_path) or {
            "present": True,
            "status": "invalid",
        }
        baseline_comparison["present"] = True
        if baseline_comparison.get("status") != "pass":
            failures.append(
                {
                    "field": "baseline_comparison.status",
                    "expected": "pass",
                    "actual": baseline_comparison.get("status"),
                }
            )

    baseline_reference_comparison = None
    if baseline_reference is not None and summary is not None:
        reference = _read_json(baseline_reference)
        if reference is None:
            failures.append(
                {
                    "field": "baseline_reference",
                    "expected": "valid JSON object",
                    "actual": "missing or invalid",
                    "path": str(baseline_reference),
                }
            )
        else:
            baseline_reference_comparison = compare_comparison_to_baseline(
                summary,
                reference,
                baseline_path=baseline_reference,
            )
            if baseline_reference_comparison["status"] != "pass":
                failures.extend(baseline_reference_comparison["mismatches"])

    report = {
        "status": "pass" if not failures else "fail",
        "comparison_dir": str(comparison_dir),
        "summary_status": None if summary is None else summary.get("status"),
        "verdict_status": verdict.get("status"),
        "phase0_invariants": {
            "passed": verdict.get("invariants_passed"),
            "count": verdict.get("invariant_count", 0),
            "failed": verdict.get("failed_invariants", []),
        },
        "artifact_invariants": {
            "passed": verdict.get("artifact_invariants_passed"),
            "count": verdict.get("artifact_invariant_count", 0),
            "failed": verdict.get("failed_artifact_invariants", []),
        },
        "hep_alpha_acceptance": verdict.get("hep_alpha_acceptance"),
        "baseline_comparison": baseline_comparison,
        "baseline_reference_comparison": baseline_reference_comparison,
        "failures": failures,
    }
    return report


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--baseline-reference", type=Path)
    parser.add_argument("--require-baseline-comparison", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check_comparison_artifacts(
        args.comparison_dir,
        baseline_reference=args.baseline_reference,
        require_baseline_comparison=args.require_baseline_comparison,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
