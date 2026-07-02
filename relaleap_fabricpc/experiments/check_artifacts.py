"""Inspect existing RelaLeap FabricPC comparison artifacts without rerunning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relaleap_fabricpc.experiments.compare import compare_comparison_to_baseline


REQUIRED_ARTIFACTS = ("summary.json", "metrics.csv", "notes.md")


def check_comparison_artifacts(
    comparison_dir: Path,
    *,
    require_baseline_comparison: bool = False,
    baseline_reference: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Check a comparison artifact tree and return a compact pass/fail report."""

    checks: list[dict[str, Any]] = []
    for name in REQUIRED_ARTIFACTS:
        checks.append(_artifact_check(comparison_dir / name, f"comparison.{name}"))

    summary = _read_json(comparison_dir / "summary.json", checks, "comparison.summary")
    runs = summary.get("runs") if isinstance(summary, dict) else []
    run_reports = []
    if isinstance(runs, list):
        for entry in runs:
            if not isinstance(entry, dict):
                continue
            run_reports.append(_run_artifact_report(comparison_dir, entry, checks))
    checks.extend(
        check
        for report in run_reports
        for check in report["artifacts"]
    )

    baseline_path = comparison_dir / "baseline_comparison.json"
    baseline_check = _artifact_check(
        baseline_path,
        "comparison.baseline_comparison.json",
    )
    baseline = None
    if baseline_check["exists"]:
        checks.append(baseline_check)
        baseline = _read_json(
            baseline_path,
            checks,
            "comparison.baseline_comparison",
        )
    elif require_baseline_comparison:
        checks.append(baseline_check)

    baseline_reference_comparison = None
    if baseline_reference is not None:
        baseline_reference_comparison = _baseline_reference_comparison(
            summary,
            baseline_reference,
            checks,
        )

    verdict = summary.get("verdict") if isinstance(summary, dict) else {}
    if not isinstance(verdict, dict):
        verdict = {}
    phase0_passed = (
        verdict.get("invariants_passed") if isinstance(verdict, dict) else None
    )
    phase0_failed = (
        verdict.get("failed_invariants", []) if isinstance(verdict, dict) else []
    )
    artifact_invariants_passed = (
        verdict.get("artifact_invariants_passed")
        if isinstance(verdict, dict)
        else None
    )
    artifact_invariants_failed = (
        verdict.get("failed_artifact_invariants", [])
        if isinstance(verdict, dict)
        else []
    )
    acceptance = (
        verdict.get("hep_alpha_acceptance", {}) if isinstance(verdict, dict) else {}
    )
    accepted_alpha = (
        acceptance.get("accepted_alpha") if isinstance(acceptance, dict) else None
    )

    failures = _artifact_failures(checks)
    failures.extend(
        failure
        for report in run_reports
        for failure in report["summary_failures"]
    )
    if isinstance(summary, dict):
        if summary.get("status") != "ok":
            failures.append(
                {
                    "field": "comparison.status",
                    "expected": "ok",
                    "actual": summary.get("status"),
                }
            )
        if verdict.get("status") != "pass":
            failures.append(
                {
                    "field": "comparison.verdict.status",
                    "expected": "pass",
                    "actual": verdict.get("status"),
                }
            )
        if phase0_passed is not True:
            failures.append(
                {
                    "field": "comparison.verdict.invariants_passed",
                    "expected": True,
                    "actual": phase0_passed,
                }
            )
        if artifact_invariants_passed is not True:
            failures.append(
                {
                    "field": "comparison.verdict.artifact_invariants_passed",
                    "expected": True,
                    "actual": artifact_invariants_passed,
                }
            )
    if isinstance(baseline, dict) and baseline.get("status") != "pass":
        failures.append(
            {
                "field": "baseline_comparison.status",
                "expected": "pass",
                "actual": baseline.get("status"),
            }
        )
    failures.extend(_baseline_reference_failures(baseline_reference_comparison))

    report = {
        "status": "pass" if not failures else "fail",
        "comparison_dir": str(comparison_dir),
        "artifacts": checks,
        "summary_status": summary.get("status") if isinstance(summary, dict) else None,
        "verdict_status": verdict.get("status") if isinstance(verdict, dict) else None,
        "phase0_invariants": {
            "passed": phase0_passed,
            "count": verdict.get("invariant_count") if isinstance(verdict, dict) else None,
            "failed_count": len(phase0_failed) if isinstance(phase0_failed, list) else None,
        },
        "artifact_invariants": {
            "passed": artifact_invariants_passed,
            "count": (
                verdict.get("artifact_invariant_count")
                if isinstance(verdict, dict)
                else None
            ),
            "failed_count": (
                len(artifact_invariants_failed)
                if isinstance(artifact_invariants_failed, list)
                else None
            ),
        },
        "hep_alpha_acceptance": {
            "status": acceptance.get("status") if isinstance(acceptance, dict) else None,
            "accepted_alpha": accepted_alpha,
        },
        "baseline_comparison": _baseline_summary(baseline),
        "baseline_reference_comparison": _baseline_reference_summary(
            baseline_reference_comparison
        ),
        "runs": run_reports,
        "failures": failures,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _run_artifact_report(
    comparison_dir: Path,
    entry: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    config_path = entry.get("config_path", "")
    experiment_id = entry.get("experiment_id")
    run_dir = comparison_dir / "runs" / Path(str(config_path)).stem
    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path, checks, f"run.{experiment_id}.summary")
    summary_failures = (
        _run_summary_failures(summary, experiment_id) if summary_path.is_file() else []
    )
    return {
        "experiment_id": experiment_id,
        "run_dir": str(run_dir),
        "artifacts": [
            _artifact_check(run_dir / name, f"run.{experiment_id}.{name}")
            for name in REQUIRED_ARTIFACTS
        ],
        "summary_status": summary.get("status") if isinstance(summary, dict) else None,
        "artifact_invariants": (
            summary.get("artifact_invariants") if isinstance(summary, dict) else None
        ),
        "summary_failures": summary_failures,
    }


def _run_summary_failures(
    summary: dict[str, Any],
    experiment_id: Any,
) -> list[dict[str, Any]]:
    failures = []
    if summary.get("experiment_id") != experiment_id:
        failures.append(
            {
                "field": f"run.{experiment_id}.summary.experiment_id",
                "expected": experiment_id,
                "actual": summary.get("experiment_id"),
            }
        )
    if summary.get("status") != "ok":
        failures.append(
            {
                "field": f"run.{experiment_id}.summary.status",
                "expected": "ok",
                "actual": summary.get("status"),
            }
        )
    artifact_invariants = summary.get("artifact_invariants")
    if not isinstance(artifact_invariants, dict):
        failures.append(
            {
                "field": f"run.{experiment_id}.artifact_invariants",
                "expected": "summary_json/metrics_csv/notes_md contract",
                "actual": artifact_invariants,
            }
        )
        return failures
    for key in ("summary_json", "metrics_csv", "notes_md"):
        if artifact_invariants.get(key) is not True:
            failures.append(
                {
                    "field": f"run.{experiment_id}.artifact_invariants.{key}",
                    "expected": True,
                    "actual": artifact_invariants.get(key),
                }
            )
    return failures


def _artifact_check(path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "exists": path.is_file(),
    }


def _artifact_failures(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for check in checks:
        if not check["exists"]:
            failures.append(
                {
                    "field": check["label"],
                    "expected": "file exists",
                    "actual": "missing",
                    "path": check["path"],
                }
            )
        if check.get("valid_json") is False:
            failures.append(
                {
                    "field": check["label"],
                    "expected": "valid JSON object",
                    "actual": check.get("error", "invalid JSON"),
                    "path": check["path"],
                }
            )
    return failures


def _read_json(
    path: Path,
    checks: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks.append(
            {
                "label": label,
                "path": str(path),
                "exists": True,
                "valid_json": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return {}
    if not isinstance(loaded, dict):
        checks.append(
            {
                "label": label,
                "path": str(path),
                "exists": True,
                "valid_json": False,
                "error": "expected JSON object",
            }
        )
        return {}
    return loaded


def _baseline_summary(baseline: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        return {
            "present": False,
            "status": None,
            "mismatch_count": None,
        }
    mismatches = baseline.get("mismatches", [])
    return {
        "present": True,
        "status": baseline.get("status"),
        "mismatch_count": len(mismatches) if isinstance(mismatches, list) else None,
    }


def _baseline_reference_comparison(
    summary: dict[str, Any],
    baseline_reference: Path,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_check = _artifact_check(baseline_reference, "baseline_reference")
    checks.append(baseline_check)
    if not baseline_check["exists"]:
        return {
            "status": "fail",
            "baseline_path": str(baseline_reference),
            "mismatches": [
                {
                    "field": "baseline_reference",
                    "reference": "file exists",
                    "candidate": "missing",
                }
            ],
        }

    reference = _read_json(baseline_reference, checks, "baseline_reference")
    if not reference:
        return {
            "status": "fail",
            "baseline_path": str(baseline_reference),
            "mismatches": [
                {
                    "field": "baseline_reference",
                    "reference": "valid baseline JSON object",
                    "candidate": "invalid",
                }
            ],
        }
    try:
        return compare_comparison_to_baseline(
            summary,
            reference,
            baseline_path=baseline_reference,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "fail",
            "baseline_path": str(baseline_reference),
            "mismatches": [
                {
                    "field": "baseline_reference.schema",
                    "reference": "compatible Phase 0 comparison baseline",
                    "candidate": f"{type(exc).__name__}: {exc}",
                }
            ],
        }


def _baseline_reference_failures(
    baseline_reference_comparison: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not baseline_reference_comparison:
        return []
    if baseline_reference_comparison.get("status") == "pass":
        return []
    failures = []
    for mismatch in baseline_reference_comparison.get("mismatches", []):
        if mismatch.get("field") == "baseline_reference":
            continue
        failures.append(
            {
                "field": f"baseline_reference.{mismatch.get('field')}",
                "expected": mismatch.get("reference"),
                "actual": mismatch.get("candidate"),
            }
        )
    return failures


def _baseline_reference_summary(
    baseline_reference_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    if not baseline_reference_comparison:
        return {
            "present": False,
            "status": None,
            "mismatch_count": None,
        }
    mismatches = baseline_reference_comparison.get("mismatches", [])
    return {
        "present": True,
        "status": baseline_reference_comparison.get("status"),
        "baseline_path": baseline_reference_comparison.get("baseline_path"),
        "mismatch_count": len(mismatches) if isinstance(mismatches, list) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an existing RelaLeap FabricPC comparison artifact directory."
    )
    parser.add_argument(
        "--comparison-dir",
        default=Path("results/comparisons/phase0"),
        type=Path,
        help="Comparison output directory to inspect without rerunning experiments.",
    )
    parser.add_argument(
        "--require-baseline-comparison",
        action="store_true",
        help="Fail if baseline_comparison.json is missing.",
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        help=(
            "Optional checked-in baseline to compare the existing summary.json "
            "against without rerunning experiments."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the artifact check JSON report.",
    )
    args = parser.parse_args()
    report = check_comparison_artifacts(
        args.comparison_dir,
        require_baseline_comparison=args.require_baseline_comparison,
        baseline_reference=args.baseline_reference,
        out_path=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
