"""Run and compare Phase 0 smoke configurations."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from relaleap_fabricpc.experiments.run import run


DEFAULT_CONFIGS = [
    Path("configs/char_smoke.yaml"),
    Path("configs/char_smoke_pc.yaml"),
    Path("configs/char_smoke_hep.yaml"),
]
DEFAULT_HEP_MAX_LOGIT_DELTA = 0.1
DEFAULT_HEP_MIN_LOSS_IMPROVEMENT = 0.0
BASELINE_SCHEMA_VERSION = 1
REQUIRED_ARTIFACT_INVARIANTS = ("summary_json", "metrics_csv", "notes_md")


def run_comparison(
    config_paths: list[Path],
    out_dir: Path,
    *,
    hep_max_logit_delta: float = DEFAULT_HEP_MAX_LOGIT_DELTA,
    hep_min_loss_improvement: float = DEFAULT_HEP_MIN_LOSS_IMPROVEMENT,
) -> dict[str, Any]:
    if len(config_paths) < 2:
        raise ValueError("comparison requires at least two config paths")
    if hep_max_logit_delta < 0.0:
        raise ValueError("hep_max_logit_delta must be non-negative")
    if hep_min_loss_improvement < 0.0:
        raise ValueError("hep_min_loss_improvement must be non-negative")

    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = out_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    entries = []
    combined_rows = []
    for config_path in config_paths:
        run_dir = run_root / config_path.stem
        summary = run(config_path, run_dir)
        metric_rows = _read_metrics(run_dir / "metrics.csv")
        entry = _comparison_entry(config_path, run_dir, summary, metric_rows)
        entries.append(entry)
        combined_rows.extend(_combined_rows(entry, metric_rows))

    status = "ok" if all(entry["status"] == "ok" for entry in entries) else "failed"
    verdict = _comparison_verdict(
        entries,
        status,
        hep_max_logit_delta=hep_max_logit_delta,
        hep_min_loss_improvement=hep_min_loss_improvement,
    )
    comparison = {
        "status": status,
        "out_dir": str(out_dir),
        "runtime_seconds": round(time.time() - start, 4),
        "loss_scale_note": (
            "Residual objectives may use different loss scales; compare each "
            "trajectory against its own initial loss."
        ),
        "verdict": verdict,
        "runs": entries,
    }
    _write_metrics(out_dir / "metrics.csv", combined_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_notes(out_dir / "notes.md", comparison)
    return comparison


def write_comparison_baseline(path: Path, comparison: dict[str, Any]) -> dict[str, Any]:
    baseline = _comparison_baseline(comparison)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return baseline


def compare_to_baseline(
    comparison: dict[str, Any],
    baseline_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    reference = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = compare_comparison_to_baseline(comparison, reference, baseline_path=baseline_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def compare_comparison_to_baseline(
    comparison: dict[str, Any],
    reference: dict[str, Any],
    *,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    candidate = _comparison_baseline(comparison)
    mismatches = _baseline_mismatches(reference, candidate)
    result = {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "reference": reference,
        "candidate": candidate,
    }
    if baseline_path is not None:
        result["baseline_path"] = str(baseline_path)
    return result


def _comparison_baseline(comparison: dict[str, Any]) -> dict[str, Any]:
    verdict = comparison["verdict"]
    acceptance = verdict["hep_alpha_acceptance"]
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "comparison_status": comparison["status"],
        "verdict_status": verdict["status"],
        "config_paths": [entry["config_path"] for entry in comparison["runs"]],
        "runs": [
            {
                "experiment_id": entry["experiment_id"],
                "config_path": entry["config_path"],
                "residual_objective": entry["residual_objective"],
                "status": entry["status"],
                "training_steps": entry["training_steps"],
                "invariant_count": len(entry.get("invariants") or {}),
                "artifact_invariants": _baseline_run_artifact_invariants(entry),
                "final_residual_loss": entry["final_residual_loss"],
            }
            for entry in comparison["runs"]
        ],
        "phase0_invariants": {
            "passed": verdict["invariants_passed"],
            "count": verdict["invariant_count"],
            "failed": verdict["failed_invariants"],
        },
        "artifact_invariants": {
            "passed": verdict["artifact_invariants_passed"],
            "count": verdict["artifact_invariant_count"],
            "failed": verdict["failed_artifact_invariants"],
        },
        "hep": {
            "best_alpha_by_loss": _baseline_hep_alpha(verdict["best_hep_alpha_by_loss"]),
            "acceptance": {
                "status": acceptance["status"],
                "max_logit_delta_from_ordinary": acceptance[
                    "max_logit_delta_from_ordinary"
                ],
                "min_loss_improvement_from_alpha0": acceptance[
                    "min_loss_improvement_from_alpha0"
                ],
                "candidate_count": acceptance["candidate_count"],
                "rejected_count": acceptance["rejected_count"],
                "baseline_alpha0": _baseline_hep_alpha(acceptance["baseline_alpha0"]),
                "accepted_alpha": _baseline_hep_alpha(acceptance["accepted_alpha"]),
            },
        },
    }


def _baseline_mismatches(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    prefix: str = "baseline",
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    keys = sorted(set(reference) | set(candidate))
    for key in keys:
        field = f"{prefix}.{key}"
        expected = reference.get(key)
        actual = candidate.get(key)
        if isinstance(expected, dict) and isinstance(actual, dict):
            mismatches.extend(_baseline_mismatches(expected, actual, prefix=field))
        elif expected != actual:
            mismatches.append({"field": field, "expected": expected, "actual": actual})
    return mismatches


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _comparison_entry(
    config_path: Path,
    run_dir: Path,
    summary: dict[str, Any],
    metric_rows: list[dict[str, str]],
) -> dict[str, Any]:
    initial = _first_metric(metric_rows, "residual_loss")
    final = _last_metric(metric_rows, "residual_loss")
    loss_delta = None if initial is None or final is None else final - initial
    loss_ratio = None
    if initial not in {None, 0.0} and final is not None:
        loss_ratio = final / initial
    phase0 = summary.get("phase0") or {}
    return {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "experiment_id": summary.get("experiment_id"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "residual_objective": phase0.get("residual_objective", ""),
        "dataset": phase0.get("dataset"),
        "num_columns": phase0.get("num_columns"),
        "atoms_per_column": phase0.get("atoms_per_column"),
        "top_k": phase0.get("top_k"),
        "support_router": phase0.get("support_router", "linear"),
        "contextual_router_hidden_dim": phase0.get("contextual_router_hidden_dim"),
        "training_steps": phase0.get("training_steps"),
        "initial_residual_loss": initial,
        "final_residual_loss": final,
        "residual_loss_delta": loss_delta,
        "residual_loss_ratio": loss_ratio,
        "base_loss": phase0.get("base_loss"),
        "zero_init_loss": phase0.get("zero_init_loss"),
        "pinned_support": phase0.get("pinned_support", False),
        "support_stress": phase0.get("support_stress", False),
        "support_stress_preset": phase0.get("support_stress_preset", False),
        "hep_update_clip_norm": phase0.get("hep_update_clip_norm"),
        "hep_settling_objective": phase0.get("hep_settling_objective", "residual_adapter"),
        "support_instability": phase0.get("support_instability") or {},
        "support_audit": phase0.get("support_audit") or {},
        "hep_alpha_sweep": phase0.get("hep_alpha_sweep") or [],
        "invariants": phase0.get("invariants") or {},
        "artifact_invariants": summary.get("artifact_invariants") or {},
    }


def _comparison_verdict(
    entries: list[dict[str, Any]],
    status: str,
    *,
    hep_max_logit_delta: float = DEFAULT_HEP_MAX_LOGIT_DELTA,
    hep_min_loss_improvement: float = DEFAULT_HEP_MIN_LOSS_IMPROVEMENT,
) -> dict[str, Any]:
    failed_invariants = []
    invariant_count = 0
    failed_artifact_invariants = []
    artifact_invariant_count = 0
    for entry in entries:
        invariants = entry.get("invariants") or {}
        invariant_count += len(invariants)
        for name, value in sorted(invariants.items()):
            if not value:
                failed_invariants.append(
                    {"experiment_id": entry["experiment_id"], "invariant": name}
                )
        artifact_invariants = entry.get("artifact_invariants")
        artifact_invariant_count += len(REQUIRED_ARTIFACT_INVARIANTS)
        for name in REQUIRED_ARTIFACT_INVARIANTS:
            if (
                not isinstance(artifact_invariants, dict)
                or artifact_invariants.get(name) is not True
            ):
                failed_artifact_invariants.append(
                    {"experiment_id": entry["experiment_id"], "artifact": name}
                )

    best_hep = _best_hep_alpha(entries)
    hep_acceptance = _hep_alpha_acceptance(
        entries,
        max_logit_delta=hep_max_logit_delta,
        min_loss_improvement=hep_min_loss_improvement,
    )
    invariants_passed = bool(invariant_count) and not failed_invariants
    artifact_invariants_passed = (
        bool(artifact_invariant_count) and not failed_artifact_invariants
    )
    verdict_status = (
        "pass"
        if status == "ok" and invariants_passed and artifact_invariants_passed
        else "fail"
    )
    return {
        "status": verdict_status,
        "invariants_passed": invariants_passed,
        "invariant_count": invariant_count,
        "failed_invariants": failed_invariants,
        "artifact_invariants_passed": artifact_invariants_passed,
        "artifact_invariant_count": artifact_invariant_count,
        "failed_artifact_invariants": failed_artifact_invariants,
        "best_hep_alpha_by_loss": best_hep,
        "hep_alpha_acceptance": hep_acceptance,
    }


def _best_hep_alpha(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for entry in entries:
        for sweep_entry in entry.get("hep_alpha_sweep") or []:
            loss = sweep_entry.get("loss")
            if loss is None:
                continue
            candidates.append(
                {
                    "experiment_id": entry["experiment_id"],
                    "alpha": float(sweep_entry["alpha"]),
                    "loss": float(loss),
                    "max_logit_delta_from_ordinary": float(
                        sweep_entry["max_logit_delta_from_ordinary"]
                    ),
                }
            )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate["loss"])


def _hep_alpha_acceptance(
    entries: list[dict[str, Any]],
    *,
    max_logit_delta: float,
    min_loss_improvement: float,
) -> dict[str, Any]:
    baselines = []
    candidates = []
    for entry in entries:
        sweep = entry.get("hep_alpha_sweep") or []
        baseline = _alpha0_baseline(entry, sweep)
        if baseline is not None:
            baselines.append(baseline)
        for sweep_entry in sweep:
            alpha = float(sweep_entry["alpha"])
            loss = sweep_entry.get("loss")
            if alpha == 0.0 or loss is None or baseline is None:
                continue
            candidate = {
                "experiment_id": entry["experiment_id"],
                "alpha": alpha,
                "loss": float(loss),
                "loss_improvement_from_alpha0": baseline["loss"] - float(loss),
                "max_logit_delta_from_ordinary": float(
                    sweep_entry["max_logit_delta_from_ordinary"]
                ),
                "alpha0_loss": baseline["loss"],
            }
            candidate["accepted"] = (
                candidate["loss_improvement_from_alpha0"] > min_loss_improvement
                and candidate["max_logit_delta_from_ordinary"] <= max_logit_delta
            )
            candidates.append(candidate)

    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    accepted_alpha = None
    if accepted_candidates:
        accepted_alpha = min(accepted_candidates, key=lambda candidate: candidate["loss"])
    status = "accepted" if accepted_alpha else "no_accepted_alpha"
    if not candidates:
        status = "no_nonzero_hep_candidates"

    return {
        "status": status,
        "max_logit_delta_from_ordinary": max_logit_delta,
        "min_loss_improvement_from_alpha0": min_loss_improvement,
        "baseline_alpha0": (
            min(baselines, key=lambda baseline: baseline["loss"]) if baselines else None
        ),
        "accepted_alpha": accepted_alpha,
        "candidate_count": len(candidates),
        "rejected_count": len(candidates) - len(accepted_candidates),
        "candidates": sorted(
            candidates,
            key=lambda candidate: (candidate["experiment_id"], candidate["alpha"]),
        ),
    }


def _alpha0_baseline(
    entry: dict[str, Any],
    sweep: list[dict[str, Any]],
) -> dict[str, Any] | None:
    alpha0_entries = [
        sweep_entry
        for sweep_entry in sweep
        if float(sweep_entry.get("alpha", -1.0)) == 0.0
        and sweep_entry.get("loss") is not None
    ]
    if not alpha0_entries:
        return None
    baseline = min(alpha0_entries, key=lambda sweep_entry: float(sweep_entry["loss"]))
    return {
        "experiment_id": entry["experiment_id"],
        "alpha": 0.0,
        "loss": float(baseline["loss"]),
        "max_logit_delta_from_ordinary": float(
            baseline["max_logit_delta_from_ordinary"]
        ),
    }


def _combined_rows(
    entry: dict[str, Any],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    combined = []
    for row in rows:
        enriched = dict(row)
        enriched["run_dir"] = entry["run_dir"]
        enriched["config_path"] = entry["config_path"]
        combined.append(enriched)
    return combined


def _first_metric(rows: list[dict[str, str]], field: str) -> float | None:
    for row in rows:
        parsed = _parse_float(row.get(field))
        if parsed is not None:
            return parsed
    return None


def _last_metric(rows: list[dict[str, str]], field: str) -> float | None:
    for row in reversed(rows):
        parsed = _parse_float(row.get(field))
        if parsed is not None:
            return parsed
    return None


def _parse_float(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, comparison: dict[str, Any]) -> None:
    verdict = comparison["verdict"]
    lines = [
        "# Phase 0 Comparison",
        "",
        f"- Status: `{comparison['status']}`",
        f"- Verdict: `{verdict['status']}`",
        f"- Phase 0 invariants passed: `{verdict['invariants_passed']}`",
        f"- Artifact invariants passed: `{verdict['artifact_invariants_passed']}`",
        f"- Best HEP alpha by loss: `{verdict['best_hep_alpha_by_loss']}`",
        f"- HEP alpha acceptance: `{verdict['hep_alpha_acceptance']['status']}`",
        "",
        "## Runs",
        "",
    ]
    for entry in comparison["runs"]:
        lines.append(
            "- "
            f"`{entry['experiment_id']}`: status `{entry['status']}`, "
            f"objective `{entry['residual_objective']}`, "
            f"final residual loss `{entry['final_residual_loss']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_run_artifact_invariants(entry: dict[str, Any]) -> dict[str, Any]:
    invariants = entry.get("artifact_invariants") or {}
    failed = [
        name
        for name in REQUIRED_ARTIFACT_INVARIANTS
        if invariants.get(name) is not True
    ]
    return {
        "passed": not failed,
        "count": len(REQUIRED_ARTIFACT_INVARIANTS),
        "failed": failed,
    }


def _baseline_hep_alpha(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    keys = [
        "experiment_id",
        "alpha",
        "loss",
        "max_logit_delta_from_ordinary",
        "loss_improvement_from_alpha0",
    ]
    return {key: entry[key] for key in keys if key in entry}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase 0 comparison.")
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        type=Path,
        help="Config path. May be passed more than once.",
    )
    parser.add_argument("--out", default=Path("results/comparisons/phase0"), type=Path)
    parser.add_argument("--baseline-out", type=Path)
    parser.add_argument("--baseline-reference", type=Path)
    parser.add_argument(
        "--hep-max-logit-delta",
        type=float,
        default=DEFAULT_HEP_MAX_LOGIT_DELTA,
    )
    parser.add_argument(
        "--hep-min-loss-improvement",
        type=float,
        default=DEFAULT_HEP_MIN_LOSS_IMPROVEMENT,
    )
    args = parser.parse_args()
    comparison = run_comparison(
        args.configs or DEFAULT_CONFIGS,
        args.out,
        hep_max_logit_delta=args.hep_max_logit_delta,
        hep_min_loss_improvement=args.hep_min_loss_improvement,
    )
    if args.baseline_out is not None:
        write_comparison_baseline(args.baseline_out, comparison)
    if args.baseline_reference is not None:
        result = compare_to_baseline(
            comparison,
            args.baseline_reference,
            args.out / "baseline_comparison.json",
        )
        if result["status"] != "pass":
            print(json.dumps(comparison, indent=2, sort_keys=True))
            raise SystemExit(1)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if comparison["status"] != "ok" or comparison["verdict"]["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
