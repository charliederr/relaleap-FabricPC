"""Compare Phase 0 FabricPC experiment runs from their standard artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from relaleap_fabricpc.experiments.run import run


DEFAULT_CONFIGS = (
    Path("configs/char_smoke.yaml"),
    Path("configs/char_smoke_pc.yaml"),
    Path("configs/char_smoke_hep.yaml"),
)
DEFAULT_HEP_MAX_LOGIT_DELTA = 0.1
DEFAULT_HEP_MIN_LOSS_IMPROVEMENT = 0.0
BASELINE_SCHEMA_VERSION = 3
REQUIRED_ARTIFACT_INVARIANTS = ("summary_json", "metrics_csv", "notes_md")


def run_comparison(
    config_paths: list[Path],
    out_dir: Path,
    *,
    hep_max_logit_delta: float = DEFAULT_HEP_MAX_LOGIT_DELTA,
    hep_min_loss_improvement: float = DEFAULT_HEP_MIN_LOSS_IMPROVEMENT,
) -> dict[str, Any]:
    """Run configs into sibling directories and write a compact comparison."""

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
    """Write a compact, stable baseline extracted from a comparison artifact."""

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
    """Compare current stable Phase 0 fields against a saved baseline."""

    reference = json.loads(baseline_path.read_text(encoding="utf-8"))
    result = compare_comparison_to_baseline(
        comparison,
        reference,
        baseline_path=baseline_path,
    )
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
    """Compare a comparison summary against an already-loaded baseline."""

    candidate = _comparison_baseline(comparison)
    mismatches = _baseline_mismatches(reference, candidate)
    result = {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "reference": _baseline_comparison_fields(reference),
        "candidate": _baseline_comparison_fields(candidate),
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
            "best_alpha_by_loss": _baseline_hep_alpha(
                verdict["best_hep_alpha_by_loss"]
            ),
            "acceptance": {
                "status": acceptance["status"],
                "max_logit_delta_from_ordinary": acceptance[
                    "max_logit_delta_from_ordinary"
                ],
                "min_loss_improvement_from_alpha0": acceptance[
                    "min_loss_improvement_from_alpha0"
                ],
                "baseline_alpha0": _baseline_hep_alpha(
                    acceptance["baseline_alpha0"]
                ),
                "accepted_alpha": _baseline_hep_alpha(
                    acceptance["accepted_alpha"],
                    include_improvement=True,
                ),
                "candidate_count": acceptance["candidate_count"],
                "rejected_count": acceptance["rejected_count"],
            },
        },
    }


def _baseline_mismatches(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    if reference.get("schema_version") != candidate.get("schema_version"):
        return [
            {
                "field": "schema_version",
                "reference": reference.get("schema_version"),
                "candidate": candidate.get("schema_version"),
            }
        ]
    checks = [
        ("comparison_status", reference["comparison_status"], candidate["comparison_status"]),
        ("verdict_status", reference["verdict_status"], candidate["verdict_status"]),
        ("config_paths", reference["config_paths"], candidate["config_paths"]),
        (
            "phase0_invariants.passed",
            reference["phase0_invariants"]["passed"],
            candidate["phase0_invariants"]["passed"],
        ),
        (
            "phase0_invariants.count",
            reference["phase0_invariants"]["count"],
            candidate["phase0_invariants"]["count"],
        ),
        (
            "phase0_invariants.failed",
            reference["phase0_invariants"]["failed"],
            candidate["phase0_invariants"]["failed"],
        ),
        (
            "artifact_invariants.passed",
            reference["artifact_invariants"]["passed"],
            candidate["artifact_invariants"]["passed"],
        ),
        (
            "artifact_invariants.count",
            reference["artifact_invariants"]["count"],
            candidate["artifact_invariants"]["count"],
        ),
        (
            "artifact_invariants.failed",
            reference["artifact_invariants"]["failed"],
            candidate["artifact_invariants"]["failed"],
        ),
        (
            "runs.artifact_invariants",
            _run_artifact_baseline_fields(reference),
            _run_artifact_baseline_fields(candidate),
        ),
        (
            "hep.acceptance.status",
            reference["hep"]["acceptance"]["status"],
            candidate["hep"]["acceptance"]["status"],
        ),
        (
            "hep.acceptance.accepted_alpha.alpha",
            _accepted_alpha_value(reference),
            _accepted_alpha_value(candidate),
        ),
    ]
    return [
        {"field": field, "reference": expected, "candidate": actual}
        for field, expected, actual in checks
        if expected != actual
    ]


def _baseline_comparison_fields(baseline: dict[str, Any]) -> dict[str, Any]:
    acceptance = baseline["hep"]["acceptance"]
    return {
        "schema_version": baseline.get("schema_version"),
        "comparison_status": baseline["comparison_status"],
        "verdict_status": baseline["verdict_status"],
        "config_paths": baseline["config_paths"],
        "phase0_invariants": baseline["phase0_invariants"],
        "artifact_invariants": baseline.get("artifact_invariants"),
        "run_artifact_invariants": _run_artifact_baseline_fields(baseline),
        "accepted_hep_alpha": acceptance["accepted_alpha"],
        "accepted_status": acceptance["status"],
    }


def _accepted_alpha_value(baseline: dict[str, Any]) -> float | None:
    accepted = baseline["hep"]["acceptance"]["accepted_alpha"]
    if accepted is None:
        return None
    return accepted["alpha"]


def _baseline_hep_alpha(
    entry: dict[str, Any] | None,
    *,
    include_improvement: bool = False,
) -> dict[str, Any] | None:
    if entry is None:
        return None
    baseline = {
        "experiment_id": entry["experiment_id"],
        "alpha": entry["alpha"],
        "loss": entry["loss"],
        "max_logit_delta_from_ordinary": entry["max_logit_delta_from_ordinary"],
    }
    if include_improvement:
        baseline["loss_improvement_from_alpha0"] = entry[
            "loss_improvement_from_alpha0"
        ]
    return baseline


def _baseline_run_artifact_invariants(entry: dict[str, Any]) -> dict[str, Any]:
    artifact_invariants = entry.get("artifact_invariants")
    failed = []
    for name in REQUIRED_ARTIFACT_INVARIANTS:
        if (
            not isinstance(artifact_invariants, dict)
            or artifact_invariants.get(name) is not True
        ):
            failed.append(name)
    return {
        "count": len(REQUIRED_ARTIFACT_INVARIANTS),
        "failed": failed,
        "passed": not failed,
    }


def _run_artifact_baseline_fields(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": entry["experiment_id"],
            "config_path": entry["config_path"],
            "artifact_invariants": entry.get("artifact_invariants"),
        }
        for entry in baseline["runs"]
    ]


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
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
        "hep_settling_objective": phase0.get(
            "hep_settling_objective",
            "residual_adapter",
        ),
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
                    {
                        "experiment_id": entry["experiment_id"],
                        "invariant": name,
                    }
                )
        artifact_invariants = entry.get("artifact_invariants")
        artifact_invariant_count += len(REQUIRED_ARTIFACT_INVARIANTS)
        for name in REQUIRED_ARTIFACT_INVARIANTS:
            if (
                not isinstance(artifact_invariants, dict)
                or artifact_invariants.get(name) is not True
            ):
                failed_artifact_invariants.append(
                    {
                        "experiment_id": entry["experiment_id"],
                        "artifact": name,
                    }
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

    accepted_candidates = [
        candidate for candidate in candidates if candidate["accepted"]
    ]
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
            key=lambda candidate: (
                candidate["experiment_id"],
                candidate["alpha"],
            ),
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
    metric_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    initial = entry["initial_residual_loss"]
    for row in metric_rows:
        residual_loss = _parse_float(row.get("residual_loss", ""))
        loss_delta = None
        if initial is not None and residual_loss is not None:
            loss_delta = residual_loss - initial
        rows.append(
            {
                "experiment_id": entry["experiment_id"],
                "config_path": entry["config_path"],
                "run_dir": entry["run_dir"],
                "residual_objective": entry["residual_objective"],
                "dataset": entry.get("dataset"),
                "num_columns": entry.get("num_columns"),
                "atoms_per_column": entry.get("atoms_per_column"),
                "top_k": entry.get("top_k"),
                "pinned_support": entry["pinned_support"],
                "support_stress": entry["support_stress"],
                "hep_update_clip_norm": entry.get("hep_update_clip_norm"),
                "hep_settling_objective": entry.get("hep_settling_objective"),
                "step": row.get("step", ""),
                "phase": row.get("phase", ""),
                "base_loss": row.get("base_loss", ""),
                "residual_loss": row.get("residual_loss", ""),
                "loss_delta_from_initial": _format_optional(loss_delta),
                "hep_alpha": row.get("hep_alpha", ""),
                "hep_loss": row.get("hep_loss", ""),
                "max_hep_logit_delta_from_ordinary": row.get(
                    "max_hep_logit_delta_from_ordinary", ""
                ),
                "hep_support_change_fraction": row.get(
                    "hep_support_change_fraction", ""
                ),
                "hep_pinned_vs_repicked_logit_delta": row.get(
                    "hep_pinned_vs_repicked_logit_delta", ""
                ),
                "status": row.get("status", entry["status"]),
            }
        )
    return rows


def _first_metric(rows: list[dict[str, str]], field: str) -> float | None:
    for row in rows:
        value = _parse_float(row.get(field, ""))
        if value is not None:
            return value
    return None


def _last_metric(rows: list[dict[str, str]], field: str) -> float | None:
    for row in reversed(rows):
        value = _parse_float(row.get(field, ""))
        if value is not None:
            return value
    return None


def _parse_float(value: str | None) -> float | None:
    if value in {"", None}:
        return None
    return float(value)


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment_id",
        "config_path",
        "run_dir",
        "residual_objective",
        "dataset",
        "num_columns",
        "atoms_per_column",
        "top_k",
        "pinned_support",
        "support_stress",
        "hep_update_clip_norm",
        "hep_settling_objective",
        "step",
        "phase",
        "base_loss",
        "residual_loss",
        "loss_delta_from_initial",
        "hep_alpha",
        "hep_loss",
        "max_hep_logit_delta_from_ordinary",
        "hep_support_change_fraction",
        "hep_pinned_vs_repicked_logit_delta",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_notes(path: Path, comparison: dict[str, Any]) -> None:
    verdict = comparison["verdict"]
    hep_acceptance = verdict["hep_alpha_acceptance"]
    lines = [
        "# RelaLeap Objective Comparison",
        "",
        "Command-driven comparison of Phase 0 residual objectives.",
        "",
        f"- Status: `{comparison['status']}`",
        f"- Verdict: `{verdict['status']}`",
        (
            f"- Phase 0 invariants: `{verdict['invariant_count']}` checked, "
            f"passed `{verdict['invariants_passed']}`"
        ),
        (
            f"- Artifact invariants: `{verdict['artifact_invariant_count']}` checked, "
            f"passed `{verdict['artifact_invariants_passed']}`"
        ),
        f"- HEP alpha acceptance: `{hep_acceptance['status']}`",
        f"- Loss scale note: {comparison['loss_scale_note']}",
        "",
        "## Runs",
        "",
        (
            "| Experiment | Objective | Pinned | Stress | Status | Initial loss | "
            "Final loss | Delta | Ratio | HEP clip | Support change | Pinned-vs-repicked |"
        ),
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in comparison["runs"]:
        support_instability = entry.get("support_instability") or {}
        row_template = (
            "| {experiment_id} | {objective} | {pinned} | {stress} | {status} | "
            "{initial} | {final} | {delta} | {ratio} | {hep_clip} | "
            "{support_change} | {pinned_vs_repicked} |"
        )
        lines.append(
            row_template.format(
                experiment_id=entry["experiment_id"],
                objective=entry["residual_objective"],
                pinned=entry.get("pinned_support"),
                stress=entry.get("support_stress"),
                status=entry["status"],
                initial=_format_note_metric(entry["initial_residual_loss"]),
                final=_format_note_metric(entry["final_residual_loss"]),
                delta=_format_note_metric(entry["residual_loss_delta"]),
                ratio=_format_note_metric(entry["residual_loss_ratio"]),
                hep_clip=_format_note_metric(entry.get("hep_update_clip_norm")),
                support_change=_format_note_metric(
                    support_instability.get("support_change_fraction")
                ),
                pinned_vs_repicked=_format_note_metric(
                    support_instability.get("pinned_vs_repicked_logit_delta")
                ),
            )
        )
    lines.extend(["", "## Artifacts", ""])
    for entry in comparison["runs"]:
        lines.append(f"- `{entry['experiment_id']}`: `{entry['run_dir']}`")
    hep_entries = [
        entry for entry in comparison["runs"] if entry.get("hep_alpha_sweep")
    ]
    if hep_entries:
        lines.extend(["", "## HEP Alpha Sweeps", ""])
        for entry in hep_entries:
            sweep = ", ".join(
                (
                    f"alpha {sweep_entry['alpha']}: "
                    f"loss {_format_note_metric(sweep_entry['loss'])}, "
                    "delta "
                    f"{_format_note_metric(sweep_entry['max_logit_delta_from_ordinary'])}, "
                    "support-change "
                    f"{_format_note_metric(sweep_entry.get('support_change_fraction', 0.0))}, "
                    "pinned-vs-repicked "
                    f"{_format_note_metric(sweep_entry.get('pinned_vs_repicked_logit_delta', 0.0))}"
                )
                for sweep_entry in entry["hep_alpha_sweep"]
            )
            lines.append(f"- `{entry['experiment_id']}`: {sweep}")
    if verdict["failed_invariants"]:
        lines.extend(["", "## Failed Invariants", ""])
        for failed in verdict["failed_invariants"]:
            lines.append(
                f"- `{failed['experiment_id']}`: `{failed['invariant']}`"
            )
    if verdict["failed_artifact_invariants"]:
        lines.extend(["", "## Failed Artifact Invariants", ""])
        for failed in verdict["failed_artifact_invariants"]:
            lines.append(
                f"- `{failed['experiment_id']}`: `{failed['artifact']}`"
            )
    if verdict["best_hep_alpha_by_loss"]:
        best = verdict["best_hep_alpha_by_loss"]
        lines.extend(
            [
                "",
                "## Verdict",
                "",
                (
                    "- Best HEP alpha by loss: "
                    f"`{best['alpha']}` in `{best['experiment_id']}` "
                    f"with loss `{_format_note_metric(best['loss'])}` "
                    "and ordinary-logit delta "
                    f"`{_format_note_metric(best['max_logit_delta_from_ordinary'])}`"
                ),
            ]
        )
        accepted = hep_acceptance["accepted_alpha"]
        lines.extend(
            [
                (
                    "- HEP acceptance policy: require nonzero alpha, loss improvement "
                    "over alpha 0 greater than "
                    f"`{_format_note_metric(hep_acceptance['min_loss_improvement_from_alpha0'])}`, "
                    "and ordinary-logit delta at or below "
                    f"`{_format_note_metric(hep_acceptance['max_logit_delta_from_ordinary'])}`"
                ),
                (
                    "- Accepted HEP alpha: "
                    + (
                        f"`{accepted['alpha']}` in `{accepted['experiment_id']}` "
                        f"with loss improvement `{_format_note_metric(accepted['loss_improvement_from_alpha0'])}` "
                        "and ordinary-logit delta "
                        f"`{_format_note_metric(accepted['max_logit_delta_from_ordinary'])}`"
                        if accepted
                        else "`none`"
                    )
                ),
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_note_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.8f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and compare RelaLeap FabricPC experiment configs."
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        type=Path,
        help="Config to include. Repeat for multiple configs.",
    )
    parser.add_argument(
        "--out",
        default=Path("results/comparisons/phase0"),
        type=Path,
    )
    parser.add_argument(
        "--hep-max-logit-delta",
        default=DEFAULT_HEP_MAX_LOGIT_DELTA,
        type=float,
        help="Maximum ordinary-logit delta allowed for accepting nonzero HEP alpha.",
    )
    parser.add_argument(
        "--hep-min-loss-improvement",
        default=DEFAULT_HEP_MIN_LOSS_IMPROVEMENT,
        type=float,
        help="Minimum loss improvement over alpha 0 required for accepting HEP alpha.",
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        help="Optional path for a compact, stable Phase 0 comparison baseline JSON.",
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        help=(
            "Optional checked-in baseline to compare against. Writes "
            "baseline_comparison.json in --out and exits nonzero on mismatch."
        ),
    )
    args = parser.parse_args()
    config_paths = args.configs or list(DEFAULT_CONFIGS)
    comparison = run_comparison(
        config_paths,
        args.out,
        hep_max_logit_delta=args.hep_max_logit_delta,
        hep_min_loss_improvement=args.hep_min_loss_improvement,
    )
    if args.baseline_out:
        write_comparison_baseline(args.baseline_out, comparison)
    baseline_comparison = None
    if args.baseline_reference:
        baseline_comparison = compare_to_baseline(
            comparison,
            args.baseline_reference,
            args.out / "baseline_comparison.json",
        )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if comparison["status"] != "ok" or comparison["verdict"]["status"] != "pass":
        raise SystemExit(1)
    if baseline_comparison and baseline_comparison["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
