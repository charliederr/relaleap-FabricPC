from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

from relaleap_fabricpc.experiments.check_artifacts import check_comparison_artifacts
from relaleap_fabricpc.experiments.compare import (
    _comparison_baseline,
    _comparison_verdict,
    compare_comparison_to_baseline,
    compare_to_baseline,
    run_comparison,
    write_comparison_baseline,
)


def _passing_artifact_invariants() -> dict[str, bool]:
    return {
        "summary_json": True,
        "metrics_csv": True,
        "notes_md": True,
    }


def _comparison_with_accepted_hep() -> dict[str, object]:
    return {
        "status": "ok",
        "verdict": _comparison_verdict(
            [
                {
                    "experiment_id": "char_smoke_hep",
                    "invariants": {"zero_init_identity": True},
                    "artifact_invariants": _passing_artifact_invariants(),
                    "hep_alpha_sweep": [
                        {
                            "alpha": 0.0,
                            "loss": 3.5,
                            "max_logit_delta_from_ordinary": 0.0,
                        },
                        {
                            "alpha": 0.25,
                            "loss": 3.4,
                            "max_logit_delta_from_ordinary": 0.05,
                        },
                    ],
                }
            ],
            "ok",
        ),
        "runs": [
            {
                "experiment_id": "char_smoke_hep",
                "config_path": "configs/char_smoke_hep.yaml",
                "residual_objective": "supervised_ce",
                "status": "ok",
                "training_steps": 10,
                "invariants": {"zero_init_identity": True},
                "artifact_invariants": _passing_artifact_invariants(),
                "final_residual_loss": 3.4,
            }
        ],
    }


def test_comparison_baseline_matches_source_schema_v3() -> None:
    baseline = _comparison_baseline(_comparison_with_accepted_hep())

    assert baseline["schema_version"] == 3
    assert baseline["comparison_status"] == "ok"
    assert baseline["verdict_status"] == "pass"
    assert baseline["phase0_invariants"] == {
        "passed": True,
        "count": 1,
        "failed": [],
    }
    assert baseline["artifact_invariants"] == {
        "passed": True,
        "count": 3,
        "failed": [],
    }
    assert baseline["runs"][0]["artifact_invariants"] == {
        "count": 3,
        "failed": [],
        "passed": True,
    }
    assert baseline["hep"]["acceptance"]["accepted_alpha"]["alpha"] == 0.25
    assert "loss_improvement_from_alpha0" in baseline["hep"]["acceptance"]["accepted_alpha"]
    assert "runtime_seconds" not in baseline
    assert "out_dir" not in baseline


def test_compare_to_baseline_reports_source_style_mismatches(tmp_path: Path) -> None:
    comparison = _comparison_with_accepted_hep()
    baseline_path = tmp_path / "baseline.json"
    out_path = tmp_path / "baseline_comparison.json"
    reference = write_comparison_baseline(baseline_path, comparison)
    reference["hep"]["acceptance"]["accepted_alpha"]["alpha"] = 0.5
    baseline_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = compare_to_baseline(comparison, baseline_path, out_path)

    assert result["status"] == "fail"
    assert result["mismatches"] == [
        {
            "field": "hep.acceptance.accepted_alpha.alpha",
            "reference": 0.5,
            "candidate": 0.25,
        }
    ]
    assert result["candidate"]["accepted_hep_alpha"]["alpha"] == 0.25
    assert out_path.is_file()


def test_schema_mismatch_short_circuits_with_comparison_fields() -> None:
    comparison = _comparison_with_accepted_hep()
    reference = _comparison_baseline(comparison)
    reference["schema_version"] = 1
    reference.pop("artifact_invariants")

    result = compare_comparison_to_baseline(comparison, reference)

    assert result["status"] == "fail"
    assert result["mismatches"] == [
        {"field": "schema_version", "reference": 1, "candidate": 3}
    ]
    assert result["reference"]["artifact_invariants"] is None


def test_run_comparison_writes_source_style_metrics_and_notes(tmp_path: Path) -> None:
    configs = [tmp_path / "a.yaml", tmp_path / "b.yaml"]
    for config in configs:
        config.write_text("run:\n  experiment_id: fake\n", encoding="utf-8")

    def fake_run(config_path: Path, run_dir: Path) -> dict[str, object]:
        run_dir.mkdir(parents=True, exist_ok=True)
        experiment_id = config_path.stem
        objective = "supervised_ce" if experiment_id == "a" else "pc_logit_mse"
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "phase",
                    "base_loss",
                    "residual_loss",
                    "hep_alpha",
                    "hep_loss",
                    "max_hep_logit_delta_from_ordinary",
                    "status",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "step": 0,
                        "phase": "initial",
                        "base_loss": "1.00000000",
                        "residual_loss": "1.00000000",
                        "hep_alpha": "",
                        "hep_loss": "",
                        "max_hep_logit_delta_from_ordinary": "",
                        "status": "ok",
                    },
                    {
                        "step": 1,
                        "phase": "residual_update",
                        "base_loss": "1.00000000",
                        "residual_loss": "0.75000000",
                        "hep_alpha": "",
                        "hep_loss": "",
                        "max_hep_logit_delta_from_ordinary": "",
                        "status": "ok",
                    },
                ]
            )
        return {
            "experiment_id": experiment_id,
            "status": "ok",
            "error": None,
            "artifact_invariants": _passing_artifact_invariants(),
            "phase0": {
                "residual_objective": objective,
                "dataset": "tiny_shakespeare_word",
                "num_columns": 24 if experiment_id == "b" else 8,
                "atoms_per_column": 4,
                "top_k": 2 if experiment_id == "b" else 1,
                "training_steps": 1,
                "base_loss": 1.0,
                "zero_init_loss": 1.0,
                "pinned_support": experiment_id == "b",
                "support_stress": experiment_id == "b",
                "support_instability": {
                    "support_change_fraction": 0.25 if experiment_id == "b" else 0.0,
                    "pinned_vs_repicked_logit_delta": 1.5 if experiment_id == "b" else 0.0,
                },
                "hep_alpha_sweep": (
                    [
                        {
                            "alpha": 0.0,
                            "loss": 0.75,
                            "max_logit_delta_from_ordinary": 0.0,
                        }
                    ]
                    if experiment_id == "b"
                    else []
                ),
                "invariants": {"zero_init_identity": True},
            },
        }

    with patch("relaleap_fabricpc.experiments.compare.run", side_effect=fake_run):
        comparison = run_comparison(configs, tmp_path / "comparison")

    assert comparison["status"] == "ok"
    assert comparison["verdict"]["status"] == "pass"

    with (tmp_path / "comparison" / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "loss_delta_from_initial" in rows[0]
    assert "dataset" in rows[0]
    assert "num_columns" in rows[0]
    assert "atoms_per_column" in rows[0]
    assert "top_k" in rows[0]
    assert rows[-1]["dataset"] == "tiny_shakespeare_word"
    assert rows[-1]["top_k"] == "2"
    assert rows[-1]["loss_delta_from_initial"] == "-0.25000000"

    notes = (tmp_path / "comparison" / "notes.md").read_text(encoding="utf-8")
    assert "## HEP Alpha Sweeps" in notes
    assert "alpha 0.0" in notes
    assert "Pinned-vs-repicked" in notes
    assert "Best HEP alpha by loss" in notes
    assert "Accepted HEP alpha" in notes


def test_artifact_checker_validates_child_run_summaries(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    _write_comparison_tree(comparison_dir)
    summary_path = comparison_dir / "runs" / "char_smoke" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["experiment_id"] = "char_smoke_pc"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = check_comparison_artifacts(comparison_dir)

    assert report["status"] == "fail"
    assert {
        "field": "run.char_smoke.summary.experiment_id",
        "expected": "char_smoke",
        "actual": "char_smoke_pc",
    } in report["failures"]


def test_artifact_checker_compares_baseline_reference(tmp_path: Path) -> None:
    comparison_dir = tmp_path / "comparison"
    _write_comparison_tree(comparison_dir, include_baseline=False)
    summary = json.loads((comparison_dir / "summary.json").read_text(encoding="utf-8"))
    baseline_path = tmp_path / "baseline.json"
    baseline = write_comparison_baseline(baseline_path, summary)
    baseline["hep"]["acceptance"]["accepted_alpha"]["alpha"] = 0.5
    baseline_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = check_comparison_artifacts(comparison_dir, baseline_reference=baseline_path)

    assert report["status"] == "fail"
    assert report["baseline_reference_comparison"]["status"] == "fail"
    assert {
        "field": "baseline_reference.hep.acceptance.accepted_alpha.alpha",
        "expected": 0.5,
        "actual": 0.25,
    } in report["failures"]


def _write_comparison_tree(
    comparison_dir: Path,
    *,
    include_baseline: bool = True,
    baseline_status: str = "pass",
) -> None:
    comparison_dir.mkdir(parents=True)
    (comparison_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
    (comparison_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (comparison_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "verdict": {
                    "status": "pass",
                    "invariants_passed": True,
                    "invariant_count": 12,
                    "failed_invariants": [],
                    "artifact_invariants_passed": True,
                    "artifact_invariant_count": 9,
                    "failed_artifact_invariants": [],
                    "best_hep_alpha_by_loss": {
                        "alpha": 1.0,
                        "experiment_id": "char_smoke_hep",
                        "loss": 3.52012658,
                        "max_logit_delta_from_ordinary": 0.20742974,
                    },
                    "hep_alpha_acceptance": {
                        "status": "accepted",
                        "max_logit_delta_from_ordinary": 0.1,
                        "min_loss_improvement_from_alpha0": 0.0,
                        "baseline_alpha0": {
                            "alpha": 0.0,
                            "experiment_id": "char_smoke_hep",
                            "loss": 3.56317067,
                            "max_logit_delta_from_ordinary": 0.0,
                        },
                        "accepted_alpha": {
                            "alpha": 0.25,
                            "experiment_id": "char_smoke_hep",
                            "loss": 3.55195642,
                            "loss_improvement_from_alpha0": 0.01,
                            "max_logit_delta_from_ordinary": 0.05,
                        },
                        "candidate_count": 2,
                        "rejected_count": 1,
                    },
                },
                "runs": [
                    {
                        "experiment_id": "char_smoke",
                        "config_path": "configs/char_smoke.yaml",
                        "residual_objective": "supervised_ce",
                        "status": "ok",
                        "training_steps": 10,
                        "final_residual_loss": 3.56,
                        "artifact_invariants": _passing_artifact_invariants(),
                        "invariants": {"zero_init_identity": True},
                    },
                    {
                        "experiment_id": "char_smoke_pc",
                        "config_path": "configs/char_smoke_pc.yaml",
                        "residual_objective": "pc_logit_mse",
                        "status": "ok",
                        "training_steps": 10,
                        "final_residual_loss": 0.03,
                        "artifact_invariants": _passing_artifact_invariants(),
                        "invariants": {"zero_init_identity": True},
                    },
                    {
                        "experiment_id": "char_smoke_hep",
                        "config_path": "configs/char_smoke_hep.yaml",
                        "residual_objective": "supervised_ce",
                        "status": "ok",
                        "training_steps": 10,
                        "final_residual_loss": 3.55,
                        "artifact_invariants": _passing_artifact_invariants(),
                        "invariants": {"zero_init_identity": True},
                        "hep_alpha_sweep": [
                            {
                                "alpha": 0.0,
                                "loss": 3.56317067,
                                "max_logit_delta_from_ordinary": 0.0,
                            },
                            {
                                "alpha": 0.25,
                                "loss": 3.55195642,
                                "max_logit_delta_from_ordinary": 0.05185753,
                            },
                        ],
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for stem in ["char_smoke", "char_smoke_pc", "char_smoke_hep"]:
        run_dir = comparison_dir / "runs" / stem
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "experiment_id": stem,
                    "status": "ok",
                    "error": None,
                    "artifact_invariants": _passing_artifact_invariants(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "metrics.csv").write_text("step,status\n0,ok\n", encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
    if include_baseline:
        (comparison_dir / "baseline_comparison.json").write_text(
            json.dumps(
                {
                    "status": baseline_status,
                    "mismatches": [] if baseline_status == "pass" else [{"field": "hep.acceptance.accepted_alpha.alpha"}],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
