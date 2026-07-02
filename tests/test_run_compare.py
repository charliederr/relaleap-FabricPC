from __future__ import annotations

import json
from pathlib import Path

from relaleap_fabricpc.experiments.check_artifacts import check_comparison_artifacts
from relaleap_fabricpc.experiments.compare import run_comparison, write_comparison_baseline
from relaleap_fabricpc.experiments.run import run


def _write_config(path: Path, experiment_id: str, objective: str = "supervised_ce") -> None:
    path.write_text(
        "\n".join(
            [
                "run:",
                f"  experiment_id: {experiment_id}",
                "  seed: 1",
                "  max_steps: 1",
                "data:",
                "  dataset: tiny_shakespeare_char",
                "  seq_len: 16",
                "training:",
                f"  residual_objective: {objective}",
                "model:",
                "  base:",
                "    layers: 1",
                "    hidden_dim: 16",
                "  columns:",
                "    num_columns: 4",
                "    atoms_per_column: 2",
                "    top_k: 1",
                "    insertion_sites: 1",
                "inference:",
                "  pc_steps: 1",
                "  hep_alpha: 0.0",
                "outputs:",
                "  require_summary_json: true",
                "  require_metrics_csv: true",
                "  require_notes_md: true",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_run_writes_required_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "char_smoke.yaml"
    out_dir = tmp_path / "run"
    _write_config(config, "char_smoke")

    summary = run(config, out_dir)

    assert summary["status"] == "ok"
    assert summary["artifact_invariants"] == {
        "summary_json": True,
        "metrics_csv": True,
        "notes_md": True,
    }
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "notes.md").is_file()


def test_comparison_and_artifact_check_pass(tmp_path: Path) -> None:
    config_a = tmp_path / "char_smoke.yaml"
    config_b = tmp_path / "char_smoke_pc.yaml"
    _write_config(config_a, "char_smoke")
    _write_config(config_b, "char_smoke_pc", objective="pc_logit_mse")

    comparison = run_comparison([config_a, config_b], tmp_path / "comparison")
    report = check_comparison_artifacts(tmp_path / "comparison")

    assert comparison["status"] == "ok"
    assert comparison["verdict"]["status"] == "pass"
    assert report["status"] == "pass"


def test_baseline_roundtrip(tmp_path: Path) -> None:
    config_a = tmp_path / "char_smoke.yaml"
    config_b = tmp_path / "char_smoke_pc.yaml"
    _write_config(config_a, "char_smoke")
    _write_config(config_b, "char_smoke_pc", objective="pc_logit_mse")

    comparison = run_comparison([config_a, config_b], tmp_path / "comparison")
    baseline = write_comparison_baseline(tmp_path / "baseline.json", comparison)

    assert json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["verdict_status"] == "pass"
