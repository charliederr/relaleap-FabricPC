from __future__ import annotations

import json
from pathlib import Path

from relaleap_fabricpc.experiments.run import _read_config
from relaleap_fabricpc.fabricpc_nodes import SUPPORT_ROUTER_CHOICES


SUPPORTED_DATASETS = {"tiny_shakespeare_char", "tiny_shakespeare_word"}
SUPPORTED_RESIDUAL_OBJECTIVES = {
    "supervised_ce",
    "supervised_ce_confidence_penalty",
    "supervised_ce_margin_penalty",
    "supervised_ce_label_smoothing",
    "supervised_ce_focal",
    "supervised_ce_temporal_consistency",
    "pc_logit_mse",
    "pc_logit_mse_ce_anchor",
}
SUPPORTED_HEP_OBJECTIVES = {
    "residual_adapter",
    "prediction_entropy_gradient",
    "temporal_consistency_gradient",
    "supervised_ce_gradient",
}


def test_source_phase0_config_inventory_is_mirrored() -> None:
    names = {path.name for path in Path("configs").glob("*.yaml")}

    assert len(names) == 92
    assert {
        "char_smoke.yaml",
        "char_smoke_pc.yaml",
        "char_smoke_hep.yaml",
        "char_smoke_hep_support_stress_clipped.yaml",
        "token_larger_support_wide_causal_contextual_router_hep_temporal_clipped_objective_gate.yaml",
    } <= names


def test_mirrored_configs_use_supported_phase0_values() -> None:
    for config_path in sorted(Path("configs").glob("*.yaml")):
        config = _read_config(config_path)
        dataset = config.get("data", {}).get("dataset", "tiny_shakespeare_char")
        residual_objective = config.get("training", {}).get("residual_objective", "supervised_ce")
        columns = config.get("model", {}).get("columns", {})
        support_router = columns.get("support_router", "linear")
        hep_objective = config.get("inference", {}).get(
            "hep_settling_objective",
            "residual_adapter",
        )

        assert dataset in SUPPORTED_DATASETS, config_path.name
        assert residual_objective in SUPPORTED_RESIDUAL_OBJECTIVES, config_path.name
        assert support_router in SUPPORT_ROUTER_CHOICES, config_path.name
        assert hep_objective in SUPPORTED_HEP_OBJECTIVES, config_path.name


def test_checked_fabricpc_phase0_baseline_records_passing_gate() -> None:
    baseline = json.loads(
        Path("baselines/phase0_fabricpc_comparison.json").read_text(encoding="utf-8")
    )

    assert baseline["schema_version"] == 3
    assert baseline["comparison_status"] == "ok"
    assert baseline["verdict_status"] == "pass"
    assert baseline["config_paths"] == [
        "configs/char_smoke.yaml",
        "configs/char_smoke_pc.yaml",
        "configs/char_smoke_hep.yaml",
    ]
    assert baseline["phase0_invariants"] == {
        "count": 12,
        "failed": [],
        "passed": True,
    }
    assert baseline["artifact_invariants"] == {
        "count": 9,
        "failed": [],
        "passed": True,
    }
    assert baseline["hep"]["acceptance"]["accepted_alpha"]["alpha"] == 0.25
