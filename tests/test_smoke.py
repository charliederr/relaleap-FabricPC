from __future__ import annotations

import copy

import jax
import jax.numpy as jnp

from relaleap_fabricpc.fabricpc_nodes import init_residual_params, residual_forward
from relaleap_fabricpc.smoke import run_phase0_smoke


CONFIG = {
    "run": {"experiment_id": "test_smoke", "seed": 1, "max_steps": 1},
    "data": {"dataset": "tiny_shakespeare_char", "seq_len": 16},
    "model": {
        "base": {"layers": 1, "hidden_dim": 16},
        "columns": {
            "num_columns": 4,
            "atoms_per_column": 2,
            "top_k": 1,
            "insertion_sites": 1,
        },
    },
    "inference": {"pc_steps": 1, "hep_alpha": 0.0},
    "outputs": {
        "require_summary_json": True,
        "require_metrics_csv": True,
        "require_notes_md": True,
    },
}


def test_phase0_invariants_pass() -> None:
    result = run_phase0_smoke(CONFIG)

    assert result.invariants["zero_init_identity"]
    assert result.invariants["frozen_base_unchanged"]
    assert result.invariants["hep_alpha_0_equivalence"]
    assert result.invariants["residual_parameters_updated"]
    assert result.training_steps == CONFIG["run"]["max_steps"]
    assert len(result.to_metric_rows()) == CONFIG["run"]["max_steps"] + 1
    assert result.support_audit["support_positions"] == 4 * 16
    assert result.support_audit["top_k"] == 1
    assert result.support_audit["num_columns"] == 4


def test_pc_objective_reports_pc_update_phase() -> None:
    pc_config = copy.deepcopy(CONFIG)
    pc_config["training"] = {"residual_objective": "pc_logit_mse"}

    result = run_phase0_smoke(pc_config)

    assert result.residual_objective == "pc_logit_mse"
    assert result.to_metric_rows()[1]["phase"] == "pc_residual_update"


def test_residual_columns_zero_init_identity_and_tie_break() -> None:
    params = init_residual_params(
        jax.random.PRNGKey(0),
        hidden_dim=4,
        num_columns=4,
        atoms_per_column=2,
        support_router="linear",
        contextual_router_hidden_dim=8,
    )
    hidden = jnp.zeros((2, 3, 4), dtype=jnp.float32)

    output, support = residual_forward(
        params,
        hidden,
        num_columns=4,
        top_k=2,
        support_router="linear",
    )

    assert jnp.array_equal(output, hidden)
    assert jnp.array_equal(support[..., 0], jnp.zeros_like(support[..., 0]))
    assert jnp.array_equal(support[..., 1], jnp.ones_like(support[..., 1]))


def test_contextual_causal_router_ignores_future_positions() -> None:
    params = init_residual_params(
        jax.random.PRNGKey(0),
        hidden_dim=4,
        num_columns=5,
        atoms_per_column=2,
        support_router="contextual_mlp_causal",
        contextual_router_hidden_dim=8,
    )
    hidden = jnp.zeros((1, 5, 4), dtype=jnp.float32)
    hidden = hidden.at[:, :, 0].set(jnp.array([0.0, 0.1, 0.2, 0.3, 0.4]))
    perturbed = hidden.at[:, 2:, 0].add(100.0)

    _, support = residual_forward(
        params,
        hidden,
        num_columns=5,
        top_k=2,
        support_router="contextual_mlp_causal",
    )
    _, perturbed_support = residual_forward(
        params,
        perturbed,
        num_columns=5,
        top_k=2,
        support_router="contextual_mlp_causal",
    )

    assert jnp.array_equal(support[:, :2, :], perturbed_support[:, :2, :])
