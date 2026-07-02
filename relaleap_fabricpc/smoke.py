"""Phase 0 smoke machinery for the JAX/FabricPC RelaLeap port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from relaleap_fabricpc.jax_setup import configure_jax

configure_jax()

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.types import NodeParams

from relaleap_fabricpc.data import build_batch
from relaleap_fabricpc.fabricpc_nodes import (
    SUPPORT_ROUTER_CHOICES,
    gelu,
    init_residual_params,
    layer_norm_last,
    residual_forward,
    score_columns,
)


@dataclass(frozen=True)
class Phase0Result:
    """Structured result returned by the Phase 0 smoke routine."""

    residual_objective: str
    ce_anchor_weight: float
    confidence_penalty_weight: float
    margin_penalty_weight: float
    target_logit_margin: float
    label_smoothing_weight: float
    focal_gamma: float
    temporal_consistency_weight: float
    dataset: str
    vocab_size: int
    seq_len: int
    batch_size: int
    num_columns: int
    atoms_per_column: int
    top_k: int
    support_router: str
    contextual_router_hidden_dim: int
    base_loss: float
    zero_init_loss: float
    initial_loss: float
    post_step_loss: float
    training_steps: int
    metric_rows: list[dict[str, float | int | str]]
    residual_parameter_delta: float
    max_zero_init_logit_delta: float
    max_hep_alpha0_logit_delta: float
    pinned_support: bool
    support_stress: bool
    support_stress_preset: bool
    support_instability: dict[str, float | int | bool | None]
    support_audit: dict[str, Any]
    hep_update_clip_norm: float | None
    hep_settling_objective: str
    hep_alpha_sweep: list[dict[str, float]]
    invariants: dict[str, bool]

    def to_summary(self) -> dict[str, Any]:
        return {
            "residual_objective": self.residual_objective,
            "ce_anchor_weight": self.ce_anchor_weight,
            "confidence_penalty_weight": self.confidence_penalty_weight,
            "margin_penalty_weight": self.margin_penalty_weight,
            "target_logit_margin": self.target_logit_margin,
            "label_smoothing_weight": self.label_smoothing_weight,
            "focal_gamma": self.focal_gamma,
            "temporal_consistency_weight": self.temporal_consistency_weight,
            "dataset": self.dataset,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "num_columns": self.num_columns,
            "atoms_per_column": self.atoms_per_column,
            "top_k": self.top_k,
            "support_router": self.support_router,
            "contextual_router_hidden_dim": self.contextual_router_hidden_dim,
            "base_loss": self.base_loss,
            "zero_init_loss": self.zero_init_loss,
            "initial_loss": self.initial_loss,
            "post_step_loss": self.post_step_loss,
            "training_steps": self.training_steps,
            "residual_parameter_delta": self.residual_parameter_delta,
            "max_zero_init_logit_delta": self.max_zero_init_logit_delta,
            "max_hep_alpha0_logit_delta": self.max_hep_alpha0_logit_delta,
            "pinned_support": self.pinned_support,
            "support_stress": self.support_stress,
            "support_stress_preset": self.support_stress_preset,
            "support_instability": self.support_instability,
            "support_audit": self.support_audit,
            "hep_update_clip_norm": self.hep_update_clip_norm,
            "hep_settling_objective": self.hep_settling_objective,
            "hep_alpha_sweep": self.hep_alpha_sweep,
            "invariants": self.invariants,
        }

    def to_metric_rows(self) -> list[dict[str, float | int | str]]:
        return self.metric_rows


def run_phase0_smoke(config: dict[str, Any]) -> Phase0Result:
    """Run local Phase 0 invariants against a tiny JAX sequence model."""

    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    base_cfg = model_cfg.get("base", {})
    column_cfg = model_cfg.get("columns", {})
    inference_cfg = config.get("inference", {})
    training_cfg = config.get("training", {})

    seed = int(run_cfg.get("seed", 1))
    max_steps = int(run_cfg.get("max_steps", 10))
    learning_rate = float(run_cfg.get("learning_rate", 1e-2))
    residual_objective = str(
        training_cfg.get(
            "residual_objective",
            run_cfg.get("residual_objective", "supervised_ce"),
        )
    )
    ce_anchor_weight = float(training_cfg.get("ce_anchor_weight", 0.1))
    confidence_penalty_weight = float(training_cfg.get("confidence_penalty_weight", 0.01))
    margin_penalty_weight = float(training_cfg.get("margin_penalty_weight", 0.01))
    target_logit_margin = float(training_cfg.get("target_logit_margin", 0.25))
    label_smoothing_weight = float(training_cfg.get("label_smoothing_weight", 0.05))
    focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
    temporal_consistency_weight = float(
        training_cfg.get("temporal_consistency_weight", 0.01)
    )
    dataset = str(data_cfg.get("dataset", "tiny_shakespeare_char"))
    seq_len = int(data_cfg.get("seq_len", 32))
    hidden_dim = int(base_cfg.get("hidden_dim", 32))
    layers = int(base_cfg.get("layers", 2))
    num_columns = int(column_cfg.get("num_columns", 8))
    atoms_per_column = int(column_cfg.get("atoms_per_column", 4))
    top_k = int(column_cfg.get("top_k", 1))
    pinned_support = bool(column_cfg.get("pinned_support", False))
    support_stress = bool(column_cfg.get("support_stress", False))
    support_stress_preset = support_stress and bool(
        column_cfg.get("support_stress_preset", True)
    )
    support_router = str(column_cfg.get("support_router", "linear"))
    contextual_router_hidden_dim = int(
        column_cfg.get("contextual_router_hidden_dim", hidden_dim * 2)
    )
    pc_steps = int(inference_cfg.get("pc_steps", 1))
    hep_alpha = float(inference_cfg.get("hep_alpha", 0.0))
    hep_update_clip_norm = _parse_optional_float(
        inference_cfg.get("hep_update_clip_norm")
    )
    hep_settling_objective = str(
        inference_cfg.get("hep_settling_objective", "residual_adapter")
    )
    hep_alpha_sweep = _parse_hep_alpha_sweep(inference_cfg, fallback_alpha=hep_alpha)

    _validate_phase0_config(
        max_steps=max_steps,
        pc_steps=pc_steps,
        hep_update_clip_norm=hep_update_clip_norm,
        ce_anchor_weight=ce_anchor_weight,
        confidence_penalty_weight=confidence_penalty_weight,
        margin_penalty_weight=margin_penalty_weight,
        target_logit_margin=target_logit_margin,
        label_smoothing_weight=label_smoothing_weight,
        focal_gamma=focal_gamma,
        temporal_consistency_weight=temporal_consistency_weight,
        residual_objective=residual_objective,
        hep_settling_objective=hep_settling_objective,
        support_router=support_router,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
        hep_alpha_sweep=hep_alpha_sweep,
    )

    inputs, targets, vocab_size = build_batch(
        dataset=dataset,
        seq_len=seq_len,
        batch_size=4,
    )
    master_key = jax.random.PRNGKey(seed)
    base_key, residual_key = jax.random.split(master_key)
    base_params = _init_base_params(
        base_key,
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        layers=layers,
    )
    residual_params = init_residual_params(
        residual_key,
        hidden_dim=hidden_dim,
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        support_router=support_router,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
    )

    base_logits = _base_forward(base_params, inputs)
    zero_init_logits = _forward_with_residual(
        base_params,
        residual_params,
        inputs,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    hep_logits = forward_with_hep_alpha(
        base_params,
        residual_params,
        inputs,
        pc_steps=pc_steps,
        hep_alpha=0.0,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
        pinned_support=pinned_support,
        hep_update_clip_norm=hep_update_clip_norm,
        hep_settling_objective=hep_settling_objective,
        targets=targets,
        vocab_size=vocab_size,
    )
    base_loss = _cross_entropy(base_logits, targets, vocab_size)
    zero_init_loss = _cross_entropy(zero_init_logits, targets, vocab_size)
    max_zero_delta = _max_abs(base_logits - zero_init_logits)
    initial_hep_alpha0_delta = _max_abs(zero_init_logits - hep_logits)

    base_snapshot = _tree_copy(base_params)
    before_residual = _tree_copy(residual_params)
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(residual_params)

    loss_kwargs = {
        "objective": residual_objective,
        "ce_anchor_weight": ce_anchor_weight,
        "confidence_penalty_weight": confidence_penalty_weight,
        "margin_penalty_weight": margin_penalty_weight,
        "target_logit_margin": target_logit_margin,
        "label_smoothing_weight": label_smoothing_weight,
        "focal_gamma": focal_gamma,
        "temporal_consistency_weight": temporal_consistency_weight,
        "num_columns": num_columns,
        "top_k": top_k,
        "support_router": support_router,
    }
    loss_fn = lambda rp: _residual_loss(
        base_params,
        rp,
        inputs,
        targets,
        vocab_size,
        **loss_kwargs,
    )
    initial_loss = float(loss_fn(residual_params))
    metric_rows: list[dict[str, float | int | str]] = [
        _metric_row(
            step=0,
            phase="initial",
            residual_objective=residual_objective,
            base_loss=float(base_loss),
            residual_loss=initial_loss,
            zero_init_loss=float(zero_init_loss),
            residual_parameter_delta=0.0,
            max_zero_init_logit_delta=max_zero_delta,
            max_hep_alpha0_logit_delta=initial_hep_alpha0_delta,
            hep_alpha="",
            hep_loss="",
            max_hep_logit_delta_from_ordinary="",
            hep_support_change_fraction="",
            hep_pinned_vs_repicked_logit_delta="",
        )
    ]

    post_step_loss = initial_loss
    max_hep_alpha0_delta = initial_hep_alpha0_delta
    for step in range(1, max_steps + 1):
        loss_value, grads = jax.value_and_grad(loss_fn)(residual_params)
        updates, opt_state = optimizer.update(grads, opt_state, residual_params)
        residual_params = optax.apply_updates(residual_params, updates)
        post_step_loss = float(loss_fn(residual_params))
        max_hep_alpha0_delta = _max_hep_delta_from_ordinary(
            base_params,
            residual_params,
            inputs,
            pc_steps=pc_steps,
            hep_alpha=0.0,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
            pinned_support=pinned_support,
            hep_update_clip_norm=hep_update_clip_norm,
            hep_settling_objective=hep_settling_objective,
            targets=targets,
            vocab_size=vocab_size,
        )
        metric_rows.append(
            _metric_row(
                step=step,
                phase=_residual_update_phase(residual_objective),
                residual_objective=residual_objective,
                base_loss=float(base_loss),
                residual_loss=post_step_loss,
                zero_init_loss=float(zero_init_loss),
                residual_parameter_delta=_tree_abs_delta(before_residual, residual_params),
                max_zero_init_logit_delta=max_zero_delta,
                max_hep_alpha0_logit_delta=max_hep_alpha0_delta,
                hep_alpha="",
                hep_loss="",
                max_hep_logit_delta_from_ordinary="",
                hep_support_change_fraction="",
                hep_pinned_vs_repicked_logit_delta="",
            )
        )
        _ = loss_value

    if support_stress_preset:
        residual_params = _apply_hep_support_stress(
            base_params,
            residual_params,
            inputs,
            num_columns=num_columns,
            top_k=top_k,
        )
        post_step_loss = float(loss_fn(residual_params))

    support_instability = _hep_support_instability(
        base_params,
        residual_params,
        inputs,
        pc_steps=pc_steps,
        hep_alpha=1.0,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
        hep_update_clip_norm=hep_update_clip_norm,
    )
    support_audit = _residual_support_audit(
        base_params,
        residual_params,
        inputs,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    hep_sweep_rows = _evaluate_hep_alpha_sweep(
        base_params,
        residual_params,
        inputs,
        targets,
        vocab_size,
        pc_steps=pc_steps,
        alphas=hep_alpha_sweep,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
        pinned_support=pinned_support,
        hep_update_clip_norm=hep_update_clip_norm,
        hep_settling_objective=hep_settling_objective,
    )
    for sweep_row in hep_sweep_rows:
        if sweep_row["alpha"] == 0.0:
            max_hep_alpha0_delta = sweep_row["max_logit_delta_from_ordinary"]
        metric_rows.append(
            _metric_row(
                step=max_steps,
                phase="hep_sweep",
                residual_objective=residual_objective,
                base_loss=float(base_loss),
                residual_loss=post_step_loss,
                zero_init_loss=float(zero_init_loss),
                residual_parameter_delta=_tree_abs_delta(before_residual, residual_params),
                max_zero_init_logit_delta=max_zero_delta,
                max_hep_alpha0_logit_delta=max_hep_alpha0_delta,
                hep_alpha=sweep_row["alpha"],
                hep_loss=sweep_row["loss"],
                max_hep_logit_delta_from_ordinary=sweep_row[
                    "max_logit_delta_from_ordinary"
                ],
                hep_support_change_fraction=sweep_row["support_change_fraction"],
                hep_pinned_vs_repicked_logit_delta=sweep_row[
                    "pinned_vs_repicked_logit_delta"
                ],
            )
        )

    residual_delta = _tree_abs_delta(before_residual, residual_params)
    invariants = {
        "zero_init_identity": max_zero_delta <= 1e-6,
        "frozen_base_unchanged": _tree_all_equal(base_snapshot, base_params),
        "hep_alpha_0_equivalence": max_hep_alpha0_delta <= 1e-6,
        "residual_parameters_updated": residual_delta > 0.0,
    }

    return Phase0Result(
        residual_objective=residual_objective,
        ce_anchor_weight=ce_anchor_weight,
        confidence_penalty_weight=confidence_penalty_weight,
        margin_penalty_weight=margin_penalty_weight,
        target_logit_margin=target_logit_margin,
        label_smoothing_weight=label_smoothing_weight,
        focal_gamma=focal_gamma,
        temporal_consistency_weight=temporal_consistency_weight,
        dataset=dataset,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=int(inputs.shape[0]),
        num_columns=num_columns,
        atoms_per_column=atoms_per_column,
        top_k=top_k,
        support_router=support_router,
        contextual_router_hidden_dim=contextual_router_hidden_dim,
        base_loss=float(base_loss),
        zero_init_loss=float(zero_init_loss),
        initial_loss=initial_loss,
        post_step_loss=post_step_loss,
        training_steps=max_steps,
        metric_rows=metric_rows,
        residual_parameter_delta=residual_delta,
        max_zero_init_logit_delta=max_zero_delta,
        max_hep_alpha0_logit_delta=max_hep_alpha0_delta,
        pinned_support=pinned_support,
        support_stress=support_stress,
        support_stress_preset=support_stress_preset,
        support_instability=support_instability,
        support_audit=support_audit,
        hep_update_clip_norm=hep_update_clip_norm,
        hep_settling_objective=hep_settling_objective,
        hep_alpha_sweep=hep_sweep_rows,
        invariants=invariants,
    )


def _validate_phase0_config(**kwargs: Any) -> None:
    if kwargs["pc_steps"] < 1:
        raise ValueError("inference.pc_steps must be at least 1")
    if kwargs["max_steps"] < 1:
        raise ValueError("run.max_steps must be at least 1 for Phase 0 smoke")
    if kwargs["hep_update_clip_norm"] is not None and kwargs["hep_update_clip_norm"] <= 0.0:
        raise ValueError("inference.hep_update_clip_norm must be positive when set")
    for alpha in kwargs["hep_alpha_sweep"]:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("HEP alpha values must be between 0.0 and 1.0")
    for name in [
        "ce_anchor_weight",
        "confidence_penalty_weight",
        "margin_penalty_weight",
        "target_logit_margin",
        "focal_gamma",
        "temporal_consistency_weight",
    ]:
        if kwargs[name] < 0.0:
            raise ValueError(f"training.{name} must be non-negative")
    if kwargs["label_smoothing_weight"] < 0.0 or kwargs["label_smoothing_weight"] >= 1.0:
        raise ValueError("training.label_smoothing_weight must be in [0.0, 1.0)")
    if kwargs["support_router"] not in SUPPORT_ROUTER_CHOICES:
        raise ValueError(
            "model.columns.support_router must be one of: "
            "linear, contextual_mlp, contextual_mlp_causal"
        )
    if kwargs["contextual_router_hidden_dim"] < 1:
        raise ValueError("model.columns.contextual_router_hidden_dim must be positive")
    if kwargs["residual_objective"] not in {
        "supervised_ce",
        "supervised_ce_confidence_penalty",
        "supervised_ce_margin_penalty",
        "supervised_ce_label_smoothing",
        "supervised_ce_focal",
        "supervised_ce_temporal_consistency",
        "pc_logit_mse",
        "pc_logit_mse_ce_anchor",
    }:
        raise ValueError("training.residual_objective is not supported")
    if kwargs["hep_settling_objective"] not in {
        "residual_adapter",
        "prediction_entropy_gradient",
        "temporal_consistency_gradient",
        "supervised_ce_gradient",
    }:
        raise ValueError("inference.hep_settling_objective is not supported")


def _init_base_params(
    key: jax.Array,
    *,
    vocab_size: int,
    seq_len: int,
    hidden_dim: int,
    layers: int,
) -> dict[str, Any]:
    keys = jax.random.split(key, layers + 3)
    params = {
        "token_embedding": jax.random.normal(
            keys[0], (vocab_size, hidden_dim), dtype=jnp.float32
        )
        * jnp.float32(0.02),
        "position_embedding": jnp.zeros((seq_len, hidden_dim), dtype=jnp.float32),
        "layers": tuple(_init_encoder_layer(k, hidden_dim) for k in keys[1 : layers + 1]),
        "final_ln_gamma": jnp.ones((hidden_dim,), dtype=jnp.float32),
        "final_ln_beta": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "lm_head": jax.random.normal(
            keys[-1], (hidden_dim, vocab_size), dtype=jnp.float32
        )
        * jnp.float32(1.0 / np.sqrt(hidden_dim)),
    }
    return params


def _init_encoder_layer(key: jax.Array, hidden_dim: int) -> dict[str, jnp.ndarray]:
    keys = jax.random.split(key, 6)
    ff_dim = hidden_dim * 4
    scale = jnp.float32(1.0 / np.sqrt(hidden_dim))
    ff_scale = jnp.float32(1.0 / np.sqrt(ff_dim))
    return {
        "W_q": jax.random.normal(keys[0], (hidden_dim, hidden_dim), dtype=jnp.float32) * scale,
        "W_k": jax.random.normal(keys[1], (hidden_dim, hidden_dim), dtype=jnp.float32) * scale,
        "W_v": jax.random.normal(keys[2], (hidden_dim, hidden_dim), dtype=jnp.float32) * scale,
        "W_o": jax.random.normal(keys[3], (hidden_dim, hidden_dim), dtype=jnp.float32) * scale,
        "W_ff1": jax.random.normal(keys[4], (hidden_dim, ff_dim), dtype=jnp.float32) * scale,
        "W_ff2": jax.random.normal(keys[5], (ff_dim, hidden_dim), dtype=jnp.float32) * ff_scale,
        "b_q": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "b_k": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "b_v": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "b_o": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "b_ff1": jnp.zeros((ff_dim,), dtype=jnp.float32),
        "b_ff2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "ln1_gamma": jnp.ones((hidden_dim,), dtype=jnp.float32),
        "ln1_beta": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "ln2_gamma": jnp.ones((hidden_dim,), dtype=jnp.float32),
        "ln2_beta": jnp.zeros((hidden_dim,), dtype=jnp.float32),
    }


def _encode(base_params: dict[str, Any], input_ids: jnp.ndarray) -> jnp.ndarray:
    seq_len = input_ids.shape[1]
    hidden = (
        base_params["token_embedding"][input_ids]
        + base_params["position_embedding"][:seq_len]
    )
    for layer in base_params["layers"]:
        hidden = _encoder_layer_forward(layer, hidden)
    return layer_norm_last(hidden, base_params["final_ln_gamma"], base_params["final_ln_beta"])


def _encoder_layer_forward(layer: dict[str, jnp.ndarray], hidden: jnp.ndarray) -> jnp.ndarray:
    batch_size, seq_len, hidden_dim = hidden.shape
    num_heads = 4
    head_dim = hidden_dim // num_heads

    q = jnp.matmul(hidden, layer["W_q"]) + layer["b_q"]
    k = jnp.matmul(hidden, layer["W_k"]) + layer["b_k"]
    v = jnp.matmul(hidden, layer["W_v"]) + layer["b_v"]

    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.float32(head_dim))
    attn = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, hidden_dim
    )
    attn_out = jnp.matmul(attn_out, layer["W_o"]) + layer["b_o"]
    hidden = layer_norm_last(hidden + attn_out, layer["ln1_gamma"], layer["ln1_beta"])
    ff = gelu(jnp.matmul(hidden, layer["W_ff1"]) + layer["b_ff1"])
    ff = jnp.matmul(ff, layer["W_ff2"]) + layer["b_ff2"]
    return layer_norm_last(hidden + ff, layer["ln2_gamma"], layer["ln2_beta"])


def _decode(base_params: dict[str, Any], hidden: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(hidden, base_params["lm_head"])


def _base_forward(base_params: dict[str, Any], input_ids: jnp.ndarray) -> jnp.ndarray:
    return _decode(base_params, _encode(base_params, input_ids))


def _forward_with_residual(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    input_ids: jnp.ndarray,
    *,
    num_columns: int,
    top_k: int,
    support_router: str,
) -> jnp.ndarray:
    hidden = _encode(base_params, input_ids)
    settled, _ = residual_forward(
        residual_params,
        hidden,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    return _decode(base_params, settled)


def forward_with_hep_alpha(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    input_ids: jnp.ndarray,
    *,
    pc_steps: int,
    hep_alpha: float,
    num_columns: int,
    top_k: int,
    support_router: str,
    pinned_support: bool = False,
    hep_update_clip_norm: float | None = None,
    hep_settling_objective: str = "residual_adapter",
    targets: jnp.ndarray | None = None,
    vocab_size: int | None = None,
) -> jnp.ndarray:
    if pc_steps < 1:
        raise ValueError("pc_steps must be at least 1")
    if hep_alpha < 0.0 or hep_alpha > 1.0:
        raise ValueError("hep_alpha must be between 0.0 and 1.0")
    if hep_update_clip_norm is not None and hep_update_clip_norm <= 0.0:
        raise ValueError("hep_update_clip_norm must be positive when set")
    if hep_settling_objective == "supervised_ce_gradient" and (
        targets is None or vocab_size is None
    ):
        raise ValueError("supervised_ce_gradient HEP settling requires targets and vocab_size")

    hidden = _encode(base_params, input_ids)
    support_indices = None
    if pinned_support:
        settled, support_indices = residual_forward(
            residual_params,
            hidden,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
        )
    else:
        settled, _ = residual_forward(
            residual_params,
            hidden,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
        )

    for _ in range(1, pc_steps):
        if hep_settling_objective == "supervised_ce_gradient":
            update = _supervised_ce_hidden_update(
                base_params, settled, targets, vocab_size, max_norm=hep_update_clip_norm
            )
        elif hep_settling_objective == "prediction_entropy_gradient":
            update = _prediction_entropy_hidden_update(
                base_params, settled, max_norm=hep_update_clip_norm
            )
        elif hep_settling_objective == "temporal_consistency_gradient":
            update = _temporal_consistency_hidden_update(
                base_params, settled, max_norm=hep_update_clip_norm
            )
        else:
            proposed, _ = residual_forward(
                residual_params,
                settled,
                num_columns=num_columns,
                top_k=top_k,
                support_router=support_router,
                support_indices=support_indices,
            )
            update = _clip_update(proposed - settled, hep_update_clip_norm)
        settled = settled + jnp.float32(hep_alpha) * update
    return _decode(base_params, settled)


def _supervised_ce_hidden_update(
    base_params: dict[str, Any],
    hidden: jnp.ndarray,
    targets: jnp.ndarray,
    vocab_size: int,
    *,
    max_norm: float | None,
) -> jnp.ndarray:
    loss_fn = lambda h: _cross_entropy(_decode(base_params, h), targets, vocab_size)
    gradient = jax.grad(loss_fn)(hidden)
    return _clip_update(-gradient, max_norm)


def _prediction_entropy_hidden_update(
    base_params: dict[str, Any],
    hidden: jnp.ndarray,
    *,
    max_norm: float | None,
) -> jnp.ndarray:
    def entropy_fn(h: jnp.ndarray) -> jnp.ndarray:
        logits = _decode(base_params, h)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        return -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

    gradient = jax.grad(entropy_fn)(hidden)
    return _clip_update(-gradient, max_norm)


def _temporal_consistency_hidden_update(
    base_params: dict[str, Any],
    hidden: jnp.ndarray,
    *,
    max_norm: float | None,
) -> jnp.ndarray:
    if hidden.shape[1] < 2:
        return jnp.zeros_like(hidden)

    def consistency_fn(h: jnp.ndarray) -> jnp.ndarray:
        logits = _decode(base_params, h)
        teacher_probs = jax.nn.softmax(jax.lax.stop_gradient(logits[:, :-1, :]), axis=-1)
        student_log_probs = jax.nn.log_softmax(logits[:, 1:, :], axis=-1)
        return -jnp.mean(jnp.sum(teacher_probs * student_log_probs, axis=-1))

    gradient = jax.grad(consistency_fn)(hidden)
    return _clip_update(-gradient, max_norm)


def _clip_update(update: jnp.ndarray, max_norm: float | None) -> jnp.ndarray:
    if max_norm is None:
        return update
    norm = jnp.linalg.norm(update, axis=-1, keepdims=True)
    scale = jnp.minimum(jnp.float32(1.0), jnp.float32(max_norm) / jnp.maximum(norm, 1e-12))
    return update * scale


def _residual_loss(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    vocab_size: int,
    *,
    objective: str,
    ce_anchor_weight: float,
    confidence_penalty_weight: float,
    margin_penalty_weight: float,
    target_logit_margin: float,
    label_smoothing_weight: float,
    focal_gamma: float,
    temporal_consistency_weight: float,
    num_columns: int,
    top_k: int,
    support_router: str,
) -> jnp.ndarray:
    logits = _forward_with_residual(
        base_params,
        residual_params,
        inputs,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    prediction_logits = logits[:, :-1, :]
    prediction_targets = targets[:, :-1]
    ce_loss = _cross_entropy_from_prediction_logits(
        prediction_logits, prediction_targets, vocab_size
    )
    if objective == "supervised_ce":
        return ce_loss
    if objective == "supervised_ce_confidence_penalty":
        log_probs = jax.nn.log_softmax(prediction_logits, axis=-1)
        probs = jax.nn.softmax(prediction_logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
        return ce_loss - jnp.float32(confidence_penalty_weight) * entropy
    if objective == "supervised_ce_margin_penalty":
        target_logits = jnp.take_along_axis(
            prediction_logits, prediction_targets[..., None], axis=-1
        )[..., 0]
        target_mask = jax.nn.one_hot(prediction_targets, vocab_size).astype(bool)
        strongest_other = jnp.max(
            jnp.where(target_mask, jnp.asarray(-jnp.inf, dtype=prediction_logits.dtype), prediction_logits),
            axis=-1,
        )
        margin_shortfall = jax.nn.relu(
            jnp.float32(target_logit_margin) - (target_logits - strongest_other)
        )
        return ce_loss + jnp.float32(margin_penalty_weight) * jnp.mean(margin_shortfall)
    if objective == "supervised_ce_label_smoothing":
        return _label_smoothed_cross_entropy(
            prediction_logits, prediction_targets, vocab_size, label_smoothing_weight
        )
    if objective == "supervised_ce_focal":
        per_token_ce = _per_token_cross_entropy(prediction_logits, prediction_targets)
        target_probs = jnp.exp(-per_token_ce)
        return jnp.mean(jnp.power(1.0 - target_probs, focal_gamma) * per_token_ce)
    if objective == "supervised_ce_temporal_consistency":
        if prediction_logits.shape[1] < 2:
            return ce_loss
        teacher_probs = jax.nn.softmax(
            jax.lax.stop_gradient(prediction_logits[:, :-1, :]), axis=-1
        )
        student_log_probs = jax.nn.log_softmax(prediction_logits[:, 1:, :], axis=-1)
        temporal_loss = -jnp.mean(jnp.sum(teacher_probs * student_log_probs, axis=-1))
        return ce_loss + jnp.float32(temporal_consistency_weight) * temporal_loss
    if objective in {"pc_logit_mse", "pc_logit_mse_ce_anchor"}:
        probs = jax.nn.softmax(prediction_logits, axis=-1)
        target_probs = jax.nn.one_hot(prediction_targets, vocab_size, dtype=probs.dtype)
        pc_loss = jnp.mean(jnp.square(probs - target_probs))
        if objective == "pc_logit_mse":
            return pc_loss
        return pc_loss + jnp.float32(ce_anchor_weight) * ce_loss
    raise ValueError(f"Unsupported residual objective: {objective}")


def _cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray, vocab_size: int) -> jnp.ndarray:
    return _cross_entropy_from_prediction_logits(logits[:, :-1, :], targets[:, :-1], vocab_size)


def _cross_entropy_from_prediction_logits(
    prediction_logits: jnp.ndarray,
    prediction_targets: jnp.ndarray,
    vocab_size: int,
) -> jnp.ndarray:
    _ = vocab_size
    return jnp.mean(_per_token_cross_entropy(prediction_logits, prediction_targets))


def _per_token_cross_entropy(
    prediction_logits: jnp.ndarray,
    prediction_targets: jnp.ndarray,
) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(prediction_logits, axis=-1)
    return -jnp.take_along_axis(log_probs, prediction_targets[..., None], axis=-1)[..., 0]


def _label_smoothed_cross_entropy(
    prediction_logits: jnp.ndarray,
    prediction_targets: jnp.ndarray,
    vocab_size: int,
    smoothing: float,
) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(prediction_logits, axis=-1)
    targets = jax.nn.one_hot(prediction_targets, vocab_size, dtype=log_probs.dtype)
    smooth = jnp.float32(smoothing)
    targets = targets * (1.0 - smooth) + smooth / vocab_size
    return -jnp.mean(jnp.sum(targets * log_probs, axis=-1))


def _parse_hep_alpha_sweep(
    inference_cfg: dict[str, Any],
    *,
    fallback_alpha: float,
) -> list[float]:
    configured = inference_cfg.get("hep_alpha_sweep")
    if configured is None:
        return [] if fallback_alpha == 0.0 else [fallback_alpha]
    if isinstance(configured, str):
        values = [part.strip() for part in configured.split(",") if part.strip()]
        return [float(value) for value in values]
    if isinstance(configured, (list, tuple)):
        return [float(value) for value in configured]
    raise ValueError("inference.hep_alpha_sweep must be a list or comma-separated string")


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _evaluate_hep_alpha_sweep(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    vocab_size: int,
    *,
    pc_steps: int,
    alphas: list[float],
    num_columns: int,
    top_k: int,
    support_router: str,
    pinned_support: bool,
    hep_update_clip_norm: float | None,
    hep_settling_objective: str,
) -> list[dict[str, float]]:
    rows = []
    ordinary_logits = _forward_with_residual(
        base_params,
        residual_params,
        inputs,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    for alpha in alphas:
        support_instability = _hep_support_instability(
            base_params,
            residual_params,
            inputs,
            pc_steps=pc_steps,
            hep_alpha=alpha,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
            hep_update_clip_norm=hep_update_clip_norm,
        )
        hep_logits = forward_with_hep_alpha(
            base_params,
            residual_params,
            inputs,
            pc_steps=pc_steps,
            hep_alpha=alpha,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
            pinned_support=pinned_support,
            hep_update_clip_norm=hep_update_clip_norm,
            hep_settling_objective=hep_settling_objective,
            targets=targets,
            vocab_size=vocab_size,
        )
        loss = _cross_entropy(hep_logits, targets, vocab_size)
        rows.append(
            {
                "alpha": float(alpha),
                "loss": float(loss),
                "max_logit_delta_from_ordinary": _max_abs(ordinary_logits - hep_logits),
                "support_change_fraction": float(
                    support_instability["support_change_fraction"]
                ),
                "support_transition_count": float(
                    support_instability["support_transition_count"]
                ),
                "pinned_vs_repicked_logit_delta": float(
                    support_instability["pinned_vs_repicked_logit_delta"]
                ),
            }
        )
    return rows


def _hep_support_instability(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    inputs: jnp.ndarray,
    *,
    pc_steps: int,
    hep_alpha: float,
    num_columns: int,
    top_k: int,
    support_router: str,
    hep_update_clip_norm: float | None = None,
) -> dict[str, float | int | bool | None]:
    if pc_steps < 1:
        raise ValueError("pc_steps must be at least 1")
    if hep_alpha < 0.0 or hep_alpha > 1.0:
        raise ValueError("hep_alpha must be between 0.0 and 1.0")
    hidden = _encode(base_params, inputs)
    repicked_settled, initial_support = residual_forward(
        residual_params,
        hidden,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    pinned_settled = repicked_settled
    changed_positions = jnp.zeros(initial_support.shape[:-1], dtype=bool)
    support_transition_count = 0
    for _ in range(1, pc_steps):
        repicked_proposed, repicked_support = residual_forward(
            residual_params,
            repicked_settled,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
        )
        pinned_proposed, _ = residual_forward(
            residual_params,
            pinned_settled,
            num_columns=num_columns,
            top_k=top_k,
            support_router=support_router,
            support_indices=initial_support,
        )
        repicked_update = _clip_update(
            repicked_proposed - repicked_settled, hep_update_clip_norm
        )
        pinned_update = _clip_update(
            pinned_proposed - pinned_settled, hep_update_clip_norm
        )
        repicked_settled = repicked_settled + jnp.float32(hep_alpha) * repicked_update
        pinned_settled = pinned_settled + jnp.float32(hep_alpha) * pinned_update
        changed = jnp.any(repicked_support != initial_support, axis=-1)
        support_transition_count += int(jnp.sum(changed))
        changed_positions = jnp.logical_or(changed_positions, changed)

    total_positions = int(changed_positions.size)
    changed_position_count = int(jnp.sum(changed_positions))
    repicked_logits = _decode(base_params, repicked_settled)
    pinned_logits = _decode(base_params, pinned_settled)
    return {
        "pc_steps": pc_steps,
        "hep_alpha": float(hep_alpha),
        "hep_update_clip_norm": hep_update_clip_norm,
        "support_positions": total_positions,
        "support_changed_positions": changed_position_count,
        "support_change_fraction": (
            changed_position_count / total_positions if total_positions else 0.0
        ),
        "support_transition_count": support_transition_count,
        "pinned_vs_repicked_logit_delta": _max_abs(pinned_logits - repicked_logits),
        "support_changed": changed_position_count > 0,
    }


def _residual_support_audit(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    inputs: jnp.ndarray,
    *,
    num_columns: int,
    top_k: int,
    support_router: str,
) -> dict[str, Any]:
    hidden = _encode(base_params, inputs)
    _, support = residual_forward(
        residual_params,
        hidden,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    support_rows = np.asarray(support).reshape(-1, top_k)
    total_positions = int(support_rows.shape[0])
    column_counts = [0 for _ in range(num_columns)]
    support_set_counts: dict[str, int] = {}
    for row in support_rows.tolist():
        normalized = tuple(sorted(int(index) for index in row))
        key = ",".join(str(index) for index in normalized)
        support_set_counts[key] = support_set_counts.get(key, 0) + 1
        for index in normalized:
            column_counts[index] += 1
    used_columns = sum(1 for count in column_counts if count > 0)
    total_slots = total_positions * top_k
    max_column_count = max(column_counts, default=0)
    return {
        "support_positions": total_positions,
        "top_k": top_k,
        "num_columns": num_columns,
        "total_support_slots": total_slots,
        "used_columns": used_columns,
        "dead_columns": num_columns - used_columns,
        "column_counts": column_counts,
        "unique_support_sets": len(support_set_counts),
        "support_set_counts": dict(sorted(support_set_counts.items())),
        "max_column_fraction": max_column_count / total_slots if total_slots else 0.0,
    }


def _max_hep_delta_from_ordinary(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    inputs: jnp.ndarray,
    *,
    pc_steps: int,
    hep_alpha: float,
    num_columns: int,
    top_k: int,
    support_router: str,
    pinned_support: bool,
    hep_update_clip_norm: float | None,
    hep_settling_objective: str,
    targets: jnp.ndarray,
    vocab_size: int,
) -> float:
    ordinary_logits = _forward_with_residual(
        base_params,
        residual_params,
        inputs,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
    )
    hep_logits = forward_with_hep_alpha(
        base_params,
        residual_params,
        inputs,
        pc_steps=pc_steps,
        hep_alpha=hep_alpha,
        num_columns=num_columns,
        top_k=top_k,
        support_router=support_router,
        pinned_support=pinned_support,
        hep_update_clip_norm=hep_update_clip_norm,
        hep_settling_objective=hep_settling_objective,
        targets=targets,
        vocab_size=vocab_size,
    )
    return _max_abs(ordinary_logits - hep_logits)


def _apply_hep_support_stress(
    base_params: dict[str, Any],
    residual_params: NodeParams,
    inputs: jnp.ndarray,
    *,
    num_columns: int,
    top_k: int,
) -> NodeParams:
    if top_k != 1:
        raise ValueError("model.columns.support_stress requires top_k: 1")
    if num_columns < 2:
        raise ValueError("model.columns.support_stress requires at least 2 columns")
    hidden = _encode(base_params, inputs)
    direction = hidden.reshape(-1, hidden.shape[-1])[0]
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-12)
    w = residual_params.weights["column_score_w"]
    w = w.at[:, 0].set(direction)
    w = w.at[:, 1].set(-direction)
    atom_values = residual_params.weights["atom_values"]
    atom_values = atom_values.at[0, 0, :].set(0.01 * direction)
    atom_values = atom_values.at[1, 0, :].set(-0.01 * direction)
    weights = dict(residual_params.weights)
    weights["column_score_w"] = w
    weights["atom_values"] = atom_values
    return NodeParams(weights=weights, biases=dict(residual_params.biases))


def _residual_update_phase(objective: str) -> str:
    if objective in {"pc_logit_mse", "pc_logit_mse_ce_anchor"}:
        return "pc_residual_update"
    return "residual_update"


def _metric_row(
    *,
    step: int,
    phase: str,
    residual_objective: str,
    base_loss: float,
    residual_loss: float,
    zero_init_loss: float,
    residual_parameter_delta: float,
    max_zero_init_logit_delta: float,
    max_hep_alpha0_logit_delta: float,
    hep_alpha: float | str,
    hep_loss: float | str,
    max_hep_logit_delta_from_ordinary: float | str,
    hep_support_change_fraction: float | str,
    hep_pinned_vs_repicked_logit_delta: float | str,
) -> dict[str, float | int | str]:
    return {
        "step": step,
        "phase": phase,
        "residual_objective": residual_objective,
        "base_loss": base_loss,
        "residual_loss": residual_loss,
        "zero_init_loss": zero_init_loss,
        "residual_parameter_delta": residual_parameter_delta,
        "max_zero_init_logit_delta": max_zero_init_logit_delta,
        "max_hep_alpha0_logit_delta": max_hep_alpha0_logit_delta,
        "hep_alpha": hep_alpha,
        "hep_loss": hep_loss,
        "max_hep_logit_delta_from_ordinary": max_hep_logit_delta_from_ordinary,
        "hep_support_change_fraction": hep_support_change_fraction,
        "hep_pinned_vs_repicked_logit_delta": hep_pinned_vs_repicked_logit_delta,
    }


def _tree_copy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)


def _tree_abs_delta(left: Any, right: Any) -> float:
    leaves_left = jax.tree_util.tree_leaves(left)
    leaves_right = jax.tree_util.tree_leaves(right)
    return float(sum(float(jnp.sum(jnp.abs(a - b))) for a, b in zip(leaves_left, leaves_right)))


def _tree_all_equal(left: Any, right: Any) -> bool:
    leaves_left = jax.tree_util.tree_leaves(left)
    leaves_right = jax.tree_util.tree_leaves(right)
    return all(bool(jnp.array_equal(a, b)) for a, b in zip(leaves_left, leaves_right))


def _max_abs(value: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(value)))
