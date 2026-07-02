"""Downstream FabricPC node implementations used by the Phase 0 port."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from relaleap_fabricpc.jax_setup import configure_jax

configure_jax()

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import InitializerBase, NormalInitializer
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec


SUPPORT_ROUTER_CHOICES = {"linear", "contextual_mlp", "contextual_mlp_causal"}


def layer_norm_last(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + 1e-5) * gamma + beta


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (
        1.0 + jnp.tanh(jnp.sqrt(jnp.array(2.0 / jnp.pi)) * (x + 0.044715 * x**3))
    )


def init_residual_params(
    key: jax.Array,
    *,
    hidden_dim: int,
    num_columns: int,
    atoms_per_column: int,
    support_router: str,
    contextual_router_hidden_dim: int,
) -> NodeParams:
    """Initialize sparse residual-column parameters."""

    if support_router not in SUPPORT_ROUTER_CHOICES:
        raise ValueError(
            "support_router must be one of: linear, contextual_mlp, contextual_mlp_causal"
        )
    if num_columns < 1:
        raise ValueError("num_columns must be positive")
    if atoms_per_column < 1:
        raise ValueError("atoms_per_column must be positive")
    if contextual_router_hidden_dim < 1:
        raise ValueError("contextual_router_hidden_dim must be positive")

    feature_dim = hidden_dim * 5 + 3
    key1, key2 = jax.random.split(key)
    weights = {
        "column_score_w": jnp.zeros((hidden_dim, num_columns), dtype=jnp.float32),
        "context_ln_gamma": jnp.ones((feature_dim,), dtype=jnp.float32),
        "context_w1": jax.random.normal(
            key1, (feature_dim, contextual_router_hidden_dim), dtype=jnp.float32
        )
        * jnp.float32(0.02),
        "context_w2": jax.random.normal(
            key2, (contextual_router_hidden_dim, num_columns), dtype=jnp.float32
        )
        * jnp.float32(0.02),
        "atom_logits": jnp.zeros(
            (num_columns, atoms_per_column), dtype=jnp.float32
        ),
        "atom_values": jnp.zeros(
            (num_columns, atoms_per_column, hidden_dim), dtype=jnp.float32
        ),
    }
    biases = {
        "context_ln_beta": jnp.zeros((feature_dim,), dtype=jnp.float32),
        "context_b1": jnp.zeros(
            (contextual_router_hidden_dim,), dtype=jnp.float32
        ),
    }
    return NodeParams(weights=weights, biases=biases)


def contextual_features(hidden: jnp.ndarray, support_router: str) -> jnp.ndarray:
    current = hidden
    previous = jnp.concatenate([current[:, :1, :], current[:, :-1, :]], axis=1)
    next_hidden = jnp.concatenate([current[:, 1:, :], current[:, -1:, :]], axis=1)
    seq_len = current.shape[1]
    if seq_len <= 1:
        normalized_position = jnp.zeros(
            (current.shape[0], seq_len, 1), dtype=current.dtype
        )
    else:
        normalized_position = jnp.linspace(
            0.0, 1.0, seq_len, dtype=current.dtype
        ).reshape(1, seq_len, 1)
        normalized_position = jnp.broadcast_to(
            normalized_position, (current.shape[0], seq_len, 1)
        )
    angle = normalized_position * (2.0 * jnp.pi)
    if support_router == "contextual_mlp_causal":
        next_hidden = jnp.zeros_like(current)
        next_delta = jnp.zeros_like(current)
    else:
        next_delta = next_hidden - current
    return jnp.concatenate(
        [
            current,
            previous,
            next_hidden,
            current - previous,
            next_delta,
            normalized_position,
            jnp.sin(angle),
            jnp.cos(angle),
        ],
        axis=-1,
    )


def score_columns(
    params: NodeParams,
    hidden: jnp.ndarray,
    *,
    support_router: str,
) -> jnp.ndarray:
    if support_router in {"contextual_mlp", "contextual_mlp_causal"}:
        features = contextual_features(hidden, support_router)
        normalized = layer_norm_last(
            features,
            params.weights["context_ln_gamma"],
            params.biases["context_ln_beta"],
        )
        h = gelu(jnp.matmul(normalized, params.weights["context_w1"]) + params.biases["context_b1"])
        return jnp.matmul(h, params.weights["context_w2"])
    return jnp.matmul(hidden, params.weights["column_score_w"])


def residual_forward(
    params: NodeParams,
    hidden: jnp.ndarray,
    *,
    num_columns: int,
    top_k: int,
    support_router: str,
    support_indices: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply sparse residual columns and return output plus selected support."""

    if top_k < 1 or top_k > num_columns:
        raise ValueError("top_k must be between 1 and num_columns")
    scores = score_columns(params, hidden, support_router=support_router)
    tie_breaker = (
        jnp.arange(num_columns, 0, -1, dtype=hidden.dtype).reshape(1, 1, num_columns)
        * jnp.asarray(1e-6, dtype=hidden.dtype)
    )
    scores = scores + tie_breaker
    if support_indices is None:
        top_values, top_indices = jax.lax.top_k(scores, top_k)
    else:
        top_indices = support_indices
        top_values = jnp.take_along_axis(scores, top_indices, axis=-1)

    column_weights = jax.nn.softmax(top_values, axis=-1)
    atom_weights = jax.nn.softmax(params.weights["atom_logits"], axis=-1)
    column_values = jnp.einsum("ca,cah->ch", atom_weights, params.weights["atom_values"])
    selected_values = column_values[top_indices]
    residual = jnp.einsum("bsk,bskh->bsh", column_weights, selected_values)
    return hidden + residual, top_indices


class ResidualColumnsNode(NodeBase):
    """FabricPC node for zero-initialized sparse residual columns."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        num_columns: int,
        atoms_per_column: int,
        top_k: int,
        support_router: str = "linear",
        contextual_router_hidden_dim: int | None = None,
        latent_init=NormalInitializer(std=0.01),
        weight_init=NormalInitializer(std=0.01),
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        **kwargs: Any,
    ) -> None:
        if len(shape) != 2:
            raise ValueError("ResidualColumnsNode shape must be (seq_len, hidden_dim)")
        if top_k < 1 or top_k > num_columns:
            raise ValueError("top_k must be between 1 and num_columns")
        if support_router not in SUPPORT_ROUTER_CHOICES:
            raise ValueError(
                "support_router must be one of: linear, contextual_mlp, contextual_mlp_causal"
            )
        hidden_dim = int(shape[-1])
        if contextual_router_hidden_dim is None:
            contextual_router_hidden_dim = hidden_dim * 2
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            num_columns=num_columns,
            atoms_per_column=atoms_per_column,
            top_k=top_k,
            support_router=support_router,
            contextual_router_hidden_dim=contextual_router_hidden_dim,
            **kwargs,
        )
        self.num_columns = num_columns
        self.atoms_per_column = atoms_per_column
        self.top_k = top_k
        self.support_router = support_router
        self.contextual_router_hidden_dim = contextual_router_hidden_dim

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        return init_residual_params(
            key,
            hidden_dim=int(node_shape[-1]),
            num_columns=int(config["num_columns"]),
            atoms_per_column=int(config["atoms_per_column"]),
            support_router=str(config.get("support_router", "linear")),
            contextual_router_hidden_dim=int(
                config.get("contextual_router_hidden_dim", int(node_shape[-1]) * 2)
            ),
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> tuple[jax.Array, NodeState]:
        edge_key = next(iter(inputs))
        hidden = inputs[edge_key]
        config = node_info.node_config
        z_mu, _ = residual_forward(
            params,
            hidden,
            num_columns=int(config["num_columns"]),
            top_k=int(config["top_k"]),
            support_router=str(config.get("support_router", "linear")),
        )
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=z_mu, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
