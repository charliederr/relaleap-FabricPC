"""Small deterministic data builders for local Phase 0 runs."""

from __future__ import annotations

import re
from typing import Any

import jax.numpy as jnp


TINY_SHAKESPEARE_EXCERPT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
""".strip()

TINY_TOKENIZED_EXCERPT = """
the residual learner observes tokens across a compact language stream .
temporal consistency compares each prediction with the following position .
entropy settling is label free but may chase overconfident local choices .
guided settling uses labels and remains an oracle for diagnostics only .
support stress changes sparse column choices during repeated settling .
the promotion gate asks for evidence beyond character level smoke runs .
""".strip()


def build_batch(dataset: str, seq_len: int, batch_size: int) -> tuple[Any, Any, int]:
    """Build a deterministic local batch with integer inputs and targets."""

    if dataset == "tiny_shakespeare_char":
        return _build_char_batch(seq_len=seq_len, batch_size=batch_size)
    if dataset == "tiny_shakespeare_word":
        return _build_word_batch(seq_len=seq_len, batch_size=batch_size)
    raise ValueError(
        "data.dataset must be one of: tiny_shakespeare_char, tiny_shakespeare_word"
    )


def _starts(length: int, seq_len: int, batch_size: int) -> list[int]:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    last = length - seq_len - 2
    if last < 0:
        raise ValueError("source text is too short for the configured seq_len")
    if batch_size == 1:
        return [0]
    return [round(i * last / (batch_size - 1)) for i in range(batch_size)]


def _build_char_batch(seq_len: int, batch_size: int) -> tuple[Any, Any, int]:
    text = (TINY_SHAKESPEARE_EXCERPT + "\n") * 16
    vocab = sorted(set(text))
    char_to_id = {char: index for index, char in enumerate(vocab)}
    encoded = jnp.array([char_to_id[char] for char in text], dtype=jnp.int32)

    rows = []
    targets = []
    for start in _starts(len(encoded), seq_len, batch_size):
        rows.append(encoded[start : start + seq_len])
        targets.append(encoded[start + 1 : start + seq_len + 1])
    return jnp.stack(rows), jnp.stack(targets), len(vocab)


def _build_word_batch(seq_len: int, batch_size: int) -> tuple[Any, Any, int]:
    tokens = re.findall(r"[a-z]+|[.,]", TINY_TOKENIZED_EXCERPT.lower())
    repeats = max(8, (seq_len * batch_size * 2) // len(tokens) + 2)
    token_stream = tokens * repeats
    vocab = sorted(set(token_stream))
    token_to_id = {token: index for index, token in enumerate(vocab)}
    encoded = jnp.array([token_to_id[token] for token in token_stream], dtype=jnp.int32)

    rows = []
    targets = []
    for start in _starts(len(encoded), seq_len, batch_size):
        rows.append(encoded[start : start + seq_len])
        targets.append(encoded[start + 1 : start + seq_len + 1])
    return jnp.stack(rows), jnp.stack(targets), len(vocab)
