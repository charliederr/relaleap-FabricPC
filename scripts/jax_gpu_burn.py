#!/usr/bin/env python3
"""Run a short JAX matmul workload without selecting a backend."""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--size", type=int, default=4096)
    args = parser.parse_args()

    print("JAX default backend:", jax.default_backend(), flush=True)
    print("JAX devices:", jax.devices(), flush=True)

    key_a, key_b = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(key_a, (args.size, args.size), dtype=jnp.float32)
    b = jax.random.normal(key_b, (args.size, args.size), dtype=jnp.float32)
    b = b / math.sqrt(args.size)

    @jax.jit
    def step(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.tanh(x @ y)

    a = step(a, b).block_until_ready()
    print("Work array devices:", a.devices(), flush=True)
    print(f"Running for {args.seconds:g}s with matrix size {args.size}...", flush=True)

    deadline = time.monotonic() + args.seconds
    iterations = 0
    while time.monotonic() < deadline:
        a = step(a, b).block_until_ready()
        iterations += 1

    checksum = float(jnp.mean(a).block_until_ready())
    print(f"Completed {iterations} iterations; checksum={checksum:.6g}", flush=True)


if __name__ == "__main__":
    main()
