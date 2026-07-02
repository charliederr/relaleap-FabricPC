"""JAX runtime defaults for local RelaLeap FabricPC commands."""

from __future__ import annotations

import os


def configure_jax() -> None:
    """Set conservative defaults before importing JAX.

    The shared environment may have CUDA wheels even when the active driver
    cannot initialize cuDNN. CPU is the default for local commands unless the
    process sets RELALEAP_FABRICPC_JAX_AUTO=1 before importing this package.
    """

    if os.environ.get("RELALEAP_FABRICPC_JAX_AUTO") != "1":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")
