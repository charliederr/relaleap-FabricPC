#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
unset JAX_PLATFORMS
export RELALEAP_FABRICPC_JAX_AUTO=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYTHON="${PYTHON:-/home/ni/repos/fpc/py3/bin/python}"
OUT="${1:-results/comparisons/phase0_gpu}"

"$PYTHON" -m relaleap_fabricpc.experiments.compare --out "$OUT"
"$PYTHON" -m relaleap_fabricpc.experiments.check_artifacts --comparison-dir "$OUT"
