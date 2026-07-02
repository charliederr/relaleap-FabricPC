#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/ni/repos/fpc/py3/bin/python}"

"$PYTHON" scripts/jax_gpu_burn.py "$@"
