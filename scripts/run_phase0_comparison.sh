#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/ni/repos/fpc/py3/bin/python}"
OUT="${1:-results/comparisons/phase0_script_$(date +%Y%m%d_%H%M%S)}"
shift || true

COMPARE_ARGS=(--out "$OUT")
CHECK_ARGS=(--comparison-dir "$OUT")

if (($# == 0)); then
  COMPARE_ARGS+=(--baseline-reference baselines/phase0_fabricpc_comparison.json)
  CHECK_ARGS+=(--baseline-reference baselines/phase0_fabricpc_comparison.json)
else
  for config in "$@"; do
    if [[ ! -f "$config" ]]; then
      echo "Config not found: $config" >&2
      exit 2
    fi
    COMPARE_ARGS+=(--config "$config")
  done
fi

echo "Python: $PYTHON"
echo "Output: $OUT"
if (($# == 0)); then
  echo "Configs: default Phase 0 comparison"
else
  printf 'Configs:\n'
  printf '  %s\n' "$@"
fi

"$PYTHON" -m relaleap_fabricpc.experiments.compare "${COMPARE_ARGS[@]}"
"$PYTHON" -m relaleap_fabricpc.experiments.check_artifacts "${CHECK_ARGS[@]}"

echo
echo "Wrote:"
echo "  $OUT/summary.json"
echo "  $OUT/metrics.csv"
echo "  $OUT/notes.md"
