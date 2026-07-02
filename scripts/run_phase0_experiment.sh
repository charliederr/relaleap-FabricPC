#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/ni/repos/fpc/py3/bin/python}"
CONFIG="${1:-configs/char_smoke_hep_support_stress_clipped.yaml}"

stem="$(basename "$CONFIG")"
stem="${stem%.yaml}"
OUT="${2:-results/runs/${stem}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

echo "Python: $PYTHON"
echo "Config: $CONFIG"
echo "Output: $OUT"

"$PYTHON" -m relaleap_fabricpc.experiments.run \
  --config "$CONFIG" \
  --out "$OUT"

echo
echo "Wrote:"
echo "  $OUT/summary.json"
echo "  $OUT/metrics.csv"
echo "  $OUT/notes.md"
