#!/bin/bash
# Run all stages in order (each stage saves to disk, so a crash can resume from the last stage).
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
for stage in attack features a2 evaluate; do
  echo "=== $stage ==="
  python -m ragdet.cli "$stage" --config "$CONFIG"
done
