#!/bin/bash
# Usage:  bash scripts/download_data.sh nq        (or hotpotqa)
# Big raw corpus -> $RAGDET_DATA/<ds>/   (disposable; Colab local disk is fine)
# Small files (qrels, targets) -> $RAGDET_ROOT/data_small/<ds>/   (durable; put on Drive)
# URLs come from the PoisonedRAG repo (prepare_dataset.py + released results folder).
set -euo pipefail
DS=${1:-nq}
ROOT=${RAGDET_ROOT:?set RAGDET_ROOT first}
DATA=${RAGDET_DATA:-$ROOT/data}
mkdir -p "$DATA" "$ROOT/data_small/$DS"
cd "$DATA"
if [ ! -f "$DS/corpus.jsonl" ]; then
  curl -L -o "$DS.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$DS.zip"
  unzip -q -o "$DS.zip" && rm -f "$DS.zip"
fi
curl -L -o "$ROOT/data_small/$DS/targets.json" \
  "https://github.com/sleeepeer/PoisonedRAG/raw/refs/heads/main/results/adv_targeted_results/$DS.json"
cp "$DS/qrels/test.tsv" "$ROOT/data_small/$DS/qrels_test.tsv"
echo "--- raw corpus (disposable):"; ls -la "$DATA/$DS"
echo "--- small durable files:";     ls -la "$ROOT/data_small/$DS"
