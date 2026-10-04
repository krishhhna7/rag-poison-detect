#!/bin/bash
# Usage:  source setup_env.sh && bash scripts/download_data.sh nq        (or hotpotqa)
# Produces $RAGDET_ROOT/data/<ds>/{corpus.jsonl, queries.jsonl, qrels/test.tsv, targets.json}
# URLs come from the PoisonedRAG repo (prepare_dataset.py) and its released results folder.
# If the cluster blocks these hosts, download on your laptop and copy the folder over.
set -euo pipefail
DS=${1:-nq}
ROOT=${RAGDET_ROOT:?run: source setup_env.sh first}
mkdir -p "$ROOT/data" && cd "$ROOT/data"
if [ ! -d "$DS" ]; then
  curl -L -o "$DS.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$DS.zip"
  unzip -q "$DS.zip" && rm "$DS.zip"
fi
curl -L -o "$DS/targets.json" \
  "https://github.com/sleeepeer/PoisonedRAG/raw/refs/heads/main/results/adv_targeted_results/$DS.json"
echo "Done. Contents:"; ls -la "$DS" "$DS/qrels"
