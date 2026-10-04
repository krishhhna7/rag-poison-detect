#!/bin/bash
# Source this on the cluster:  source setup_env.sh
# Every value marked CONFIRM must be checked against the cluster's own documentation.

# CONFIRM: large-storage path (guess: /data/<upi>). Keep code in git, big files here.
export RAGDET_ROOT="${RAGDET_ROOT:-/data/$USER/ragdet}"
mkdir -p "$RAGDET_ROOT"

# CONFIRM: proxy needed for pip/conda/huggingface downloads (unverified guess below).
# export PROXY="http://squid.auckland.ac.nz:3128"
if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY"
fi

# Keep model weights and caches off the (probably small) home quota.
export HF_HOME="$RAGDET_ROOT/hf_cache"
export PIP_CACHE_DIR="$RAGDET_ROOT/pip_cache"
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Environment (created once): python -m venv "$RAGDET_ROOT/venv"
if [ -f "$RAGDET_ROOT/venv/bin/activate" ]; then source "$RAGDET_ROOT/venv/bin/activate"; fi
echo "RAGDET_ROOT=$RAGDET_ROOT  HF_HOME=$HF_HOME"
