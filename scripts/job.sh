#!/bin/bash
# TEMPLATE. We ASSUME the cluster uses Slurm; confirm with `which sbatch; sinfo` and adjust
# partition/GPU/time flags to what the cluster docs say. Submit with:  sbatch scripts/job.sh
#SBATCH --job-name=ragdet
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs
source setup_env.sh
CONFIG=${CONFIG:-configs/default.yaml}
STAGE=${STAGE:-attack}
python -m ragdet.cli "$STAGE" --config "$CONFIG"
