#!/bin/bash
#SBATCH --job-name=kd_boost
#SBATCH --output=run_boosteval_%j.out
#SBATCH --error=run_boosteval_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:30:00

# Lorentz-boost robustness eval — runs eval_boosted.py over 20 β values.
#
# Inputs (env vars):
#   BE_TEACHER     1 = run teacher boost-eval (overrides BE_CHECKPOINT)
#   BE_CHECKPOINT  path to student checkpoint (relative to $SLURM_SUBMIT_DIR)
#   BE_OUTDIR      output directory for boost_results.json (relative)
#   BE_BS          batch size (default 256)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TEACHER="${BE_TEACHER:-0}"
CHECKPOINT="${BE_CHECKPOINT:-}"
OUTDIR="${BE_OUTDIR:-runs/boost_default}"
BS="${BE_BS:-256}"

mkdir -p "$OUTDIR"

echo "[$(date '+%F %T')] boost-eval  teacher=$TEACHER  ckpt=$CHECKPOINT  out=$OUTDIR  bs=$BS"
echo "[$(date '+%F %T')] jobid=$SLURM_JOB_ID  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

if [[ "$TEACHER" == "1" ]]; then
    $PY -u eval_boosted.py \
        --teacher \
        --out-dir "$OUTDIR" \
        --batch-size "$BS" \
        --num-workers 8
else
    if [[ -z "$CHECKPOINT" ]]; then
        echo "ERROR: must set either BE_TEACHER=1 or BE_CHECKPOINT=<path>" >&2
        exit 2
    fi
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
        exit 2
    fi
    $PY -u eval_boosted.py \
        --checkpoint "$CHECKPOINT" \
        --out-dir "$OUTDIR" \
        --batch-size "$BS" \
        --num-workers 8
fi

echo "[$(date '+%F %T')] boost-eval done."
