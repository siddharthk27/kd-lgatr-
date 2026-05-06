#!/bin/bash
#SBATCH --job-name=kd_phase3
#SBATCH --output=run_phase3_%j.out
#SBATCH --error=run_phase3_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=05:00:00

# Phase 3: penultimate-invariant hint distillation.
# 1) If teacher_invariants_{train,val}.pt are missing, builds them once
#    (loads teacher with hook, runs through train+val, ~10 min).
# 2) Trains the student with the hint loss enabled.
#
# Inputs (env vars; sensible defaults baked in):
#   HINT_BETA   : weight on the hint MSE term (default 0.5)
#   HINT_ALPHA  : KD weight α (default 0.7, same as v1)
#   HINT_TEMP   : KD temperature T (default 2.0, ablations showed it's degenerate
#                 for our logit-MSE loss, but kept for parametric continuity)
#   HINT_TAG    : tag for the run subdir under phase3/ (default "beta05")
#   HINT_EPOCHS : training epochs (default 30)
#   HINT_SEED   : seed (default 42)

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p phase3

BETA="${HINT_BETA:-0.5}"
ALPHA="${HINT_ALPHA:-0.7}"
TEMP="${HINT_TEMP:-2.0}"
TAG="${HINT_TAG:-beta05}"
EPOCHS="${HINT_EPOCHS:-30}"
SEED="${HINT_SEED:-42}"

OUT_DIR="phase3/$TAG"
mkdir -p "$OUT_DIR"

echo "[$(date '+%F %T')] phase3 run=$TAG  beta=$BETA  alpha=$ALPHA  T=$TEMP  epochs=$EPOCHS  seed=$SEED"
echo "[$(date '+%F %T')] jobid=$SLURM_JOB_ID  host=$(hostname)  out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# v1's cache lives in this directory (cwd of the experiment).
CACHE_DIR="$PWD"
if [[ ! -f "$CACHE_DIR/teacher_logits_train.pt" || ! -f "$CACHE_DIR/teacher_logits_val.pt" ]]; then
    echo "ERROR: v1 logit cache missing — run sbatch run_deepsets.sh first." >&2
    exit 2
fi

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

$PY -u mlp_kd_deepsets.py \
    --epochs "$EPOCHS" \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --alpha "$ALPHA" \
    --temperature "$TEMP" \
    --hint-beta "$BETA" \
    --num-workers 8 \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --teacher-cache-dir "$CACHE_DIR"

echo "[$(date '+%F %T')] training done. eval ..."
$PY -u eval_deepsets.py \
    --checkpoint "$OUT_DIR/deepsets_student_best.pt" \
    --batch-size 1024 \
    --num-workers 8 \
    --out-dir "$OUT_DIR"

echo "[$(date '+%F %T')] phase3 $TAG done."
