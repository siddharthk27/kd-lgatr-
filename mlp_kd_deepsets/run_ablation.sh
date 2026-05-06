#!/bin/bash
#SBATCH --job-name=kd_abl
#SBATCH --output=ablations/abl_%j.out
#SBATCH --error=ablations/abl_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00

# Phase 2 ablation runner — student-only training using v1's cached teacher logits.
# Smaller resource footprint than v1 since we skip teacher loading + cache precompute.
# Inputs (env vars from submit_ablations.sh):
#   ABL_ALPHA  : KD weight α
#   ABL_TEMP   : KD temperature T
#   ABL_TAG    : tag for the run subdir (e.g. "T2_a07")
#   ABL_CACHE  : path to v1 dir containing teacher_logits_{train,val}.pt
#   ABL_EPOCHS : (optional) override training epochs, default 30
#   ABL_SEED   : (optional) override seed, default 42

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p ablations

: "${ABL_ALPHA:?must set ABL_ALPHA}"
: "${ABL_TEMP:?must set ABL_TEMP}"
: "${ABL_TAG:?must set ABL_TAG}"
: "${ABL_CACHE:?must set ABL_CACHE (path to v1 dir with teacher_logits_*.pt)}"
EPOCHS="${ABL_EPOCHS:-30}"
SEED="${ABL_SEED:-42}"

OUT_DIR="ablations/$ABL_TAG"
mkdir -p "$OUT_DIR"

echo "[$(date '+%F %T')] ablation=$ABL_TAG  alpha=$ABL_ALPHA  T=$ABL_TEMP  epochs=$EPOCHS  seed=$SEED"
echo "[$(date '+%F %T')] cache=$ABL_CACHE  out=$OUT_DIR  jobid=$SLURM_JOB_ID  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$ABL_CACHE/teacher_logits_train.pt" || ! -f "$ABL_CACHE/teacher_logits_val.pt" ]]; then
    echo "ERROR: teacher logits cache missing under $ABL_CACHE — run v1 first." >&2
    exit 2
fi

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

$PY -u mlp_kd_deepsets.py \
    --epochs "$EPOCHS" \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --alpha "$ABL_ALPHA" \
    --temperature "$ABL_TEMP" \
    --num-workers 8 \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --teacher-cache-dir "$ABL_CACHE"

echo "[$(date '+%F %T')] training done. eval ..."
$PY -u eval_deepsets.py \
    --checkpoint "$OUT_DIR/deepsets_student_best.pt" \
    --batch-size 1024 \
    --num-workers 8 \
    --out-dir "$OUT_DIR"

echo "[$(date '+%F %T')] ablation $ABL_TAG done."
