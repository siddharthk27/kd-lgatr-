#!/bin/bash
#SBATCH --job-name=kd_pair
#SBATCH --output=run_pairwise_%j.out
#SBATCH --error=run_pairwise_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00

# Pairwise-feature student: same DeepSets architecture as v1, but with 4 extra
# per-particle features summarizing each particle's relationship to the rest of
# the jet (dR_min, log_kT_min, n_close_03, dR_to_hardest). Reuses v1's teacher
# logit cache; does not need the invariant cache.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAG="${PAIR_TAG:-v1pair}"
EPOCHS="${PAIR_EPOCHS:-30}"
SEED="${PAIR_SEED:-42}"
ALPHA="${PAIR_ALPHA:-0.7}"
TEMP="${PAIR_TEMP:-2.0}"

OUT_DIR="pairwise/$TAG"
mkdir -p "$OUT_DIR"

echo "[$(date '+%F %T')] pairwise run=$TAG  alpha=$ALPHA  T=$TEMP  epochs=$EPOCHS  seed=$SEED"
echo "[$(date '+%F %T')] jobid=$SLURM_JOB_ID  host=$(hostname)  out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

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
    --use-pairwise \
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

echo "[$(date '+%F %T')] pairwise $TAG done."
