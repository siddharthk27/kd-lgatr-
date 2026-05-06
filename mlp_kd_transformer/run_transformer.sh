#!/bin/bash
#SBATCH --job-name=kd_xform
#SBATCH --output=run_transformer_%j.out
#SBATCH --error=run_transformer_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=04:00:00

# Option 2b: small transformer student (4 blocks, d_model=64, 4 heads, FFN=256).
# Reuses v1's teacher logit cache from ../mlp_kd_deepsets/. Pairwise features
# enabled by default (--use-pairwise).
#
# Optional env vars:
#   XF_TAG       run subdir under ./runs/  (default "v1")
#   XF_EPOCHS    epochs (default 30)
#   XF_SEED      seed (default 42)
#   XF_HINT_BETA hint loss weight, 0 disables (default 0)
#   XF_DMODEL / XF_NHEADS / XF_NBLOCKS / XF_FFN  override architecture

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAG="${XF_TAG:-v1}"
EPOCHS="${XF_EPOCHS:-30}"
SEED="${XF_SEED:-42}"
HINT_BETA="${XF_HINT_BETA:-0.0}"
DMODEL="${XF_DMODEL:-64}"
NHEADS="${XF_NHEADS:-4}"
NBLOCKS="${XF_NBLOCKS:-4}"
FFN="${XF_FFN:-256}"

OUT_DIR="runs/$TAG"
mkdir -p "$OUT_DIR"

# Teacher cache lives in the deepsets sibling directory (built by v1).
CACHE_DIR="$(realpath ../mlp_kd_deepsets)"

echo "[$(date '+%F %T')] xform run=$TAG  d_model=$DMODEL nheads=$NHEADS nblocks=$NBLOCKS ffn=$FFN"
echo "[$(date '+%F %T')] beta=$HINT_BETA  epochs=$EPOCHS  seed=$SEED  cache=$CACHE_DIR"
echo "[$(date '+%F %T')] jobid=$SLURM_JOB_ID  host=$(hostname)  out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$CACHE_DIR/teacher_logits_train.pt" || ! -f "$CACHE_DIR/teacher_logits_val.pt" ]]; then
    echo "ERROR: teacher logit cache missing under $CACHE_DIR — run sbatch ../mlp_kd_deepsets/run_deepsets.sh first." >&2
    exit 2
fi

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

$PY -u mlp_kd_transformer.py \
    --epochs "$EPOCHS" \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --alpha 0.7 \
    --temperature 2.0 \
    --d-model "$DMODEL" \
    --num-heads "$NHEADS" \
    --num-blocks "$NBLOCKS" \
    --ffn-dim "$FFN" \
    --dropout 0.1 \
    --hint-beta "$HINT_BETA" \
    --num-workers 8 \
    --seed "$SEED" \
    --out-dir "$OUT_DIR" \
    --teacher-cache-dir "$CACHE_DIR"

echo "[$(date '+%F %T')] training done. eval ..."
$PY -u eval_transformer.py \
    --checkpoint "$OUT_DIR/transformer_student_best.pt" \
    --batch-size 1024 \
    --num-workers 8 \
    --out-dir "$OUT_DIR"

echo "[$(date '+%F %T')] xform $TAG done."
