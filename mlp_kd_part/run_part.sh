#!/bin/bash
#SBATCH --job-name=kd_part
#SBATCH --output=run_part_%j.out
#SBATCH --error=run_part_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=06:00:00

# ParT-style transformer student with pairwise-feature attention bias.
# Reuses v1's teacher logit cache from ../mlp_kd_deepsets/.
#
# Inputs (env vars; sensible defaults):
#   PT_TAG       run subdir under runs/  (default "v1")
#   PT_EPOCHS    epochs (default 50)
#   PT_BS        batch size (default 256 — pair tensors are memory-heavy)
#   PT_SEED      seed (default 42)
#   PT_ALPHA     KD soft-loss weight (default 0.7); set to 0.0 for from-scratch baseline
#   PT_HINT_BETA hint loss weight, 0 disables (default 0)
#   PT_KD_LOSS   KD soft-loss form: logit_mse | softmax_kl  (default logit_mse)
#   PT_TEMPERATURE  KD temperature (default 2.0; tunable only for softmax_kl)
#   PT_DMODEL / PT_NHEADS / PT_NBLOCKS / PT_FFN  override architecture

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAG="${PT_TAG:-v1}"
EPOCHS="${PT_EPOCHS:-50}"
BS="${PT_BS:-256}"
SEED="${PT_SEED:-42}"
ALPHA="${PT_ALPHA:-0.7}"
HINT_BETA="${PT_HINT_BETA:-0.0}"
KD_LOSS="${PT_KD_LOSS:-logit_mse}"
TEMPERATURE="${PT_TEMPERATURE:-2.0}"
DMODEL="${PT_DMODEL:-64}"
NHEADS="${PT_NHEADS:-4}"
NBLOCKS="${PT_NBLOCKS:-4}"
FFN="${PT_FFN:-256}"

OUT_DIR="runs/$TAG"
mkdir -p "$OUT_DIR"

CACHE_DIR="$(realpath ../mlp_kd_deepsets)"

echo "[$(date '+%F %T')] part run=$TAG  d_model=$DMODEL nheads=$NHEADS nblocks=$NBLOCKS ffn=$FFN"
echo "[$(date '+%F %T')] alpha=$ALPHA  hint_beta=$HINT_BETA  kd_loss=$KD_LOSS  T=$TEMPERATURE  epochs=$EPOCHS  bs=$BS  seed=$SEED  cache=$CACHE_DIR"
echo "[$(date '+%F %T')] jobid=$SLURM_JOB_ID  host=$(hostname)  out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$CACHE_DIR/teacher_logits_train.pt" || ! -f "$CACHE_DIR/teacher_logits_val.pt" ]]; then
    echo "ERROR: teacher logit cache missing under $CACHE_DIR — run sbatch ../mlp_kd_deepsets/run_deepsets.sh first." >&2
    exit 2
fi

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

$PY -u mlp_kd_part.py \
    --epochs "$EPOCHS" \
    --batch-size "$BS" \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --alpha "$ALPHA" \
    --temperature "$TEMPERATURE" \
    --kd-loss "$KD_LOSS" \
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
$PY -u eval_part.py \
    --checkpoint "$OUT_DIR/part_student_best.pt" \
    --batch-size "$BS" \
    --num-workers 8 \
    --out-dir "$OUT_DIR"

echo "[$(date '+%F %T')] part $TAG done."
