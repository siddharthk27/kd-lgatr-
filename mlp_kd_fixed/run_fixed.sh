#!/bin/bash
#SBATCH --job-name=kd_mlp_fix
#SBATCH --output=run_fixed_%j.out
#SBATCH --error=run_fixed_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=04:00:00

# Re-run the original mlp_kd_engd flat-MLP student under the *fixed* L-GATr
# teacher interface (in_s_channels=8, 7 tagging features + global flag,
# 3 spurions, /20 momentum scaling, mass regularization). Teacher logit
# cache is reused from ../mlp_kd_deepsets/ so the teacher never touches GPU.
#
# Inputs (env vars; sensible defaults):
#   MK_TAG          run subdir under runs/  (default "run")
#   MK_KD_LOSS      softmax_kl | logit_mse  (default softmax_kl, the headline)
#   MK_TEMPERATURE  KD temperature (default 4.0; ignored under logit_mse - F2)
#   MK_ALPHA        soft-loss weight (default 0.7)
#   MK_EPOCHS       epochs (default 30)
#   MK_BS           batch size (default 512)
#   MK_SEED         seed (default 42)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAG="${MK_TAG:-run}"
KD_LOSS="${MK_KD_LOSS:-softmax_kl}"
TEMPERATURE="${MK_TEMPERATURE:-4.0}"
ALPHA="${MK_ALPHA:-0.7}"
EPOCHS="${MK_EPOCHS:-30}"
BS="${MK_BS:-512}"
SEED="${MK_SEED:-42}"

OUT_DIR="runs/$TAG"
mkdir -p "$OUT_DIR"

DATA_PATH="/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/data/toptagging/toptagging_full.npz"
TEACHER_CKPT="/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/lloca-experiments/runs/topt_lgatr/seed1001/models/model_run0_it174999.pt"
LLOCA_REPO="/nfs_home/users/dhruvk/jay_agarwal/lgatr/repos/lloca-experiments"
CACHE_DIR="$(realpath ../mlp_kd_deepsets)"

echo "[$(date '+%F %T')] mlp_kd_fixed run=$TAG  kd_loss=$KD_LOSS  T=$TEMPERATURE  alpha=$ALPHA"
echo "[$(date '+%F %T')] epochs=$EPOCHS  bs=$BS  seed=$SEED  cache=$CACHE_DIR"
echo "[$(date '+%F %T')] jobid=$SLURM_JOB_ID  host=$(hostname)  out=$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$CACHE_DIR/teacher_logits_train.pt" || ! -f "$CACHE_DIR/teacher_logits_val.pt" ]]; then
    echo "ERROR: teacher logit cache missing under $CACHE_DIR — run sbatch ../mlp_kd_deepsets/run_deepsets.sh first." >&2
    exit 2
fi

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

$PY -u mlp_kd_fixed.py \
    --data-path        "$DATA_PATH" \
    --teacher-ckpt     "$TEACHER_CKPT" \
    --lloca-repo       "$LLOCA_REPO" \
    --teacher-cache-dir "$CACHE_DIR" \
    --kd-loss          "$KD_LOSS" \
    --temperature      "$TEMPERATURE" \
    --alpha            "$ALPHA" \
    --epochs           "$EPOCHS" \
    --batch-size       "$BS" \
    --seed             "$SEED" \
    --num-workers      8 \
    --out-dir          "$OUT_DIR"

echo "[$(date '+%F %T')] mlp_kd_fixed $TAG done."
