#!/bin/bash
#SBATCH --job-name=kd_deepsets
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G
#SBATCH --time=07:30:00

# v1 distillation: PFN-style DeepSets student trained against L-GATr teacher
# (seed1001/it174999). Runs first a teacher-AUC smoke test, then caches
# teacher logits over train+val once, then trains the student for 30 epochs.
# Stays inside gpu-short limits: 8 hr cap, 1 A100, ≤24 CPU, ≤96 GB RAM.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
echo "[$(date '+%F %T')] starting on $(hostname) jobid=$SLURM_JOB_ID"
echo "[setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

ENV=/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging
PY=$ENV/bin/python

# ---- Step 1: teacher smoke test (fast; aborts the whole job if teacher is broken) ----
echo "[$(date '+%F %T')] teacher smoke test ..."
$PY -u mlp_kd_deepsets.py --smoke-test --num-workers 4 \
    --attention-backend xformers
echo "[$(date '+%F %T')] smoke test passed."

# ---- Step 2: full training (caches teacher logits + trains student) ----
echo "[$(date '+%F %T')] training ..."
$PY -u mlp_kd_deepsets.py \
    --epochs 30 \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --alpha 0.7 \
    --temperature 2.0 \
    --num-workers 8 \
    --seed 42 \
    --attention-backend xformers \
    --out-dir .

echo "[$(date '+%F %T')] training done. Running final eval..."

# ---- Step 3: standalone eval against best-val checkpoint ----
$PY -u eval_deepsets.py \
    --checkpoint deepsets_student_best.pt \
    --batch-size 1024 \
    --num-workers 8 \
    --out-dir .

echo "[$(date '+%F %T')] all done."
