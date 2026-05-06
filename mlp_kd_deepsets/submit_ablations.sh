#!/bin/bash
# Submit Phase 2 KD ablations: sweep T and alpha around the v1 baseline (T=2, α=0.7).
# Stays at 5 concurrent jobs (per-user CSIS cap). Run from this directory AFTER v1 has finished.
#
# Sweep design (8 runs total, two crossed sweeps sharing the v1 baseline):
#   - alpha sweep at T=2:    α ∈ {0.3, 0.5, 0.9}     (3 runs)
#   - T sweep at α=0.7:      T ∈ {1, 3, 4}           (3 runs)
#   - Two off-diagonal points:  (T=1, α=0.5), (T=3, α=0.9)
#
# v1 (T=2, α=0.7) is the reference; do NOT re-submit it here.

set -euo pipefail

# Path to the v1 run directory containing teacher_logits_{train,val}.pt
V1_DIR="${V1_DIR:-$PWD}"

if [[ ! -f "$V1_DIR/teacher_logits_train.pt" || ! -f "$V1_DIR/teacher_logits_val.pt" ]]; then
    echo "ERROR: cache not found in V1_DIR=$V1_DIR" >&2
    echo "  Either run v1 first (sbatch run_deepsets.sh) or set V1_DIR=/abs/path/to/v1" >&2
    exit 2
fi

# Tagged sweep: (tag, alpha, T)
RUNS=(
    "T2_a03   0.3   2.0"
    "T2_a05   0.5   2.0"
    "T2_a09   0.9   2.0"
    "T1_a07   0.7   1.0"
    "T3_a07   0.7   3.0"
    "T4_a07   0.7   4.0"
    "T1_a05   0.5   1.0"
    "T3_a09   0.9   3.0"
)

mkdir -p ablations
echo "Submitting ${#RUNS[@]} ablations using cache at $V1_DIR"
echo

for line in "${RUNS[@]}"; do
    read -r tag alpha temp <<<"$line"
    if [[ -d "ablations/$tag" && -f "ablations/$tag/final_test_metrics.json" ]]; then
        echo "  [skip] $tag already complete"
        continue
    fi
    jid=$(ABL_TAG="$tag" ABL_ALPHA="$alpha" ABL_TEMP="$temp" ABL_CACHE="$V1_DIR" \
          sbatch --parsable run_ablation.sh)
    echo "  [submit] $tag  alpha=$alpha  T=$temp  -> jobid=$jid"
done

echo
echo "Submitted. Monitor with: squeue -u \$USER"
echo "Summarize when done: python summarize_ablations.py"
