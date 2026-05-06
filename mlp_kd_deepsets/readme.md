# mlp_kd_deepsets — v1 distillation pipeline

DeepSets / PFN-style student distilled from the L-GATr top-tagging teacher.

## Why this experiment exists

The previous KD experiments (`mlp_kd/`, `mlp_kd_engd/`) plateaued at AUC ≈ 0.79.
Two structural problems explain the floor:

1. **Architecture.** A flat MLP over a flattened `[N_max, 4]` jet has no
   permutation invariance, pads sparsely, and can't represent "set of
   particles" — the inductive bias L-GATr is built around. Engineered features
   helped a little (+0.04 AUC) but didn't fix this.
2. **Teacher interface (the bigger one).** Both prior scripts loaded the
   teacher with `in_s_channels=1`, a single `[1,0,0,1]` beam token, and zero
   tagging features in the scalar channel. The actual teacher
   (`runs/topt_lgatr/seed1001`) was trained with `in_s_channels=8` (7 tagging
   features + 1 global-token bit), 3 spurions (2 lightlike beams + 1 time
   reference), mass regularization, and `/20` momentum scaling. With
   `strict=False` loading, the input-projection mismatch was silent. The
   teacher's KD signal in the prior runs was very likely noise.

This experiment fixes both: DeepSets architecture + teacher used through its
real wrapper (`LGATrWrapper` from `lloca-experiments`), with `embed_tagging_data`
producing exactly the inputs it was trained on.

## Design (v1)

| | |
|---|---|
| Student arch | per-particle MLP `7 → 128 → 128 → 128`, masked **sum + mean + max** pool (→ 384), head `384 → 256 → 128 → 1` |
| Norm / activation | LayerNorm + GELU (no BatchNorm); dropout 0.1 in head only |
| Params | ~150k |
| Student input | the same 7 tagging features the teacher uses internally (`log_pt, log_E, log_pt_rel, log_E_rel, dphi, deta, dr`), pre-normalized via the same `(x − mean) * factor` table |
| Teacher | `runs/topt_lgatr/seed1001/models/model_run0_it174999.pt` (best by val); test AUC 0.9870 / rej@εS=0.5 = 587 |
| Teacher pipeline | imported, not reimplemented (`LGATrWrapper` + `embed_tagging_data` from `lloca-experiments`) |
| KD loss | `α · MSE(student_logit/T, teacher_logit/T) · T²  +  (1 − α) · BCE(student_logit, label)` with `T=2`, `α=0.7` |
| Why logit-MSE | binary classifier with confident teacher; BCE-on-soft-probs over-smooths at T=4 |
| Optimizer | AdamW lr 3e-4, weight decay 1e-4 |
| LR schedule | CosineAnnealingLR over total steps |
| Mixed precision | `torch.cuda.amp` autocast + GradScaler |
| Grad clip | global norm 1.0 |
| Seed | 42 |
| Validation | val AUC tracked every epoch; **best-val** checkpoint saved (not last-epoch) |
| Eval metric | AUC + rej@εS={0.3, 0.5, 0.8} on test, matching teacher reporting |

## Success criteria

- **v1 target**: AUC ≥ 0.93, rej@εS=0.5 ≥ 200. Roughly: match a non-distilled
  PFN baseline. If we hit this, KD ablations (T, α, hint-loss) become
  meaningful.
- **stretch**: AUC ≥ 0.96, rej@εS=0.5 ≥ 600 (ParticleNet territory).
- **gate**: smoke test reproducing teacher AUC ~0.987 on a 2k-jet val slice.
  If smoke fails, abort; teacher interface is wrong and KD is moot.

## Dataset & paths

- Data: `/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/data/toptagging/toptagging_full.npz`
  (1.5 GB; train 1.21M / val 403k / test 404k; keys `kinematics_{train,val,test}`, `labels_{train,val,test}`)
- Teacher: same `~/scratch` tree, under `lloca-experiments/runs/topt_lgatr/seed1001/models/`
- Both live in `~/scratch` → 30-day auto-delete. Read in-place.

## Files in this directory

| | |
|---|---|
| `mlp_kd_deepsets.py`     | training script — teacher load, smoke test, teacher-logits cache, student train, val-tracked best-checkpoint save, final test eval. Skips teacher entirely when `--teacher-cache-dir` already has both `teacher_logits_*.pt`. |
| `eval_deepsets.py`       | standalone test-set eval against a saved checkpoint; writes `roc_curve_deepsets.png`, `prob_dist_deepsets.png`, `test_metrics.json` |
| `run_deepsets.sh`        | v1 sbatch wrapper for CSIS `gpu-short` (1×A100, 24 CPU, 80 GB RAM, 7h30m). Runs smoke test → train → eval. |
| `run_ablation.sh`        | Phase 2 ablation sbatch (smaller: 16 CPU, 48 GB, 3 hr) — student-only, reuses v1 cache. Driven by env vars `ABL_TAG`, `ABL_ALPHA`, `ABL_TEMP`, `ABL_CACHE`. |
| `submit_ablations.sh`    | Login-side launcher that queues all 8 ablation jobs (T/α sweep). Skips combos that already finished. |
| `summarize_ablations.py` | Scans `ablations/<tag>/` and prints a sorted table of (T, α, val AUC, test AUC, rejection); writes `ablations/summary.csv`. |
| `run_phase3.sh`          | Phase 3 sbatch — penultimate-invariant hint distillation. Driven by env vars `HINT_BETA`, `HINT_ALPHA`, `HINT_TEMP`, `HINT_TAG`. First run builds `teacher_invariants_{train,val}.pt` (~10 min once) and reuses v1's logit cache. |

Outputs from a successful run:
- `teacher_logits_train.pt`, `teacher_logits_val.pt` (cached once, reused across reruns)
- `deepsets_student_best.pt` (best-val checkpoint with metadata)
- `history.json` (per-epoch loss + val AUC + rej)
- `final_test_metrics.json`, `test_metrics.json`
- `run_<jobid>.out` / `run_<jobid>.err`

## Launch

From this directory on the CSIS login node:

```
sbatch run_deepsets.sh
```

This stays inside `gpu-short` limits (8 hr cap, 1×A100, 24 CPU, 96 GB RAM).
A 30-epoch run including the one-time teacher-logits cache should fit
comfortably (~1–2 hr expected total).

## Phase 2 — KD ablations

After v1 finishes, sweep T and α around the (T=2, α=0.7) baseline:

```
./submit_ablations.sh
# ... watch with squeue -u $USER ...
python summarize_ablations.py
```

Sweep design (8 runs, two crossed sweeps sharing the v1 baseline):
- α ∈ {0.3, 0.5, 0.9} at T=2
- T ∈ {1, 3, 4} at α=0.7
- Two off-diagonal: (T=1, α=0.5), (T=3, α=0.9)

Each ablation run:
- Reuses v1's `teacher_logits_{train,val}.pt` (no teacher load, no cache pass)
- Trains the student for 30 epochs (~30 min on A100)
- Writes outputs to `ablations/<tag>/`
- Stays inside `gpu-short` with a 3 hr cap and 16 CPU / 48 GB

Per-user concurrent job cap on CSIS is 5, so 5 of the 8 run in parallel and the rest queue automatically.

## Phase 3 — penultimate-invariant hint distillation

**What's added.** A second supervisory signal beyond the teacher's logit: a 48-dim plain-scalar target taken from the **penultimate hidden state at the global token** of the L-GATr teacher. Specifically:

- 16 numbers = scalar component (idx 0) of each of the 16 hidden multivector channels at the global token, *just before* `linear_out`
- 32 numbers = the 32 hidden scalar-stream channels at the global token, same layer
- All 48 are plain scalars (not Lorentz-equivariant), so a non-equivariant student doesn't have to memorize lab-frame outputs

The student gets a small projection head (`HintProjector`: pooled 384 → 128 → 48). Loss becomes:

```
L = α · MSE(s_logit/T, t_logit/T) · T²   +   (1 − α) · BCE(s_logit, label)
                                          +   β · MSE(proj(s_pool), t_invariants)
```

**Why penultimate, not output.** The teacher's *output* multivector at the global token is 16-dim, but only the scalar component receives gradient (it becomes the logit). The other 15 components are quasi-arbitrary — matching them with MSE = student wastes capacity learning noise. The penultimate state is the deepest representation the teacher actually built.

**Why scalars only, not the full multivector at penultimate.** A DeepSets student has no equivariance; predicting equivariant features would force lab-frame memorization. Scalar components are invariants by structure (or plain scalars in the s-stream), so the student can target them without the structure-mismatch tax.

**Caveats**

- Phase 3 makes T meaningful again only if the hint loss dominates. With α=0.7 and β=0.5 the logit-MSE side still has T cancellation (Phase 2 finding); the hint loss has no T at all. So a meaningful T sweep would need a different KD term (BCE-on-soft-probs). We're not redoing that — the hint is the bigger lever.
- Once the invariant cache is built, every subsequent Phase 3 run is student-only training. ~30 min on A100.

**Launch (single β=0.5 run; sweep later if useful):**

```
sbatch run_phase3.sh                                 # default β=0.5, α=0.7, T=2
HINT_BETA=0.1 HINT_TAG=beta01 sbatch run_phase3.sh   # sweep β if first run is promising
HINT_BETA=1.0 HINT_TAG=beta10 sbatch run_phase3.sh
```

The first run will build `teacher_invariants_{train,val}.pt` next to v1's logit cache (~310 MB extra, in `$HOME`). Subsequent runs reuse it.

Outputs land in `phase3/<tag>/` with the same files as v1 (`deepsets_student_best.pt`, `history.json`, `final_test_metrics.json`, `roc_curve_deepsets.png`, `prob_dist_deepsets.png`).

**Expected gain over v1.** The biggest available lever for this student. v1 hit AUC 0.9833 / rej@εS=0.5 = 290 with logit-only KD. The hint adds 48 numbers of representational supervision per jet vs. 1 (the logit). Realistic target: AUC 0.985+ / rej@εS=0.5 in the 350–500 range. Closing fully to teacher (rej@0.5 = 587) is unlikely without two-particle interaction features in the student input.

## Notes / next steps
- **Phase 3 (hint distillation)**: implemented (see Phase 3 section above) — penultimate-invariant target via a forward hook on the teacher's last block, plus a 48-dim hint MSE term.
- **Attention backend**: the teacher's training config used `xformers`. The
  sbatch wrapper passes `--attention-backend torch` for portability; switch
  to `xformers` if benchmarking shows it matters (it shouldn't, since the
  teacher only runs during the cache pass).
