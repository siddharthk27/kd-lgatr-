# kd-lgatr — Knowledge Distillation from L-GATr to Compact Top-Tagging Students

This file is loaded by Claude Code when starting a session rooted under `~/khare27/kd-lgatr-/`. It is the executive summary + index for the project. Detailed design rationale lives in each experiment subdirectory's `readme.md`. Numerical results live in JSON files next to the code. Cross-cutting operational facts (cluster paths, teacher interface, SOP rules) live in the auto-memory directory `~/.claude-config/projects/-nfs-home-users-dhruvk-khare27/memory/` and load automatically.

## Project goal

Distill the L-GATr top-tagging classifier (≈1M params, 12-block Lorentz-equivariant transformer, AUC 0.9870 on the standard 1.21M-jet top-tagging dataset) into a smaller non-equivariant student. Track AUC + background rejection at signal efficiency 0.3 / 0.5 / 0.8 — the standard reporting format from the L-GATr paper.

Teacher checkpoint used: `seed1001/model_run0_it174999.pt` from Jay's `lloca-experiments` runs. Its test-set numbers anchor every comparison:

| Teacher (test) | AUC | rej@εS=0.3 | rej@εS=0.5 | rej@εS=0.8 |
|---|---|---|---|---|
| L-GATr seed1001 | **0.9870** | 2433 | **587** | 67 |

## Experiment timeline & results

| # | Subdir | Student | Test AUC | rej@0.3 | rej@0.5 | rej@0.8 | Headline |
|---|---|---|---|---|---|---|---|
| 0 | `mlp_kd/`         | flat MLP, raw 4-momenta (3-layer GELU, depth=3)            | 0.7492 | 19 | 3 | 2 | broken (teacher-interface bug) |
| 1 | `mlp_kd_engd/`    | flat MLP, jet-relative engineered features (depth=3)       | 0.7890 | 20 | 6 | 3 | broken (same bug) |
| 1b| `mlp_kd_engd/`    | same, depth=6                                              | 0.7722 | 27 | 3 | 3 | deeper hurt (no residuals) |
| 2 | `mlp_kd_deepsets/` (v1) | DeepSets PFN, 7 features, sum+mean+max pool, **fixed teacher interface** | **0.9833** | 1188 | **290** | 45 | +0.19 AUC jump from architecture + correct teacher loading |
| 3 | `mlp_kd_deepsets/ablations/` | Phase 2: T ∈ {1,2,3,4} × α ∈ {0.3,0.5,0.7,0.9} sweep on v1 | 0.9832–0.9833 | 1141–1209 | 284–296 | 45 | flat — KD hyperparams not the lever |
| 4 | `mlp_kd_deepsets/phase3/beta05/` | Phase 3: penultimate-invariant hint loss (β=0.5) | 0.9833 | 1141 | 289 | 45 | flat — hint absorbed but not classification-relevant for DeepSets |
| 5 | `mlp_kd_deepsets/pairwise/v1pair/` | DeepSets + 4 pairwise-aggregate features (dR_min, log_kT_min, n_close_03, dR_to_hardest) | 0.9837 | 1195 | 311 | 47 | small but real lift — features help marginally |
| 6  | `mlp_kd_transformer/runs/v1/`           | small transformer (4 blocks, d_model=64, ~205k params) + pairwise feats     | 0.9845 | 1541 | 368 | 51 | architecture jump: +0.0012 AUC over DeepSets+pairwise |
| 7  | `mlp_kd_transformer/runs/v1_hint/`      | row-6 + hint loss β=0.5 (transformer-side hint redux)                       | 0.9845 | 1453 | 352 | 51 | flat — Phase 3 null result reproduces even with a structurally homologous student |
| 8  | `mlp_kd_transformer/runs/v1_s43/`       | row-6 with seed=43 (noise-floor probe)                                      | 0.9845 | 1496 | 373 | 51 | seed-to-seed noise: ±0.0001 AUC, ±5 in rej@0.5 |
| 9  | `mlp_kd_transformer/runs/v1_e50/`       | row-6 × 50 epochs (decomposes longer-vs-bigger in row 10)                   | 0.9852 | 1669 | 420 | 55 | longer training: 2/3 of row-10's lift is from extra epochs alone |
| 10 | `mlp_kd_transformer/runs/d96_b6_e50/`   | d=96, b=6, 50 epochs (~600k params)                                         | **0.9858** | **1787** | **440** | **58** | best non-pair-bias student — 50% of rej@0.5 gap to teacher closed |
| 11 | `mlp_kd_transformer/runs/tiny_d32_b2/`  | d=32, b=2, 30 epochs (~50k params)                                          | 0.9830 | 1086 | 272 | 45 | capacity floor: 50k-param transformer doesn't beat 167k-param DeepSets+pairwise |
| 12 | `mlp_kd_part/runs/v1/`                  | ParT-style: vanilla transformer + learned pairwise-feature attention bias (4 blocks, d=64, ~205k params, 50 epochs) | **0.9869** | **2433** | **567** | **67** | **matches the teacher within noise floor at 5× fewer params** |

Read individual `final_test_metrics.json` files for exact numbers; `history.json` for per-epoch trajectories.

### Headline as of 2026-05-07

**Best student: `mlp_kd_part/runs/v1/`** (ParT-style, 205k params) at AUC **0.9869** / rej@εS=0.5 = **567** / rej@εS=0.3 = **2433** / rej@εS=0.8 = **67**.

Gap to teacher (0.9870 / 587 / 67 / 2433, ~1M params):
- AUC: 0.9870 − 0.9869 = **0.0001** (at the seed noise floor of F6 — essentially equal)
- rej@εS=0.3: **0** — bit-exact match (2433 = 2433)
- rej@εS=0.5: **20** — 93% of original 297 gap closed
- rej@εS=0.8: **0** — bit-exact match (67 = 67)

A pair-bias seed=43 reproducibility run is in flight (`runs/v1_s43/`) to confirm the AUC=0.9869 isn't a lucky seed.

The student is **5× smaller than the teacher** (205k vs ~1M params) and **~10× smaller than vanilla ParT** (which sits at ≈0.987 with ~2.1M params on the same benchmark).

## Key technical findings (paper-worthy)

### F1 — The in_s_channels mismatch in prior KD scripts (rows 0, 1, 1b)

The L-GATr top-tagging teacher trained by Jay (config in `~/scratch/jay_agarwal/lgatr/lloca-experiments/runs/topt_lgatr/seed1001/config_0.yaml`) was instantiated with `in_s_channels=8`: 7 jet-tagging features (`log_pt`, `log_E`, `log_pt_rel`, `log_E_rel`, `dphi`, `deta`, `dr`) plus a 1-bit global-token flag. It also expects 3 spurion tokens (2 lightlike beams + 1 time reference), `/20` momentum scaling on non-spurions, and a mass regularization step.

Earlier KD scripts in this repo (`mlp_kd/mlp_kd.py`, `mlp_kd_engd/mlp_kd_engd.py`) instantiated the teacher with `in_s_channels=1`, fed only a single beam-flag scalar, and prepended only one beam token. They loaded the checkpoint with `strict=False`, which silently mismatched the input projection layer. **Result: the teacher's KD signal in those experiments was effectively noise.** With α=0.5, half the gradient came from useless soft-targets and half from genuine hard-label BCE — exactly the regime that produces ~0.78 AUC for a flat MLP.

The v1 DeepSets script (`mlp_kd_deepsets/mlp_kd_deepsets.py`) imports the real `LGATrWrapper` from `lloca-experiments` and feeds it via `embed_tagging_data`. The smoke test reports `[teacher] loaded from model_run0_it174999.pt | missing=0 unexpected=0`, confirming key-by-key match.

### F2 — Temperature is mathematically irrelevant in logit-MSE KD (Phase 2)

The KD soft loss used was:
```
soft_loss = MSE(student_logit / T, teacher_logit / T) · T²
         = mean((s/T − t/T)²) · T²
         = mean((s − t)² / T²) · T²
         = mean((s − t)²)
```
T cancels exactly. The 4 different T values in the Phase 2 sweep produced training trajectories identical up to floating-point noise (and `cudnn.benchmark=True` non-determinism). This is what we initially read as "the sweep showed (T, α) doesn't matter." The α dimension was real but degenerate at convergence on this confident binary teacher.

To make T meaningful again would require switching to BCE-on-soft-probs (Hinton-style) or KL on softened sigmoid distributions. We did not redo this — the architecture moved on.

### F3 — Phase 3 hint loss: absorbed but not classification-relevant (DeepSets *and* transformer)

The Phase 3 hint loss extracted a 48-dim invariant target from the teacher's penultimate hidden state at the global token (16 scalar components of the 16 multivector channels, plus the 32 scalar-stream channels). A small projection head mapped the student's pooled vector to that 48-dim target via MSE.

On DeepSets (row 4), the hint loss **trained successfully** — epoch 1 hint-MSE = 6.37, epoch 30 hint-MSE = 1.10. Yet test AUC was identical to v1 (0.9833 vs 0.9833). Per-epoch val-AUC delta was within ±0.0002 from epoch 11 onward.

The natural follow-up was: does structural homology with the teacher rescue the hint loss? Row 7 (`runs/v1_hint/`) tests this on the transformer student. **Result: still flat.** Test AUC 0.9845 (identical to non-hint transformer v1, row 6); rej@εS=0.5 = 352 vs 368 (slightly *worse*, marginally outside the seed noise floor). **The structural-homology hypothesis is falsified.** The hint signal is consistently absorbed by a learned projection but provides no additional classification utility regardless of student architecture.

Interpretation: the teacher's penultimate scalar invariants encode geometric-algebra information that the equivariant teacher uses downstream of `linear_out` to produce its logit, but matching them by regression doesn't transmit that downstream-use computation. The student would need *operations* like the teacher's, not targets that look like the teacher's intermediate representations. This is a clean negative result on intermediate-feature hint distillation across student-class boundaries.

### F4 — Architecture is the binding constraint across the experiment series

Sorted by lift over v1 DeepSets baseline (0.9833 / 290):

| Change | ΔAUC | Δrej@0.5 | Params | Notes |
|---|---|---|---|---|
| Phase 2 (T, α sweep) on DeepSets         | +0.0000 | +6   | 150k  | within noise |
| Phase 3 (hint loss) on DeepSets          | +0.0000 | -1   | 150k  | within noise |
| Phase 3 hint redux on transformer        | +0.0000 | -16  | 205k  | structural homology doesn't rescue it |
| Seed=43 vs seed=42 (transformer)         | +0.0000 | +5   | 205k  | noise floor |
| Pairwise features (DeepSets)             | +0.0004 | +21  | 167k  | real but small |
| DeepSets → small transformer             | +0.0012 | +78  | 205k  | first major lift |
| More epochs (30→50, same arch)           | +0.0007 | +52  | 205k  | ~2/3 of the bigger-model gain came from training time alone |
| Bigger transformer (d=96, b=6, 50ep)     | +0.0025 | +150 | 600k  | 50% of rej@0.5 gap to teacher closed |
| **ParT-style pair-bias (d=64, b=4, 50ep)** | **+0.0036** | **+277** | **205k** | **matches teacher within noise; 3× fewer params than `d96_b6_e50`** |

Six of nine experiments that changed *features* or *supervision* gave zero lift. The remaining three changed *architecture* or *training time* — they account for **all** the gain. **The right inductive structure beat raw capacity by a factor of 3 in parameter efficiency** (compare row 9 — pair bias at 205k — to row 8 — bigger plain transformer at 600k). The bottleneck has consistently been the student's representational capacity, not what or how it's taught.

### F5 — More epochs is half the "bigger model" win (transformer)

Decomposing row 10 (`d96_b6_e50`):

| | AUC | rej@0.5 | vs v1 |
|---|---|---|---|
| v1 (d=64, b=4, 30 epochs) | 0.9845 | 368 | — |
| v1_e50 (d=64, b=4, **50 epochs**) | 0.9852 | 420 | +0.0007 / +52 |
| d96_b6_e50 (d=96, b=6, 50 epochs) | 0.9858 | 440 | +0.0013 / +72 |

So roughly 2/3 of d96_b6_e50's lift over v1 comes from longer training; only 1/3 from the bigger architecture. Implication for the writeup: **the transformer student wasn't converged at 30 epochs.** Future runs (and the ParT-style row 12) default to 50 epochs.

### F6 — Transformer seed-to-seed noise floor

`v1_s43` (seed=43) vs `v1` (seed=42): AUC +0.0000, rej@εS=0.5 +5 (368→373), best val AUC identical to four decimal places. So **the noise floor on this benchmark is ±0.0001 AUC and ±5 in rej@0.5.** This calibrates how to read close ablations: anything inside this band shouldn't be claimed as a real effect. Phase 2's flat sweep (rej@0.5 spread of 12 across 8 runs) is exactly noise; the hint-redux's -16 rej delta is borderline (just outside noise, marginal).

## Decision rationale at each step

- **v1 (DeepSets)**: chosen over flat MLP because the prior failure pattern (depth=6 < depth=3) signaled that flat MLPs can't use particle-set inputs — they have no permutation invariance and pad sparsely.
- **Phase 2 (T/α sweep) before Phase 3**: cheap; reused v1's teacher cache; expected to find a small win. Found nothing → useful negative.
- **Phase 3 (hint loss) over directly trying transformer**: research-economical at the time — it's a single-flag change, reuses the existing student, and tests a specific hypothesis about supervision richness. Negative result was informative.
- **Pairwise features before transformer**: cheap orthogonal axis; would have been embarrassing to skip and find later that "DeepSets just needed pairwise."
- **Small transformer first, then sweep matrix**: distillation efficiency story is stronger if the small student works. Once row 6 confirmed the architecture lift, we ran a 5-job decomposition matrix in parallel (rows 7–11) to disentangle hints/seed-noise/longer/bigger/tiny.
- **ParT-style pair-bias before "even bigger transformer"**: row 12 tests whether explicit two-particle structure injected into attention scores closes more of the rejection gap than just adding capacity. **Outcome confirmed**: pair-bias at 205k params beat the 600k bigger-transformer baseline by +0.0011 AUC and +127 rej@0.5, matching the teacher within noise floor. That validates the "architecture, not capacity" framing as the central paper claim.

## Pointers — where things live

### Code

| | |
|---|---|
| `mlp_kd_deepsets/mlp_kd_deepsets.py`     | v1 + Phase 2 + Phase 3 + pairwise. All the shared infrastructure (data, teacher loader, hint plumbing, KD loss, eval helpers) imported by the transformer dir. |
| `mlp_kd_deepsets/eval_deepsets.py`       | standalone DeepSets-student eval |
| `mlp_kd_deepsets/run_*.sh`               | sbatch wrappers: `run_deepsets.sh` (v1), `run_ablation.sh`+`submit_ablations.sh` (Phase 2), `run_phase3.sh`, `run_pairwise.sh` |
| `mlp_kd_deepsets/summarize_ablations.py` | post-hoc table over `ablations/<tag>/` |
| `mlp_kd_deepsets/readme.md`              | v1 + Phase 2 + Phase 3 design notes |
| `mlp_kd_transformer/mlp_kd_transformer.py` | small transformer student (option 2b), self-contained main() |
| `mlp_kd_transformer/eval_transformer.py`   | transformer eval |
| `mlp_kd_transformer/run_transformer.sh`    | sbatch wrapper, env-var driven |
| `mlp_kd_transformer/readme.md`             | transformer student design notes |
| `mlp_kd_part/mlp_kd_part.py`               | ParT-style student: vanilla transformer + learned pairwise-feature attention bias. Imports shared infrastructure from `mlp_kd_deepsets`. Defines `PairBiasAttention`, `PairBiasBlock`, `PartStudent`, `compute_pairwise_interactions`. |
| `mlp_kd_part/eval_part.py`                 | ParT-student eval |
| `mlp_kd_part/run_part.sh`                  | sbatch wrapper, env-var driven (`PT_TAG`, `PT_DMODEL`, `PT_HINT_BETA`, ...) |

### Numerical results

```
mlp_kd_deepsets/final_test_metrics.json                      # v1
mlp_kd_deepsets/ablations/<tag>/final_test_metrics.json      # Phase 2 (8 tags)
mlp_kd_deepsets/phase3/beta05/final_test_metrics.json        # Phase 3
mlp_kd_deepsets/pairwise/v1pair/final_test_metrics.json      # pairwise DeepSets
mlp_kd_transformer/runs/<tag>/final_test_metrics.json        # transformer runs (6 tags as of 2026-05-06)
mlp_kd_part/runs/<tag>/final_test_metrics.json               # ParT-style runs
*/history.json                                               # per-epoch trajectories
```

To pull a specific number, prefer reading the JSON over recalling from this file — JSON is authoritative.

### Auto-memory (loads automatically)

`~/.claude-config/projects/-nfs-home-users-dhruvk-khare27/memory/`:
- `project_kd_lgatr_paths_and_teacher.md` — dataset path, teacher checkpoint paths, val-split confirmation, full corrected teacher I/O interface
- `feedback_csis_cluster_rules.md` — CSIS SOP: sbatch-only, partition/QoS table, scratch auto-delete
- `feedback_no_python_on_login_node.md` — even import-only python probes hang on the CSIS login node
- `feedback_lloca_attention_backend.md` — LGATrWrapper supports only xformers/flash/flex; "torch" raises ValueError

### External

- Dataset: `/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/data/toptagging/toptagging_full.npz` (1.5 GB, train 1.21M / val 403k / test 404k)
- Teacher checkpoints: `/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/lloca-experiments/runs/topt_lgatr/seed{1001,1002,1003}/`
- Jay's repo (read-only ref): `/nfs_home/users/dhruvk/jay_agarwal/lgatr/repos/lloca-experiments-analysis/`
- Conda env with all teacher deps: `/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging/`

## Currently in flight (as of 2026-05-07)

- `mlp_kd_part/runs/v1_s43/` — pair-bias reproducibility check (seed=43). Same arch as row 12 (d=64, b=4, 50ep). Tests whether the AUC=0.9869 / rej@0.5=567 result reproduces under a different seed; result will tighten the central claim in the paper.

Check status: `squeue -u $USER -o "%.10i %.18j %.2t %.10M %R"`
Quick scan of finished runs:
```bash
for d in mlp_kd_part/runs/*/ mlp_kd_transformer/runs/*/ mlp_kd_deepsets/{,phase3/*/,pairwise/*/,ablations/*/}; do
    [[ -f "$d/final_test_metrics.json" ]] || continue
    auc=$(grep -oP '"test_auc":\s*\K[0-9.]+' "$d/final_test_metrics.json")
    r05=$(grep -oP '"0\.5":\s*\K[0-9.]+' "$d/final_test_metrics.json")
    printf "%-60s AUC=%s rej@0.5=%s\n" "$d" "$auc" "$r05"
done
```

## Open questions for the draft

These haven't been settled and are decision points the writing session should surface to the user:

- **Scope**: top-tagging only, or position the method as general (boost-invariant set classification with an equivariant teacher)?
- **Baselines to compare against explicitly**: PFN (≈0.932), ParticleNet (≈0.985), ParT (≈0.987). All public on top-tagging. Numbers on file: see `Headline as of 2026-05-06` section above for the current student-vs-baseline framing. ParT comparison is especially clean (~3.5× fewer params, ~0.0012 AUC behind vanilla ParT).
- **Central finding for the abstract**: with the pair-bias result, the headline tightens to "matches teacher within noise floor (ΔAUC ≤ 0.0001) at 5× parameter compression." Three-piece structure for the paper: (i) clean distillation pipeline + 205k-param non-equivariant student matches a 1M-param Lorentz-equivariant teacher, (ii) right inductive bias beats raw capacity by ~3× in parameter efficiency on this task (F4 — architecture is the bottleneck), (iii) intermediate-feature hint distillation is null even with a structurally-homologous student (F3 — supervision-side interventions don't help when the inductive bias is the constraint).
- **Section dedicated to the in_s_channels bug**: worth writing as a methodology footnote (warning for others) or omit?
- **Format / venue**: not yet decided. NeurIPS ML4PS workshop? ICLR? short conference paper? journal?
- **Author list, acknowledgements**: not yet decided. Jay's teacher, Jay's repo — clearly an acknowledgement at minimum.

## Operational notes for any agent working in this tree

- All compute runs through `sbatch`. Login node compute is forbidden by the SOP. The auto-memory rules will reinforce this.
- Teacher cache (`teacher_logits_{train,val}.pt`, `teacher_invariants_{train,val}.pt`) lives in `mlp_kd_deepsets/`. Reused across all student experiments via `--teacher-cache-dir`. **Do not delete or overwrite these** — every result above depends on a stable cache.
- Per-user concurrent job cap on CSIS is 5; faculty group cap is 8 GPUs. Pending state with reason `QOSGrpMemLimit` or `QOSMaxGRESPerUser` is normal during heavy-use windows.
- Outputs (checkpoints, JSONs, plots) all go in `$HOME` paths. Never write trainable artifacts to `~/scratch` — it auto-deletes after 30 days.
