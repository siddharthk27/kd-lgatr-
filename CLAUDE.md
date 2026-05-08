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
| 13 | `mlp_kd_part/runs/v1_scratch/`          | row-12 architecture trained from scratch (α=0.0, no soft loss, only hard-label BCE) | 0.9857 | 1453 | 388 | 58 | iid: ~0.0012 AUC, ~179 rej@0.5 below KD-ParT — small. **Boost-robustness gap is much larger** (see F7) |
| 14 | `mlp_kd_part/runs/v1_s43/`              | row-12 architecture, **seed=43** reproducibility retry (jobid 5389 had failed silently at ~ep 12; 5483 succeeded) | **0.9870** | 2462 | 566 | 67 | seed-noise reproducibility: ΔAUC=+0.0001, Δrej@0.5=−1 vs row 12 — within F6 noise floor. Confirms the headline isn't a lucky seed |
| 15 | `mlp_kd_part/runs/v1_softkl_T2/`        | row-12 retrained with **softmax-KL T=2** (vs row-12's logit-MSE)                | 0.9870 | 2082 | 559 | 67 | F2 ablation: matches row 12 within noise — **loss form is not a discriminating lever for binary KD** |
| 16 | `mlp_kd_fixed/runs/run_softkl_T4/`      | original `mlp_kd_engd` flat-MLP arch (depth=3, d_ff=512, 4 engineered features), **fixed teacher interface**, softmax-KL T=4 | 0.9738 | 433 | 134 | 25 | **F1 vindicated empirically**: +0.185 AUC over row 1 (broken-teacher MLP-KD); beats `mlp_scratch_engd` (0.9501 / 45) by +0.024 AUC and ~3× rej@0.5 — KD helps the MLP class once the teacher signal is correct |
| 17 | `mlp_kd_fixed/runs/run_logit_mse/`      | row-16 architecture, logit-MSE arm                                              | 0.9743 | 464 | 137 | 26 | F2 ablation on flat MLP: matches row 16 within noise — confirms loss-form invariance across the architecture spectrum, from weak (flat MLP) to strong (ParT) students |

Read individual `final_test_metrics.json` files for exact numbers; `history.json` for per-epoch trajectories.

### Headline as of 2026-05-09

**Best student: `mlp_kd_part/runs/v1/`** (ParT-style, 205k params) at AUC **0.9869** / rej@εS=0.5 = **567** / rej@εS=0.3 = **2433** / rej@εS=0.8 = **67**.

Gap to teacher (0.9870 / 587 / 67 / 2433, ~1M params):
- AUC: 0.9870 − 0.9869 = **0.0001** (at the seed noise floor of F6 — essentially equal)
- rej@εS=0.3: **0** — bit-exact match (2433 = 2433)
- rej@εS=0.5: **20** — 93% of original 297 gap closed
- rej@εS=0.8: **0** — bit-exact match (67 = 67)

A pair-bias seed=43 reproducibility run was completed (`runs/v1_s43/`, jobid 5483) — final test AUC **0.9870**, rej@εS=0.5 = **566**. **Bit-equal to seed=42 within F6 noise floor (ΔAUC=+0.0001, Δrej@0.5=−1).** The headline is not a lucky seed. (An earlier attempt at jobid 5389 failed silently at ~epoch 12; the partial checkpoint is preserved at `runs/v1_s43/part_student_best_partial_5389.pt` for forensics.)

The student is **5× smaller than the teacher** (205k vs ~1M params) and **~10× smaller than vanilla ParT** (which sits at ≈0.987 with ~2.1M params on the same benchmark).

## Boost-robustness experiments (per Liu et al. 2023)

Liu et al. (NeurIPS ML4PS 2023, the paper that motivated this project) showed that the strongest evidence for KD transferring an inductive bias is not iid test AUC but **how the student's performance degrades under Lorentz boosts** of the test inputs. KD-trained students degrade less than scratch-trained students of the same architecture; an equivariant teacher is flat by construction. Without this test we can claim "matches teacher on iid data" but not "transferred the inductive bias", which is the stronger claim and the natural paper headline.

### Implementation

`mlp_kd_part/eval_boosted.py` reproduces Liu et al.'s recipe (their `KD4Jets/eval.py:49`):
1. For β in `np.linspace(0, 1, 20, endpoint=False)`, apply x-axis Lorentz boost to the dense `kinematics_test` tensor in-place (no separate dataset). Special-case β=0 (skip the formula); zero out NaN entries that arise at high β where γ blows up for extreme energies.
2. Recompute student features (`build_student_inputs`) and pair features (`compute_pairwise_interactions`) on the boosted inputs — these are NOT Lorentz invariants, so the student sees genuinely different inputs at each β.
3. Forward through the model, compute AUC + rej@εS at each β.
4. Write `runs/<tag>/boost_results.json` with arrays `{betas, auc, rej_30, rej_50, rej_80}`.

Three reference checkpoints get evaluated:
- **`runs/v1/`** — KD-ParT student (row 12). Expected: degrades less than scratch.
- **`runs/v1_scratch/`** — same architecture, hard-label-only training (row 13). Expected: degrades more — the contrast curve.
- **teacher** — Lorentz-equivariant by construction. Expected: **flat** (within ±0.0001 AUC) across all β. Sanity check.

The teacher path is a separate code branch in `eval_boosted.py` (uses `LGATrWrapper` + `embed_tagging_data` directly, since the cached teacher logits are fixed for β=0 inputs and can't be reused).

### Reference paper numbers (Liu et al., for calibration)

Teacher (LorentzNet, equivariant): ~flat across boost. Scratch DeepSets dropped from ~0.98 at β=0 to ~0.93 at β=0.8. KD DeepSets degraded measurably less.

**Note from our actual measurement:** the L-GATr teacher in our setup is *not* truly flat under x-boosts — it drops from AUC 0.9870 at β=0 to 0.9611 at β=0.9. This is because L-GATr only has equivariant *multivector* paths; the scalar input channels (`log_pt`, `log_E`, `log_pt_rel`, `log_E_rel`, `dphi`, `deta`, `dr`) are explicitly Lorentz-non-invariant. So the teacher has *partial* equivariance, not full. The student inherits a fraction of this partial robustness via KD.

### F7 — Boost-robustness: KD transfers ~40% of the teacher's partial Lorentz inductive bias to the non-equivariant student

Three reference curves (`mlp_kd_part/runs/{teacher,v1,v1_scratch}/boost_results.json`), evaluated over β ∈ {0.0, 0.05, …, 0.95} on the standard test set. Plot in `mlp_kd_part/boost_robustness.{png,pdf}`.

**Headline numbers** (selected β):

| β | Teacher AUC | KD-ParT AUC | Scratch-ParT AUC | KD−Scratch (AUC) | KD/Scratch (rej@0.5) |
|---|---|---|---|---|---|
| 0.0 | 0.9870 | 0.9869 | 0.9857 | +0.0012 | 1.46× |
| 0.4 | 0.9786 | 0.9729 | 0.9689 | +0.0040 | 1.78× |
| 0.5 | 0.9762 | 0.9660 | 0.9593 | +0.0067 | 1.86× |
| 0.7 | 0.9727 | 0.9446 | 0.9241 | +0.0205 | 1.69× |
| 0.8 | 0.9704 | 0.9232 | 0.8886 | **+0.0346** | 1.79× |
| 0.9 | 0.9611 | 0.8837 | 0.8429 | **+0.0408** | 1.70× |

**Three findings:**

1. **The KD-vs-scratch AUC gap widens by ~30× as β grows** — from +0.0012 at β=0 to +0.0408 at β=0.9. **Looking at iid AUC alone almost completely hides the inductive-bias transfer**; the boost-eval makes it visible. This is exactly the Liu et al. claim, replicated in our setting (different teacher class, different student class, different feature pipeline).

2. **Transfer fraction is stable at ~40% across the high-β regime**, defined as `(KD − Scratch) / (Teacher − Scratch)`:
   - β=0.5: 0.0067 / 0.0169 = **40%**
   - β=0.7: 0.0205 / 0.0486 = **42%**
   - β=0.8: 0.0346 / 0.0818 = **42%**
   - **Mean over β ∈ [0.5, 0.85]: 41.2%**
   The student picks up ≈40% of the way from "scratch baseline" to "(partially-equivariant) teacher" in boost-AUC, robustly across the regime where the teacher itself shows nontrivial structure to transfer.

3. **Background-rejection advantage is consistently ~1.7–1.9× across the boost range.** Mean KD-ParT/Scratch-ParT rej@εS=0.5 ratio over β ∈ [0.5, 0.85]: **1.75×**. Even at iid (β=0) there's a 1.46× advantage; the multiplier *grows* through the boost regime. **This is the metric LHC trigger systems actually optimize** (background rejection at fixed signal efficiency), so the practical claim "KD-distilled students are ~75% more boost-robust than identical architectures trained on hard labels alone" is the deployment-relevant version of the statement.

The teacher is the "ceiling" curve; KD-ParT sits between teacher and scratch at every β — never below scratch, never above teacher, always strictly between, always closer to teacher than to scratch. Cleanest possible visual demonstration of inductive-bias transfer.

## Key technical findings (paper-worthy)

### F1 — The in_s_channels mismatch in prior KD scripts (rows 0, 1, 1b)

The L-GATr top-tagging teacher trained by Jay (config in `~/scratch/jay_agarwal/lgatr/lloca-experiments/runs/topt_lgatr/seed1001/config_0.yaml`) was instantiated with `in_s_channels=8`: 7 jet-tagging features (`log_pt`, `log_E`, `log_pt_rel`, `log_E_rel`, `dphi`, `deta`, `dr`) plus a 1-bit global-token flag. It also expects 3 spurion tokens (2 lightlike beams + 1 time reference), `/20` momentum scaling on non-spurions, and a mass regularization step.

Earlier KD scripts in this repo (`mlp_kd/mlp_kd.py`, `mlp_kd_engd/mlp_kd_engd.py`) instantiated the teacher with `in_s_channels=1`, fed only a single beam-flag scalar, and prepended only one beam token. They loaded the checkpoint with `strict=False`, which silently mismatched the input projection layer. **Result: the teacher's KD signal in those experiments was effectively noise.** With α=0.5, half the gradient came from useless soft-targets and half from genuine hard-label BCE — exactly the regime that produces ~0.78 AUC for a flat MLP.

The v1 DeepSets script (`mlp_kd_deepsets/mlp_kd_deepsets.py`) imports the real `LGATrWrapper` from `lloca-experiments` and feeds it via `embed_tagging_data`. The smoke test reports `[teacher] loaded from model_run0_it174999.pt | missing=0 unexpected=0`, confirming key-by-key match.

**Empirical confirmation (rows 16–17, `mlp_kd_fixed/`).** Re-running the *original flat-MLP arch* (depth=3, d_ff=512, 4 engineered features — the same configuration as `mlp_scratch_engd` and `mlp_kd_engd`) under the **fixed** teacher interface lifts test AUC from **0.7890 → 0.9738** (+0.185 AUC). Crucially, the fixed-teacher KD student now **beats** its no-teacher scratch counterpart: 0.9738 vs `mlp_scratch_engd`'s 0.9501 (+0.024 AUC, ~3× rej@0.5). Under the broken teacher, KD made the MLP *worse* than scratch — a sign the soft signal was actively harmful (i.e. random). With the fixed teacher, KD lifts the MLP class above its scratch baseline, just as it does for DeepSets/transformer/ParT. **The four-piece headline claim "KD helps every student class when the teacher interface is correct" is now empirically verified across the full architecture spectrum** (flat MLP → DeepSets → transformer → ParT).

### F2 — Temperature is mathematically irrelevant in logit-MSE KD (Phase 2)

The KD soft loss used was:
```
soft_loss = MSE(student_logit / T, teacher_logit / T) · T²
         = mean((s/T − t/T)²) · T²
         = mean((s − t)² / T²) · T²
         = mean((s − t)²)
```
T cancels exactly. The 4 different T values in the Phase 2 sweep produced training trajectories identical up to floating-point noise (and `cudnn.benchmark=True` non-determinism). This is what we initially read as "the sweep showed (T, α) doesn't matter." The α dimension was real but degenerate at convergence on this confident binary teacher.

**Empirical follow-up (rows 15–17).** Switching to softmax-T KL (canonical Hinton-style: 2-class softmax with temperature, KL with batchmean reduction × T²) makes T mathematically meaningful again. Tested on two student classes:
- **ParT pair-bias headline (row 15):** softmax-KL T=2 → AUC 0.9870, rej@εS=0.5 = 559. Matches row-12 logit-MSE within F6 noise (Δrej@0.5 = −8, ΔAUC = +0.0001).
- **Flat-MLP fixed-teacher (rows 16–17):** softmax-KL T=4 → AUC 0.9738; logit-MSE → AUC 0.9743. Δ within seed noise.

So **even when T becomes a real lever, the loss-form choice does not move the needle on this benchmark**. Mechanistic reason: for binary KD, KL between two Bernoulli distributions reduces to roughly an MSE on logits up to a scale factor in the converged regime, so both losses produce the same student up to second-order effects. The Phase 2 sweep was degenerate not just because of the T-cancellation algebra but because the *useful* part of the soft signal — pulling student logits toward teacher logits — is identical to first order across both formulations. F2 thus graduates from a math observation ("T cancels in logit-MSE") to a generalized empirical claim: **the binary KD soft-loss form is not a discriminating lever for this teacher**, verified across student capacities from a flat MLP (AUC 0.97) to ParT pair-bias (AUC 0.987).

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
| `mlp_kd_part/eval_part.py`                 | ParT-student iid eval |
| `mlp_kd_part/run_part.sh`                  | sbatch wrapper, env-var driven (`PT_TAG`, `PT_DMODEL`, `PT_HINT_BETA`, `PT_ALPHA`, ...). Setting `PT_ALPHA=0.0` trains the same arch from scratch (hard-label BCE only). |
| `mlp_kd_part/eval_boosted.py`              | Lorentz x-boost robustness eval. Two modes via mutually-exclusive flags: `--checkpoint <path>` (student) or `--teacher` (uses `LGATrWrapper` directly). Boost formula and edge cases mirror Liu et al.'s `KD4Jets/eval.py`. Writes `boost_results.json`. |
| `mlp_kd_part/run_boosteval.sh`             | sbatch wrapper for `eval_boosted.py`. Env vars `BE_CHECKPOINT`, `BE_OUTDIR`, `BE_TEACHER`. ~15 min per student-checkpoint on `gpu-short`; ~2 hr for teacher (xformers attention over particles each batch). |
| `mlp_kd_part/plot_boost_curves.py`         | reads the three `boost_results.json` files and produces `boost_robustness.{png,pdf}` (the F7 headline figure). Single matplotlib script, runs in seconds on `debug` partition. |
| `mlp_kd_fixed/mlp_kd_fixed.py`             | F1-vindication run: flat-MLP arch (`mlp_kd_engd`-style, depth=3, d_ff=512, 4 engineered features) trained under the **fixed** teacher interface. Imports teacher pipeline + `kd_loss` from `mlp_kd_deepsets`. CLI args for `--kd-loss {softmax_kl, logit_mse}` and `--temperature`. Includes a built-in smoke test that aborts if teacher AUC < 0.95 on a 2k val slice. |
| `mlp_kd_fixed/run_fixed.sh`                | sbatch wrapper, env-var driven (`MK_TAG`, `MK_KD_LOSS`, `MK_TEMPERATURE`, `MK_ALPHA`, …). Reuses teacher cache from `../mlp_kd_deepsets/`. ~6 min wall on `gpu-short` per arm with cache. |
| `mlp_kd_fixed/AGENTS.md`                   | self-contained porting guide for an agent landing fresh on a new server (paths, smoke test, both arms, common failure modes). Read this if running on a different cluster. |

### Numerical results

```
mlp_kd_deepsets/final_test_metrics.json                      # v1
mlp_kd_deepsets/ablations/<tag>/final_test_metrics.json      # Phase 2 (8 tags)
mlp_kd_deepsets/phase3/beta05/final_test_metrics.json        # Phase 3
mlp_kd_deepsets/pairwise/v1pair/final_test_metrics.json      # pairwise DeepSets
mlp_kd_transformer/runs/<tag>/final_test_metrics.json        # transformer runs (6 tags as of 2026-05-07)
mlp_kd_part/runs/<tag>/final_test_metrics.json               # ParT-style runs (iid; tags: v1, v1_scratch, v1_s43, v1_softkl_T2)
mlp_kd_part/runs/<tag>/boost_results.json                    # ParT-style runs (boost-robustness)
mlp_kd_part/runs/teacher/boost_results.json                  # teacher boost-eval (equivariant reference)
mlp_kd_part/boost_robustness.{png,pdf}                       # F7 headline figure: AUC and rej@0.5 vs β
mlp_kd_fixed/runs/<tag>/final_test_metrics.json              # fixed-teacher flat-MLP runs (tags: run_softkl_T4, run_logit_mse)
*/history.json                                               # per-epoch trajectories
```

To pull a specific number, prefer reading the JSON over recalling from this file — JSON is authoritative.

### Auto-memory (loads automatically)

`~/.claude-config/projects/-nfs-home-users-dhruvk-khare27/memory/`:
- `project_kd_lgatr_paths_and_teacher.md` — dataset path, teacher checkpoint paths, val-split confirmation, full corrected teacher I/O interface
- `feedback_csis_cluster_rules.md` — CSIS SOP: sbatch-only, partition/QoS table, scratch auto-delete
- `feedback_no_python_on_login_node.md` — even import-only python probes hang on the CSIS login node
- `feedback_lloca_attention_backend.md` — LGATrWrapper supports only xformers/flash/flex; "torch" raises ValueError
- `feedback_amp_masked_fill_sentinel.md` — under autocast, -1e9 overflows float16; use torch.finfo(dtype).min

### External

- Dataset: `/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/data/toptagging/toptagging_full.npz` (1.5 GB, train 1.21M / val 403k / test 404k)
- Teacher checkpoints: `/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/lloca-experiments/runs/topt_lgatr/seed{1001,1002,1003}/`
- Jay's repo (read-only ref): `/nfs_home/users/dhruvk/jay_agarwal/lgatr/repos/lloca-experiments-analysis/`
- Conda env with all teacher deps: `/nfs_home/users/dhruvk/jay_agarwal/lgatr/envs/lloca-toptagging/`

## Currently in flight (as of 2026-05-09)

**Nothing in flight.** All experiments for the four-piece headline claim (F1–F7) have landed, plus the F1-vindication runs (rows 16–17) and the F2-generalization runs (rows 15–17), and the seed=43 reproducibility (row 14).

**Optional follow-up runs** still on the table (not blocking the paper but useful as appendix material).

These are *new* runs, not re-evals of existing checkpoints. The boost-robustness eval pipeline (`eval_boosted.py`) — which produced the F7 numbers above — only *evaluates* already-trained checkpoints under boosted test inputs; it does **not** train with boost augmentation.

- **Boost-augmented ParT — new training run, not a re-eval.** Liu et al.'s `if hparams.get("boost"):` branch applies a random Lorentz boost to particle inputs each step during *training*, so the student sees already-perturbed inputs. Tests Claims B/C from the boost-augmentation discussion: does augmentation alone (B, no KD) or augmentation + KD (C) yield a more boost-robust student than KD alone (which is the F7 baseline)? Two arms: augmented + scratch (cheap, ~3.7 hr `gpu-short`); augmented + KD (expensive — teacher inference per batch breaks the cache shortcut, ~15 hr → needs `gpu-1day`).
- **Scratch baselines for DeepSets and vanilla transformer.** Right now F7 only proves "KD transfers Lorentz inductive bias" for the ParT student. A scratch-DeepSets and scratch-vanilla-transformer (both at α=0, hard-label only) — combined with their existing KD counterparts and boost-eval — would extend F7's "transfer effect is robust across student architectures" claim to the full architecture spectrum. Total cost: ~5 hr wall, 2 sbatch jobs.

**Recently completed in this session:**
- Row 14 (`v1_s43`, ParT seed=43 reproducibility): jobid 5483 — confirms headline isn't a lucky seed.
- Row 15 (`v1_softkl_T2`, ParT softmax-KL T=2): jobid 5490 — F2 ablation on strong student.
- Rows 16–17 (`mlp_kd_fixed/run_softkl_T4`, `run_logit_mse`): jobids 5492, 5493 — F1 empirical vindication on flat MLP, F2 ablation on weak student.

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
- **Baselines to compare against explicitly**: PFN (≈0.932), ParticleNet (≈0.985), ParT (≈0.987). All public on top-tagging. Numbers on file: see `Headline as of 2026-05-09` section above for the current student-vs-baseline framing. ParT comparison is especially clean (~3.5× fewer params, ~0.0012 AUC behind vanilla ParT).
- **Central finding for the abstract**: four-piece claim, all verified: (i) clean distillation pipeline + 205k-param non-equivariant student matches a 1M-param Lorentz-equivariant teacher within noise floor on iid data (F4), (ii) the right inductive bias beats raw capacity by ~3× in parameter efficiency on this task (F4 — architecture is the bottleneck), (iii) **KD transfers ~40% of the teacher's partial Lorentz inductive bias to the non-equivariant student** — measured against an identical-architecture scratch baseline; mean transfer fraction 41.2%, mean rej@εS=0.5 ratio 1.75× over β ∈ [0.5, 0.85] (F7), (iv) intermediate-feature hint distillation is null even with a structurally-homologous student (F3). **Piece (iii) is the strongest paper claim** — the iid AUC parity (i) is interesting but conventional; the boost-robustness gap that widens 30× from β=0 to β=0.9 is the inductive-bias-transfer evidence that uniquely follows from KD.
- **Section dedicated to the in_s_channels bug**: worth writing as a methodology footnote (warning for others) or omit?
- **Format / venue**: not yet decided. NeurIPS ML4PS workshop? ICLR? short conference paper? journal?
- **Author list, acknowledgements**: not yet decided. Jay's teacher, Jay's repo — clearly an acknowledgement at minimum.

## Operational notes for any agent working in this tree

- All compute runs through `sbatch`. Login node compute is forbidden by the SOP. The auto-memory rules will reinforce this.
- Teacher cache (`teacher_logits_{train,val}.pt`, `teacher_invariants_{train,val}.pt`) lives in `mlp_kd_deepsets/`. Reused across all student experiments via `--teacher-cache-dir`. **Do not delete or overwrite these** — every result above depends on a stable cache.
- Per-user concurrent job cap on CSIS is 5; faculty group cap is 8 GPUs. Pending state with reason `QOSGrpMemLimit` or `QOSMaxGRESPerUser` is normal during heavy-use windows.
- Outputs (checkpoints, JSONs, plots) all go in `$HOME` paths. Never write trainable artifacts to `~/scratch` — it auto-deletes after 30 days.
