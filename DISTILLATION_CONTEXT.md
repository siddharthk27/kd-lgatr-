# L-GATr → MLP Distillation: Handoff Context

This document is a handoff for a Claude Code agent that will continue investigating
**distillation of an L-GATr top-tagger into an MLP student**. The agent has local
access to the `lgatr` repo (https://github.com/heidelberg-hepml/lgatr) at the same
commit as the conversation that produced this doc (HEAD around commit 87ea16b on
`main`, plus `lgatrpaper.pdf` at the repo root).

The doc is organized as:

1. Codebase map (what lives where, with file paths)
2. End-to-end trace of one particle through the architecture (top tagging)
3. Paper facts that are not obvious from the code alone
4. Distillation strategy (the actual investigation plan)
5. Open decisions and ablations
6. References

Read sections 1–3 to get oriented; sections 4–6 are the work.

---

## 1. Codebase map

The package is small and clean. Top-level layout:

- [lgatr/](lgatr/)
  - [__init__.py](lgatr/__init__.py) — re-exports `LGATr`, `LGATrSlim`, `Conditional*`, and the `embed_/extract_` helpers.
  - [interface/](lgatr/interface/) — embedding/extraction between physical objects and 16-dim multivectors.
    - [scalar.py](lgatr/interface/scalar.py): `embed_scalar`, `extract_scalar` → MV index 0.
    - [vector.py](lgatr/interface/vector.py): `embed_vector`, `extract_vector` → MV indices 1:5.
    - [pseudoscalar.py](lgatr/interface/pseudoscalar.py): index 15.
    - [axialvector.py](lgatr/interface/axialvector.py): indices 11:15.
    - [spurions.py](lgatr/interface/spurions.py): the **beam-direction bivector** and **time-direction vector** symmetry-breaking tokens used in top tagging.
  - [primitives/](lgatr/primitives/) — Lorentz-equivariant operations as pure functions.
    - [config.py](lgatr/primitives/config.py): global `gatr_config` flags — `use_fully_connected_subgroup`, `use_bivector`, `use_geometric_product`.
    - [linear.py](lgatr/primitives/linear.py): `equi_linear` (Eq. 1 of paper) using a precomputed basis tensor; `grade_project`, `reverse`, `grade_involute`.
    - [linear_basis_subgroup.pt](lgatr/primitives/linear_basis_subgroup.pt) / [linear_basis_full.pt](lgatr/primitives/linear_basis_full.pt): precomputed equivariant linear basis (10 elements for SO⁺(1,3), 5 for O(1,3)).
    - [geometric_product.pt](lgatr/primitives/geometric_product.pt): the (16,16,16) GA multiplication tensor.
    - [bilinear.py](lgatr/primitives/bilinear.py): `geometric_product`.
    - [normalization.py](lgatr/primitives/normalization.py): `equi_layer_norm` (Eq. 3, with the per-grade absolute-value trick).
    - [invariants.py](lgatr/primitives/invariants.py): `abs_squared_norm`, inner-product factors used to convert GA inner product → Euclidean dot product for attention.
    - [attention.py](lgatr/primitives/attention.py): `sdp_attention`, `scaled_dot_product_attention` — wraps the Flash-Attention-compatible backends.
    - [attention_backends/](lgatr/primitives/attention_backends/): backend dispatch (PyTorch SDPA, xformers, etc.).
    - [nonlinearities.py](lgatr/primitives/nonlinearities.py): `gated_gelu`, `gated_relu`, `gated_sigmoid`, `gated_silu`.
    - [dropout.py](lgatr/primitives/dropout.py): grade-dropout primitive.
  - [layers/](lgatr/layers/) — `nn.Module` wrappers around the primitives.
    - [linear.py](lgatr/layers/linear.py): `EquiLinear` — wraps `equi_linear`, handles MV↔scalar mixing, has 4 init schemes (`default`, `small`, `unit_scalar`, `almost_unit_scalar`).
    - [layer_norm.py](lgatr/layers/layer_norm.py): `EquiLayerNorm`.
    - [dropout.py](lgatr/layers/dropout.py): `GradeDropout`.
    - [attention/](lgatr/layers/attention/)
      - [config.py](lgatr/layers/attention/config.py): `SelfAttentionConfig`, `CrossAttentionConfig`.
      - [qkv.py](lgatr/layers/attention/qkv.py): `QKVModule` (multi-head) and `MultiQueryQKVModule` (multi-query). **Paper uses multi-head** (Appendix B footnote 9).
      - [attention.py](lgatr/layers/attention/attention.py): `GeometricAttention`.
      - [self_attention.py](lgatr/layers/attention/self_attention.py): `SelfAttention` — QKV → attention → out-projection (`output_init="small"`) → optional dropout.
      - [cross_attention.py](lgatr/layers/attention/cross_attention.py): used by conditional variants.
    - [mlp/](lgatr/layers/mlp/)
      - [config.py](lgatr/layers/mlp/config.py): `MLPConfig`.
      - [geometric_bilinears.py](lgatr/layers/mlp/geometric_bilinears.py): `GeometricBilinear` — two parallel `EquiLinear`s feed `geometric_product`, then `EquiLinear` + `EquiLayerNorm`.
      - [nonlinearities.py](lgatr/layers/mlp/nonlinearities.py): `ScalarGatedNonlinearity` — gates every grade by `f(⟨x⟩₀)`.
      - [mlp.py](lgatr/layers/mlp/mlp.py): `GeoMLP` — `GeometricBilinear → (Gated → EquiLinear)*` stack.
    - [lgatr_block.py](lgatr/layers/lgatr_block.py): `LGATrBlock` — pre-LN attention + residual, pre-LN MLP + residual.
    - [conditional_lgatr_block.py](lgatr/layers/conditional_lgatr_block.py): conditional variant.
  - [nets/](lgatr/nets/)
    - [lgatr.py](lgatr/nets/lgatr.py): `LGATr` — `linear_in → blocks → linear_out`. Supports `reinsert_mv_channels`/`reinsert_s_channels` for re-injecting input features at every Q/K computation.
    - [lgatr_slim.py](lgatr/nets/lgatr_slim.py): smaller variant.
    - [conditional_lgatr.py](lgatr/nets/conditional_lgatr.py), [conditional_lgatr_slim.py](lgatr/nets/conditional_lgatr_slim.py): conditional variants.
  - [utils/](lgatr/utils/) — `einsum`, misc helpers.
- [examples/](examples/)
  - [demo_lgatr.ipynb](examples/demo_lgatr.ipynb): minimal usage demo (NOT the top-tagging setup; uses `M=1, M_h=8, S_h=16, blocks=2, heads=2`).
  - [demo_lgatr_slim.ipynb](examples/demo_lgatr_slim.ipynb), [demo_conditional_lgatr.ipynb](examples/demo_conditional_lgatr.ipynb).
- [tests/](tests/) — equivariance tests; useful templates if you need to verify your distillation pipeline keeps the teacher equivariant.
- [lgatrpaper.pdf](lgatrpaper.pdf) — the NeurIPS 2024 paper. Sec. 3 = architecture; Sec. 4.2 + Appendix C.2 = top-tagging setup; Appendix A = geometric algebra; Appendix B = baselines.

**Important:** no training loop, no top-tagging dataset loader, and no classification head live in this repo. All of those live in a separate (un-public) experiments repo associated with the paper. You will need to write them.

---

## 2. End-to-end trace: one particle through the top-tagger

Hyperparameters from Appendix C.2 of the paper:
- 12 attention blocks
- 16 multivector channels, 32 scalar channels
- 8 attention heads
- ~1.1 M learnable parameters
- BCE loss, LION optimizer, weight decay 0.2, batch 128, cosine annealing LR with peak 3·10⁻⁴, 2·10⁵ steps

### Stage 0 — raw datum

Kasieczka top-tagging dataset ([Zenodo 2603256](https://doi.org/10.5281/zenodo.2603256)). Per event: a list of reconstructed-particle 4-momenta `(E, px, py, pz)` (and a particle-type label) and a binary jet label `y ∈ {top, qcd}`. ~1.2 M train, 4·10⁵ each val/test.

### Stage 1 — preprocessing

Single global scalar division: `p ← p / σ` with `σ ≈ 200 GeV` (paper rescales by the std of all momenta in the dataset; for the generative experiments they use 206.6 GeV — same idea). **Per-component standardization is not allowed** because it breaks Lorentz equivariance — the same σ must scale all four components.

### Stage 2 — tokenization

Per particle:

```python
mv  = embed_vector(p)        # shape (1, 16); fills indices 1:5, rest zero
s   = one_hot(particle_type) # shape (K,)
```

Then prepend three **special tokens**:

| Token | MV content | Purpose |
|---|---|---|
| Global / CLS | zeros | classification readout (will end up holding jet-level info) |
| Beam | bivector encoding plane orthogonal to beam | breaks SO⁺(1,3) → SO(1,2) along beam axis |
| Time | grade-1 vector `(1,0,0,0)` | further breaks down to SO(3) |

See [lgatr/interface/spurions.py](lgatr/interface/spurions.py) for the exact spurion construction.

After Stage 2:
```
multivectors : (B, N+3, 1, 16)
scalars      : (B, N+3, K)
```

### Stage 3 — `linear_in` (initial EquiLinear)

`EquiLinear(in_mv=1, out_mv=16, in_s=K, out_s=32)`. Implements paper Eq. 1:

$$\text{Linear}(x) = \sum_{k=0}^{4} v_k \langle x \rangle_k + \sum_{k=0}^{4} w_k\, e_{0123}\, \langle x \rangle_k$$

— 10 weights per `(in_mv, out_mv)` pair, plus standard `nn.Linear` weights for `s2mvs`, `mvs2s`, `s2s`. Bias only allowed on the scalar (and pseudoscalar, in `SO⁺(1,3)` mode) slots. Output:

```
h_mv : (B, N+3, 16, 16)
h_s  : (B, N+3, 32)
```

### Stage 4 — 12 × `LGATrBlock`

Each block is the paper's:

```
x̄ = LayerNorm(x)
AttentionBlock(x) = Linear ∘ Attention(Linear(x̄), Linear(x̄), Linear(x̄)) + x
MLPBlock(x)       = Linear ∘ GatedGELU ∘ Linear ∘ GP(Linear(x̄), Linear(x̄)) + x
Block(x)          = MLPBlock(AttentionBlock(x))
```

Sub-step details:

1. **EquiLayerNorm** ([primitives/normalization.py:10](lgatr/primitives/normalization.py#L10)): divides by `sqrt(mean_c Σ_k |⟨⟨x_c⟩_k, ⟨x_c⟩_k⟩| + ε)`. Per-grade absolute value avoids cancellation between positive- and negative-norm grades in the Minkowski signature.
2. **QKV projection** ([attention/qkv.py](lgatr/layers/attention/qkv.py)): one big `EquiLinear` to `3 · hidden · num_heads`, sliced into Q/K/V; each re-normed.
3. **Attention** ([primitives/attention.py:11](lgatr/primitives/attention.py#L11)): the GA inner product is rewritten as a Euclidean dot product after multiplying Q by a fixed diagonal of `±1`s (the metric, in `_load_inner_product_factors`). MV channels are flattened, scalar channels concatenated, fed through `scaled_dot_product_attention` (Flash-Attention-compatible). This is the engineering trick that makes the architecture scale.
4. **Out-projection**: `EquiLinear` with `output_init="small"` ([linear.py:259](lgatr/layers/linear.py#L259)) — initialized 10× smaller so the residual stream is not overwhelmed at init.
5. **GeometricBilinear** ([mlp/geometric_bilinears.py:12](lgatr/layers/mlp/geometric_bilinears.py#L12)): `left = L(x̄)`, `right = R(x̄)` (R init `"almost_unit_scalar"`), `gp = geometric_product(left, right)`, optional zero-out of bivector grade if `use_bivector=False`, `EquiLinear` + `EquiLayerNorm`. **This is the only place in the network that creates bivector / pseudoscalar content from purely-vector inputs.**
6. **GatedGELU** ([mlp/nonlinearities.py:14](lgatr/layers/mlp/nonlinearities.py#L14)): `out = GELU(x[..., 0:1]) * x` (gates entire MV by invariant scalar grade).
7. **Final EquiLinear** + **residual**.

Shapes preserved across blocks. After 12 blocks, the global token has aggregated jet-level information through 12 rounds of attention.

### Stage 5 — `linear_out`

`EquiLinear(16 → 1, 32 → 0)` (or whatever the head needs). Output: `(B, N+3, 1, 16)`.

### Stage 6 — readout

Index into the global token, take its scalar grade:

```python
logit = output_mv[:, idx_global, 0, 0]   # (B,)
```

That single scalar — the grade-0 component of the CLS token — is the jet's classification logit. It is automatically Lorentz-invariant (grade 0 is the trivial rep) and permutation-invariant w.r.t. the particles (attention is a sum). With the beam+time spurions in place, the residual symmetry is exactly `SO(3)`, which matches the physics of a fixed-frame detector measurement.

### Stage 7 — loss

```
L = BCE_with_logits(logit, y)
```

### End-to-end shape ladder

```
raw event              (N, 4) + type (N,)
÷ σ                    (N, 4)
embed_vector           (N, 1, 16)
+ types as scalars     (N, K)
add 3 spurion tokens   (N+3, 1, 16), (N+3, K)
linear_in              (N+3, 16, 16), (N+3, 32)
[block × 12]           same shapes; equivariance preserved
linear_out             (N+3, 1, 16), (N+3, 0)
extract_scalar(global) → 1 scalar per jet
BCE                    against y
```

---

## 3. Paper facts not obvious from the code

These are the things you need to know that the code alone doesn't tell you:

1. **Multi-head, not multi-query, in the paper.** Code supports both via `SelfAttentionConfig.multi_query`; the published top-tagger uses multi-head (Appendix B fn 9).
2. **The 10-element basis** in [linear_basis_subgroup.pt](lgatr/primitives/linear_basis_subgroup.pt) is exactly Proposition 1 of the paper: 5 grade projections + 5 grade projections multiplied by the pseudoscalar `e₀₁₂₃`. This pseudoscalar mixing is what allows scalar↔pseudoscalar and vector↔axial-vector swapping under SO⁺(1,3).
3. **Symmetry breaking is architectural, not preprocessing.** The detector frame is broken into the network as two extra **tokens** (beam bivector + time vector), not as input features on each particle. This keeps every layer manifestly Lorentz-equivariant; the network learns how much to use the broken directions.
4. **Classification readout = CLS-token scalar grade.** Paper does not pool over particles for top-tagging — it reads the scalar component of the global token. (The demo notebook is misleading on this point: it uses mean-pool style readout for illustration.)
5. **Input rescaling is uniform across the four components.** Per-component standardization breaks Lorentz equivariance. The paper rescales by a single global σ.
6. **Top-tagging benchmark numbers** (Table 1):
   - L-GATr: AUC 0.9870 ± 0.0001, accuracy 0.9423 ± 0.0002, 1/εB at εS=0.5 = 540 ± 20, at εS=0.3 = 2240 ± 70.
   - Reference baselines: PELICAN 0.9870 / 2250, LorentzNet 0.9868 / 2195, ParT 0.9858 / 1602, ParticleNet 0.9858 / 1615.
7. **Inference scaling** (Fig. 7): Transformer-style attention makes L-GATr 10× faster than equivariant message-passing nets at >10³ particles, but ~10× slower than a vanilla Transformer at small N because the linear layers are heavier.
8. **Equivariance under reflections (parity, time reversal) is *not* preserved by default** because `use_fully_connected_subgroup=True`. This is correct for the LHC (parity is violated by the weak interaction) but matters if you ever care about the full Pin group.

---

## 4. Distillation strategy: L-GATr → MLP

### 4.1 Goal

Train an MLP student to match an L-GATr top-tagger as closely as possible on the
Kasieczka top-tagging benchmark. Compare on AUC, accuracy, 1/εB at εS=0.5 and
0.3, and **boost-test robustness** (the diagnostic the paper baselines did not run).

### 4.2 Why this is non-trivial

The teacher's strength is *architectural*: Lorentz equivariance, permutation
invariance, set-valued inputs, and the geometric product as a non-linear bilinear.
An MLP has none of those. Distillation can transfer them only indirectly:

- **Through inputs**: feed the student Lorentz-invariant features (smuggling equivariance into the input space).
- **Through data augmentation**: re-evaluate the teacher on Lorentz-transformed events and train the student to match — *behavioral* equivariance.
- **Through soft logits**: standard KD; gives "dark knowledge" but doesn't directly transfer structure.
- **Through hidden-state hints**: regress to invariant teacher features (CLS-token scalar channels).

Inputs and augmentation are where the result is decided; KD and hints add ~10–20 % of the lift.

### 4.3 Pipeline

**Step 1 — reproduce/load the teacher.**
- Implement training loop matching Appendix C.2: 12 blocks, 16 MV / 32 s channels, 8 heads, beam+time spurions, global token init zero, BCE, LION (or AdamW as fallback), 2·10⁵ steps, batch 128, cosine LR peak 3·10⁻⁴.
- Confirm AUC ≈ 0.987 on test before proceeding.
- Expose two hooks on the teacher: (a) the final logit, (b) the **scalar channels of the global token at the last block** (32 numbers per jet, all Lorentz-invariant). These will feed the hint loss.

**Step 2 — student input: invariant feature block.**
- Sort jet constituents by `pT` descending; truncate/zero-pad to `K=64` (covers >99 % of Kasieczka jets).
- Build per-jet feature vector:
  - Per-particle `pT_i, η_i, φ_i, m_i² = p_i · p_i, log E_i, log pT_i` (latter two break Lorentz but match what the teacher sees via its spurions, which already break SO⁺(1,3) → SO(3)).
  - Pairwise Minkowski inner products `p_i · p_j` for `i < j ≤ K` (≈ K(K−1)/2 numbers; this is PELICAN's input).
  - Pairwise `ΔR_ij`, `m_ij² = (p_i + p_j)²`.
  - Global: jet 4-momentum's mass and pT, total scalar pT, sphericity, planarity, N-subjettiness `τ₁, τ₂, τ₃`.
- Ablation: also try raw padded `(64, 4)` momenta to quantify how much of the gap closes from the input choice alone.

**Step 3 — distillation loss.**

```
L = α · BCE(z_S, y)
  + β · T² · KL(softmax(z_T / T) ‖ softmax(z_S / T))
  + γ · ‖ proj(h_S) − scalar_channels(teacher_global) ‖²
```

Defaults: `α = 0.5, β = 0.5, γ = 0.1, T = 4`. Anneal `T → 1` over training. `proj` is a small `nn.Linear` matching student-hidden-dim → 32.

**Step 4 — Lorentz augmentation (the secret sauce).**

For every minibatch:
1. Draw `Λ = R · B` where `R ∈ SO(3)` is a random rotation about the beam axis and `B` is a random boost with `β ~ U(0, 0.9)` along a random spatial direction. (Limit boosts to the residual SO(3) of the detector if you want to match the teacher's effective symmetry exactly; or use full SO⁺(1,3) and rely on the teacher's beam/time spurions to capture the broken directions — try both.)
2. Apply `Λ` to all 4-momenta of the event.
3. Recompute the student's invariant input features.
4. Re-evaluate the teacher on the boosted event to get fresh `z_T` and global-token features.
5. Train the student against these.

This roughly doubles training cost but gives the student "free" supervision under
the teacher's symmetry. **The expected biggest single contributor to closing the
AUC gap.** If GPU budget is tight, pre-cache teacher outputs on a fixed pool of
5–10× the dataset under random transformations.

**Step 5 — student architecture.** Two regimes:
- **Param-matched** (~1 M params, e.g. 5 hidden layers × 1024 + GELU + dropout 0.1) — answers "is there an equivariance gap once capacity is controlled?"
- **Deployment-sized** (~50 k params, e.g. 4 × 128) — answers "does distillation actually buy a deploy-cheap model?"

**Step 6 — schedule.** AdamW `lr = 1e-3`, cosine to `1e-5`, 100 epochs, batch 1024. Standard.

**Step 7 — evaluation.**
- AUC, accuracy, 1/εB at εS ∈ {0.5, 0.3} on the standard test set.
- **Boost-test**: apply a fixed `γ = 2` boost to the test set. Teacher AUC unchanged by construction. Student AUC degradation quantifies how much equivariance was *behaviorally* transferred.
- Inference throughput on 1× H100: forward time per jet, batch=1024.
- Calibration: ECE, Brier, reliability diagram.

### 4.4 Expected outcomes (priors before running)

- Param-matched MLP-on-invariants student with full Lorentz augmentation: **AUC ≈ 0.984–0.986** (paper teacher 0.9870; PELICAN 0.9870; ParT 0.9858).
- Without augmentation: AUC drops by ~0.002–0.005.
- Without invariant features (raw padded 4-mom): drops further by ~0.003–0.008.
- Boost-test AUC drop without augmentation: **0.01–0.05**. With heavy augmentation: target < 0.005.
- Deployment-sized student should still beat similarly-sized non-distilled MLPs by 0.005–0.01 AUC; this is the practical case for distillation.

### 4.5 Tradeoffs to flag

- **Invariant features ≈ smuggling the inductive bias into the input.** A purist objection: this isn't "really" MLP distillation. Honest framing: *some* part of the equivariance must move from architecture to either inputs or augmentation; the question is the mix. Run both extremes.
- **Augmented teacher re-evaluation roughly 2× training cost.** Mitigation: cache teacher outputs.
- **Calibration check**: BCE-trained L-GATr is usually well-calibrated; verify on val before locking `T = 4`.
- **Spurion handling**: the teacher uses beam/time tokens to break SO⁺(1,3) → SO(3). The student's invariant features are SO(3)-invariant by construction (they use η/φ); this is the right level of symmetry to match. If you instead use full Lorentz invariants (only Minkowski products, no η), the student is *more* symmetric than the teacher and may underperform.

---

## 5. Open decisions & ablation plan

Decisions to make early:

- [ ] Which optimizer to use for teacher reproduction (LION as paper, or AdamW for simplicity)?
- [ ] How to handle variable-length jets in the student input — truncate or attention-style mask in a "Deep Sets pre-aggregator + MLP" baseline?
- [ ] Should the teacher be retrained, or is a published checkpoint available? (Check the experiments repo at https://github.com/heidelberg-hepml/lorentz-gatr if it exists separately.)
- [ ] Random seeds: 5 seeds per run as in the paper.

Ablation grid (the result lives or dies by these):

| Ablation | Hypothesis |
|---|---|
| Inputs: raw 4-mom / invariants / both | invariants close ≥ 80 % of gap |
| KD components: hard / +soft / +hint | each adds ~0.001 AUC |
| Aug: none / SO(3) only / full SO⁺(1,3) | full aug adds 0.002–0.005 |
| Boost test (γ=2): on/off | aug closes gap to <0.005 |
| Student size: 1 M vs 50 k params | distillation worth more at small sizes |
| Teacher temperature: 1, 2, 4, 8 | 4 is usually optimal |
| `T` annealing: on/off | small effect |

Recommended order: build invariant-feature pipeline first (everything keys off its
output shape) → reproduce teacher → vanilla KD baseline → add augmentation → add
hint loss → ablate.

---

## 6. References

- Spinner, Bresó et al., *"Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics"*, NeurIPS 2024. Local copy: [lgatrpaper.pdf](lgatrpaper.pdf). arXiv: 2405.14806.
- Kasieczka et al., *"The machine learning landscape of top taggers"*, SciPost Phys. 7(1):014, 2019.
- Kasieczka et al., Top quark tagging reference dataset, https://doi.org/10.5281/zenodo.2603256
- PELICAN: Bogatskiy et al., 2022 — closest competitor; uses pairwise Minkowski products as inputs to a permutation-equivariant net. Useful template for the invariant-feature student input.
- LorentzNet: Gong et al., JHEP 07(2022)030.
- GATr (E(3) version): Brehmer et al., NeurIPS 2023.
- FitNets / hint distillation: Romero et al., ICLR 2015.
- Hinton et al., *"Distilling the knowledge in a neural network"*, NIPS 2014 workshop.
