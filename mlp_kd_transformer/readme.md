# mlp_kd_transformer — small transformer student (option 2b)

Distillation from L-GATr top-tagging teacher to a small transformer student. This is the architecture-side follow-up to `../mlp_kd_deepsets/`, motivated by what the prior experiments showed:

- **v1 (DeepSets, 7 features)**: AUC 0.9833 / rej@εS=0.5 = 290.
- **Phase 2 (T/α sweep)**: flat. KD hyperparameters not the bottleneck.
- **Phase 3 (penultimate-invariant hint)**: flat. The hint signal was absorbed by a learned projection but didn't help classification — the student lacks structure to use it productively.
- **Pairwise features (DeepSets, 11 features)**: AUC 0.9837 / rej@εS=0.5 = 311. Real lift, but small.

The pattern says the student **architecture** is the binding constraint. Three of the four prior experiments only changed *what* the student is supervised on or fed; only the architecture actually moved the needle (and only modestly, because the DeepSets pool is still the bottleneck). This experiment changes the architecture: a small transformer with self-attention over particles.

## Why a transformer

- Structurally homologous to the teacher's 12 LGATrBlocks (attention + FFN, residual, LN).
- Can model multi-step inter-particle interactions — exactly what DeepSets can't.
- Can natively *use* the pairwise input features through attention, instead of squashing them through a fixed sum/mean/max pool.
- Gives hint distillation a real chance: the student now has analogous representations to match against the teacher's penultimate state. Phase-3 hint loss may stop being degenerate.

## Architecture (default)

```
input (B, N, in_dim=11)            ← 7 base + 4 pairwise features
        ↓
nn.Linear → d_model=64
        ↓
nn.TransformerEncoder × 4 blocks   ← pre-LN, GELU, MHA(4 heads), FFN(256), dropout 0.1
        ↓
LayerNorm
        ↓
masked-mean pool over particles → (B, 64)
        ↓
Linear(64→64) → GELU → Dropout → Linear(64→1)
        ↓
logit
```

- **Params**: ~210k (vs DeepSets v1 ~150k, pairwise 167k; teacher ~1M)
- **Pre-LN**: more stable training than post-LN at this depth
- **Masked-mean pool**: simpler than CLS-token; respects variable-length particle sets

## What's reused from `../mlp_kd_deepsets/`

The training script imports rather than duplicates:

| | source |
|---|---|
| Data loader, NPZ paths | `TopTaggingNPZ`, `DATA_PATH` |
| Per-particle features, pairwise features | `compute_student_features`, `compute_pairwise_features`, `build_student_inputs`, `student_in_dim` |
| Teacher loader + hooked-teacher for hint extraction | `build_teacher`, `HookedTeacher`, `teacher_logits` |
| Teacher logit + invariant cache | `precompute_teacher_logits` |
| KD loss (with optional hint term) | `kd_loss` |
| Student-eval helper | `eval_split` |
| Hint-projection head | `HintProjector` |

The teacher logit cache built by v1 (`../mlp_kd_deepsets/teacher_logits_{train,val}.pt`) is reused directly. **No new teacher pass needed for the headline run.**

If you opt into hint distillation (`--hint-beta > 0`), it'll use the existing `teacher_invariants_{train,val}.pt` from Phase 3 if present, else build them once (~10 min).

## Files in this directory

| | |
|---|---|
| `mlp_kd_transformer.py` | training script (self-contained `main()`, all helpers imported) |
| `eval_transformer.py`   | standalone eval against a saved checkpoint |
| `run_transformer.sh`    | sbatch wrapper for `gpu-short` (4 hr cap, 1 A100, 24 CPU, 80 GB) |

## Launch

```
cd /nfs_home/users/dhruvk/khare27/kd-lgatr-/mlp_kd_transformer
sbatch run_transformer.sh
```

Defaults: 30 epochs, BS 512, lr 3e-4, α=0.7, T=2, pairwise features ON, hint OFF.

To run with hint distillation (now that the student arch is appropriate for it):
```
XF_TAG=v1_hint XF_HINT_BETA=0.5 sbatch run_transformer.sh
```

To explore the transformer's depth/width:
```
XF_TAG=d96_b6 XF_DMODEL=96 XF_NBLOCKS=6 sbatch run_transformer.sh
```

Outputs land in `runs/<tag>/` with the same files as v1 (`transformer_student_best.pt`, `history.json`, `final_test_metrics.json`, `roc_curve_transformer.png`, `prob_dist_transformer.png`).

## What success looks like

- **Plausible target** (without hints): AUC 0.985–0.986, rej@εS=0.5 ≈ 350–400. ParT-class numbers on this dataset.
- **Stretch** (with hints, if Phase-3 logic is rescued by structural homology): AUC 0.986–0.987, rej@εS=0.5 ≈ 400–500.
- **Teacher ceiling**: AUC 0.9870, rej@εS=0.5 = 587. We won't reach it without Lorentz equivariance and/or two-particle attention biases (ParT does the latter), but closing 30–50% of the rejection gap is the goal.

## Notes / open follow-ups

- ParT-style **pairwise attention bias** (precomputed `Δr_ij`, `log(k_T_ij)`, `log(m²_ij)` injected into the attention scores) is a clear next move if this baseline closes most of the AUC gap but rejection lags. Same architecture, just `attn_bias[i,j] += MLP(pair_features_ij)`.
- Lorentz-equivariant student (a smaller LGATr — say 4 blocks, hidden_mv=8) is the only architecturally principled way to close the last ~0.001 AUC. Saved for after we know whether vanilla transformer hits its expected plateau.
