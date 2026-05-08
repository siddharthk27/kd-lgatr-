# `mlp_kd_fixed/` — agent instructions

This directory contains a single training script, `mlp_kd_fixed.py`, that re-runs
the MLP-KD experiment with the **corrected** L-GATr teacher interface (the
original `mlp_kd/` and `mlp_kd_engd/` runs had `in_s_channels=1` instead of
`8` and silently mismatched the teacher's input projection — see `F1` in the
parent `CLAUDE.md`).

The student architecture is **identical** to `../mlp_scratch_engd/`
(depth=3, d_ff=512, engineered features rel_pT/deta/dphi/rel_E, max 128
particles, LeakyReLU + BatchNorm), so the KD-vs-scratch comparison is a
one-variable contrast. The default KD loss is **softmax-T KL** with T=4
(Hinton); `--kd-loss logit_mse` is exposed as an ablation arm (in logit-MSE
the temperature cancels exactly — see F2).

This document is for an agent that has just cloned `kd-lgatr-/` onto a new
server and is told "run this." Paths to the dataset, teacher checkpoint, and
`lloca-experiments` repo will almost certainly **not** match the original
hardcoded paths in `../mlp_kd_deepsets/mlp_kd_deepsets.py`. That's expected —
this script accepts all of them via CLI args.

---

## Step 1 — verify dependencies

```bash
python -c "import torch, numpy, sklearn; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import xformers; print('xformers', xformers.__version__)"
```

If `xformers` is missing, you can pass `--attention-backend flash` or
`--attention-backend flex` instead. **Do not** try `--attention-backend torch`
— the LGATrWrapper raises ValueError on it (see auto-memory).

The `lgatr` package itself comes from `lloca-experiments` (its top-level
`lgatr/` folder), so it does not need to be installed separately as long as
you pass `--lloca-repo` correctly in step 3.

---

## Step 2 — locate the three required paths

You need three paths the user must supply (or you must locate). Ask the user
if any are unclear; do not invent paths.

### a. `--data-path` — `toptagging_full.npz`

The standard top-tagging dataset. ~1.5 GB. Inside it:
- `kinematics_{train,val,test}`: float arrays, shape `(N_jets, 200, 4)` =
  `(E, px, py, pz)`, zero-padded to 200 particles per jet.
- `labels_{train,val,test}`: 0/1 arrays.

Sizes: train 1.21M, val 403k, test 404k. If any of those splits is missing
(e.g., a "no-val" variant), abort and ask the user — `mlp_kd_fixed.py` uses
the val split for early-stopping checkpoint selection.

Try common locations:
```bash
find ~ /data /scratch -name 'toptagging_full.npz' 2>/dev/null | head
```

### b. `--teacher-ckpt` — L-GATr teacher weights

A `.pt` file from one of Jay's seeded runs. We have always used
`seed1001/models/model_run0_it174999.pt` for headline numbers; seeds 1002/1003
exist as alternates but produce slightly different teacher AUCs.

Look for a directory matching `runs/topt_lgatr/seed{1001,1002,1003}/models/`:
```bash
find ~ /data /scratch -path '*topt_lgatr*model_run0_it*.pt' 2>/dev/null | head
```

The script's smoke test (built-in, runs before training) confirms this
checkpoint loads correctly: it expects teacher AUC ≥ 0.95 on a 2k val slice
(target ~0.987). If smoke AUC < 0.95, the script aborts — that means the
checkpoint, attention backend, or feature pipeline doesn't match the
training-time configuration.

### c. `--lloca-repo` — clone of `lloca-experiments`

This repo provides the actual teacher class and embedding pipeline:
- `experiments.tagging.wrappers.LGATrWrapper`
- `experiments.tagging.embedding.embed_tagging_data`
- `lloca.framesnet.nonequi_frames.IdentityFrames`
- the `lgatr` package itself (`LGATr`, `MLPConfig`, `SelfAttentionConfig`)

Do not try to substitute the older `lgatr` interface that lives in
`../mlp_kd/mlp_kd.py` — that's the broken interface (`in_s_channels=1`)
the whole point of this script is to avoid.

```bash
find ~ /data /scratch -type d -name 'lloca-experiments' 2>/dev/null | head
```

If it's not on the server, the user will need to clone it. The original repo
is at `/nfs_home/users/dhruvk/scratch/jay_agarwal/lgatr/lloca-experiments` on
the source machine — coordinate with the user.

---

## Step 3 — smoke-test the teacher load

**Always** run the smoke test before launching a full training. It loads the
teacher, runs it over 2k val jets, and prints AUC. If it's clearly broken
(AUC ~0.5 or smoke aborts), you save 30+ minutes of wasted training.

```bash
python mlp_kd_fixed.py \
    --data-path        /YOUR/PATH/toptagging_full.npz \
    --teacher-ckpt     /YOUR/PATH/seed1001/models/model_run0_it174999.pt \
    --lloca-repo       /YOUR/PATH/lloca-experiments \
    --smoke-test
```

Expected output ends with:
```
[smoke] teacher AUC on 2048 val jets = 0.98XX  (expected ~0.987; aborts if << 0.95)
```

If smoke AUC < 0.95, **stop and diagnose** — do not blindly retry. Likely
causes: wrong checkpoint, attention backend mismatch, or a stale
`mlp_kd_deepsets/mlp_kd_deepsets.py` (the script imports the teacher pipeline
from there).

---

## Step 4 — full training: two arms

For clean attribution between "fixing the teacher" and "switching to softmax-T KL,"
run both arms. Arm A is the headline; Arm B is the F2 ablation.

### Arm A — softmax-T KL, T=4 (default, headline)

```bash
python mlp_kd_fixed.py \
    --data-path        /YOUR/PATH/toptagging_full.npz \
    --teacher-ckpt     /YOUR/PATH/seed1001/models/model_run0_it174999.pt \
    --lloca-repo       /YOUR/PATH/lloca-experiments \
    --out-dir          ./run_softmax_kl_T4
```

The first run pre-computes teacher logits for train+val (~5–15 min on a
decent GPU) and caches them to `./run_softmax_kl_T4/teacher_logits_{train,val}.pt`.
Then 30 epochs of student training (~15–60 min depending on GPU).

### Arm B — logit-MSE (T cancels, F2 ablation)

Reuse Arm A's teacher cache so this is student-training-only:

```bash
python mlp_kd_fixed.py \
    --data-path        /YOUR/PATH/toptagging_full.npz \
    --teacher-ckpt     /YOUR/PATH/seed1001/models/model_run0_it174999.pt \
    --lloca-repo       /YOUR/PATH/lloca-experiments \
    --teacher-cache-dir ./run_softmax_kl_T4 \
    --kd-loss          logit_mse \
    --out-dir          ./run_logit_mse
```

### Reusing an existing cache from elsewhere

If `../mlp_kd_deepsets/teacher_logits_{train,val}.pt` were copied over with
the repo (they're gitignored, so they don't come with `git clone` — must be
copied manually), point both arms at that:

```bash
--teacher-cache-dir ../mlp_kd_deepsets
```

The cache files are tiny (~5 MB train + ~1.5 MB val); ask the user whether
to copy them over rather than recomputing. The teacher is deterministic up to
floating-point noise, so the cache is a frozen artifact and safe to reuse.

---

## Step 5 — read the results

Each arm writes:
- `<out_dir>/mlp_student_best.pt` — best-val checkpoint
- `<out_dir>/history.json` — per-epoch trajectory
- `<out_dir>/final_test_metrics.json` — `test_auc`, `test_rej` at εS={0.3, 0.5, 0.8}, `best_val_auc`, plus the KD config used

Compare against the existing scratch baseline:

```
mlp_scratch_engd  (../mlp_scratch_engd/eval.log)
    AUC = 0.9501 | rej@0.3 = 110 | rej@0.5 = 45 | rej@0.8 = 13
```

…and the original (broken-teacher) MLP-KD numbers from `CLAUDE.md`:

```
mlp_kd       AUC 0.7492 (raw 4-momenta MLP, broken teacher)
mlp_kd_engd  AUC 0.7890 (engineered MLP, depth=3, broken teacher)
```

**What "success" looks like**: Arm A beats `mlp_scratch_engd` (AUC > 0.9501).
That validates the four-piece headline claim "KD helps every student class
when the teacher interface is correct," because previously the MLP row of
that claim was inverted.

**What "interesting" looks like**: Arm A beats Arm B by a noticeable margin.
That separately supports the F2 footnote — that on weak students, switching
from logit-MSE to softmax-T KL with a real T does buy something.

---

## Common failure modes

| Symptom | Most likely cause | Fix |
|---|---|---|
| `ImportError: experiments.tagging.wrappers` | `--lloca-repo` not pointed at a real `lloca-experiments` checkout | re-locate or clone the repo |
| `ValueError: attention backend ... not supported` | tried `--attention-backend torch` | use `xformers`, `flash`, or `flex` |
| smoke AUC ~0.5 | wrong checkpoint or wrong attention backend | inspect checkpoint path; try a different backend |
| smoke AUC ~0.7-0.9 | partial mismatch — likely wrong `cfg_data` (spurions, mass_reg, scaling) | the script uses `make_cfg_data()` from `mlp_kd_deepsets`; if that file was edited locally, restore it |
| `RuntimeError: train cache size N != dataset size M` | stale cache from a different dataset variant | delete `<cache_dir>/teacher_logits_*.pt` and rerun |
| training loss nan after a few epochs | AMP underflow on a particular batch; rare | rerun with a different seed; if persistent, switch to fp32 (remove the autocast block in `main()`) |

---

## Constraints — please respect

- **Do not edit** `../mlp_kd_deepsets/mlp_kd_deepsets.py`. This script imports
  the teacher pipeline (`build_teacher`, `make_cfg_data`, `dense_to_sparse`,
  `precompute_teacher_logits`, `kd_loss`) from there. Editing it would
  invalidate all downstream experiment results in the repo.
- **Do not change** the student architecture defaults
  (`--depth 3 --d-ff 512 --max-particles 128`). They are chosen specifically
  to match `mlp_scratch_engd` so the KD-vs-scratch comparison is single-variable.
  Changing them produces a different experiment that does not address
  the gap this script exists to close.
- **Do not skip Arm B**. The F2 ablation matters for the paper claim.
- If results are surprising or inconsistent with the targets above, **stop
  and report** — do not proactively launch more runs. The next decision is
  the user's.
