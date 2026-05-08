"""
Flat-MLP knowledge distillation from L-GATr top-tagging teacher
— corrected teacher interface (in_s_channels=8, 7 tagging features +
  global flag, 3 spurions, /20 momentum scaling, mass regularization),
  plus default Hinton-style softmax-T KL.

Why a re-run is needed:
    The original mlp_kd/ and mlp_kd_engd/ scripts instantiated the teacher
    with in_s_channels=1 and a single beam token, then loaded the checkpoint
    with strict=False. The input projection silently mismatched, so the soft
    targets were noise. Resulting AUCs (0.7492, 0.7890, 0.7722) were below
    the scratch MLP baselines (mlp_scratch_raw: 0.9411, mlp_scratch_engd:
    0.9501) — i.e. KD made the MLP *worse*, which is a teacher-loading bug
    rather than a real claim about KD.

What this script does:
    - Imports the verified teacher pipeline from mlp_kd_deepsets — same
      LGATrWrapper construction, same embed_tagging_data path, same
      pre-computed teacher-logit cache layout.
    - Trains a flat MLP whose architecture, feature pipeline, and tokenizer
      match mlp_scratch_engd EXACTLY (depth=3, d_ff=512, max_particles=128,
      engineered per-particle features rel_pT/deta/dphi/rel_E, LeakyReLU +
      BatchNorm, no dropout). This makes the KD-vs-scratch contrast a single-
      variable comparison.
    - Default KD loss is softmax-T KL with T=4 (Hinton). Logit-MSE is exposed
      via --kd-loss logit_mse for an ablation arm; F2 in CLAUDE.md notes that
      T cancels exactly in logit-MSE, so that arm is implicitly T=1.
    - Caches teacher logits on disk; --teacher-cache-dir can point at an
      existing cache (e.g. mlp_kd_deepsets/) to skip the teacher entirely.

Run (example):
    python mlp_kd_fixed.py \\
        --data-path /path/to/toptagging_full.npz \\
        --teacher-ckpt /path/to/seed1001/models/model_run0_it174999.pt \\
        --lloca-repo /path/to/lloca-experiments \\
        --out-dir ./run_softmax_kl_T4

Outputs (in --out-dir):
    mlp_student_best.pt         best-val checkpoint
    history.json                per-epoch trajectory
    final_test_metrics.json     test-set AUC + rejection at eS=0.3/0.5/0.8
"""
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent           # kd-lgatr-/
EPS = 1e-5


# ============================================================
# Student: flat MLP, identical to mlp_scratch_engd
# ============================================================
class MLPStudent(nn.Module):
    def __init__(self, d_input=4, d_ff=512, depth=3, dropout=0.0, max_particles=128):
        super().__init__()
        layers = []
        d = d_input * max_particles
        for _ in range(depth - 1):
            layers += [
                nn.Linear(d, d_ff, bias=True),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(d_ff),
            ]
            d = d_ff
        self.mlp  = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1, bias=True)

    def forward(self, x_flat):
        return self.head(self.mlp(x_flat)).squeeze(-1)


# ============================================================
# Engineered per-particle features (matches mlp_scratch_engd):
#   rel_pT, deta, dphi, rel_E
# Returns (B, max_particles*4) flat input.
# ============================================================
def compute_engineered_features(kin, mask):
    """
    kin  : (B, N, 4)  E, px, py, pz  (zero-padded)
    mask : (B, N)     bool
    Returns: (B, N, 4)  zeroed on padded rows
    """
    eps = 1e-8
    E  = kin[..., 0]
    px = kin[..., 1]
    py = kin[..., 2]
    pz = kin[..., 3]

    pjet = (kin * mask.unsqueeze(-1).to(kin.dtype)).sum(dim=1, keepdim=True)   # (B,1,4)
    jE   = pjet[..., 0]
    jpx  = pjet[..., 1]
    jpy  = pjet[..., 2]
    jpz  = pjet[..., 3]

    pt_part = torch.sqrt(px * px + py * py + eps)
    pt_jet  = torch.sqrt(jpx * jpx + jpy * jpy + eps)
    rel_pT  = pt_part / (pt_jet + eps)

    p_mag_part = torch.sqrt(px * px + py * py + pz * pz + eps)
    p_mag_jet  = torch.sqrt(jpx * jpx + jpy * jpy + jpz * jpz + eps)

    eta_part = torch.atanh((pz / (p_mag_part + eps)).clamp(-0.999, 0.999))
    eta_jet  = torch.atanh((jpz / (p_mag_jet + eps)).clamp(-0.999, 0.999))
    deta = eta_part - eta_jet

    phi_part = torch.atan2(py, px)
    phi_jet  = torch.atan2(jpy, jpx)
    dphi = phi_part - phi_jet
    dphi = ((dphi + math.pi) % (2 * math.pi)) - math.pi

    rel_E = E / (jE + eps)

    feats = torch.stack([rel_pT, deta, dphi, rel_E], dim=-1)
    feats = feats * mask.unsqueeze(-1).to(feats.dtype)
    return feats


def build_mlp_input(kin, mask, max_particles):
    feats = compute_engineered_features(kin, mask)
    B, N, D = feats.shape
    if N >= max_particles:
        feats = feats[:, :max_particles, :]
    else:
        pad = torch.zeros(B, max_particles - N, D, device=feats.device, dtype=feats.dtype)
        feats = torch.cat([feats, pad], dim=1)
    return feats.reshape(B, -1)


# ============================================================
# Eval
# ============================================================
@torch.no_grad()
def eval_split(student, dataset, device, max_particles, batch_size=1024, num_workers=4,
               collate_fn=None):
    student.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    all_lbl, all_pred = [], []
    for kin, lbl, _idx in loader:
        kin  = kin.to(device, non_blocking=True)
        mask = (kin.abs() > EPS).all(dim=-1)
        x    = build_mlp_input(kin, mask, max_particles)
        logits = student(x)
        all_lbl.append(lbl.numpy())
        all_pred.append(torch.sigmoid(logits).float().cpu().numpy())
    y_true = np.concatenate(all_lbl)
    y_pred = np.concatenate(all_pred)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    rej = {}
    for eps_S in (0.3, 0.5, 0.8):
        i = int(np.argmin(np.abs(tpr - eps_S)))
        rej[eps_S] = float("inf") if fpr[i] == 0 else 1.0 / float(fpr[i])
    return auc, rej


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--data-path",     type=str, required=True,
                   help="toptagging_full.npz")
    p.add_argument("--teacher-ckpt",  type=str, required=True,
                   help="L-GATr teacher checkpoint, e.g. seed1001/models/model_run0_it174999.pt")
    p.add_argument("--lloca-repo",    type=str, required=True,
                   help="Path to lloca-experiments repo (provides experiments.tagging.wrappers etc.)")
    p.add_argument("--out-dir",       type=str, default=str(SCRIPT_DIR / "run"))
    p.add_argument("--teacher-cache-dir", type=str, default=None,
                   help="Where teacher_logits_{train,val}.pt live (point at "
                        "mlp_kd_deepsets/ to reuse the verified cache).")
    # KD knobs
    p.add_argument("--kd-loss",       type=str, default="softmax_kl",
                   choices=["softmax_kl", "logit_mse"],
                   help="softmax_kl = Hinton KL on softened sigmoid (T is meaningful); "
                        "logit_mse = MSE in logit space (T cancels exactly, F2).")
    p.add_argument("--alpha",         type=float, default=0.7)
    p.add_argument("--temperature",   type=float, default=4.0)
    # student
    p.add_argument("--depth",         type=int,   default=3)
    p.add_argument("--d-ff",          type=int,   default=512)
    p.add_argument("--max-particles", type=int,   default=128)
    p.add_argument("--dropout",       type=float, default=0.0)
    # training
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--num-workers",   type=int,   default=8)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--attention-backend", type=str, default="xformers",
                   help="LGATr attention backend: xformers | flash | flex (matches teacher training).")
    p.add_argument("--smoke-test",    action="store_true",
                   help="Verify teacher AUC on a small val slice and exit before training.")
    args = p.parse_args()

    # --- inject lloca-experiments + repo root onto sys.path BEFORE heavy imports ---
    if not os.path.isdir(args.lloca_repo):
        raise FileNotFoundError(f"--lloca-repo not a directory: {args.lloca_repo}")
    if args.lloca_repo not in sys.path:
        sys.path.insert(0, args.lloca_repo)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Override the DeepSets module's hardcoded paths so its internals use ours.
    # (We import from it for the teacher pipeline, dataset class, KD loss, etc.)
    from mlp_kd_deepsets import mlp_kd_deepsets as ds   # noqa: E402
    ds.DATA_PATH    = args.data_path
    ds.TEACHER_CKPT = args.teacher_ckpt

    # --- determinism / device ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} | torch={torch.__version__} | seed={args.seed}")
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)} | "
              f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    out_dir   = Path(args.out_dir);   out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.teacher_cache_dir) if args.teacher_cache_dir else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "teacher_logits_train.pt"
    val_cache   = cache_dir / "teacher_logits_val.pt"

    cfg_data = ds.make_cfg_data()
    print(f"[setup] cfg_data={vars(cfg_data)}")
    print(f"[setup] kd_loss={args.kd_loss} | alpha={args.alpha} | T={args.temperature}")
    print(f"[setup] teacher cache dir={cache_dir}")

    # --- data + teacher cache ---
    val_ds = ds.TopTaggingNPZ(args.data_path, mode="val")
    cache_complete = train_cache.exists() and val_cache.exists() and not args.smoke_test

    if cache_complete:
        print("[teacher] cache present — skipping teacher load")
        train_ds = ds.TopTaggingNPZ(args.data_path, mode="train")
        train_teacher_logits = torch.load(train_cache, map_location="cpu")
        val_teacher_logits   = torch.load(val_cache,   map_location="cpu")
        if train_teacher_logits.numel() != len(train_ds):
            raise RuntimeError(f"train cache size {train_teacher_logits.numel()} "
                               f"!= dataset size {len(train_ds)}; delete and retry")
        if val_teacher_logits.numel() != len(val_ds):
            raise RuntimeError(f"val cache size {val_teacher_logits.numel()} "
                               f"!= dataset size {len(val_ds)}; delete and retry")
    else:
        teacher = ds.build_teacher(args.teacher_ckpt, args.attention_backend, device)

        # smoke test: teacher AUC on a 2k val slice — must be ~0.987
        smoke_n = min(2048, len(val_ds))
        smoke_loader = DataLoader(
            torch.utils.data.Subset(val_ds, list(range(smoke_n))),
            batch_size=256, shuffle=False, num_workers=2,
            collate_fn=ds._collate_for_teacher_only,
        )
        print(f"[smoke] running teacher over {smoke_n} val jets ...")
        t_logits, t_labels = [], []
        for kin, lbl, _idx in smoke_loader:
            kin  = kin.to(device, non_blocking=True)
            mask = (kin.abs() > EPS).all(dim=-1)
            fm, sc, ptr = ds.dense_to_sparse(kin, mask)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                tl = ds.teacher_logits(teacher, fm, sc, ptr, cfg_data)
            t_logits.append(tl.float().cpu().numpy())
            t_labels.append(lbl.numpy())
        smoke_auc = roc_auc_score(np.concatenate(t_labels), np.concatenate(t_logits))
        print(f"[smoke] teacher AUC on {smoke_n} val jets = {smoke_auc:.4f}  "
              f"(expected ~0.987; aborts if << 0.95)")
        if args.smoke_test:
            return
        if smoke_auc < 0.95:
            raise RuntimeError(f"Teacher AUC {smoke_auc:.4f} too low — interface mismatch.")

        train_ds = ds.TopTaggingNPZ(args.data_path, mode="train")
        train_teacher_logits = ds.precompute_teacher_logits(
            "train", train_ds, teacher, cfg_data,
            cache_path=train_cache, device=device,
            batch_size=256, num_workers=args.num_workers,
        )
        val_teacher_logits = ds.precompute_teacher_logits(
            "val", val_ds, teacher, cfg_data,
            cache_path=val_cache, device=device,
            batch_size=256, num_workers=args.num_workers,
        )
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_auc_teacher = roc_auc_score(val_ds.lbl.numpy(), val_teacher_logits.numpy())
    print(f"[teacher] full val AUC (from cache) = {val_auc_teacher:.4f}")

    # --- student ---
    student = MLPStudent(
        d_input=4, d_ff=args.d_ff, depth=args.depth,
        dropout=args.dropout, max_particles=args.max_particles,
    ).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[student] flat MLP | depth={args.depth} d_ff={args.d_ff} "
          f"max_p={args.max_particles} | params={n_params:,}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=ds._collate_for_teacher_only, drop_last=False,
    )

    history = []
    best_val_auc = -1.0
    best_path = out_dir / "mlp_student_best.pt"

    print(f"\n--- Training: {args.epochs} epochs, BS {args.batch_size}, "
          f"kd_loss={args.kd_loss}, alpha={args.alpha}, T={args.temperature} ---\n")
    for epoch in range(1, args.epochs + 1):
        student.train()
        t0 = time.time()
        running = {"loss": 0.0, "soft": 0.0, "hard": 0.0, "n": 0}

        for kin, lbl, idx in train_loader:
            kin = kin.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            t_log = train_teacher_logits[idx].to(device, non_blocking=True)
            mask  = (kin.abs() > EPS).all(dim=-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                x = build_mlp_input(kin, mask, args.max_particles)
                s_log = student(x)
                loss, soft, hard, _ = ds.kd_loss(
                    s_log, t_log, lbl,
                    alpha=args.alpha, T=args.temperature,
                    loss_type=args.kd_loss,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = lbl.numel()
            running["loss"] += loss.item() * bs
            running["soft"] += soft.item() * bs
            running["hard"] += hard.item() * bs
            running["n"]    += bs

        avg_loss = running["loss"] / running["n"]
        avg_soft = running["soft"] / running["n"]
        avg_hard = running["hard"] / running["n"]

        val_auc, val_rej = eval_split(
            student, val_ds, device, args.max_particles,
            batch_size=1024, num_workers=args.num_workers,
            collate_fn=ds._collate_for_teacher_only,
        )
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} (soft={avg_soft:.4f} hard={avg_hard:.4f}) | "
              f"val AUC={val_auc:.4f} rej@0.3={val_rej[0.3]:.0f} rej@0.5={val_rej[0.5]:.0f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        history.append(dict(
            epoch=epoch, loss=avg_loss, soft=avg_soft, hard=avg_hard,
            val_auc=val_auc, val_rej_30=val_rej[0.3],
            val_rej_50=val_rej[0.5], val_rej_80=val_rej[0.8],
            lr=lr_now, time_s=elapsed,
        ))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state_dict": student.state_dict(),
                "epoch": epoch, "val_auc": val_auc, "val_rej": val_rej,
                "args": vars(args),
            }, best_path)
            print(f"  -> new best val AUC, saved to {best_path}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --- final test eval ---
    print(f"\n[final] loading best-val checkpoint (val AUC={best_val_auc:.4f}) ...")
    state = torch.load(best_path, map_location=device)
    student.load_state_dict(state["model_state_dict"])
    test_ds = ds.TopTaggingNPZ(args.data_path, mode="test")
    test_auc, test_rej = eval_split(
        student, test_ds, device, args.max_particles,
        batch_size=1024, num_workers=args.num_workers,
        collate_fn=ds._collate_for_teacher_only,
    )
    print(f"[final] test AUC={test_auc:.4f} | "
          f"rej@0.3={test_rej[0.3]:.0f}  rej@0.5={test_rej[0.5]:.0f}  rej@0.8={test_rej[0.8]:.0f}")

    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump({
            "test_auc": test_auc,
            "test_rej": {str(k): v for k, v in test_rej.items()},
            "best_val_auc": best_val_auc,
            "kd_loss": args.kd_loss,
            "alpha": args.alpha,
            "temperature": args.temperature,
        }, f, indent=2)


if __name__ == "__main__":
    main()
