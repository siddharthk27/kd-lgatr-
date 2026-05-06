"""
Transformer student distilled from L-GATr top-tagging teacher (option 2b).

Structurally homologous to the teacher (12 transformer-like blocks with
attention) but smaller and without Lorentz equivariance. Self-contained main()
that reuses the data pipeline, teacher loader, KD loss, and (optional) hint
distillation machinery from ../mlp_kd_deepsets/mlp_kd_deepsets.py.

Defaults track what the prior experiments learned:
  - --use-pairwise on by default (modest but real lift in the pairwise run)
  - hint distillation off by default (--hint-beta 0); enable to test whether
    structural homology with the teacher rescues the Phase-3 negative result
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# ---- pull shared infrastructure from the sibling deepsets directory ----
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_THIS, "..", "mlp_kd_deepsets")))

from mlp_kd_deepsets import (                                # noqa: E402
    DATA_PATH, TEACHER_CKPT, EPS, N_FEAT, N_PAIR_FEAT,
    TopTaggingNPZ, _collate_for_teacher_only,
    build_student_inputs, student_in_dim,
    make_cfg_data, build_teacher, HookedTeacher,
    teacher_logits, dense_to_sparse, precompute_teacher_logits,
    HintProjector, kd_loss, eval_split,
)


# ============================================================
# Transformer student (pre-LN, masked mean pool, small head)
# ============================================================
class TransformerStudent(nn.Module):
    def __init__(self, in_dim=11, d_model=64, num_heads=4, num_blocks=4,
                 ffn_dim=256, p_drop=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=p_drop, activation="gelu",
            batch_first=True, norm_first=True,         # pre-LN: stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model, 1),
        )
        self.d_model = d_model

    def forward(self, feats, mask, return_pooled=False):
        """
        feats : (B, N, in_dim)
        mask  : (B, N) bool, True where particle is real
        """
        x = self.input_proj(feats)                                  # (B, N, d_model)
        # PyTorch convention: True in key_padding_mask = ignore that key
        kpm = ~mask
        x = self.encoder(x, src_key_padding_mask=kpm)               # (B, N, d_model)
        x = self.out_norm(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # masked mean
        logits = self.head(pooled).squeeze(-1)                      # (B,)
        if return_pooled:
            return logits, pooled
        return logits


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--alpha",          type=float, default=0.7)
    parser.add_argument("--temperature",    type=float, default=2.0)
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--seed",           type=int,   default=42)
    # arch
    parser.add_argument("--d-model",        type=int,   default=64)
    parser.add_argument("--num-heads",      type=int,   default=4)
    parser.add_argument("--num-blocks",     type=int,   default=4)
    parser.add_argument("--ffn-dim",        type=int,   default=256)
    parser.add_argument("--dropout",        type=float, default=0.1)
    # features
    parser.add_argument("--use-pairwise",      action="store_true", default=True,
                        help="Include 4 pairwise-aggregate features (default ON; pass --no-pairwise to disable).")
    parser.add_argument("--no-pairwise", dest="use_pairwise", action="store_false")
    # hints (optional Phase 3-style)
    parser.add_argument("--hint-beta",      type=float, default=0.0,
                        help="Weight on penultimate-invariant hint loss. 0 disables.")
    parser.add_argument("--hint-projector-hidden", type=int, default=128)
    # io
    parser.add_argument("--out-dir",        type=str,   default=".")
    parser.add_argument("--teacher-cache-dir", type=str, default=None,
                        help="Where teacher_logits_{train,val}.pt (and optionally "
                             "teacher_invariants_{train,val}.pt) live. Defaults to --out-dir.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--attention-backend", type=str, default="xformers",
                        help="LGATr teacher attention backend (xformers | flash | flex).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.teacher_cache_dir) if args.teacher_cache_dir else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "teacher_logits_train.pt"
    val_cache   = cache_dir / "teacher_logits_val.pt"
    train_inv_cache = cache_dir / "teacher_invariants_train.pt"
    val_inv_cache   = cache_dir / "teacher_invariants_val.pt"
    cache_complete = train_cache.exists() and val_cache.exists()
    hint_mode = args.hint_beta > 0.0
    inv_cache_complete = train_inv_cache.exists() and val_inv_cache.exists()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} | torch={torch.__version__} | seed={args.seed}")
    if device.type == "cuda":
        print(f"[setup] gpu={torch.cuda.get_device_name(0)} | "
              f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    cfg_data = make_cfg_data()
    print(f"[setup] cfg_data={vars(cfg_data)}")
    print(f"[setup] cache_dir={cache_dir} | cache_complete={cache_complete} | "
          f"hint_mode={hint_mode} | inv_cache_complete={inv_cache_complete}")

    val_ds = TopTaggingNPZ(DATA_PATH, mode="val")
    train_teacher_invariants = None
    val_teacher_invariants   = None
    can_skip_teacher = (cache_complete
                        and (not hint_mode or inv_cache_complete)
                        and not args.smoke_test)

    if can_skip_teacher:
        print("[teacher] cache present — skipping teacher load + smoke test")
        train_ds = TopTaggingNPZ(DATA_PATH, mode="train")
        train_teacher_logits = torch.load(train_cache, map_location="cpu")
        val_teacher_logits   = torch.load(val_cache,   map_location="cpu")
        if train_teacher_logits.numel() != len(train_ds):
            raise RuntimeError("train logit cache size mismatch — delete and rebuild")
        if val_teacher_logits.numel() != len(val_ds):
            raise RuntimeError("val logit cache size mismatch — delete and rebuild")
        if hint_mode:
            train_teacher_invariants = torch.load(train_inv_cache, map_location="cpu")
            val_teacher_invariants   = torch.load(val_inv_cache,   map_location="cpu")
    else:
        teacher = build_teacher(TEACHER_CKPT, args.attention_backend, device)
        hooked = HookedTeacher(teacher) if hint_mode else None

        smoke_n = min(2048, len(val_ds))
        smoke_loader = DataLoader(
            torch.utils.data.Subset(val_ds, list(range(smoke_n))),
            batch_size=256, shuffle=False, num_workers=2,
            collate_fn=_collate_for_teacher_only,
        )
        print(f"[smoke] running teacher over {smoke_n} val jets ...")
        t_logits_arr, t_labels_arr = [], []
        for kin, lbl, _idx in smoke_loader:
            kin = kin.to(device, non_blocking=True)
            mask = (kin.abs() > EPS).all(dim=-1)
            fm, sc, ptr = dense_to_sparse(kin, mask)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                tl = teacher_logits(teacher, fm, sc, ptr, cfg_data)
            t_logits_arr.append(tl.float().cpu().numpy())
            t_labels_arr.append(lbl.numpy())
        smoke_auc = roc_auc_score(np.concatenate(t_labels_arr),
                                  np.concatenate(t_logits_arr))
        print(f"[smoke] teacher AUC on {smoke_n} val jets = {smoke_auc:.4f}")
        if args.smoke_test:
            return
        if smoke_auc < 0.95:
            raise RuntimeError(f"Teacher smoke AUC {smoke_auc:.4f} too low — abort.")

        train_ds = TopTaggingNPZ(DATA_PATH, mode="train")
        train_out = precompute_teacher_logits(
            "train", train_ds, teacher, cfg_data,
            cache_path=train_cache, device=device, batch_size=256,
            num_workers=args.num_workers,
            hooked=hooked, invariant_cache_path=(train_inv_cache if hint_mode else None),
        )
        val_out = precompute_teacher_logits(
            "val", val_ds, teacher, cfg_data,
            cache_path=val_cache, device=device, batch_size=256,
            num_workers=args.num_workers,
            hooked=hooked, invariant_cache_path=(val_inv_cache if hint_mode else None),
        )
        if hint_mode:
            train_teacher_logits, train_teacher_invariants = train_out
            val_teacher_logits,   val_teacher_invariants   = val_out
        else:
            train_teacher_logits = train_out
            val_teacher_logits   = val_out

        if hooked is not None:
            hooked.remove_hook()
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_auc_teacher = roc_auc_score(val_ds.lbl.numpy(), val_teacher_logits.numpy())
    print(f"[teacher] full val AUC (from cache) = {val_auc_teacher:.4f}")

    # ---- student ----
    in_dim = student_in_dim(with_pairwise=args.use_pairwise)
    student = TransformerStudent(
        in_dim=in_dim, d_model=args.d_model, num_heads=args.num_heads,
        num_blocks=args.num_blocks, ffn_dim=args.ffn_dim, p_drop=args.dropout,
    ).to(device)
    hint_proj = (HintProjector(args.d_model, HookedTeacher.INVARIANT_DIM,
                               hidden=args.hint_projector_hidden).to(device)
                 if hint_mode else None)

    n_params_s = sum(p.numel() for p in student.parameters())
    n_params_h = sum(p.numel() for p in hint_proj.parameters()) if hint_proj else 0
    print(f"[student] Transformer | in_dim={in_dim} | "
          f"d_model={args.d_model} num_heads={args.num_heads} num_blocks={args.num_blocks} "
          f"ffn={args.ffn_dim} | params={n_params_s:,}"
          + (f" + hint_proj={n_params_h:,}" if hint_proj else ""))

    train_params = list(student.parameters())
    if hint_proj is not None:
        train_params += list(hint_proj.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate_for_teacher_only, drop_last=False,
    )

    history = []
    best_val_auc = -1.0
    best_path = out_dir / "transformer_student_best.pt"

    print(f"\n--- Training: {args.epochs} epochs, BS {args.batch_size}, "
          f"alpha={args.alpha}, T={args.temperature}, use_pairwise={args.use_pairwise}"
          + (f", hint_beta={args.hint_beta}" if hint_mode else "")
          + " ---\n")
    for epoch in range(1, args.epochs + 1):
        student.train()
        if hint_proj is not None:
            hint_proj.train()
        t0 = time.time()
        running = {"loss": 0.0, "soft": 0.0, "hard": 0.0, "hint": 0.0, "n": 0}
        for kin, lbl, idx in train_loader:
            kin = kin.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            t_log = train_teacher_logits[idx].to(device, non_blocking=True)
            t_inv = (train_teacher_invariants[idx].to(device, non_blocking=True)
                     if hint_mode else None)
            mask = (kin.abs() > EPS).all(dim=-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                feats = build_student_inputs(kin, mask, with_pairwise=args.use_pairwise)
                if hint_mode:
                    s_log, s_pool = student(feats, mask, return_pooled=True)
                    s_hint = hint_proj(s_pool)
                else:
                    s_log = student(feats, mask)
                    s_hint = None
                loss, soft, hard, hint = kd_loss(
                    s_log, t_log, lbl,
                    alpha=args.alpha, T=args.temperature,
                    student_hint=s_hint, teacher_hint=t_inv, beta=args.hint_beta,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bs = lbl.numel()
            running["loss"] += loss.item() * bs
            running["soft"] += soft.item() * bs
            running["hard"] += hard.item() * bs
            if hint is not None:
                running["hint"] += hint.item() * bs
            running["n"]    += bs

        avg_loss = running["loss"] / running["n"]
        avg_soft = running["soft"] / running["n"]
        avg_hard = running["hard"] / running["n"]
        avg_hint = running["hint"] / running["n"] if hint_mode else 0.0

        val_auc, val_rej, _, _ = eval_split(
            student, val_ds, device,
            batch_size=1024, num_workers=args.num_workers,
            with_pairwise=args.use_pairwise,
        )
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        components = f"soft={avg_soft:.4f} hard={avg_hard:.4f}"
        if hint_mode:
            components += f" hint={avg_hint:.4f}"
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"loss={avg_loss:.4f} ({components}) | "
              f"val AUC={val_auc:.4f} rej@0.3={val_rej[0.3]:.0f} rej@0.5={val_rej[0.5]:.0f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        hist_entry = dict(epoch=epoch, loss=avg_loss, soft=avg_soft, hard=avg_hard,
                          val_auc=val_auc, val_rej_30=val_rej[0.3],
                          val_rej_50=val_rej[0.5], val_rej_80=val_rej[0.8],
                          lr=lr_now, time_s=elapsed)
        if hint_mode:
            hist_entry["hint"] = avg_hint
        history.append(hist_entry)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            ckpt = {
                "model_state_dict": student.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_rej": val_rej,
                "args": vars(args),
            }
            if hint_proj is not None:
                ckpt["hint_proj_state_dict"] = hint_proj.state_dict()
            torch.save(ckpt, best_path)
            print(f"  -> new best val AUC, saved to {best_path}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- final test eval using best-val checkpoint ----
    print(f"\n[final] loading best-val checkpoint (val AUC={best_val_auc:.4f}) ...")
    state = torch.load(best_path, map_location=device)
    student.load_state_dict(state["model_state_dict"])
    test_ds = TopTaggingNPZ(DATA_PATH, mode="test")
    test_auc, test_rej, _, _ = eval_split(
        student, test_ds, device,
        batch_size=1024, num_workers=args.num_workers,
        with_pairwise=args.use_pairwise,
    )
    print(f"[final] test AUC={test_auc:.4f} | "
          f"rej@0.3={test_rej[0.3]:.0f}  rej@0.5={test_rej[0.5]:.0f}  rej@0.8={test_rej[0.8]:.0f}")

    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump({"test_auc": test_auc,
                   "test_rej": {str(k): v for k, v in test_rej.items()},
                   "best_val_auc": best_val_auc}, f, indent=2)


if __name__ == "__main__":
    main()
