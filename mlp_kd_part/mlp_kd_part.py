"""
ParT-style transformer student: vanilla transformer encoder with a learned
pairwise-feature attention bias (Qu, Li, Qian 2022).

Difference from ../mlp_kd_transformer/:
  - For each pair of particles (i, j) we compute 4 physics-motivated pair
    features  U_ij = (log ΔR_ij, log k_T_ij, log z_ij, log m_ij)  once per
    batch, pass them through a tiny MLP to produce a per-head learned bias
    of shape (B, num_heads, N, N), and add that bias to attention scores
    before softmax in every encoder block. This lets attention "see" two-
    particle structure directly instead of inferring it from per-particle
    features.

  - We use a hand-written attention block (not nn.MultiheadAttention)
    because PyTorch's built-in API doesn't cleanly expose a per-head
    additive bias.

Reuses everything else from ../mlp_kd_deepsets/mlp_kd_deepsets.py:
  data loader, teacher loader, KD loss, hint plumbing, eval helpers,
  per-particle feature pipeline (7 base + 4 aggregate features).
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
# Pairwise interaction features  (B, N, N, 4)
# ParT eq. 7:  U_ij = (log ΔR_ij, log k_T_ij, log z_ij, log m_ij)
# ============================================================
N_PAIR_INT = 4

def compute_pairwise_interactions(kin_dense, mask_dense, eps=1e-8):
    """
    kin_dense  : (B, N, 4)  E, px, py, pz   zero-padded
    mask_dense : (B, N) bool

    Returns
        pair    : (B, N, N, 4)  pre-normalized log features, zero on invalid pairs
        pair_mask : (B, N, N) bool  True where both i and j are real
    """
    B, N, _ = kin_dense.shape
    device = kin_dense.device
    dtype  = kin_dense.dtype

    E  = kin_dense[..., 0]
    px = kin_dense[..., 1]
    py = kin_dense[..., 2]
    pz = kin_dense[..., 3]
    pt = torch.sqrt(px * px + py * py + eps)                                  # (B, N)
    p  = torch.sqrt(px * px + py * py + pz * pz + eps)
    eta = torch.atanh((pz / p).clamp(-0.999, 0.999))
    phi = torch.atan2(py, px)

    pair_mask = mask_dense.unsqueeze(2) & mask_dense.unsqueeze(1)             # (B, N, N)

    deta = eta.unsqueeze(2) - eta.unsqueeze(1)
    dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
    dphi = ((dphi + math.pi) % (2 * math.pi)) - math.pi
    dR   = torch.sqrt(deta * deta + dphi * dphi + eps)                        # (B, N, N)

    pt_min = torch.minimum(pt.unsqueeze(2), pt.unsqueeze(1))                  # (B, N, N)
    pt_sum = pt.unsqueeze(2) + pt.unsqueeze(1)                                # (B, N, N)
    kT     = pt_min * dR
    z      = pt_min / (pt_sum + eps)

    # invariant mass squared of the (i, j) two-body system
    E_sum  = E.unsqueeze(2) + E.unsqueeze(1)
    px_sum = px.unsqueeze(2) + px.unsqueeze(1)
    py_sum = py.unsqueeze(2) + py.unsqueeze(1)
    pz_sum = pz.unsqueeze(2) + pz.unsqueeze(1)
    m2 = E_sum * E_sum - (px_sum * px_sum + py_sum * py_sum + pz_sum * pz_sum)
    m  = torch.sqrt(m2.clamp(min=0.0) + eps)                                  # (B, N, N)

    log_dR = dR.clamp(min=eps).log()
    log_kT = kT.clamp(min=eps).log()
    log_z  = z.clamp(min=eps).log()
    log_m  = m.clamp(min=eps).log()

    pair = torch.stack([log_dR, log_kT, log_z, log_m], dim=-1)                # (B, N, N, 4)
    pair = pair * pair_mask.unsqueeze(-1).to(dtype)
    return pair, pair_mask


# ============================================================
# Pairwise-bias attention block (pre-LN, per-head additive bias)
# ============================================================
class PairBiasAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.out  = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        x                : (B, N, d_model)
        attn_bias        : (B, num_heads, N, N) or None  — added to scores pre-softmax
        key_padding_mask : (B, N) bool, True where padded
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                                          # each (B, N, H, D)
        q = q.transpose(1, 2)                                                # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)          # (B, H, N, N)
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]                        # broadcast over (H, N_q)
            # Use dtype-aware negative sentinel: -1e9 overflows float16 under AMP.
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v                                                    # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out(out)


class PairBiasBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = PairBiasAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        # pre-LN residual
        x = x + self.drop(self.attn(self.norm1(x), attn_bias=attn_bias,
                                    key_padding_mask=key_padding_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ============================================================
# ParT-style student
# ============================================================
class PartStudent(nn.Module):
    def __init__(self, in_dim, d_model=64, num_heads=4, num_blocks=4,
                 ffn_dim=256, pair_in_dim=N_PAIR_INT, pair_hidden=32,
                 dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        # Pair bias MLP: (B, N, N, 4) -> (B, N, N, num_heads)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, num_heads),
        )
        self.blocks = nn.ModuleList([
            PairBiasBlock(d_model, num_heads, ffn_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.num_heads = num_heads
        self.d_model   = d_model

    def forward(self, feats, mask, pair_feats, pair_mask, return_pooled=False):
        """
        feats      : (B, N, in_dim)
        mask       : (B, N)        True where real
        pair_feats : (B, N, N, 4)
        pair_mask  : (B, N, N)
        """
        x = self.input_proj(feats)                                            # (B, N, d_model)

        # Pair bias: shared across all blocks (ParT's design)
        bias = self.pair_mlp(pair_feats)                                       # (B, N, N, H)
        bias = bias.permute(0, 3, 1, 2).contiguous()                          # (B, H, N, N)
        # zero out bias on invalid pair entries; key_padding_mask in attention
        # already kills attention to padded keys, but masking the bias keeps
        # the MLP from feeding on garbage gradient through padded entries
        bias = bias.masked_fill(~pair_mask.unsqueeze(1), 0.0)

        kpm = ~mask                                                            # (B, N) True where padded
        for blk in self.blocks:
            x = blk(x, attn_bias=bias, key_padding_mask=kpm)
        x = self.out_norm(x)

        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)             # masked mean
        logits = self.head(pooled).squeeze(-1)
        if return_pooled:
            return logits, pooled
        return logits


# ============================================================
# Pair-aware eval helper (parallels mlp_kd_deepsets.eval_split)
# ============================================================
@torch.no_grad()
def eval_split_pair(student, dataset, device, batch_size=1024, num_workers=4,
                    with_pairwise=True):
    student.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=_collate_for_teacher_only,
    )
    all_lbl, all_pred = [], []
    for kin, lbl, _idx in loader:
        kin  = kin.to(device, non_blocking=True)
        mask = (kin.abs() > EPS).all(dim=-1)
        feats = build_student_inputs(kin, mask, with_pairwise=with_pairwise)
        pair_feats, pair_mask = compute_pairwise_interactions(kin, mask)
        logits = student(feats, mask, pair_feats, pair_mask)
        all_lbl.append(lbl.numpy())
        all_pred.append(torch.sigmoid(logits).float().cpu().numpy())

    from sklearn.metrics import roc_curve
    y_true = np.concatenate(all_lbl)
    y_pred = np.concatenate(all_pred)
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    rej = {}
    for eps_S in (0.3, 0.5, 0.8):
        i = int(np.argmin(np.abs(tpr - eps_S)))
        rej[eps_S] = float("inf") if fpr[i] == 0 else 1.0 / float(fpr[i])
    return auc, rej, y_true, y_pred


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch-size",     type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--alpha",          type=float, default=0.7)
    parser.add_argument("--temperature",    type=float, default=2.0)
    parser.add_argument("--num-workers",    type=int,   default=8)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--d-model",        type=int,   default=64)
    parser.add_argument("--num-heads",      type=int,   default=4)
    parser.add_argument("--num-blocks",     type=int,   default=4)
    parser.add_argument("--ffn-dim",        type=int,   default=256)
    parser.add_argument("--pair-hidden",    type=int,   default=32)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--use-pairwise",      action="store_true", default=True,
                        help="Include 4 pairwise-aggregate per-particle features (default ON).")
    parser.add_argument("--no-pairwise",        dest="use_pairwise", action="store_false")
    parser.add_argument("--hint-beta",      type=float, default=0.0)
    parser.add_argument("--hint-projector-hidden", type=int, default=128)
    parser.add_argument("--out-dir",        type=str,   default=".")
    parser.add_argument("--teacher-cache-dir", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--attention-backend", type=str, default="xformers")
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
    student = PartStudent(
        in_dim=in_dim, d_model=args.d_model, num_heads=args.num_heads,
        num_blocks=args.num_blocks, ffn_dim=args.ffn_dim,
        pair_in_dim=N_PAIR_INT, pair_hidden=args.pair_hidden,
        dropout=args.dropout,
    ).to(device)
    hint_proj = (HintProjector(args.d_model, HookedTeacher.INVARIANT_DIM,
                               hidden=args.hint_projector_hidden).to(device)
                 if hint_mode else None)

    n_params_s = sum(p.numel() for p in student.parameters())
    n_params_h = sum(p.numel() for p in hint_proj.parameters()) if hint_proj else 0
    print(f"[student] ParT-style | in_dim={in_dim} | "
          f"d_model={args.d_model} num_heads={args.num_heads} num_blocks={args.num_blocks} "
          f"ffn={args.ffn_dim} pair_hidden={args.pair_hidden} | params={n_params_s:,}"
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
    best_path = out_dir / "part_student_best.pt"

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
                pair_feats, pair_mask = compute_pairwise_interactions(kin, mask)
                if hint_mode:
                    s_log, s_pool = student(feats, mask, pair_feats, pair_mask,
                                            return_pooled=True)
                    s_hint = hint_proj(s_pool)
                else:
                    s_log = student(feats, mask, pair_feats, pair_mask)
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

        val_auc, val_rej, _, _ = eval_split_pair(
            student, val_ds, device,
            batch_size=512, num_workers=args.num_workers,
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
    test_auc, test_rej, _, _ = eval_split_pair(
        student, test_ds, device,
        batch_size=512, num_workers=args.num_workers,
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
