"""
Apply x-axis Lorentz boosts to test inputs at varying β, evaluate the model at each β,
and write `boost_results.json`. Reproduces the central robustness experiment from
Liu et al. 2023 (`KD4Jets/eval.py:49`).

Modes:
  * Student: --checkpoint <path>
       Auto-detects ParT-style architecture from checkpoint args (mirrors eval_part.py).
  * Teacher: --teacher
       Builds the LGATrWrapper from `mlp_kd_deepsets.build_teacher`, runs `embed_tagging_data`
       per batch (boosted inputs are passed in; spurions are inserted post-boost which is correct
       — they are reference vectors of the new frame).

For both modes:
  - boost is applied to the dense kinematics tensor (B, N, 4) BEFORE feature extraction
  - mask is refreshed to drop entries that became non-finite (high-β regime)
  - β=0 short-circuits to the original tensor (no √0 in the rsqrt path)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_THIS, "..", "mlp_kd_deepsets")))

from mlp_kd_deepsets import (                                # noqa: E402
    TopTaggingNPZ, build_student_inputs, student_in_dim,
    _collate_for_teacher_only, DATA_PATH, TEACHER_CKPT, EPS,
    make_cfg_data, build_teacher, dense_to_sparse, teacher_logits,
)
from mlp_kd_part import (                                     # noqa: E402
    PartStudent, compute_pairwise_interactions, N_PAIR_INT,
)


# ============================================================
# Lorentz boost along x — matches KD4Jets/kd4jets/knowledge_distillation_base.py:324
# ============================================================
def boost_x(kin, mask, beta):
    """
    kin  : (B, N, 4)   E, px, py, pz   (zero-padded)
    mask : (B, N) bool
    beta : float in (-1, 1)
    Returns (boosted_kin, refreshed_mask)
    """
    if beta == 0.0:
        return kin, mask
    if not (-1.0 < beta < 1.0):
        raise ValueError(f"beta must be in (-1, 1); got {beta}")
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    E  = kin[..., 0]
    px = kin[..., 1]
    Ep  = gamma * (E  - beta * px)
    pxp = gamma * (px - beta * E)
    boosted = torch.stack([Ep, pxp, kin[..., 2], kin[..., 3]], dim=-1)
    # zero out non-finite entries (high β + extreme E) and re-apply the original mask
    finite = torch.isfinite(boosted).all(dim=-1)
    keep = mask & finite
    boosted = boosted.masked_fill(~keep.unsqueeze(-1), 0.0)
    return boosted, keep


# ============================================================
# Student boost-eval
# ============================================================
@torch.no_grad()
def eval_student_boosted(student, dataset, device, betas, batch_size, num_workers, use_pairwise):
    student.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=_collate_for_teacher_only)

    results = {"betas": [float(b) for b in betas],
               "auc": [], "rej_30": [], "rej_50": [], "rej_80": []}
    for beta in betas:
        all_y, all_p = [], []
        for kin, lbl, _idx in loader:
            kin = kin.to(device, non_blocking=True)
            mask = (kin.abs() > EPS).all(dim=-1)
            kin_b, mask_b = boost_x(kin, mask, float(beta))
            feats = build_student_inputs(kin_b, mask_b, with_pairwise=use_pairwise)
            pair_feats, pair_mask = compute_pairwise_interactions(kin_b, mask_b)
            logits = student(feats, mask_b, pair_feats, pair_mask)
            all_y.append(lbl.numpy())
            all_p.append(torch.sigmoid(logits).float().cpu().numpy())
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_p)
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        rej = {}
        for eps_S in (0.3, 0.5, 0.8):
            i = int(np.argmin(np.abs(tpr - eps_S)))
            rej[eps_S] = float("inf") if fpr[i] == 0 else 1.0 / float(fpr[i])
        results["auc"].append(float(auc))
        results["rej_30"].append(float(rej[0.3]))
        results["rej_50"].append(float(rej[0.5]))
        results["rej_80"].append(float(rej[0.8]))
        print(f"  beta={beta:.4f}  auc={auc:.4f}  rej@0.3={rej[0.3]:.0f} "
              f" rej@0.5={rej[0.5]:.0f}  rej@0.8={rej[0.8]:.0f}")
    return results


# ============================================================
# Teacher boost-eval — uses LGATrWrapper via mlp_kd_deepsets infrastructure
# ============================================================
@torch.no_grad()
def eval_teacher_boosted(teacher, cfg_data, dataset, device, betas, batch_size, num_workers):
    teacher.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=_collate_for_teacher_only)

    results = {"betas": [float(b) for b in betas],
               "auc": [], "rej_30": [], "rej_50": [], "rej_80": []}
    for beta in betas:
        all_y, all_p = [], []
        for kin, lbl, _idx in loader:
            kin = kin.to(device, non_blocking=True)
            mask = (kin.abs() > EPS).all(dim=-1)
            kin_b, mask_b = boost_x(kin, mask, float(beta))
            fm, sc, ptr = dense_to_sparse(kin_b, mask_b)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                logits = teacher_logits(teacher, fm, sc, ptr, cfg_data)
            all_y.append(lbl.numpy())
            all_p.append(torch.sigmoid(logits).float().cpu().numpy())
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_p)
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        rej = {}
        for eps_S in (0.3, 0.5, 0.8):
            i = int(np.argmin(np.abs(tpr - eps_S)))
            rej[eps_S] = float("inf") if fpr[i] == 0 else 1.0 / float(fpr[i])
        results["auc"].append(float(auc))
        results["rej_30"].append(float(rej[0.3]))
        results["rej_50"].append(float(rej[0.5]))
        results["rej_80"].append(float(rej[0.8]))
        print(f"  beta={beta:.4f}  auc={auc:.4f}  rej@0.3={rej[0.3]:.0f} "
              f" rej@0.5={rej[0.5]:.0f}  rej@0.8={rej[0.8]:.0f}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a ParT-style student checkpoint (auto-detects arch).")
    g.add_argument("--teacher", action="store_true",
                   help="Evaluate the L-GATr teacher itself (Lorentz-equivariant reference curve).")
    p.add_argument("--out-dir",     type=str, default=".")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--betas",       type=str, default=None,
                   help="Comma-separated beta values; default mirrors Liu et al.: "
                        "np.linspace(0, 1, 20, endpoint=False)")
    p.add_argument("--attention-backend", type=str, default="xformers")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[boost-eval] device={device}")

    if args.betas is not None:
        betas = [float(b) for b in args.betas.split(",")]
    else:
        betas = list(np.linspace(0, 1, 20, endpoint=False))
    print(f"[boost-eval] betas: {[f'{b:.4f}' for b in betas]}")

    test_ds = TopTaggingNPZ(DATA_PATH, mode="test")
    cfg_data = make_cfg_data()

    if args.teacher:
        print(f"[boost-eval] mode=teacher, attention_backend={args.attention_backend}")
        teacher = build_teacher(TEACHER_CKPT, args.attention_backend, device)
        results = eval_teacher_boosted(teacher, cfg_data, test_ds, device,
                                       betas, args.batch_size, args.num_workers)
        results["mode"] = "teacher"
    else:
        print(f"[boost-eval] mode=student, checkpoint={args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
        use_pairwise = bool(ckpt_args.get("use_pairwise", True))
        in_dim = student_in_dim(with_pairwise=use_pairwise)
        student = PartStudent(
            in_dim=in_dim,
            d_model=ckpt_args.get("d_model", 64),
            num_heads=ckpt_args.get("num_heads", 4),
            num_blocks=ckpt_args.get("num_blocks", 4),
            ffn_dim=ckpt_args.get("ffn_dim", 256),
            pair_in_dim=N_PAIR_INT,
            pair_hidden=ckpt_args.get("pair_hidden", 32),
            dropout=ckpt_args.get("dropout", 0.1),
        ).to(device)
        if isinstance(state, dict) and "model_state_dict" in state:
            student.load_state_dict(state["model_state_dict"])
            print(f"[boost-eval] loaded ckpt epoch={state.get('epoch', '?')} "
                  f"val_auc={state.get('val_auc', '?')}")
        else:
            student.load_state_dict(state)
        results = eval_student_boosted(student, test_ds, device, betas,
                                       args.batch_size, args.num_workers, use_pairwise)
        results["mode"] = "student"
        results["ckpt_args"] = {k: v for k, v in ckpt_args.items()
                                if isinstance(v, (int, float, bool, str, type(None)))}

    out_path = out_dir / "boost_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[boost-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
