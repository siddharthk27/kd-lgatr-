"""
Evaluate the saved DeepSets student against the test split.

Reports AUC, calibrated accuracy (Youden-J threshold), and background
rejection at signal efficiency 0.3 / 0.5 / 0.8 — the same set of metrics
the L-GATr teacher reports in its training log.

Saves:
  - roc_curve_deepsets.png
  - prob_dist_deepsets.png
  - test_metrics.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from mlp_kd_deepsets import (
    DeepSetsStudent, TopTaggingNPZ, build_student_inputs, student_in_dim,
    _collate_for_teacher_only, DATA_PATH, EPS,
)
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="deepsets_student_best.pt")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out-dir",     type=str, default=".")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    # Inspect the checkpoint to recover whether it was trained with --use-pairwise.
    state = torch.load(args.checkpoint, map_location=device)
    ckpt_args = state.get("args", {}) if isinstance(state, dict) else {}
    use_pairwise = bool(ckpt_args.get("use_pairwise", False))
    in_dim = student_in_dim(with_pairwise=use_pairwise)
    print(f"[eval] use_pairwise={use_pairwise} | in_dim={in_dim}")

    student = DeepSetsStudent(in_dim=in_dim, phi_hidden=128,
                              rho_hidden=(256, 128), p_drop=0.1).to(device)
    if isinstance(state, dict) and "model_state_dict" in state:
        student.load_state_dict(state["model_state_dict"])
        print(f"[eval] loaded ckpt from epoch {state.get('epoch','?')} "
              f"(val AUC={state.get('val_auc','?')})")
    else:
        student.load_state_dict(state)
        print(f"[eval] loaded raw state_dict")
    student.eval()

    # test data
    test_ds = TopTaggingNPZ(DATA_PATH, mode="test")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=_collate_for_teacher_only)

    all_y, all_p = [], []
    with torch.no_grad():
        for kin, lbl, _idx in loader:
            kin = kin.to(device, non_blocking=True)
            mask = (kin.abs() > EPS).all(dim=-1)
            feats = build_student_inputs(kin, mask, with_pairwise=use_pairwise)
            probs = torch.sigmoid(student(feats, mask))
            all_y.append(lbl.numpy())
            all_p.append(probs.float().cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    j = tpr - fpr
    best = int(np.argmax(j))
    best_thresh = float(thr[best])
    best_acc = accuracy_score(y_true, (y_pred >= best_thresh).astype(int))

    rej = {}
    for eps_S in (0.3, 0.5, 0.8):
        i = int(np.argmin(np.abs(tpr - eps_S)))
        rej[eps_S] = float("inf") if fpr[i] == 0 else 1.0 / float(fpr[i])

    print("\n" + "=" * 50)
    print(" KD DEEPSETS STUDENT — TEST RESULTS ")
    print("=" * 50)
    print(f"AUC:                     {auc:.4f}")
    print(f"Optimal Threshold:       {best_thresh:.4f}")
    print(f"Calibrated Accuracy:     {best_acc:.4f}")
    print("-" * 50)
    print("Background Rejection (1 / e_B):")
    print(f"  @ 30% Signal Eff:      {rej[0.3]:.0f}")
    print(f"  @ 50% Signal Eff:      {rej[0.5]:.0f}")
    print(f"  @ 80% Signal Eff:      {rej[0.8]:.0f}")
    print("=" * 50)

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"auc": auc, "calibrated_accuracy": best_acc,
                   "optimal_threshold": best_thresh,
                   "rej": {str(k): v for k, v in rej.items()}}, f, indent=2)

    # ROC
    plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, lw=2, color="darkorange",
             label=f"DeepSets KD student (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=2, color="navy", linestyle="--", label="Random")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — DeepSets KD Student")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(out_dir / "roc_curve_deepsets.png", dpi=200, bbox_inches="tight")
    plt.close()

    # score distribution
    plt.figure(figsize=(7.5, 6))
    plt.hist(y_pred[y_true == 0], bins=60, alpha=0.5, color="red",
             label="QCD background", density=True)
    plt.hist(y_pred[y_true == 1], bins=60, alpha=0.5, color="blue",
             label="Top signal", density=True)
    plt.axvline(best_thresh, color="black", linestyle="--",
                label=f"Threshold ({best_thresh:.2f})")
    plt.xlabel("Network output (sigmoid)")
    plt.ylabel("Density")
    plt.title("Score distribution — DeepSets KD Student")
    plt.legend(loc="upper center"); plt.grid(alpha=0.3)
    plt.savefig(out_dir / "prob_dist_deepsets.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[eval] wrote {out_dir / 'roc_curve_deepsets.png'}")
    print(f"[eval] wrote {out_dir / 'prob_dist_deepsets.png'}")
    print(f"[eval] wrote {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
