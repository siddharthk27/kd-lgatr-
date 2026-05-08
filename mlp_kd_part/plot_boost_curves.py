"""
Generate the headline boost-robustness figure: AUC and rej@εS=0.5 vs β,
three curves (teacher / KD-ParT / scratch-ParT), shaded gap showing the
KD-vs-scratch advantage (the inductive-bias-transfer evidence).

Reads:
  runs/teacher/boost_results.json
  runs/v1/boost_results.json
  runs/v1_scratch/boost_results.json

Writes:
  boost_robustness.png
  boost_robustness.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def main():
    T = load("runs/teacher/boost_results.json")
    K = load("runs/v1/boost_results.json")
    S = load("runs/v1_scratch/boost_results.json")
    betas   = np.array(T["betas"])
    auc_T   = np.array(T["auc"]);   r5_T = np.array(T["rej_50"])
    auc_K   = np.array(K["auc"]);   r5_K = np.array(K["rej_50"])
    auc_S   = np.array(S["auc"]);   r5_S = np.array(S["rej_50"])

    # transfer fraction at each β (clipped: only meaningful where teacher-scratch gap > 0)
    gap_TS = auc_T - auc_S
    gap_KS = auc_K - auc_S
    transfer = np.where(gap_TS > 1e-4, gap_KS / np.maximum(gap_TS, 1e-12), np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---------- AUC panel ----------
    ax1.plot(betas, auc_T, color="C0", lw=2, marker="o", markersize=4,
             label="Teacher (L-GATr, partial-equivariance)")
    ax1.plot(betas, auc_K, color="C1", lw=2, marker="s", markersize=4,
             label="KD-ParT student (205k params)")
    ax1.plot(betas, auc_S, color="C3", lw=2, marker="^", markersize=4, linestyle="--",
             label="Scratch ParT (same arch, 205k params)")
    ax1.fill_between(betas, auc_S, auc_K, color="C1", alpha=0.15,
                     label="KD−Scratch (transferred bias)")
    ax1.set_xlabel(r"Lorentz boost $\beta = v/c$ (x-axis)", fontsize=12)
    ax1.set_ylabel("Test AUC", fontsize=12)
    ax1.set_title("AUC vs boost β", fontsize=13)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.82, 0.99)

    # annotation: transfer fraction at high β (mean over β ≥ 0.5)
    mask = (betas >= 0.5) & (betas <= 0.85)
    tf_mean = float(np.nanmean(transfer[mask]))
    ax1.text(0.05, 0.83, f"KD recovers ≈ {tf_mean*100:.0f}% of (Teacher − Scratch) AUC gap\n"
                          f"averaged over β ∈ [0.5, 0.85]",
             transform=ax1.transAxes, fontsize=10,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.9))

    # ---------- rejection panel ----------
    ax2.semilogy(betas, r5_T, color="C0", lw=2, marker="o", markersize=4,
                 label="Teacher")
    ax2.semilogy(betas, r5_K, color="C1", lw=2, marker="s", markersize=4,
                 label="KD-ParT student")
    ax2.semilogy(betas, r5_S, color="C3", lw=2, marker="^", markersize=4, linestyle="--",
                 label="Scratch ParT")
    ax2.set_xlabel(r"Lorentz boost $\beta = v/c$ (x-axis)", fontsize=12)
    ax2.set_ylabel(r"Background rejection at $\varepsilon_S = 0.5$  (1 / $\varepsilon_B$)", fontsize=12)
    ax2.set_title(r"Rejection at $\varepsilon_S = 0.5$ vs boost β", fontsize=13)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.3, which="both")
    ax2.set_xlim(0, 1)

    # annotation: KD/scratch ratio at high β
    ratio_mean = float(np.mean(r5_K[mask] / r5_S[mask]))
    ax2.text(0.05, 0.05, f"KD-ParT / Scratch rejection ratio\n≈ {ratio_mean:.2f}× over β ∈ [0.5, 0.85]",
             transform=ax2.transAxes, fontsize=10,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6", alpha=0.9))

    fig.suptitle("Knowledge distillation transfers Lorentz inductive bias to a non-equivariant student",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = Path(f"boost_robustness.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out.resolve()}")

    print(f"\nSummary statistics:")
    print(f"  Mean transfer fraction (β ∈ [0.5, 0.85]) : {tf_mean*100:.1f}%")
    print(f"  Mean KD/Scratch rej@0.5 ratio (β same)    : {ratio_mean:.2f}x")


if __name__ == "__main__":
    main()
