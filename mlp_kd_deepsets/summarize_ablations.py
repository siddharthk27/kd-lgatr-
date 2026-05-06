"""
Scan ablations/ and print a sorted table of (tag, T, alpha, val_auc, test_auc, rej@0.5).

Reads:
  ablations/<tag>/final_test_metrics.json   - test AUC + rejection
  ablations/<tag>/history.json              - per-epoch val curve
  ablations/<tag>/deepsets_student_best.pt  - best-val AUC + epoch + args

Also writes ablations/summary.csv for plotting later.
"""
import csv
import json
from pathlib import Path

import torch


def main():
    abl_root = Path("ablations")
    if not abl_root.is_dir():
        print("no ablations/ directory found")
        return

    rows = []
    for sub in sorted(abl_root.iterdir()):
        if not sub.is_dir():
            continue
        tag = sub.name
        ftm = sub / "final_test_metrics.json"
        ckpt = sub / "deepsets_student_best.pt"
        hist = sub / "history.json"

        row = {"tag": tag, "T": None, "alpha": None,
               "best_val_auc": None, "best_epoch": None,
               "test_auc": None, "rej_30": None, "rej_50": None, "rej_80": None,
               "epochs_trained": None, "status": "incomplete"}

        if ckpt.exists():
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                row["best_val_auc"] = state.get("val_auc")
                row["best_epoch"]   = state.get("epoch")
                a = state.get("args", {}) or {}
                row["T"]     = a.get("temperature")
                row["alpha"] = a.get("alpha")
            except Exception as e:
                row["status"] = f"ckpt-load-error: {type(e).__name__}"

        if hist.exists():
            try:
                h = json.loads(hist.read_text())
                row["epochs_trained"] = len(h)
            except Exception:
                pass

        if ftm.exists():
            try:
                m = json.loads(ftm.read_text())
                row["test_auc"] = m.get("test_auc")
                rej = m.get("test_rej", {})
                row["rej_30"] = rej.get("0.3")
                row["rej_50"] = rej.get("0.5")
                row["rej_80"] = rej.get("0.8")
                row["status"] = "complete"
            except Exception as e:
                row["status"] = f"metrics-parse-error: {type(e).__name__}"

        rows.append(row)

    # sort by test AUC desc (None last)
    rows.sort(key=lambda r: (-(r["test_auc"] or -1), r["tag"]))

    # print table
    hdr = ["tag", "T", "alpha", "best_val_auc", "test_auc",
           "rej_30", "rej_50", "rej_80", "best_epoch", "epochs", "status"]
    widths = {h: max(len(h), 8) for h in hdr}
    for r in rows:
        for h in hdr:
            v = r.get(h if h != "epochs" else "epochs_trained")
            if isinstance(v, float):
                s = f"{v:.4f}" if v < 100 else f"{v:.0f}"
            else:
                s = "-" if v is None else str(v)
            widths[h] = max(widths[h], len(s))

    def fmt(v, h):
        if v is None: return "-"
        if isinstance(v, float):
            return f"{v:.4f}" if v < 100 else f"{v:.0f}"
        return str(v)

    line = " | ".join(h.ljust(widths[h]) for h in hdr)
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = []
        for h in hdr:
            v = r.get(h if h != "epochs" else "epochs_trained")
            cells.append(fmt(v, h).ljust(widths[h]))
        print(" | ".join(cells))

    # csv
    csv_path = abl_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr + ["epochs_trained"])
        w.writeheader()
        for r in rows:
            row = {k: r.get(k if k != "epochs" else "epochs_trained") for k in hdr}
            row["epochs_trained"] = r.get("epochs_trained")
            w.writerow(row)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
