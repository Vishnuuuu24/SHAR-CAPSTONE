#!/usr/bin/env python3
"""RESULT 1 closure table (P2C): every degradation x method across the classical, learned, comparator and D-62 runs.

Reads the four run directories' summary.json / classification_runs.json and writes results/p2/result1_denoise.csv:
one row per (degradation, method), with learned methods also aggregated over their training seeds. Deltas are
paired per fold against the same degradation left unrestored (identity rows, bit-identical across the runs).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

RUNS = ("results/p2/p2-classical", "results/p2/p2b-learned", "results/p2/p2b-nafnet", "results/p2/p2b-extra")


def fold_means(runs: list[dict], condition: str, metric: str) -> np.ndarray:
    folds = sorted({r["fold"] for r in runs})
    return np.array([np.mean([r[metric] for r in runs if r["condition"] == condition and r["fold"] == f]) for f in folds])


def rows_for(run_dir: Path) -> list[dict]:
    summary = json.loads((run_dir / "summary.json").read_text())
    runs = json.loads((run_dir / "classification_runs.json").read_text())
    groups: dict[tuple[str, str], list[str]] = {}
    for name in summary:
        deg, method = name.split("__")
        if method == "identity" and run_dir.name != "p2-classical":
            continue  # unrestored rows are reported once, from the classical run
        groups.setdefault((deg, re.sub(r"_seed\d+$", "", method)), []).append(name)
    out = []
    for (deg, method), names in groups.items():
        base = f"{deg}__identity"
        d_f1 = np.mean([fold_means(runs, n, "macro_f1") - fold_means(runs, base, "macro_f1") for n in names], axis=0)
        d_acc = np.mean([fold_means(runs, n, "accuracy") - fold_means(runs, base, "accuracy") for n in names], axis=0)
        psnr = [summary[n]["psnr_db"] for n in names]
        out.append({"degradation": deg, "method": method, "run": run_dir.name, "training_seeds": len(names),
                    "psnr_db": "inf" if None in psnr else round(float(np.mean(psnr)), 2),
                    "ssim": round(float(np.mean([summary[n]["ssim"] for n in names])), 4),
                    "macro_f1": round(float(np.mean([summary[n]["macro_f1"] for n in names])), 4),
                    "accuracy": round(float(np.mean([summary[n]["accuracy"] for n in names])), 4),
                    "macro_f1_delta_vs_unrestored": round(float(d_f1.mean()), 4),
                    "accuracy_delta_vs_unrestored": round(float(d_acc.mean()), 4),
                    "folds_improved": int((d_f1 > 0).sum()), "folds": len(d_f1),
                    "per_fold_macro_f1_delta": " ".join(f"{v:+.3f}" for v in d_f1)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, default=Path("results/p2/result1_denoise.csv"))
    args = ap.parse_args(argv)
    rows = [r for d in RUNS for r in rows_for(args.root / d)]
    order = list(dict.fromkeys(r["degradation"] for r in rows))
    rows.sort(key=lambda r: (order.index(r["degradation"]), r["method"] != "identity"))
    with (args.root / args.output).open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
