"""Error breakdown of one head's saved video-level confusion matrices (CPU, no training)."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

GROUPS = {
    "aggression": ["Abuse", "Assault", "Fighting"],
    "theft": ["Burglary", "Robbery", "Shoplifting", "Stealing"],
}


def load_mean_confusion(run_dir, head):
    files = sorted(glob.glob(f"{run_dir}/fold*/{head}-seed-*.json"))
    total, labels, seeds = None, None, set()
    for path in files:
        data = json.load(open(path))
        metrics = data["probe_result"]["video_level_metrics"]
        matrix = np.array(metrics["confusion_matrix"], float)
        labels = metrics["labels"]
        seeds.add(data["seed"])
        total = matrix if total is None else total + matrix
    return total / len(seeds), labels, len(files)


def breakdown(matrix, labels):
    normal = labels.index("Normal_Videos_event")
    classes = []
    for i, name in enumerate(labels):
        support = matrix[i].sum()
        others = [(labels[j], float(matrix[i, j])) for j in np.argsort(-matrix[i]) if j != i and matrix[i, j] > 0]
        classes.append({"class": name, "support": float(support), "recall": float(matrix[i, i] / support),
                        "to_normal": float(matrix[i, normal]) if i != normal else 0.0, "top_confusions": others[:3]})
    anomaly = [i for i in range(len(labels)) if i != normal]
    errors = sum(matrix[i].sum() - matrix[i, i] for i in anomaly)
    summary = {"anomaly_errors": float(errors),
               "anomaly_errors_into_normal": float(sum(matrix[i, normal] for i in anomaly))}
    for group, members in GROUPS.items():
        idx = [labels.index(m) for m in members]
        group_errors = sum(matrix[i].sum() - matrix[i, i] for i in idx)
        within = sum(matrix[i, j] for i in idx for j in idx if i != j)
        summary[group] = {"errors": float(group_errors), "within_group": float(within),
                          "within_share": float(within / group_errors),
                          "chance_share": (len(members) - 1) / (len(labels) - 1)}
    return classes, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/p1c/p1c-head-comparison-stepctl")
    parser.add_argument("--head", default="step_permuted")
    parser.add_argument("--out-dir", default="results/p1c/p1c-error-breakdown")
    args = parser.parse_args()
    matrix, labels, runs = load_mean_confusion(args.run_dir, args.head)
    classes, summary = breakdown(matrix, labels)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "breakdown.json").write_text(json.dumps(
        {"run_dir": args.run_dir, "head": args.head, "runs": runs,
         "counts": "videos, summed over folds, mean over seeds", "classes": classes, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
