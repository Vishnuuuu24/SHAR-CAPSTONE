#!/usr/bin/env python3
"""Equal-weight probability fusion of inner-cross-validation arms, tested against a base arm (plan Step 2 stacking).

    scripts/fuse_inner_cv.py --base RUN_DIR:ARM --members RUN_DIR:ARM RUN_DIR:ARM ... --output DIR --name NAME

Every member's held-out probabilities are averaged with equal weights (fixed in advance, never tuned) and
the fused arm is scored against the base with the D-74 rule. All inputs must come from the same inner
splits (same folds, repeats and videos), which is checked row by row. No training; no test videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(spec: str):
    from core.failure_analysis import load_arm_predictions

    run_dir, arm = spec.rsplit(":", 1)
    return load_arm_predictions(Path(run_dir) / "probs.csv", arm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args(argv)

    from core import inner_cv_report as report

    base = _load(args.base)
    members = [_load(spec) for spec in args.members]
    for spec, member in zip(args.members, members):
        same = (np.array_equal(member.fold, base.fold) and np.array_equal(member.seed, base.seed) and np.array_equal(member.video, base.video)
                and np.array_equal(member.truth, base.truth) and member.classes == base.classes)
        if not same:
            raise SystemExit(f"{spec} does not hold the same (fold, repeat, video) rows as the base")
    fused = np.mean([m.probs for m in members], axis=0)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "probs.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "fold", "seed", "video", "true", *[f"p_{c}" for c in base.classes]])
        for name, probs in (("base", base.probs), (args.name, fused)):
            for i in range(len(base.video)):
                writer.writerow([name, int(base.fold[i]), int(base.seed[i]), base.video[i], base.classes[int(base.truth[i])], *[f"{p:.6f}" for p in probs[i]]])
    contrast = report.paired_contrasts(path, [(args.name, "base")], [args.name, "base"], args.draws)[0]
    (args.output / "summary.json").write_text(json.dumps({
        "base": args.base, "members": args.members, "weights": "equal, fixed in advance", "contrast": contrast,
        "rule": f"paired 95% interval excludes 0 and gain >= {report.MIN_GAIN}"}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
