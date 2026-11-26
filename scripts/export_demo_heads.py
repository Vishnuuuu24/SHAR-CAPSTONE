#!/usr/bin/env python3
"""Recreate and save the 12 evaluated D-53 heads (4 folds x 3 seeds) for the browser demo, plus video membership.

Same recipe as `scripts/run_headline_test.py` (train on the fold's training videos, best epoch on its validation
videos); the test view is not read. Each head's validation macro-F1 is checked against the value recorded in
`results/p1c/headline-14class-test/test.json` before anything is saved. Writes `.cache/demo/d53_heads.pt` (git-ignored,
~340 MB) and `results/demo/{membership.json, export.json}`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.demo_inference import HEADS_PATH, MEMBERSHIP_PATH  # noqa: E402

TOLERANCE = 0.02  # D-63 saw 2 of 12 cells differ (by +0.0125 / -0.0006) from multithreaded float order


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    import torch

    from core.p1c_oracle_interval import predict_video_probs
    from core.specialist import GroupMap, remap_sequences
    from scripts.run_p1c_xd_extra_probe import FRAMES, train_head
    from scripts.run_subset7_search import load_fold_features, score, sequences

    torch.set_num_threads(args.threads)
    recorded = {(r["fold"], r["seed"]): r for r in json.loads((REPO_ROOT / "results/p1c/headline-14class-test/test.json").read_text())["runs"]}
    heads, checks, membership, classes = [], [], {}, None
    for fold in (1, 2, 3, 4):
        fd = load_fold_features(fold, ("train", "validation"))
        classes = fd["class_names"]
        gm = GroupMap(classes, classes)
        for split in ("train", "validation"):
            for r in fd[split][1]:
                rec = membership.setdefault(r["source_video_id"], {"label": r["label"], "seen_by_folds": [], "role": {}})
                if fold not in rec["seen_by_folds"]:
                    rec["seen_by_folds"].append(fold)
                rec["role"][str(fold)] = split
        for seed in (0, 1, 2):
            tr, va = (remap_sequences(sequences(fd, s, seed), gm) for s in ("train", "validation"))
            result, head = train_head(seed, tr, va, classes, None)
            preds = {k: predict_video_probs(head, seq, FRAMES) for k, (seq, _r) in va.items()}
            f1, _t, _p = score(gm, va, preds)
            want = recorded[(fold, seed)]["val_macro_f1"]
            checks.append({"fold": fold, "seed": seed, "val_macro_f1": f1, "recorded_val_macro_f1": want,
                           "abs_diff": abs(f1 - want), "best_epoch": int(result.best_epoch),
                           "recorded_best_epoch": recorded[(fold, seed)]["best_epoch"]})
            print(json.dumps(checks[-1]), flush=True)
            heads.append({"fold": fold, "seed": seed, "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}})
    bad = [c for c in checks if c["abs_diff"] > TOLERANCE]
    if bad:
        raise SystemExit(f"{len(bad)} head(s) do not reproduce the recorded validation score: {bad}")
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    out = REPO_ROOT / HEADS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"classes": classes, "heads": heads, "git_revision": rev, "pipeline": "D-53 (D-57 recipe)"}, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (REPO_ROOT / MEMBERSHIP_PATH).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / MEMBERSHIP_PATH).write_text(json.dumps({"note": "seen_by_folds = folds whose heads trained on or chose their epoch with this video",
                                                         "videos": membership}, indent=0) + "\n")
    (REPO_ROOT / "results/demo/export.json").write_text(json.dumps({"git_revision": rev, "heads_sha256": sha, "heads_path": HEADS_PATH,
                                                                    "tolerance": TOLERANCE, "checks": checks,
                                                                    "protocol_videos": len(membership)}, indent=1) + "\n")
    print(json.dumps({"saved": str(out), "sha256": sha, "max_abs_diff": max(c["abs_diff"] for c in checks),
                      "exact": sum(c["abs_diff"] == 0 for c in checks)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
