#!/usr/bin/env python3
"""Clip-level event/background probe on cached DINOv2 features (CPU only).

Thin CLI over `core.p1c_event_clips`: tiles annotated `--label` videos into
fixed-length clips (`build_event_clips`), builds one or more registered clip
representations (`clip_features`), and scores each with a leave-one-video-out
logistic regression (`leave_one_video_out_clips`). All three do the real
work; this script only parses arguments, loads the cache, and prints/writes
the result.

Intervals are evaluation labels only -- see `core.p1c_event_clips`'s module
docstring for why that is not circular evaluation (docs/PROJECT_RULES.md rule 2).

    OMP_NUM_THREADS=4 taskset -c 0-3 \\
    ./.venv/bin/python scripts/run_p1c_event_clip_probe.py \\
        --label Shooting --representation single_frame dense_mean dense_mean_delta
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals  # noqa: E402
from core.p1c_event_clips import (  # noqa: E402
    CLIP_REPRESENTATIONS,
    build_event_clips,
    clip_features,
    leave_one_video_out_clips,
)
from scripts.diagnose_p1c_event_separability import ANNOTATION  # noqa: E402


def load_deduplicated_cache(cache_roots: list[Path]) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Concatenate `rows.csv`/`features.float16.npy` across `cache_roots`,
    keeping each video's rows only from the first root it appears in.

    A video assigned to more than one fold cache (should not normally
    happen, but is not asserted against upstream) would otherwise be double
    counted; first occurrence wins, matching the diagnostic script's
    `collect_frames`. Per-video row contiguity is preserved: a kept video's
    rows all come from the same root, in that root's original order.
    """

    rows: list[dict[str, Any]] = []
    feature_blocks: list[np.ndarray] = []
    seen_videos: set[str] = set()

    for root in cache_roots:
        rows_path, features_path = root / "rows.csv", root / "features.float16.npy"
        if not rows_path.exists():
            continue
        with open(rows_path, newline="") as handle:
            root_rows = list(csv.DictReader(handle))
        features = np.load(features_path, mmap_mode="r")

        keep = [i for i, row in enumerate(root_rows) if row["source_video_id"] not in seen_videos]
        seen_videos.update(root_rows[i]["source_video_id"] for i in keep)
        if not keep:
            continue
        rows.extend(root_rows[i] for i in keep)
        feature_blocks.append(np.asarray(features[keep]))

    if not feature_blocks:
        return rows, np.zeros((0, 0), dtype=np.float32)
    return rows, np.concatenate(feature_blocks, axis=0)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-glob", default="results/p1c/p1c-dinov2-feature-cache-ar-fold*/train")
    parser.add_argument("--label", default="Shooting")
    parser.add_argument(
        "--representation", choices=CLIP_REPRESENTATIONS, nargs="+", action="extend",
        help="repeatable; defaults to all three registered representations",
    )
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--positive-overlap-threshold", type=float, default=0.5)
    parser.add_argument("--negative-margin", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/p1c/event-clip-probe.json"))
    args = parser.parse_args(argv)
    if not args.representation:
        args.representation = list(CLIP_REPRESENTATIONS)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    intervals = load_intervals(ANNOTATION)
    roots = sorted(Path().glob(args.cache_glob))
    if not roots:
        raise SystemExit(f"no cache roots matched {args.cache_glob!r}")
    rows, features = load_deduplicated_cache(roots)

    build = build_event_clips(
        rows, intervals, label=args.label, clip_length=args.clip_length, frame_stride=args.frame_stride,
        positive_overlap_threshold=args.positive_overlap_threshold, negative_margin=args.negative_margin,
        seed=args.seed,
    )
    report = build.report()
    print(f"label={args.label}  cache videos={len(rows) and len({r['source_video_id'] for r in rows})}")
    print(f"clips: positive={report['positive']}  negative={report['negative']}  ambiguous={report['ambiguous']}")
    for video_id, counts in sorted(report["by_video"].items()):
        print(f"  {video_id:<28} positive={counts['positive']:<4} negative={counts['negative']:<4} "
              f"ambiguous={counts['ambiguous']}")

    usable = build.usable
    results_by_representation: dict[str, Any] = {}
    for representation in args.representation:
        x = clip_features(features, rows, usable, representation)
        result = leave_one_video_out_clips(usable, x, seed=args.seed)
        results_by_representation[representation] = result
        pooled, video_level = result["pooled"], result["video_level"]
        print(
            f"\n{representation} (D={x.shape[1]}): pooled f1={pooled['f1']:.3f} "
            f"precision={pooled['precision']:.3f} recall={pooled['recall']:.3f} "
            f"roc_auc={pooled['roc_auc']}  pr_auc={pooled['pr_auc']}  "
            f"video-level recall={video_level['recall']} ({video_level['n_predicted_positive']}/{video_level['n_videos']})"
        )

    payload = {
        "label": args.label,
        "clip_length": args.clip_length,
        "frame_stride": args.frame_stride,
        "positive_overlap_threshold": args.positive_overlap_threshold,
        "negative_margin": args.negative_margin,
        "seed": args.seed,
        "cache_glob": args.cache_glob,
        "cache_roots": [str(root) for root in roots],
        "annotation_source": str(ANNOTATION),
        "clip_report": report,
        "results_by_representation": results_by_representation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
