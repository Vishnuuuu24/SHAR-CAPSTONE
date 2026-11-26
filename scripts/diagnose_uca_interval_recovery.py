#!/usr/bin/env python3
"""Diagnostic: how well do UCA caption segments recover the official UCF-Crime event interval (121 videos)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals  # noqa: E402
from core.p1c_oracle_interval import (  # noqa: E402
    CLASS_LEXICON, fps_from_cache, frames_to_seconds, interval_iou, keyword_hits, spans_overlap, summarise,
    total_length, uca_segment_spans, video_class,
)

ANNOTATION = "Datasets/UCF Crime Official/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt"
CACHE_TEMPLATE = "results/p1c/p1c-dinov2-feature-cache-ar-fold{fold}/{split}/rows.csv"


def _max_cached_frames(folds: list[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for fold in folds:
        for split in ("train", "validation"):
            with Path(CACHE_TEMPLATE.format(fold=fold, split=split)).open(newline="") as handle:
                for row in csv.DictReader(handle):
                    out[row["source_video_id"]] = max(out.get(row["source_video_id"], -1), int(row["source_frame_index"]))
    return out


def _hull(spans):
    return [(min(s for s, _ in spans), max(e for _, e in spans))] if spans else []


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uca-dir", type=Path, default=Path("Datasets/UCA"))
    parser.add_argument("--annotation", type=Path, default=Path(ANNOTATION))
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=Path("results/p1c/p1c-uca-interval-recovery"))
    args = parser.parse_args(argv)

    uca: dict = {}
    for split in ("Train", "Val", "Test"):
        uca.update(json.loads((args.uca_dir / f"UCFCrime_{split}.json").read_text()))
    uca = {name.rsplit(".", 1)[0]: v for name, v in uca.items()}
    max_frame = _max_cached_frames(args.folds)
    intervals = load_intervals(args.annotation)

    coverage = []
    for video, last in max_frame.items():
        if video in uca:
            segs = [seg for seg, _ in uca_segment_spans(uca[video])]
            coverage.append(total_length(segs) / uca[video]["duration"])

    rows = []
    for video in sorted(max_frame):
        if video not in intervals or video not in uca:
            continue
        entry = uca[video]
        fps = fps_from_cache(max_frame[video], entry["duration"])
        official = frames_to_seconds(intervals[video], fps)
        segs = uca_segment_spans(entry)
        overlapping = [s for s, _ in segs if spans_overlap(s, official)]
        cls = video_class(video)
        hits = [s for s, sentence in segs if keyword_hits(sentence, cls)] if cls in CLASS_LEXICON else []
        rows.append({
            "video": video, "class": cls, "duration_s": entry["duration"], "fps": fps,
            "official_fraction": total_length(official) / entry["duration"],
            "uca_segments": len(segs), "segments_overlapping_official": len(overlapping),
            "iou_all_overlapping": interval_iou(overlapping, official),
            "iou_all_overlapping_hull": interval_iou(_hull(overlapping), _hull(official)),
            "iou_keyword_hull": interval_iou(_hull(hits), _hull(official)),
            "keyword_hit_segments": len(hits), "iou_keyword": interval_iou(hits, official) if hits else 0.0,
        })

    hit_rows = [r for r in rows if r["keyword_hit_segments"] > 0]
    summary = {
        "n_videos_with_official_and_uca": len(rows),
        "n_cached_videos": len(max_frame), "n_cached_videos_matched_to_uca": len(coverage),
        "uca_caption_coverage_of_duration": summarise(coverage),
        "official_event_fraction_of_video": summarise([r["official_fraction"] for r in rows]),
        "uca_segments_per_video": summarise([r["uca_segments"] for r in rows]),
        "segments_overlapping_official": summarise([r["segments_overlapping_official"] for r in rows]),
        "iou_all_overlapping_vs_official": summarise([r["iou_all_overlapping"] for r in rows]),
        "iou_keyword_vs_official_no_hit_counted_as_0": summarise([r["iou_keyword"] for r in rows]),
        "iou_keyword_vs_official_hit_videos_only": summarise([r["iou_keyword"] for r in hit_rows]),
        "variant_hull_iou_all_overlapping": summarise([r["iou_all_overlapping_hull"] for r in rows]),
        "variant_hull_iou_keyword_no_hit_as_0": summarise([r["iou_keyword_hull"] for r in rows]),
        "videos_without_keyword_hit": len(rows) - len(hit_rows),
        "fraction_keyword_iou_ge_0.3_all_videos": sum(r["iou_keyword"] >= 0.3 for r in rows) / len(rows),
        "fps_rule": "(max_cached_source_frame_index + 5) / uca_duration",
        "lexicon": CLASS_LEXICON,
        "note": "diagnostic only; official intervals are the reference, UCA captions are not ground truth",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output / "per_video.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "lexicon"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
