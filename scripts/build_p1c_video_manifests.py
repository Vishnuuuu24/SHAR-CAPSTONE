#!/usr/bin/env python3
"""Build train/validation/test manifests from the D-34/D-35 video blob.

Regenerates the P1B manifest set against video-derived frames instead of the
retired Kaggle mirror. The real work lives in `data.video_manifests` and
`data.official_splits`; this file parses arguments and orchestrates.

**The protocol is a required argument, deliberately.** The official download
ships two incompatible protocols (see the two-protocol row in
`docs/EVIDENCE_REGISTER.md`):

  --protocol action_recognition --fold N
      14 balanced classes over 700 videos, four official disjoint folds. This
      is SHAR's RESULT 2 task verbatim, and the citable option.

  --protocol anomaly_detection
      1,610 train + 290 test = 1,900 videos, the temporal/binary task's corpus.
      Using it for 14-class classification means a self-authored split and no
      official Normal class.

There is no default. Choosing between them is a research decision with real
consequences for what the results can claim, and this tool refuses to make it
silently.

Every emitted row is `label_scope=video_inherited` (docs/PROJECT_RULES.md rule 1), and the
train/validation carve is by whole source video with an explicit leakage check
before anything is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from data.official_splits import ACTION_RECOGNITION, ANOMALY_DETECTION, PROTOCOLS  # noqa: E402
from data.video_blob import VideoBlobClosure  # noqa: E402
from data.video_manifests import (  # noqa: E402
    assert_consistent_label_vocabulary,
    assert_no_video_leakage,
    carve_validation,
    load_protocol_split,
    manifest_rows,
    read_videos_csv,
    write_manifest,
)

BUILDER_CODE_VERSION = "p1c-video-manifests-v1"
DEFAULT_UCF_ROOT = REPO_ROOT / "Datasets" / "UCF Crime Official" / "raw" / "UCF_Crimes"
DEFAULT_CLOSURE = REPO_ROOT / "results" / "p1c" / "p1c-video-frames" / "extract_closure.json"


def frame_indices_by_video(
    closure: VideoBlobClosure, videos: dict
) -> dict[tuple[str, str], list[int]]:
    """Map each ``(directory, stem)`` to its kept `source_frame_index` values.

    Keyed by the pair for the same reason `read_videos_csv` is: stems repeat
    across directories in this corpus.
    """
    index = np.load(closure.frame_index_path)
    by_id: dict[int, list[int]] = {}
    for video_id, source_frame_index in zip(
        index["video_id"].tolist(), index["source_frame_index"].tolist()
    ):
        by_id.setdefault(video_id, []).append(source_frame_index)
    return {key: sorted(by_id.get(row.video_id, [])) for key, row in videos.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--fold", type=int, default=None, help="required for action_recognition (1-4)")
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--ucf-root", type=Path, default=DEFAULT_UCF_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def parser_error(message: str) -> None:
    """Exit 2 with a usage-style message, matching argparse's own convention."""
    print(f"{Path(sys.argv[0]).name}: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Validate the argument combination before touching the filesystem, so a
    # missing --fold reports the missing --fold rather than whichever file
    # happens to be opened first.
    if args.protocol == ACTION_RECOGNITION and args.fold is None:
        parser_error("--protocol action_recognition requires --fold (1-4)")
    if args.protocol == ANOMALY_DETECTION and args.fold is not None:
        parser_error("--protocol anomaly_detection has no folds; omit --fold")
    if not 0.0 < args.validation_fraction < 1.0:
        parser_error(f"--validation-fraction must be in (0, 1), got {args.validation_fraction}")

    closure = VideoBlobClosure.load(args.closure)
    videos = read_videos_csv(closure.videos_csv_path)
    indices = frame_indices_by_video(closure, videos)

    official = load_protocol_split(args.ucf_root, protocol=args.protocol, fold=args.fold)
    train_entries, validation_entries = carve_validation(
        official["train"], fraction=args.validation_fraction, seed=args.seed
    )
    splits = {
        "train": train_entries,
        "validation": validation_entries,
        "test": official["test"],
    }
    # Before writing anything: no source video may cross a split boundary.
    assert_no_video_leakage(splits)
    # And no split may carry a label the others do not: a model must not be
    # trained on a vocabulary its evaluation cannot score.
    assert_consistent_label_vocabulary(splits)

    if args.protocol == ACTION_RECOGNITION:
        label_source = f"official-action-recognition-split-fold{args.fold}"
        annotation_version = "ucf-crime-action-recognition-splits-2017"
    else:
        label_source = "official-anomaly-detection-split"
        annotation_version = "ucf-crime-anomaly-detection-splits-2018"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, object]] = {}
    for split_name, entries in splits.items():
        path = args.output_dir / f"{split_name}_manifest.csv"
        count, digest = write_manifest(
            path,
            manifest_rows(
                entries,
                videos=videos,
                frame_indices=indices,
                split=split_name,
                label_source=label_source,
                annotation_version=annotation_version,
            ),
        )
        written[split_name] = {
            "path": path.name,
            "row_count": count,
            "video_count": len(entries),
            "sha256": digest,
        }
        print(f"{split_name:11s} {len(entries):5d} videos  {count:9d} frames  {digest[:12]}")

    manifest_closure = {
        "kind": "P1B manifests derived from the D-34/D-35 video blob",
        "builder_code_version": BUILDER_CODE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": args.protocol,
        "fold": args.fold,
        "label_scope": "video_inherited",
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "blob_closure_path": str(args.closure.resolve()),
        "blob_sha256": closure.blob_sha256,
        "stride": closure.stride,
        "splits": written,
        "grouping": "train/validation carved by whole source video; leakage checked before write",
    }
    closure_path = args.output_dir / "manifest_closure.json"
    closure_path.write_text(json.dumps(manifest_closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"closed: {closure_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
