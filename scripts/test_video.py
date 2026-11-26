#!/usr/bin/env python3
"""Run the evaluated D-53 classifier on your own videos and write a report.

  scripts/test_video.py clip.mp4 --label Fighting        one video, compared with your label
  scripts/test_video.py my_videos/                       a whole folder
  scripts/test_video.py my_videos/ --labels labels.csv   CSV columns: video,label
  scripts/test_video.py by_class/ --labels-from-folders  by_class/Fighting/a.mp4 is labelled Fighting
  scripts/test_video.py --self-check                     check this code path against recorded scores

On the DGX use scripts/dgx/test_video.sh, which runs this inside a pod (D-39).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.video_report import apply_label, canonical_label, console_line, discover_videos, read_labels_csv, summarize, write_outputs  # noqa: E402

SHARED_HEAD_NODE = "<head-node>"


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*", type=Path, help="video files or folders")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--label", help="the true class, for one video")
    source.add_argument("--labels", type=Path, help="CSV with columns video,label")
    source.add_argument("--labels-from-folders", action="store_true", help="use each video's parent folder name as its class")
    ap.add_argument("--out", type=Path, help="output folder (default results/video-tests/<UTC time>)")
    ap.add_argument("--device", default="cuda:0", help="cuda:0 (falls back to CPU when there is no GPU) or cpu")
    ap.add_argument("--self-check", action="store_true", help="score one fold's validation videos and compare with the recorded values")
    ap.add_argument("--fold", type=int, default=1, choices=(1, 2, 3, 4), help="fold for --self-check")
    return ap.parse_args(argv)


def _provenance() -> dict[str, str]:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    export = json.loads((REPO_ROOT / "results/demo/export.json").read_text())
    return {"git_revision": revision, "heads_sha256": export["heads_sha256"]}


def _given_label(path: Path, args: argparse.Namespace, table: dict[str, str], classes) -> str | None:
    if args.label:
        return canonical_label(args.label, classes)
    if args.labels_from_folders:
        return canonical_label(path.parent.name, classes)
    return table.get(path.name)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if socket.gethostname().startswith(SHARED_HEAD_NODE):
        raise SystemExit("this is the shared head node (D-39); run it in a pod: scripts/dgx/test_video.sh " + " ".join(sys.argv[1:]))
    if args.self_check == bool(args.videos):
        raise SystemExit("give video files or folders, or --self-check")
    videos = [] if args.self_check else discover_videos(args.videos)
    if args.label and len(videos) != 1:
        raise SystemExit("--label is for one video; use --labels or --labels-from-folders for several")

    from core.demo_inference import DemoModel

    model = DemoModel(REPO_ROOT, device=args.device)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or REPO_ROOT / "results/video-tests" / stamp
    if args.self_check:
        from core.video_selfcheck import run_self_check

        report = {**_provenance(), "device": str(model.device), **run_self_check(model, REPO_ROOT, args.fold)}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "self_check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=1))
        print("SELF-CHECK", "PASSED" if report["passed"] else "FAILED", "->", out_dir / "self_check.json")
        return 0 if report["passed"] else 1

    table = read_labels_csv(args.labels, model.classes) if args.labels else {}
    results = []
    for path in videos:
        given = _given_label(path, args, table, model.classes)
        try:
            result = apply_label(model.predict(path, path.name), given)
        except ValueError as error:  # unreadable video: report it and carry on with the rest
            result = {"filename": path.name, "error": str(error)}
        results.append(result)
        print(console_line(result), flush=True)
    summary = summarize(results)
    meta = {"created_utc": stamp, **_provenance(), "backbone_sha256": model.weight_sha, "head_seeds": "0,1,2 (fixed)", "device": str(model.device)}
    paths = write_outputs(out_dir, results, summary, meta, model.classes)
    if summary["scored"]:
        print(f"Scored {summary['scored']}: accuracy {summary['accuracy']:.1%}" + (" (small sample: an example, not a measurement)" if summary["small_sample"] else ""))
    print("Report:", paths["report"])
    return 0 if summary["errors"] < summary["videos"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
