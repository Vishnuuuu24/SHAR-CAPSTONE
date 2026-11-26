#!/usr/bin/env python3
"""Per-frame E3 features and a causal window timeline for a few showcase videos (GPU pod only).

For each named UCF video: re-extract the four frozen E3 encoders frame by frame from the registered blob, check
the whole-video average against the saved E3 replay (vectors and probabilities), then score trailing windows with
the same held-out fold members the replay used. Writes `features/<key>.npz` (git-ignored; lets windows be
re-scored on CPU) and `timelines/<key>.json`. See core/showcase_timeline.py for what a window score means.

Usage (inside a pod):
  .venv/bin/python scripts/build_showcase_timeline.py --out results/demo/showcase-20261124 Explosion021_x264 ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
E3 = REPO / "results/demo/e3-comparison-20261116"

import core.showcase_timeline as st  # noqa: E402
from core.e3_inference import E3Encoders, check_encoder_sources, e3_support, extract_per_frame, load_members  # noqa: E402


def git_revision() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=32, help="window length in selected (stride-5) frames")
    ap.add_argument("--step", type=int, default=6, help="window hop in selected frames")
    ap.add_argument("--blob-closure", type=Path, default=REPO / "results/p1c/p1c-video-frames/extract_closure.json")
    args = ap.parse_args(argv)

    from data.video_blob import VideoBlobReader

    support = e3_support(REPO)
    CONFIG, CONFIG_SHA, predict = support.CONFIG, support.CONFIG_SHA, support.predict
    check_encoder_sources(REPO, Path(CONFIG["code_root"]))
    classes = json.loads((E3 / "class_names.json").read_text())
    rows = {r["key"]: r for r in CONFIG["ucf"]}
    models = load_members(REPO)
    t0 = time.monotonic()
    encoders = E3Encoders(REPO)
    # verify_digest=False: hashing the 84 GiB blob is skipped (as E3's calibration did); the per-video parity
    # check against the saved replay vectors below is the content check instead.
    reader = VideoBlobReader(args.blob_closure, verify_digest=False)
    print(json.dumps({"ready": True, "startup_s": round(time.monotonic() - t0, 1)}), flush=True)
    (args.out / "features").mkdir(parents=True, exist_ok=True)
    (args.out / "timelines").mkdir(parents=True, exist_ok=True)
    summary = []
    for key in args.keys:
        row = rows[key]
        answer = json.loads((E3 / "answers/ucf" / f"{key}.json").read_text())
        folds = answer["selected_folds"]
        t = time.monotonic()
        first = row["first_blob_row"]
        rows_ = range(first, first + row["kept_frames"])
        per_frame, per_clip = extract_per_frame(encoders, (reader.frame(i) for i in rows_))
        src_idx = np.array([reader.provenance(i)["source_frame_index"] for i in rows_])
        np.savez(args.out / "features" / f"{key}.npz", source_frame_index=src_idx, VJ=per_clip,
                 **{k: v.astype(np.float16) for k, v in per_frame.items()})

        full = st.pooled(per_frame, per_clip)
        probs = predict(models, full, folds, len(classes))
        with np.load(E3 / "vectors" / f"{answer['source_sha256']}.npz") as d:
            saved = {k: d[k] for k in ("B", "L", "S", "VJ")}
        parity = st.full_video_parity(full, saved, probs, np.array([answer["probs"][c] for c in classes]))

        fps = float(row["fps"])
        windows = []
        for s, e, c in st.trailing_windows(len(src_idx), args.width, args.step):
            p = predict(models, st.pooled(per_frame, per_clip, slice(s, e), c), folds, len(classes))
            top = np.argsort(-p)[:3]
            windows.append({"start_s": float(src_idx[s] / fps), "end_s": float((src_idx[e - 1] + 1) / fps),
                            "top3": [{"label": classes[int(k)], "prob": float(p[k])} for k in top],
                            "probs": [round(float(x), 5) for x in p]})
        out = {
            "key": key, "label": row["label"], "source_sha256": answer["source_sha256"], "fps": fps,
            "duration_s": float((src_idx[-1] + 1) / fps), "selected_frames": int(len(src_idx)),
            "e3_config_sha256": CONFIG_SHA, "selected_folds": folds, "exposure": answer["exposure"],
            "members_used": 9 * len(folds), "classes": classes,
            "video": {"prediction": classes[int(np.argmax(probs))], "confidence": float(np.max(probs)),
                      "top3": [{"label": classes[int(k)], "prob": float(probs[k])} for k in np.argsort(-probs)[:3]]},
            "saved_e3_answer": {"prediction": answer["prediction"], "confidence": answer["confidence"]},
            "parity": parity, "window": {"width_frames": args.width, "step_frames": args.step, "causal": True},
            "windows": windows, "git_revision": git_revision(), "seconds": round(time.monotonic() - t, 1),
        }
        (args.out / "timelines" / f"{key}.json").write_text(json.dumps(out, indent=1) + "\n")
        brief = {"key": key, "label": row["label"], "pred": out["video"]["prediction"],
                 "conf": round(out["video"]["confidence"], 3), "parity": parity, "windows": len(windows),
                 "seconds": out["seconds"]}
        summary.append(brief)
        print(json.dumps(brief), flush=True)
    (args.out / "build_summary.json").write_text(json.dumps(
        {"git_revision": git_revision(), "args": {k: str(v) for k, v in vars(args).items()}, "videos": summary}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
