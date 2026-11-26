#!/usr/bin/env python3
"""Zero-shot Qwen2.5-VL-7B probe on the AR validation videos (validation only; test untouched).

Each video: 8 temporally-spread native-resolution frames (core.p1c_video_sampling, deterministic), a 14-option
multiple-choice prompt (core.vlm_probe), one forward pass; class probabilities = softmax over the 14 answer-letter
logits, averaged over a forward and a reversed option order (position-bias control). Scored alone and fused with the
registered DINOv2 step_permuted probabilities at a fixed weight (no tuning). Resumable per (video, order).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.confusion_groups import load_runs  # noqa: E402
from core.validation_frames import select_validation_frames  # noqa: E402
from core.vlm_probe import LETTERS, ORDERS, build_question, evaluate, letter_probs_to_classes, option_order  # noqa: E402

CLOSURE = "results/p1c/p1c-video-frames/extract_closure.json"
BASELINE = "results/p1c/p1c-late-fusion/dinov2_probs.csv"
FRAMES = 8


def select_frames(folds, classes) -> dict[str, dict]:
    return select_validation_frames(REPO_ROOT, folds, classes, FRAMES)


def load_model(model_dir: Path, quant: str):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    q = None
    if quant == "8bit":
        q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"])
    elif quant == "4bit":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
                               llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda:0",
                                                               quantization_config=q)
    model.eval()
    return model, AutoProcessor.from_pretrained(model_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--model-dir", type=Path, default=Path(".cache/weights/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--quant", choices=("8bit", "4bit", "none"), default="8bit")
    ap.add_argument("--weight", type=float, default=0.5, help="fusion weight on the VLM, fixed before the run")
    ap.add_argument("--limit", type=int, default=None, help="first N videos only (smoke)")
    ap.add_argument("--output", type=Path, default=Path("results/vlm/qwen25vl7b-zeroshot"))
    args = ap.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image

    classes, runs = load_runs(REPO_ROOT / BASELINE)
    videos = select_frames(args.folds, classes)
    todo = sorted(videos)[: args.limit]
    per_video = args.output / "per_video.csv"
    done = set()
    if per_video.exists():
        done = {(r["video"], r["order"]) for r in csv.DictReader(per_video.open())}
    started = time.time()
    model, processor = load_model(args.model_dir, args.quant)
    letter_ids = [processor.tokenizer.encode(L, add_special_tokens=False) for L in LETTERS]
    if any(len(t) != 1 for t in letter_ids):
        raise RuntimeError(f"answer letters are not single tokens: {letter_ids}")
    letter_ids = [t[0] for t in letter_ids]
    new_file = not per_video.exists()
    with per_video.open("a", newline="") as h:
        w = csv.writer(h)
        if new_file:
            w.writerow(["video", "label", "fold_source", "order", "frames", "letter_mass", "top_token", "seconds"] + [f"p_{c}" for c in classes])
        for vid in todo:
            v = videos[vid]
            images = None
            for order in ORDERS:
                if (vid, order) in done:
                    continue
                if images is None:
                    images = [Image.open(io.BytesIO(v["dataset"].raw_jpeg(i))).convert("RGB") for i in v["rows"]]
                opts = option_order(classes, order)
                content = [{"type": "image"} for _ in images] + [{"type": "text", "text": build_question(opts, len(images))}]
                text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = model(**inputs).logits[0, -1].float()
                full = torch.softmax(logits, -1)
                probs = letter_probs_to_classes(logits[letter_ids].cpu().numpy(), opts, classes)
                w.writerow([vid, v["label"], v["fold"], order, " ".join(map(str, v["frames"])), f"{float(full[letter_ids].sum()):.4f}",
                            processor.tokenizer.decode([int(logits.argmax())]), f"{time.perf_counter() - t0:.2f}"]
                           + [f"{x:.6f}" for x in probs])
                h.flush()
            print(json.dumps({"video": vid, "label": v["label"], "elapsed_s": round(time.time() - started)}), flush=True)

    rows = list(csv.DictReader(per_video.open()))
    acc = defaultdict(list)
    for r in rows:
        acc[r["video"]].append(np.array([float(r[f"p_{c}"]) for c in classes]))
    vlm = {k: np.mean(v, 0) for k, v in acc.items() if len(v) == len(ORDERS)}
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    import transformers, bitsandbytes  # noqa: E401
    config = {"git_revision": rev, "model": "Qwen/Qwen2.5-VL-7B-Instruct", "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
              "weights_verified": "results/vlm/qwen25vl7b-sha256-verify.json", "quant": args.quant, "fusion_weight": args.weight,
              "frames_per_video": FRAMES, "frame_rule": "sample_rows_per_video(seed=None) over the validation manifest rows, native 320x240 JPEG",
              "orders": ORDERS, "prompt_example": build_question(option_order(classes, "forward"), FRAMES),
              "frame_blob_sha256": json.loads((REPO_ROOT / CLOSURE).read_text())["blob_sha256"], "blob_digest_reverified": False,
              "baseline_probs_sha256": hashlib.sha256((REPO_ROOT / BASELINE).read_bytes()).hexdigest(),
              "transformers": transformers.__version__, "bitsandbytes": bitsandbytes.__version__, "videos_scored": len(vlm),
              "limit": args.limit, "runtime_s": round(time.time() - started), "test_split": "untouched"}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.limit is None:
        summary = evaluate(classes, runs, vlm, args.weight)
        letter_mass = float(np.mean([float(r["letter_mass"]) for r in rows]))
        (args.output / "summary.json").write_text(json.dumps({"mean_letter_mass": letter_mass, **summary}, indent=2, default=float) + "\n")
        for name, s in summary.items():
            print(name, json.dumps({k: s[k] for k in s if k in ("candidate_mean", "baseline_mean", "mean_delta", "per_fold_delta", "folds_meeting_gate", "passes_gate")}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
