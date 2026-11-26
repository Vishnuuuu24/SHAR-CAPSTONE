#!/usr/bin/env python3
"""A5: the final Shooting experiment -- partial fine-tuning with source-video
domain-adversarial training, one factor per arm.

    arm frozen    DINOv2 frozen; heads only              (matched baseline)
    arm finetune  + last 2 blocks unfrozen, no domain loss
    arm dann      + GRL source-video discriminator

**Single fixed split, lower statistical power than the LOVO experiments.**
`results/p1c/p1c-exp-A5-dann/split.json`: 16 annotated Shooting + 52
non-Shooting training videos (= 68 source-video DOMAINS), 5 annotated Shooting +
13 non-Shooting held-out videos. Video-disjoint, asserted at load. The domain is
SOURCE VIDEO ID -- the dataset has no camera ID (videos.csv carries none).

Training clips: 16 consecutive cached rows (2.67 s). Shooting training videos
are labelled event/background from the official intervals through
`core.p1c_event_clips.build_event_clips` (same 0.5 overlap / 16-row margin as
A2/A3); non-Shooting training videos are tiled the same way and every clip is
a negative. Held-out event labels are used ONLY at evaluation.

lambda: the DANN ramp `2/(1+e^{-10p})-1` (Ganin & Lempitsky 2015) scaled by
`lambda_max`, which is MEASURED on the first real training batch as
event_grad_norm / domain_grad_norm -- the smoke test showed a 68-way CE
gradient 33x the binary BCE gradient at lambda=1, so the paper's unscaled
ramp would let the reversed domain gradient swamp the event objective.

Learning rates: heads 3e-4 (the registered head recipe); unfrozen backbone
blocks 3e-5, the 10x-lower discriminative-LR convention this project's own
backlog item B-7 proposed. Fixed epochs, FINAL checkpoint evaluated -- no
model selection on held-out data.

Evaluation on held-out videos only. Localisation: event vs background clip ROC /
PR on the 5 Shooting videos, pooled and per video. Classification: max clip
score per video, Shooting vs not, ROC / PR; the F1 threshold is chosen on the
TRAINING videos' max-scores (best F1 there) and applied unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.p1c_background_mask import load_intervals
from core.p1c_domain_adversarial import (DomainAdversarialModel, dann_lambda, domain_loss, event_loss,
                                         save_checkpoint, smoke_test)
from core.p1c_event_clips import _window_starts, build_event_clips
from scripts.build_p1c_feature_cache import _load_backbone
from scripts.diagnose_p1c_event_separability import ANNOTATION
from scripts.run_p1c_frame_baseline import _load_config, _make_datasets

CONFIGS = [f"configs/experiments/p1c/P1C-VIDEO-DINOV2-B2-AR-FOLD{f}.yaml" for f in (1, 2, 3, 4)]
CLIP = dict(clip_length=16, frame_stride=1, positive_overlap_threshold=0.5, negative_margin=16, seed=0)


def _locate_videos(wanted: set[str], closure: Path | None):
    """{video: (fold dataset, [dataset indices ascending by frame], [source frames])}."""
    found, datasets = {}, {}
    for config in CONFIGS:
        _c, proto, _r = _load_config(Path(config))
        if closure is not None:
            proto["data"]["video_frames"]["extract_closure_path"] = str(closure)
        ds = _make_datasets(proto, seed=0)[0]
        vids, ref, fr = ds._videos, ds._video_ref, ds._frame_index   # noqa: SLF001 metadata only
        for i in range(len(ds)):
            v = vids[int(ref[i])]["source_video_id"]
            if v in wanted and v not in found:
                found.setdefault(v, [config, [], []])
            if v in found and found[v][0] == config:
                found[v][1].append(i); found[v][2].append(int(fr[i]))
        datasets[config] = ds
        if all(v in found for v in wanted):
            break
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"videos not found in any train cache: {sorted(missing)}")
    return {v: (datasets[c], idx, frs) for v, (c, idx, frs) in found.items()}


def _clips_for(video, label, ds_idx, frames, intervals):
    """[(dataset indices x16, event_y)] for one video, tiled like A2/A3."""
    rows = [{"source_dataset": "d", "source_video_id": video, "source_frame_index": f, "target": "0", "label": label}
            for f in frames]
    if label == "Shooting" and video in intervals:
        b = build_event_clips(rows, intervals, label="Shooting", **CLIP)
        return [([ds_idx[i] for i in c.row_indices], 1 if c.is_positive else 0) for c in b.usable]
    starts, span = _window_starts(len(rows), CLIP["clip_length"], CLIP["frame_stride"])
    return [([ds_idx[s + k * CLIP["frame_stride"]] for k in range(CLIP["clip_length"])], 0) for s in starts]


class ClipDataset(Dataset):
    def __init__(self, items):   # items: (ds, [idx x16], event_y, domain_y, video)
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        ds, idx, ey, dy, video = self.items[k]
        return torch.stack([ds[i]["image"] for i in idx]), torch.tensor(float(ey)), torch.tensor(int(dy)), video


def _collate(batch):
    x, ey, dy, v = zip(*batch)
    return torch.stack(x), torch.stack(ey), torch.stack(dy), list(v)


@torch.no_grad()
def _score(model, loader, device):
    model.eval(); out = []
    for x, ey, _dy, v in loader:
        logit = model(x.to(device, non_blocking=True))[0].float().cpu().numpy()
        out += [(vid, float(l), int(e)) for vid, l, e in zip(v, logit, ey.numpy())]
    return out


def _evaluate(scores, held_shoot, labels):
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    by_video = {}
    for v, s, e in scores: by_video.setdefault(v, []).append((s, e))
    # localisation on held-out Shooting videos
    loc_pooled_s, loc_pooled_y, per_video = [], [], {}
    for v in held_shoot:
        s = np.array([t[0] for t in by_video[v]]); y = np.array([t[1] for t in by_video[v]])
        loc_pooled_s.append(s); loc_pooled_y.append(y)
        if y.sum() and (1 - y).sum():
            per_video[v] = {"roc": float(roc_auc_score(y, s)), "pr": float(average_precision_score(y, s)),
                            "n_event": int(y.sum()), "n_bg": int((1 - y).sum())}
    S, Y = np.concatenate(loc_pooled_s), np.concatenate(loc_pooled_y)
    loc = {"pooled_roc": float(roc_auc_score(Y, S)), "pooled_pr": float(average_precision_score(Y, S)),
           "base_rate": float(Y.mean()), "per_video": per_video,
           "per_video_roc_mean": float(np.mean([p["roc"] for p in per_video.values()])),
           "per_video_roc_median": float(np.median([p["roc"] for p in per_video.values()]))}
    # classification: max clip score per video, Shooting vs not
    vids = sorted(by_video); mx = np.array([max(t[0] for t in by_video[v]) for v in vids])
    yv = np.array([1 if labels[v] == "Shooting" else 0 for v in vids])
    cls = {"pooled_roc": float(roc_auc_score(yv, mx)), "pooled_pr": float(average_precision_score(yv, mx)),
           "n_shooting": int(yv.sum()), "n_not": int((1 - yv).sum()), "max_scores": dict(zip(vids, mx.tolist()))}
    return loc, cls, yv, mx


def _threshold_from_train(train_scores, labels):
    """Best-F1 threshold on TRAINING videos' max clip scores; applied unchanged to held-out."""
    from sklearn.metrics import f1_score
    by_video = {}
    for v, s, _e in train_scores: by_video[v] = max(by_video.get(v, -1e9), s)
    vids = sorted(by_video); mx = np.array([by_video[v] for v in vids]); y = np.array([labels[v] == "Shooting" for v in vids], int)
    cands = np.unique(mx); best = max(cands, key=lambda t: f1_score(y, (mx >= t).astype(int), zero_division=0))
    return float(best)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["frozen", "finetune", "dann"], required=True)
    p.add_argument("--split", type=Path, default=Path("results/p1c/p1c-exp-A5-dann/split.json"))
    p.add_argument("--output", type=Path, default=Path("results/p1c/p1c-exp-A5-dann"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-clips", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--lr-heads", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extract-closure-path", type=Path, default=None)
    p.add_argument("--limit-clips", type=int, default=None, help="debug")
    args = p.parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    split = json.load(open(args.split)); labels = split["labels"]; domains = split["domains"]
    intervals = load_intervals(ANNOTATION)
    train_v = split["train_shooting"] + split["train_non_shooting"]
    held_v = split["held_out_shooting"] + split["held_out_non_shooting"]
    assert not set(train_v) & set(held_v), "train/held-out videos overlap"
    assert set(domains) == set(train_v), "domain labels must be exactly the training videos"
    assert not set(domains) & set(held_v), "a held-out video appears in the domain labels"

    located = _locate_videos(set(train_v + held_v), args.extract_closure_path)
    dom_index = {v: i for i, v in enumerate(domains)}
    train_items, held_items = [], []
    for v in train_v:
        ds, idx, frs = located[v]
        train_items += [(ds, ci, ey, dom_index[v], v) for ci, ey in _clips_for(v, labels[v], idx, frs, intervals)]
    for v in held_v:
        ds, idx, frs = located[v]
        held_items += [(ds, ci, ey, -1, v) for ci, ey in _clips_for(v, labels[v], idx, frs, intervals)]
    if args.limit_clips:   # debug: cap TRAIN clips; keep 2 clips per held-out video so evaluation still runs
        train_items = train_items[: args.limit_clips]
        keep, seen = [], {}
        for t in held_items:
            if seen.get(t[4], 0) < 2: keep.append(t); seen[t[4]] = seen.get(t[4], 0) + 1
        held_items = keep
    n_ev = sum(1 for t in train_items if t[2] == 1)
    print(f"[{args.arm}] train clips {len(train_items)} ({n_ev} event, {len(train_items)-n_ev} negative) over "
          f"{len(train_v)} videos / {len(domains)} domains | held-out clips {len(held_items)} over {len(held_v)} videos", flush=True)
    assert all(t[3] == -1 for t in held_items), "held-out clips must carry no domain label"

    _c, proto, _r = _load_config(Path(CONFIGS[0]))
    unfreeze = 0 if args.arm == "frozen" else 2
    model = DomainAdversarialModel(_load_backbone(proto["backbone"]), unfreeze_last_blocks=unfreeze,
                                   n_domains=len(domains), grl_lambda=0.0).to(device)
    train_loader = DataLoader(ClipDataset(train_items), batch_size=args.batch_clips, shuffle=True, num_workers=args.num_workers,
                              collate_fn=_collate, pin_memory=device.type == "cuda", drop_last=False)
    held_loader = DataLoader(ClipDataset(held_items), batch_size=args.batch_clips, shuffle=False, num_workers=args.num_workers,
                             collate_fn=_collate, pin_memory=device.type == "cuda")
    train_eval_loader = DataLoader(ClipDataset(train_items), batch_size=args.batch_clips, shuffle=False,
                                   num_workers=args.num_workers, collate_fn=_collate)

    head_params = list(model.event_head.parameters()) + list(model.domain_head.parameters())
    bb_params = [p_ for n, p_ in model.backbone.named_parameters() if p_.requires_grad]
    groups = [{"params": head_params, "lr": args.lr_heads}]
    if bb_params: groups.append({"params": bb_params, "lr": args.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    # smoke on the first real batch; for dann this also fixes lambda_max
    x0, ey0, dy0, _ = next(iter(train_loader))
    model.grl.lambda_ = 1.0
    sm = smoke_test(model, x0.to(device), ey0.to(device), dy0.to(device))
    lambda_max = 0.0
    if args.arm == "dann":
        lambda_max = float(sm["event_grad_norm"] / max(sm["domain_grad_norm"], 1e-12))
    model.grl.lambda_ = 0.0
    print(f"[{args.arm}] smoke: " + ", ".join(f"{k}={v}" for k, v in sm.items() if isinstance(v, bool)), flush=True)
    print(f"[{args.arm}] first-batch grad norms: event {sm['event_grad_norm']:.4e}  domain {sm['domain_grad_norm']:.4e}  "
          f"-> lambda_max {lambda_max:.4f} ({'measured ratio' if args.arm=='dann' else 'domain loss off'})", flush=True)

    log = []; steps_total = args.epochs * len(train_loader); step = 0; t0 = time.time()
    for epoch in range(args.epochs):
        model.train(); el = dl = 0.0; n = 0
        for x, ey, dy, _v in train_loader:
            progress = step / max(steps_total - 1, 1)
            model.grl.lambda_ = lambda_max * dann_lambda(progress) if args.arm == "dann" else 0.0
            x, ey, dy = x.to(device, non_blocking=True), ey.to(device), dy.to(device)
            logit, dlogits, _rep = model(x)
            le = event_loss(logit, ey)
            ld = domain_loss(dlogits, dy) if args.arm == "dann" else torch.zeros((), device=device)
            loss = le + ld
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            el += float(le) * len(ey); dl += float(ld) * len(ey); n += len(ey); step += 1
        # per-epoch gradient norms on one batch, for the lambda log
        g = smoke_test(model, x0.to(device), ey0.to(device), dy0.to(device))
        rec = {"epoch": epoch, "event_loss": el / n, "domain_loss": dl / n, "lambda": model.grl.lambda_,
               "event_grad_norm": g["event_grad_norm"], "domain_grad_norm": g["domain_grad_norm"],
               "total_grad_norm": g["total_grad_norm"], "elapsed_s": time.time() - t0}
        log.append(rec)
        print(f"[{args.arm}] epoch {epoch} event_loss {rec['event_loss']:.4f} domain_loss {rec['domain_loss']:.4f} "
              f"lambda {rec['lambda']:.4f} grads e/d/t {rec['event_grad_norm']:.3e}/{rec['domain_grad_norm']:.3e}/{rec['total_grad_norm']:.3e} "
              f"({rec['elapsed_s']/60:.1f} min)", flush=True)

    out = args.output / args.arm; out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, str(out / "final.ckpt"))
    held_scores = _score(model, held_loader, device)
    train_scores = _score(model, train_eval_loader, device)
    loc, cls, yv, mx = _evaluate(held_scores, split["held_out_shooting"], labels)
    thr = _threshold_from_train(train_scores, labels)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    pred = (mx >= thr).astype(int)
    cls.update({"threshold_from_train": thr, "f1": float(f1_score(yv, pred, zero_division=0)),
                "precision": float(precision_score(yv, pred, zero_division=0)),
                "recall": float(recall_score(yv, pred, zero_division=0)), "accuracy": float(accuracy_score(yv, pred))})
    result = {"arm": args.arm, "unfreeze_last_blocks": unfreeze, "n_domains": len(domains), "lambda_max": lambda_max,
              "train_clips": len(train_items), "train_event_clips": n_ev, "held_out_clips": len(held_items),
              "train_videos": len(train_v), "held_out_videos": len(held_v), "epochs": args.epochs,
              "lr_heads": args.lr_heads, "lr_backbone": args.lr_backbone, "seed": args.seed,
              "smoke": {k: v for k, v in sm.items() if not k.endswith("_evidence")}, "log": log,
              "localisation": loc, "classification": cls, "runtime_s": time.time() - t0,
              "held_out_clip_scores": held_scores}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"[{args.arm}] LOCALISATION pooled ROC {loc['pooled_roc']:.3f} PR {loc['pooled_pr']:.3f} per-video ROC mean {loc['per_video_roc_mean']:.3f}")
    print(f"[{args.arm}] CLASSIFICATION pooled ROC {cls['pooled_roc']:.3f} PR {cls['pooled_pr']:.3f} F1 {cls['f1']:.3f} "
          f"prec {cls['precision']:.3f} rec {cls['recall']:.3f} acc {cls['accuracy']:.3f} (thr {thr:.3f} from train)")
    print(f"[{args.arm}] runtime {(time.time()-t0)/60:.1f} min; wrote {out/'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
