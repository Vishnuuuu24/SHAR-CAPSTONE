"""D-67: partial fine-tuning of DINOv2-B (last 2 or last 4 blocks) under the D-65 protocol.

Arms (same frames, seeds, head, optimiser budget, inner monitor and scoring):
  B_frozen  D-53 `step_permuted` head on the cached frozen features (control; D-66's B_ref)
  B_ft2     same head, blocks 10-11 + final norm trainable
  B_ft4     same head, blocks 8-11 + final norm trainable
  B_ft2_lr0 B_ft2's pipeline with backbone LR 0 (fold 1 seed 0 only; a pipeline check, not a contrast)

Blocks 0-7 never change, so their output for the training + monitor frames is computed once per fold and kept on the
GPU (`h8`); B_ft2 trains from `h10` = frozen blocks 8-9 of `h8`. The backbone LR is chosen per fold and arm from
`LR_GRID` by mean inner-monitor macro-F1 over seeds (ties -> lower LR); only the chosen runs are scored on the 108
validation videos, in one decode + trunk pass per fold that feeds every chosen tail.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

ARMS = ("B_frozen", "B_ft2", "B_ft4")
CONTRASTS = (("B_ft2", "B_frozen"), ("B_ft4", "B_frozen"))
TRAINABLE_BLOCKS = {"B_ft2": 2, "B_ft4": 4, "B_ft2_lr0": 2}
LR_GRID = (1e-5, 3e-5, 1e-4)
TRUNK_SPLIT = 8
CHECK_FRAMES = 128


def select_lr(monitor_f1: dict[float, Sequence[float]]) -> float:
    """LR with the highest mean monitor macro-F1 over seeds; ties go to the lower LR."""

    return min(monitor_f1, key=lambda lr: (-float(np.mean(monitor_f1[lr])), lr))


def index_source(rows: list) -> dict:
    """A `sequences`-compatible split whose 'features' are row indices, so sequence order matches the control's."""

    return {"train": (np.arange(len(rows), dtype=np.float32)[:, None], rows)}


def _prefetched(batcher, blob: np.ndarray, batch: int):
    """Yield (start, images) with the next batch decoding while the current one is on the GPU."""

    with ThreadPoolExecutor(1) as ahead:
        starts = list(range(0, len(blob), batch))
        nxt = ahead.submit(batcher, blob[starts[0]:starts[0] + batch]) if starts else None
        for i, s in enumerate(starts):
            images = nxt.result()
            if i + 1 < len(starts):
                nxt = ahead.submit(batcher, blob[starts[i + 1]:starts[i + 1] + batch])
            yield s, images


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def run_fold(fold: int, seeds: Sequence[int], arms: Sequence[str] = ARMS, *, device: str = "cuda:0", epochs: int | None = None,
             lr_grid: Sequence[float] = LR_GRID, lr0_check: bool = True, threads: int = 8, batch: int = 64,
             max_validation_frames: int | None = None, log=print) -> tuple[list[dict], list[dict], dict]:
    import torch

    from core.backbone_cache import frame_key
    from core.backbone_experiment import MONITOR_PER_CLASS, inner_monitor_videos
    from core.confusion_groups import macro_f1
    from core.frame_batches import FrameBatcher
    from core.p1c_head_variants import train_sequence_head
    from core.p1c_oracle_interval import predict_video_probs
    from core.vit_split import CachedTailHead, Tail, frozen_digest, state_digest, trunk_tokens
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from scripts.run_p1c_xd_extra_probe import FRAMES, _build_head, train_head
    from scripts.run_subset7_search import load_fold_features, sequences

    t_fold = time.time()
    fd = load_fold_features(fold, ("train", "validation"))
    classes = fd["class_names"]
    train_rows = fd["train"][1]
    train_targets = {k[1]: int(v[1]["target"]) for k, v in sequences(fd, "train", 0).items()}
    monitor = inner_monitor_videos(train_targets, MONITOR_PER_CLASS, seed=fold)
    val_rows = fd["validation"][1] if max_validation_frames is None else fd["validation"][1][:max_validation_frames]
    if max_validation_frames is not None:
        fd["validation"] = (fd["validation"][0][:max_validation_frames], val_rows)
    kw = {} if epochs is None else {"epochs": epochs}
    records, probs, heads = [], [], {}

    def split_train(seqs):
        return {k: v for k, v in seqs.items() if k[1] not in monitor}, {k: v for k, v in seqs.items() if k[1] in monitor}

    def emit(arm, seed, head, val_seq, extra):
        keys = sorted(val_seq)
        p = np.stack([predict_video_probs(head, val_seq[k][0], FRAMES) for k in keys])
        truth = np.array([int(val_seq[k][1]["target"]) for k in keys])
        records.append({"arm": arm, "fold": fold, "seed": seed, "macro_f1": macro_f1(truth, p.argmax(1), len(classes)),
                        "accuracy": float((p.argmax(1) == truth).mean()), **extra})
        probs.extend({"arm": arm, "fold": fold, "seed": seed, "video": k[1], "true": classes[int(val_seq[k][1]["target"])],
                      **{f"p_{c}": float(x) for c, x in zip(classes, row)}} for k, row in zip(keys, p))

    # 1. control on the cached frozen features (CPU, as D-66)
    if "B_frozen" in arms:
        for seed in seeds:
            tr, mo = split_train(sequences(fd, "train", seed))
            result, head = train_head(seed, tr, mo, classes, epochs)
            emit("B_frozen", seed, head, sequences(fd, "validation", seed),
                 {"best_epoch": int(result.best_epoch), "monitor_macro_f1": float(result.best_validation_macro_f1),
                  "epochs_run": len(result.epochs), "train_loss_at_best": float(result.epochs[result.best_epoch]["train_mean_loss"]),
                  "train_videos": len(tr), "monitor_videos": len(mo)})
        log({"fold": fold, "stage": "control", "seconds": round(time.time() - t_fold)})

    ft_arms = [a for a in arms if a != "B_frozen"]
    if lr0_check and fold == 1 and "B_ft2" in ft_arms:
        ft_arms.append("B_ft2_lr0")
    checks: dict = {"fold": fold}
    if not ft_arms:
        return records, probs, checks

    # 2. backbone, trunk states for every training + monitor frame, and the two equivalence checks
    dev = torch.device(device)
    backbone, _pre, weight_sha = _load_backbone_and_preprocessing(dev)
    net, npt = backbone.network, backbone.num_prefix_tokens
    for p in net.parameters():
        p.requires_grad_(False)
    digest0 = state_digest(net.state_dict().items())
    batcher = FrameBatcher(threads)
    blob = batcher.blob_rows([frame_key(r) for r in train_rows])
    h8 = None
    t0 = time.time()
    with torch.no_grad():
        identity = Tail(net, TRUNK_SPLIT, 0, npt).to(dev).eval()
        for s, images in _prefetched(batcher, blob, batch):
            x = images.to(dev)
            h = trunk_tokens(net, x, TRUNK_SPLIT)
            if h8 is None:
                h8 = torch.empty((len(blob), *h.shape[1:]), dtype=torch.float32, device=dev)
            h8[s:s + len(x)] = h
            if s == 0:
                n = min(CHECK_FRAMES, len(x))
                whole = backbone.pooled_features(x[:n]).cpu().numpy()
                split = identity(h[:n]).cpu().numpy()
                cached = np.asarray(fd["train"][0][:n], dtype=np.float32)
                checks["pixel_path_min_cosine"] = float(_cosine(whole, cached).min())
                checks["split_vs_whole_max_abs"] = float(np.abs(split - whole).max())
                checks["split_vs_whole_min_cosine"] = float(_cosine(split, whole).min())
        del identity
    checks["trunk_cache_frames"], checks["trunk_cache_seconds"] = len(blob), round(time.time() - t0, 1)
    checks["trunk_frames_per_s"] = round(len(blob) / max(time.time() - t0, 1e-9), 1)
    log({"fold": fold, "stage": "trunk_cache", **checks})
    if checks["pixel_path_min_cosine"] < 0.9999 or checks["split_vs_whole_min_cosine"] < 0.9999:
        raise RuntimeError(f"equivalence check failed: {checks}")

    # 3. fine-tuned arms: LR grid x seeds, chosen per arm by mean monitor macro-F1
    idx = index_source(train_rows)
    chosen: dict[tuple[str, int], dict] = {}
    caches = {TRUNK_SPLIT: h8}
    del h8
    order = sorted(ft_arms, key=lambda a: -TRAINABLE_BLOCKS[a])  # 4-block first, then derive h10 and free h8
    for arm in order:
        k = TRAINABLE_BLOCKS[arm]
        start = 12 - k
        if start not in caches:
            with torch.no_grad():
                src = caches[TRUNK_SPLIT]
                h = torch.empty_like(src)
                for s in range(0, len(src), batch):
                    y = src[s:s + batch]
                    for blk in net.blocks[TRUNK_SPLIT:start]:
                        y = blk(y)
                    h[s:s + batch] = y
                caches[start] = h
            del src, h
            if not any(12 - TRAINABLE_BLOCKS[a] == TRUNK_SPLIT for a in order[order.index(arm):]):
                del caches[TRUNK_SPLIT]
                torch.cuda.empty_cache()
        cache = caches[start]
        grid = (0.0,) if arm == "B_ft2_lr0" else tuple(lr_grid)
        arm_seeds = [seeds[0]] if arm == "B_ft2_lr0" else list(seeds)
        runs: dict[tuple[float, int], dict] = {}
        for lr in grid:
            for seed in arm_seeds:
                t_run = time.time()
                tr, mo = split_train(sequences(idx, "train", seed))
                torch.manual_seed(seed)  # identical head init to the control (train_head does the same)
                head = _build_head("step_permuted", len(classes), FRAMES)
                tail = Tail(net, start, k, npt).to(dev)
                model = CachedTailHead(tail, head, cache)

                def groups(m, lr=lr):
                    return [{"params": list(m.head.parameters())},
                            {"params": [p for p in m.tail.parameters() if p.requires_grad], "lr": lr}]

                result = train_sequence_head(model, tr, mo, classes, frames=FRAMES, seed=seed, optimizer_param_groups=groups, **kw)
                model.eval()
                runs[(lr, seed)] = {
                    "head": {k2: v.detach().cpu().clone() for k2, v in head.state_dict().items()},
                    "tail": tail.trainable_state(),
                    "record": {"arm": arm, "fold": fold, "seed": seed, "backbone_lr": lr, "trainable_blocks": k,
                               "best_epoch": int(result.best_epoch), "monitor_macro_f1": float(result.best_validation_macro_f1),
                               "epochs_run": len(result.epochs), "train_loss_at_best": float(result.epochs[result.best_epoch]["train_mean_loss"]),
                               "train_videos": len(tr), "monitor_videos": len(mo), "seconds": round(time.time() - t_run, 1),
                               "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                               "trainable_tensors": sorted(tail.trainable_names())[:2] + ["..."]}}
                log({k2: v for k2, v in runs[(lr, seed)]["record"].items() if k2 != "trainable_tensors"})
                del model, tail, head
                torch.cuda.empty_cache()
        lr_star = select_lr({lr: [runs[(lr, s)]["record"]["monitor_macro_f1"] for s in arm_seeds] for lr in grid})
        checks[f"{arm}_selected_lr"] = lr_star
        for (lr, seed), run in runs.items():
            run["record"]["selected"] = lr == lr_star
            if lr == lr_star:
                chosen[(arm, seed)] = run
            else:
                records.append({**run["record"], "macro_f1": None, "accuracy": None, "scored": False})
    del caches
    torch.cuda.empty_cache()

    # 4. one decode + trunk pass over the validation frames, every chosen tail fed from it
    tails = {}
    for (arm, seed), run in chosen.items():
        k = TRAINABLE_BLOCKS[arm]
        t = Tail(net, 12 - k, k, npt).to(dev).eval()
        t.load_state_dict({**t.state_dict(), **run["tail"]}, strict=True)
        tails[(arm, seed)] = t
    vblob = batcher.blob_rows([frame_key(r) for r in val_rows])
    feats = {key: np.empty((len(vblob), 2 * net.embed_dim), dtype=np.float32) for key in tails}
    t0 = time.time()
    with torch.no_grad():
        for s, images in _prefetched(batcher, vblob, batch):
            h = trunk_tokens(net, images.to(dev), TRUNK_SPLIT)
            h10 = None
            for key, t in tails.items():
                if t.start == TRUNK_SPLIT:
                    feats[key][s:s + len(h)] = t(h).cpu().numpy()
                else:
                    if h10 is None:
                        h10 = h
                        for blk in net.blocks[TRUNK_SPLIT:t.start]:
                            h10 = blk(h10)
                    feats[key][s:s + len(h)] = t(h10).cpu().numpy()
            if s and (s // batch) % 100 == 0:
                log({"fold": fold, "stage": "validation", "frames_done": s + len(h), "of": len(vblob),
                     "fps": round((s + len(h)) / (time.time() - t0), 1)})
    checks["validation_frames"], checks["validation_seconds"] = len(vblob), round(time.time() - t0, 1)
    for (arm, seed), run in chosen.items():
        torch.manual_seed(seed)
        head = _build_head("step_permuted", len(classes), FRAMES)
        head.load_state_dict(run["head"])
        head.eval()
        val_seq = sequences({"validation": (feats[(arm, seed)], val_rows)}, "validation", seed)
        emit(arm, seed, head, val_seq, {**{k2: v for k2, v in run["record"].items() if k2 not in ("arm", "fold", "seed")}, "scored": True})

    # 5. freeze audit: the shared backbone (trunk, and the original blocks 8-9 B_ft2 reads) is bit-identical after
    #    every run -- tails train deep copies, never `net` itself
    checks["backbone_state_sha256"] = state_digest(net.state_dict().items())
    checks["frozen_trunk_unchanged"] = checks["backbone_state_sha256"] == digest0
    checks["frozen_trunk_blocks_0_7_sha256"] = frozen_digest(net, TRUNK_SPLIT)
    checks["weights_sha256"] = weight_sha
    checks["fold_seconds"] = round(time.time() - t_fold)
    if not checks["frozen_trunk_unchanged"]:
        raise RuntimeError("frozen trunk changed during training")
    log({"fold": fold, "stage": "done", **checks})
    return records, probs, checks
