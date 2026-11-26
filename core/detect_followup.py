"""Project MIL adaptations on frozen features, not reproductions of published networks.

All settings are prospective in docs/DETECTOR_FOLLOWUP_2026-11-12.md.
Only training-video labels and held-out training-video AUC enter training.
"""
from __future__ import annotations

import copy
import math
import time
import numpy as np
import torch
from torch import nn
from core import detect_model as base

VARIANTS = ('control', 'magnitude', 'selftrain', 'uncertainty', 'language')


class FrameScorer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Dropout(.5),
                                      nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.5))
        self.readout = nn.Linear(128, 1)

    def forward(self, x):
        return self.readout(self.features(x))


def selected_mean(values, ranking, mask, *, largest=True):
    """Mean on top/bottom ceil(T/16) valid indices; padding can never be selected."""
    if ((~mask).sum(1) == 0).any():
        raise ValueError('empty video bag')
    n = (~mask).sum(1)
    k = (n + base.TOPK_DIVISOR - 1) // base.TOPK_DIVISOR
    indices = ranking.masked_fill(mask, -torch.inf if largest else torch.inf).argsort(1, descending=largest)
    selected = values.gather(1, indices)
    keep = torch.arange(values.shape[1], device=values.device)[None, :] < k[:, None]
    return selected.masked_fill(~keep, 0).sum(1) / k


def magnitude_loss(latent, logits, mask, labels):
    norms = latent.norm(dim=-1)
    probabilities = logits.sigmoid()
    bag = selected_mean(probabilities, norms, mask)
    mil = nn.functional.binary_cross_entropy(bag.clamp(1e-6, 1-1e-6), labels)
    mag = selected_mean(norms, norms, mask)
    # Fixed project margin and weight, not a claim to reproduce RTFM's architecture/loss.
    separation = (10 - mag[labels.bool()]).relu().square().mean() + mag[~labels.bool()].square().mean()
    return mil + .001 * separation


def pseudo_instance_loss(logits, teacher_samples, mask, labels, uncertainty=False):
    """Detached EMA-teacher pseudo targets; normal bags have true negative instance labels.

    Anomalous bags supply only confident pseudo positives/negatives; ambiguous frames are ignored.
    Teacher outputs are training auxiliaries, never evaluation truth.
    """
    samples = teacher_samples.detach()
    mean = samples.mean(0)
    normal = (~labels.bool())[:, None].expand_as(mask)
    chosen = (~mask) & (normal | (mean >= .9) | (mean <= .1))
    targets = ((mean >= .9) & ~normal).float()
    weights = (1 - samples.var(0, unbiased=False) / .25).clamp(.1, 1) if uncertainty else torch.ones_like(mean)
    weights = weights * chosen
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def fit_statistics(store, fit):
    """Fit preprocessing on fitting videos only, excluding the epoch-selection holdout."""
    total = np.zeros(store.dim, dtype=np.float64)
    square = total.copy()
    count = 0
    for video in fit:
        a,b = store.spans[video.relative_path]
        for start in range(a,b,2048):
            x = np.concatenate([np.asarray(v[start:min(start+2048,b)], dtype=np.float64) for v in store.arrays], axis=1)
            total += x.sum(0)
            square += np.square(x).sum(0)
            count += len(x)
    if not count:
        raise ValueError('no fitting frames')
    store.mean = (total/count).astype(np.float32)
    store.std = np.sqrt(np.maximum(square/count - (total/count)**2, 1e-12)).astype(np.float32)
    return store


def train(store, fit, held, *, variant, seed, device, epochs=60, patience=10, batch_videos=32, checkpoint_path=None, log=print):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if not epochs or not fit or not held:
        raise ValueError('training requires nonempty fit/holdout and positive epoch budget')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FrameScorer(store.dim*2).to(device)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    positives = [v for v in fit if v.anomaly]
    negatives = [v for v in fit if not v.anomaly]
    if not positives or not negatives:
        raise ValueError('both classes required')
    steps = math.ceil(max(len(positives),len(negatives))/batch_videos)
    best, state, best_epoch, stale, history = -1., None, -1, 0, []
    start_epoch = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path,map_location=device,weights_only=False)
        if saved['identity'] != (variant,seed,store.dim,epochs):
            raise ValueError('resume identity differs')
        model.load_state_dict(saved['model'])
        teacher.load_state_dict(saved['teacher'])
        optimizer.load_state_dict(saved['optimizer'])
        best,state,best_epoch,stale,history = saved['selection']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        start_epoch = saved['next_epoch']
    for epoch in range(start_epoch,epochs):
        if stale >= patience:
            break
        epoch_started = time.monotonic()
        model.train()
        pa, pn = rng.permutation(len(positives)), rng.permutation(len(negatives))
        running = 0.
        for step in range(steps):
            # Both classes cycle, so no majority-class tail is dropped each epoch.
            indices = np.arange(step*batch_videos,(step+1)*batch_videos)
            videos = [positives[pa[i % len(pa)]] for i in indices] + [negatives[pn[i % len(pn)]] for i in indices]
            xs = np.zeros((len(videos),base.SEQUENCE_LENGTH,store.dim*2),np.float32)
            masks = np.ones(xs.shape[:2],bool)
            for row,v in enumerate(videos):
                frames = store.frames(v.relative_path,'causal')
                x = frames[base._subsample(len(frames),base.SEQUENCE_LENGTH,rng)]
                xs[row,:len(x)] = x
                masks[row,:len(x)] = False
            x = torch.from_numpy(xs).to(device)
            mask = torch.from_numpy(masks).to(device)
            labels = torch.tensor([float(v.anomaly) for v in videos],device=device)
            latent = model.features(x)
            logits = model.readout(latent).squeeze(-1)
            if variant == 'magnitude':
                loss = magnitude_loss(latent,logits,mask,labels)
            else:
                bag = base.topk_video_score(logits.sigmoid(),mask)
                loss = nn.functional.binary_cross_entropy(bag.clamp(1e-6,1-1e-6),labels)
                if variant in ('selftrain','uncertainty') and epoch >= 5:
                    teacher.train(variant == 'uncertainty')
                    with torch.no_grad():
                        samples = torch.stack([teacher(x).squeeze(-1).sigmoid() for _ in range(4 if variant=='uncertainty' else 1)])
                    loss = loss + .5*pseudo_instance_loss(logits,samples,mask,labels,variant=='uncertainty')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for t,p in zip(teacher.parameters(),model.parameters()):
                    t.lerp_(p,.01)
            running += float(loss.detach())
        auc = base._video_auc(model,store,held,'causal',device,'topk')
        record = {'epoch':epoch,'loss':running/steps,'heldout_video_auc':auc,'epoch_seconds':time.monotonic()-epoch_started}
        history.append(record)
        log({'variant':variant,'seed':seed,**record})
        if auc > best+1e-4:
            best,state,best_epoch,stale = auc,copy.deepcopy(model.state_dict()),epoch,0
        else:
            stale += 1
        if checkpoint_path is not None:
            temporary = checkpoint_path.with_suffix('.tmp')
            torch.save({'identity':(variant,seed,store.dim,epochs),'model':model.state_dict(),
                'teacher':teacher.state_dict(),'optimizer':optimizer.state_dict(),
                'selection':(best,state,best_epoch,stale,history),'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
                'next_epoch':epoch+1},temporary)
            temporary.replace(checkpoint_path)
        if stale >= patience:
            break
    model.load_state_dict(state)
    return model,history,best_epoch
