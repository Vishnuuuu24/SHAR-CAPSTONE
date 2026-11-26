"""P3B: imbalance-oriented loss functions for `core.p1c_head_training`.

Why this module exists
-----------------------
`docs/PHASE_PLAN.md` sec.6 (P3B) asks for cross-entropy against three
imbalance-oriented alternatives: inverse-frequency weighted cross-entropy,
focal loss (Lin et al. 2017), and Balanced Softmax (Ren et al. 2020). Under
the pipeline this project actually validated (D-46: frozen DINOv2, 8
frames/video via `core.p1c_video_sampling.sample_rows_per_video(rows, 8,
seed=0)`), the fold 1 AR training set is EXACTLY class-balanced -- 272 rows
per class, all 14 classes equal. That single fact makes two of the four
losses provably identical to plain cross-entropy, and this module states the
proof rather than letting it be rediscovered by staring at a null result:

1. **Inverse-frequency weighted cross-entropy.** The per-class weight is
   `1/count_c`, normalised here to mean 1.0 (`inverse_frequency_weights`).
   When every `count_c` is equal, every `1/count_c` is equal, so every
   normalised weight is exactly 1.0 -- `nn.CrossEntropyLoss(weight=ones)` is
   `nn.CrossEntropyLoss()` elementwise. Proof: weight_c = (1/count_c) /
   mean_c(1/count_c); counts equal => 1/count_c is a constant k for every c
   => mean_c(k) = k => weight_c = k/k = 1 for every c.

2. **Balanced Softmax** (Ren, Yu, Sha 2020, NeurIPS): logit_c is adjusted by
   `+log(prior_c)` before softmax, where `prior_c = count_c / sum(counts)`
   (`log_prior_adjustment`). When every `count_c` is equal, every `prior_c`
   is equal, so every logit gets the SAME additive constant. Softmax (and
   therefore cross-entropy on it) is invariant to adding a constant to every
   logit of an example: `softmax(z + a*1) = softmax(z)` for scalar `a`. Proof:
   `exp(z_i + a) / sum_j exp(z_j + a) = exp(a)*exp(z_i) / (exp(a)*sum_j
   exp(z_j)) = exp(z_i) / sum_j exp(z_j)`. So under equal priors, Balanced
   Softmax is cross-entropy.

3. **Focal loss** (Lin et al. 2017) is NOT vacuous under class balance: it
   reweights each example by `(1 - p_t)^gamma`, where `p_t` is the model's
   *predicted* probability of the true class for THAT example -- a
   per-example difficulty signal, not a per-class frequency signal. Two
   examples of the same (perfectly balanced) class can have wildly different
   `p_t`, so focal loss changes gradients even when every class count is
   identical.

`scripts/run_p1c_loss_ablation.py` measures (1) and (2) empirically against
(3)'s cross-fold effect, and its `vacuity_check.json` asserts the equality
this docstring claims rather than merely asserting it in prose.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#: The four loss arms P3B compares. Order matches the brief and
#: `scripts/run_p1c_loss_ablation.py`'s `--losses` default.
LOSS_NAMES = ("cross_entropy", "weighted_cross_entropy", "focal", "balanced_softmax")

_REDUCTIONS = ("mean", "none")


def class_counts(rows: Sequence[dict[str, Any]], class_names: Sequence[str]) -> np.ndarray:
    """Per-class row counts, in `class_names` order.

    `row["target"]` is read as-is and cast with `int(...)`: the cache loader
    (`core.feature_cache.load_feature_cache`) stores it as a CSV string, so a
    fixture that already stores an int must not break this.
    """

    counts = np.zeros(len(class_names), dtype=np.int64)
    for row in rows:
        counts[int(row["target"])] += 1
    return counts


def inverse_frequency_weights(counts: np.ndarray, *, normalise: bool = True) -> np.ndarray:
    """`1/count_c` per class, normalised to mean 1.0 by default.

    Normalising to mean 1.0 is what makes the balanced-count case exact: if
    every `count_c` is equal to some `n`, every raw weight is `1/n`, their
    mean is `1/n`, and dividing by that mean yields 1.0 for every class --
    i.e. `nn.CrossEntropyLoss(weight=inverse_frequency_weights(balanced))` is
    elementwise identical to unweighted `nn.CrossEntropyLoss()`. This is the
    weighted-cross-entropy half of this module's vacuity argument.
    """

    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty 1-D array")
    if np.any(counts <= 0):
        raise ValueError("every class count must be > 0 to compute an inverse-frequency weight")
    weights = 1.0 / counts
    if normalise:
        weights = weights / weights.mean()
    return weights


def log_prior_adjustment(counts: np.ndarray) -> np.ndarray:
    """`log(count_c / sum(counts))` per class, for Balanced Softmax.

    Added to each class's logit before softmax (Ren et al. 2020). When every
    `count_c` is equal, every prior is equal, so this returns a constant
    vector -- adding the same constant to every logit of an example leaves
    softmax (and cross-entropy on it) unchanged. This is the Balanced-Softmax
    half of this module's vacuity argument.
    """

    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("counts must be a non-empty 1-D array")
    if np.any(counts <= 0):
        raise ValueError("every class count must be > 0 to compute a log-prior adjustment")
    priors = counts / counts.sum()
    return np.log(priors)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    alpha: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """`(1 - p_t)^gamma * CE`, Lin et al. 2017, numerically stable.

    `p_t` is read off a `log_softmax` (never `softmax` then `log`, which loses
    precision and can hand back `log(0) = -inf` from an underflowed
    probability for a confidently-wrong example): `log_pt = log_softmax(
    logits)[target]`, `p_t = exp(log_pt)`. At `gamma=0` this is exactly
    `-log_pt`, i.e. elementwise cross-entropy. `alpha`, when given, is a
    per-class weight tensor of shape `(num_classes,)`, indexed by `targets`
    exactly as `nn.CrossEntropyLoss(weight=...)` would.
    """

    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {reduction!r}")
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = log_pt.exp()
    loss = -((1.0 - p_t) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * alpha.gather(0, targets)
    if reduction == "mean":
        return loss.mean()
    return loss


def build_loss(
    name: str,
    *,
    counts: np.ndarray,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a `(logits, targets) -> Tensor` callable for one of `LOSS_NAMES`.

    `"cross_entropy"` returns the literal `nn.CrossEntropyLoss(reduction=...)`
    object -- the same object/arithmetic path every earlier P1C run used --
    so the control arm of the ablation is bit-identical to every number in
    `docs/TRAINING_LOG.md`.
    """

    if name not in LOSS_NAMES:
        raise ValueError(f"unknown loss {name!r}; expected one of {LOSS_NAMES}")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}; got {reduction!r}")

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(reduction=reduction)

    if name == "weighted_cross_entropy":
        weight = torch.as_tensor(inverse_frequency_weights(counts), dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    if name == "focal":
        def _focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return focal_loss(logits, targets, gamma=gamma, reduction=reduction)

        return _focal

    # balanced_softmax
    adjustment = torch.as_tensor(log_prior_adjustment(counts), dtype=torch.float32)

    def _balanced_softmax(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + adjustment, targets, reduction=reduction)

    return _balanced_softmax
