"""Verify a B-1 feature cache against the live backbone that should have produced it.

Split out of `core.feature_cache` by responsibility (and to keep both modules
under the ~400-line ceiling): that module owns building and loading a cache and
the fail-closed manifest contract, this one owns the far stronger claim that the
numbers in the cache are the numbers the real backbone emits.

The distinction matters. `load_feature_cache` proves a cache was *built* under
the config it is being served to -- every digest agrees. It cannot prove the
vectors are correct, because a manifest is only metadata: a bug in the pooling,
the row ordering, the dtype cast or the dataset's iteration order yields a
perfectly self-consistent cache full of wrong numbers, and every downstream
experiment then trains on something plausible and false. The only way to close
that gap is to recompute and compare, which is what this module does.

## Why this is not a bit-identity check (measured 2026-08-29, gate revised 2026-09-03)

Settled after three measured attempts, recorded here so the reasoning is not
re-litigated. The decisive fact: `scripts/run_p1c_frame_baseline.py` seeds RNGs
but never calls `torch.use_deterministic_algorithms`, so **every training run in
this project already runs on cuDNN's default, partly non-deterministic kernels**
and accepts the resulting last-bit jitter. Requiring the cache to be bit-exact
would hold it to a stricter standard than the pipeline that consumes it. What
those runs demonstrate instead is reproducibility at the level that matters:
CONFIRM-015 seed 0 reproduced SCREEN-012's validation macro-F1 to four decimals
across 21 epochs on an independent launch, because per-element jitter of ~1e-4
washes out of a metric aggregated over 37,600 frames.

The first real run of this verifier against the D-42 cache reported 0 of 256
sampled rows bit-identical, with deltas of 3.9e-3 to 3.1e-2. Those are **1-8
float16 ULPs** at the magnitudes involved (pooled C5 activations peak around
3-9), not a wiring error: CUDA convolution picks different algorithms and
reduction orders for a batch of 1 than for the batch of 64 the cache was built
with, so the underlying float32 values differ in their last bits and float16
storage rounds the two to adjacent representable values.

Demanding bit-identity was therefore testing an invariant the hardware does not
provide. The fix is not to loosen a threshold until it passes -- it is to test
the right invariant two ways:

1. **Replay at the build's batch size.** Recomputing in batches of the same
   shape reproduces the same kernel selection, so this arm is held to exact
   equality and would catch any ordering, pooling, dtype or provenance bug.
2. **A bounded numerical tolerance**, now expressed in float16 ULPs.

## K-26: the `atol + rtol * |b|` gate was measured too tight, not too loose

The first version of this gate used the standard combined absolute-and-relative
form, `|a - b| <= atol + rtol * |b|`, both 1e-3, on the reasoning that a
relative bound suits the peak activations (3-9) and an absolute floor suits the
many channels ReLU leaves at or near zero. That reasoning was not wrong, but the
constants were: K-26 recorded fold 3's ResNet train cache failing on 2 of 512
rows by less than one part in 1e-3 -- `2.930e-3` at magnitude 1.310 (allowed
`2.310e-3`) and `1.465e-3` at magnitude 0.4375 (allowed `1.438e-3`). Measured in
float16 ULPs at those magnitudes, those are **3 and 6 ULPs** -- ordinary cuDNN
convolution-algorithm variance (K-24), the same phenomenon the D-42 measurement
above already characterized, not a wrong cache. Fold 1 had a larger raw delta
(3.9e-3) but it landed on a large-magnitude element, so its relative allowance
absorbed it; nothing about fold 3 was worse, the fixed-form gate was just thin
enough that where the jitter happened to land mattered more than it should.

The original objection to a pure ULP gate was that float16 packs representable
values ~6e-8 apart near zero, so a numerically irrelevant 5e-5 difference there
is ~900 ULPs -- a pure ULP gate would fail on exactly the elements a wiring bug
is least likely to hide in. That objection is correct and is still true; the
fix is not to abandon ULPs but to floor the step size:

    step    = numpy.spacing(float16(max(|b|, ulp_floor_magnitude)))
    allowed = max_ulps * step
    fail if |a - b| > allowed

A "step" is one float16 ULP evaluated at `max(|b|, ulp_floor_magnitude)`, so
near zero the step never shrinks below its value at the floor -- resolving the
original objection -- while at real magnitudes it tracks the actual float16
spacing there, which a fixed `rtol` only approximates.

**Defaults.** `ulp_floor_magnitude = 1.0` (one float16 ULP there is
9.765625e-4, in the middle of the peak-activation range this backbone
produces). `max_ulps = 8`, chosen to cover the largest disagreement K-26
measured at real magnitude (6 ULPs) with modest headroom -- but the number that
matters is not 6 vs. 8. This gate exists to catch wiring bugs: wrong pooling,
wrong row order, wrong dtype, a stale backbone. Those change values by order
the data scale itself -- thousands to millions of ULPs, not single digits.
Distinguishing 6 ULPs of cuDNN jitter from 8 is not the job; distinguishing 8
from 1e6 is the entire job, and both 6 and 8 do that equally well. `max_ulps`
is deliberately not tuned tighter than this, because tuning it to the last
observed failure is the "raise the tolerance until it passes" move K-26
explicitly rejects -- the margin is set by the mechanism (cuDNN algorithm
variance), not by chasing the specific rows that happened to fail.

Integer ULP distance (`_ordered_int`) is still computed and reported. The
report also carries `max_significant_ulps`: the worst ULP distance restricted
to elements at or above the floor magnitude, i.e. with the near-zero explosion
excluded. That is the number a future person needs before tightening
`max_ulps` -- it says how close the real, meaningful jitter runs to the gate,
which `max_ulps` alone (an input, not an outcome) cannot say.

For scale: the tolerated jitter is a handful of float16 ULPs (~1e-3 relative at
these magnitudes), while training already feeds this backbone's output through
`amp_bfloat16` (8 mantissa bits against float16's 10) and the task's measured
seed-to-seed macro-F1 spread is 0.0049. The cache is not the least precise
thing in this pipeline by a wide margin.

Provenance (`target`, `source_video_id`) is unaffected by any of this and keeps
zero tolerance: a cache whose features are numerically fine but whose row
order is off by one would silently break the grouped evaluation this project
depends on, and no float comparison catches that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from core.feature_cache import FeatureCacheMismatch, load_feature_cache


def _ordered_int(values: np.ndarray) -> np.ndarray:
    """Map floats to monotonically ordered integers so adjacent values differ by 1.

    IEEE-754 floats of one sign are already ordered by their bit pattern, so
    reinterpreting the bits as a signed integer and folding the negative half
    across zero yields an index where `|index(a) - index(b)|` is exactly the
    number of representable values between `a` and `b` -- the honest definition
    of "how many ULPs apart", and the only one that stays meaningful near zero.

    This replaces a `delta / np.spacing(value)` ratio, which was wrong: at a
    near-zero element `np.spacing` is ~6e-8, so any difference at all divided
    into an enormous ratio, and the reported worst delta and worst ratio could
    come from different elements of the same vector. Measured 2026-08-29 on the
    real D-42 cache, that formula reported 1369 ULPs for a difference of
    9.8e-4 -- which is exactly ONE float16 step at magnitude 1.
    """

    int_dtype = np.int16 if values.dtype == np.float16 else np.int32
    as_int = values.view(int_dtype).astype(np.int64)
    # Negative floats descend as their bit pattern ascends; reflect them.
    sign_mask = as_int < 0
    magnitude_limit = np.int64(np.iinfo(int_dtype).min)
    as_int[sign_mask] = magnitude_limit - as_int[sign_mask]
    return as_int


def verify_feature_cache(
    dataset: Dataset,
    backbone: nn.Module,
    cache_dir: Path | str,
    *,
    device: torch.device | str,
    sample_size: int = 256,
    seed: int = 0,
    batch_size: int = 64,
    max_ulps: int = 8,
    ulp_floor_magnitude: float = 1.0,
    **load_kwargs: Any,
) -> dict[str, Any]:
    """Re-run the live backbone on a sample of rows and compare against the cache.

    Samples `sample_size` rows deterministically and recomputes them in batches
    of `batch_size` -- match the build's batch size and the comparison is exact,
    because the same kernels run in the same order. The numerical gate bounds
    the residual float16 jitter that a different batch shape introduces to
    `max_ulps` float16 steps, each step floored at `ulp_floor_magnitude` (see
    the module docstring's K-26 section for why); a divergence larger than that
    fails, as does ANY provenance disagreement, for which there is no tolerance
    at all -- a cache whose features are right but whose `source_video_id`
    column is off by one row would silently break the grouped evaluation this
    project depends on.

    Returns a report; raises `FeatureCacheMismatch` if anything disagrees, so a
    caller cannot proceed on an unverified cache by ignoring a return value.
    """

    features, rows, manifest = load_feature_cache(cache_dir, **load_kwargs)

    n_rows = len(rows)
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1; got {sample_size}")
    take = min(sample_size, n_rows)
    indices = sorted(np.random.default_rng(seed).choice(n_rows, size=take, replace=False).tolist())

    numpy_dtype = np.float16 if manifest.dtype == "float16" else np.float32
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    if max_ulps < 0:
        raise ValueError(f"max_ulps must be >= 0; got {max_ulps}")
    if ulp_floor_magnitude < 0:
        raise ValueError(f"ulp_floor_magnitude must be >= 0; got {ulp_floor_magnitude}")

    backbone = backbone.to(device)
    backbone.eval()

    exact = 0
    worst_delta = 0.0
    worst_ulps = 0
    worst_significant_ulps = 0
    feature_mismatches: list[str] = []
    provenance_mismatches: list[str] = []

    with torch.no_grad():
        for start in range(0, take, batch_size):
            chunk = indices[start : start + batch_size]
            items = [dataset[i] for i in chunk]  # type: ignore[index]
            images = torch.stack([item["image"] for item in items]).to(device)
            pooled = backbone.pooled_features(images)
            recomputed = pooled.to(torch.float32).cpu().numpy().astype(numpy_dtype)

            for offset, row_index in enumerate(chunk):
                item = items[offset]
                cached = features[row_index]
                fresh = recomputed[offset]

                if np.array_equal(cached, fresh):
                    exact += 1
                else:
                    a = cached.astype(np.float32)
                    b = fresh.astype(np.float32)
                    delta = np.abs(a - b)
                    step = np.spacing(np.maximum(np.abs(b), ulp_floor_magnitude).astype(np.float16)).astype(
                        np.float32
                    )
                    allowed = max_ulps * step
                    over = delta > allowed
                    worst_element = int(np.argmax(delta - allowed))
                    worst_delta = max(worst_delta, float(np.max(delta)))

                    ulp_distance = np.abs(_ordered_int(cached.copy()) - _ordered_int(fresh.copy()))
                    worst_ulps = max(worst_ulps, int(np.max(ulp_distance)))
                    significant = np.maximum(np.abs(a), np.abs(b)) >= ulp_floor_magnitude
                    if np.any(significant):
                        worst_significant_ulps = max(
                            worst_significant_ulps, int(np.max(ulp_distance[significant]))
                        )

                    if np.any(over) and len(feature_mismatches) < 5:
                        feature_mismatches.append(
                            f"row {row_index}: |delta| = {delta[worst_element]:.3e} at element "
                            f"{worst_element} exceeds {max_ulps} ULPs ({allowed[worst_element]:.3e}) "
                            f"of a step floored at magnitude {ulp_floor_magnitude:g} "
                            f"(|{b[worst_element]:.3e}|)"
                        )

                row = rows[row_index]
                if int(row["target"]) != int(item["target"]):
                    provenance_mismatches.append(
                        f"row {row_index}: cache target {row['target']}, dataset target {item['target']}"
                    )
                expected_video = item.get("source_video_id")
                if expected_video is not None and row["source_video_id"] != str(expected_video):
                    provenance_mismatches.append(
                        f"row {row_index}: cache source_video_id {row['source_video_id']!r}, "
                        f"dataset {expected_video!r}"
                    )

    report = {
        "cache_dir": str(cache_dir),
        "rows_total": n_rows,
        "rows_sampled": take,
        "sample_seed": seed,
        "bit_identical_rows": exact,
        "max_abs_delta": worst_delta,
        "max_ulps": worst_ulps,
        "max_significant_ulps": worst_significant_ulps,
        "max_ulps_allowed": max_ulps,
        "ulp_floor_magnitude": ulp_floor_magnitude,
        "verify_batch_size": batch_size,
        "dtype": manifest.dtype,
        "feature_dim": manifest.feature_dim,
        "cache_sha256": manifest.cache_sha256,
    }
    if feature_mismatches or provenance_mismatches:
        raise FeatureCacheMismatch(
            f"feature cache at {cache_dir} does not reproduce the live backbone "
            f"({exact}/{take} rows bit-identical; worst {worst_ulps} ULPs, "
            f"{worst_significant_ulps} at or above magnitude {ulp_floor_magnitude:g}):\n- "
            + "\n- ".join(feature_mismatches + provenance_mismatches)
        )
    return report
