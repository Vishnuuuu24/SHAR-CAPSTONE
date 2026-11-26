#!/usr/bin/env python3
"""B-9: fit classical probes on the B-1 feature cache and score them like a run.

Thin CLI. It resolves the cache identity through
`scripts.build_p1c_feature_cache._load_kwargs_for_config` -- the same single
definition the builder and the verifier use, so a probe cannot be served a
cache built under a different backbone, preprocessing or manifest -- and hands
the loaded features to `core.p1c_probes`, which does the fitting and scoring.

No GPU. The whole point of B-1 is that a head-only experiment is a matmul; per
D-39 this must stay off the shared node's other cores, so run it under
`taskset -c 0-3` with the BLAS thread environment pinned to match:

    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    NUMEXPR_NUM_THREADS=4 taskset -c 0-3 \\
    ./.venv/bin/python scripts/run_p1c_linear_probes.py --config ... --cache-root ...

`--extract-closure-path` exists because a config registers the pod emptyDir
path a *training* run reads pixels from, and that path does not exist off-pod.
A probe reads no pixels at all -- it needs the closure only for the blob
SHA-256 that identifies the cache -- and the same closure is kept durably at
`results/p1c/p1c-video-frames/extract_closure.json`. This is not a loosened
check: whatever digest the substituted closure declares is still compared
against the one the cache manifest recorded, so pointing at the wrong closure
fails the load rather than passing it.

Scope is inherited from the cache and is not negotiable here: AR fold 1,
validation split, inherited video-level labels (D-40, docs/PROJECT_RULES.md rule 1). Test
is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.feature_cache import load_feature_cache
from core.p1c_probes import FEATURE_SCALINGS, fit_and_score_probe, stride_rows_within_videos
from scripts.build_p1c_feature_cache import _load_kwargs_for_config
from scripts.run_p1c_frame_baseline import _load_config

PROBES = ("logistic_regression", "linear_svm", "knn")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the registered config the cache was built from")
    parser.add_argument("--cache-root", type=Path, required=True, help="directory holding train/ and validation/ caches")
    parser.add_argument("--probe", action="append", choices=PROBES, help="repeatable; default all three")
    parser.add_argument("--scaling", action="append", choices=FEATURE_SCALINGS,
                        help="repeatable; default standardize")
    parser.add_argument(
        "--train-stride", type=int, default=1,
        help="keep every Nth frame of each source video (1 = all 391,014); k-NN cost is linear in this",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None, help="directory to write one JSON report per probe into")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                        help="override a probe parameter, e.g. C=0.1 or n_neighbors=50")
    parser.add_argument(
        "--sweep", default=None, metavar="KEY=V1,V2,...",
        help="fit once per value of one parameter, e.g. C=0.01,0.1,1,10. Exists because "
             "loading and hashing the 1.6 GB cache costs minutes and a sweep should pay "
             "that once, not once per value",
    )
    parser.add_argument(
        "--extract-closure-path", type=Path, default=None,
        help="read the D-34/D-35 video-blob closure from here instead of the path the config "
             "registers (see module docstring)",
    )
    return parser.parse_args(argv)


def _parse_overrides(pairs: list[str]) -> dict:
    overrides: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--param expects KEY=VALUE; got {pair!r}")
        key, _, raw = pair.partition("=")
        overrides[key] = json.loads(raw) if raw[:1].isdigit() or raw in ("true", "false", "null") else raw
    return overrides


def _sweep_points(sweep: str | None) -> list[tuple[str | None, object, str]]:
    """`(key, value, filename_suffix)` for each fit, or one no-op point when unswept."""

    if sweep is None:
        return [(None, None, "")]
    if "=" not in sweep:
        raise ValueError(f"--sweep expects KEY=V1,V2,...; got {sweep!r}")
    key, _, raw = sweep.partition("=")
    values = [json.loads(value) for value in raw.split(",")]
    return [(key, value, f"-{key}{value}") for value in values]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _config, protocol, _run_class = _load_config(args.config)
    trainable = protocol["backbone"].get("trainable", "full")
    class_names = protocol["model"]["class_names"]
    overrides = _parse_overrides(args.param)
    if args.extract_closure_path is not None:
        protocol["data"]["video_frames"]["extract_closure_path"] = str(args.extract_closure_path)

    train_features, train_rows, train_manifest = load_feature_cache(
        args.cache_root / "train", **_load_kwargs_for_config(protocol, "train", trainable)
    )
    validation_features, validation_rows, _ = load_feature_cache(
        args.cache_root / "validation", **_load_kwargs_for_config(protocol, "validation", trainable)
    )

    keep = stride_rows_within_videos(train_rows, args.train_stride)
    if args.train_stride > 1:
        train_features = train_features[keep]
        train_rows = [train_rows[i] for i in keep]
    print(
        f"train {len(train_rows)} rows (stride {args.train_stride}), "
        f"validation {len(validation_rows)} rows, dim {train_manifest.feature_dim}, "
        f"scaling {args.scaling}, cache {train_manifest.cache_sha256[:12]}",
        flush=True,
    )

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    for probe in args.probe or list(PROBES):
        for scaling in args.scaling or ["standardize"]:
            for sweep_key, sweep_value, suffix in _sweep_points(args.sweep):
                label = f"{probe}-{scaling}{suffix}"
                print(f"\n--- {label} ---", flush=True)
                result = fit_and_score_probe(
                    train_features, train_rows, validation_features, validation_rows, class_names,
                    probe=probe, scaling=scaling, seed=args.seed,
                    bootstrap_iterations=args.bootstrap_iterations,
                    **(overrides | ({sweep_key: sweep_value} if sweep_key else {})),
                )
                ci = result.video_clustered_macro_f1_ci
                print(
                    f"frame macro-F1 {result.frame_metrics['macro_f1']:.4f} "
                    f"(clustered 95% CI {ci['lower_95']:.4f}-{ci['upper_95']:.4f}), "
                    f"accuracy {result.frame_metrics['accuracy']:.4f}, "
                    f"video-level macro-F1 {result.video_level_metrics['macro_f1']:.4f} "
                    f"over {result.video_count} videos ({result.tie_broken_video_count} tie-broken), "
                    f"fit {result.fit_seconds:.1f}s predict {result.predict_seconds:.1f}s, "
                    f"converged={result.converged}",
                    flush=True,
                )
                if args.output is not None:
                    payload = asdict(result) | {
                        "cache_sha256": train_manifest.cache_sha256,
                        "config": str(args.config),
                        "train_stride": args.train_stride,
                    }
                    path = args.output / f"{label}.json"
                    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
