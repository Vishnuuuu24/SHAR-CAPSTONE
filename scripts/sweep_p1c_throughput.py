#!/usr/bin/env python3
"""Sweep the registered throughput options at the chosen 224x224 / batch-64 point.

Sweep 009 fixed the science-bearing parameters: 224x224 at batch 64 under
bfloat16, measured at 400.5 img/s and a 19.33-22.69 h per-seed ETA. This sweep
holds all of those constant and varies only `training.throughput`, whose three
options change which CUDA kernels run without changing what they compute:

  channels_last     NHWC activation layout, which is what tensor cores consume.
                    An NCHW tensor is transposed on the way into every
                    convolution; setting the layout once removes that.
  compile           torch.compile / inductor: fuses pointwise work into the
                    convolution epilogues and cuts kernel launches.
  fused_optimizer   AdamW's fused CUDA implementation: one kernel for the whole
                    parameter update instead of one per tensor.

**Why the grid is shaped this way.** V1, V2 and V3 each differ from the V0
control in exactly one option, so each one's delta is attributable to that
option alone (docs/PROJECT_RULES.md: change one material factor per comparison). V4 and V5
are combinations, present because these options are not guaranteed to compose
additively -- channels_last changes the layout inductor then compiles against,
so V5 is not predictable from V1 and V3.

**These are not accuracy-comparable runs.** Different kernels accumulate in
different orders, so a throughput policy is deterministic within itself but not
bitwise identical to another. This measures throughput and device memory only.

**Why isolated subprocesses.** Each point pays a fresh 84.3 GiB blob
verification (~112 s) because the memo dies with the process. That is the price
of not repeating K-18/K-19, where persistent workers accumulated across
in-process points and destabilised the pool. It also matters more here than in
sweep 009: torch.compile installs process-wide inductor state and its caches
would otherwise leak across points and flatter whichever ran second.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_CONFIG = REPO_ROOT / "configs/experiments/p1c/sweep-009/P1C-VIDEO-SWEEP-009-R224-B64.yaml"
SWEEP_DIR = REPO_ROOT / "configs/experiments/p1c/sweep-010"
SWEEP_ID = "P1C-VIDEO-SWEEP-010"

# The anchor line the generated throughput block is inserted after. Chosen
# because it is the last field of `training` that every sweep-009 config
# already carries, and because a config that somehow lacks it is a config this
# sweep must refuse rather than silently emit unchanged.
_ANCHOR = "    precision: amp_bfloat16"


@dataclass(frozen=True)
class Variant:
    """One registered throughput policy plus what it is meant to isolate."""

    key: str
    channels_last: bool
    compile_mode: str
    fused_optimizer: bool
    rationale: str

    @property
    def experiment_id(self) -> str:
        return f"{SWEEP_ID}-{self.key}"

    def block(self) -> str:
        return (
            "    throughput:\n"
            f"      channels_last: {str(self.channels_last).lower()}\n"
            f"      compile: {self.compile_mode}\n"
            f"      fused_optimizer: {str(self.fused_optimizer).lower()}\n"
        )


# Ordered cheapest-to-riskiest: the control first so that a later point failing
# still leaves a reproduction of the sweep-009 number on record, and the
# compile points last because inductor is the option most likely to conflict
# with the deterministic-algorithms mode `cuda_strict` enables.
VARIANTS = (
    Variant("V0-CONTROL", False, "none", False,
            "reproduces the sweep-009 R224-B64 measurement with the throughput block "
            "explicitly all-off; anchors every delta below"),
    Variant("V1-CHLAST", True, "none", False,
            "single factor vs V0: NHWC activation layout only"),
    Variant("V2-FUSED", False, "none", True,
            "single factor vs V0: fused AdamW only"),
    Variant("V3-COMPILE", False, "none", False,
            "single factor vs V0: torch.compile only"),
    Variant("V4-CHLAST-FUSED", True, "none", True,
            "the two low-risk options together; expected to be roughly additive"),
    Variant("V5-ALL", True, "default", True,
            "all three; not predictable from V1/V3 because inductor compiles "
            "against whichever layout it is given"),
)
# V3 carries compile in its own field rather than in the tuple above so that
# the single-factor claim stays literally true in the generated header.
VARIANTS = tuple(
    Variant(v.key, v.channels_last, "default" if v.key == "V3-COMPILE" else v.compile_mode,
            v.fused_optimizer, v.rationale)
    for v in VARIANTS
)


def write_config(variant: Variant, *, base_text: str, output_dir: Path) -> Path:
    """Materialise one variant as a registered config."""
    if _ANCHOR not in base_text:
        raise ValueError(f"base config lacks the expected anchor line {_ANCHOR!r}")
    text = base_text.replace(
        "  id: P1C-VIDEO-SWEEP-009-R224-B64", f"  id: {variant.experiment_id}"
    ).replace(
        "  parent_id: P1C-VIDEO-PRECISION-CALIBRATION-007-BF16",
        "  parent_id: P1C-VIDEO-SWEEP-009-R224-B64",
    ).replace(_ANCHOR, _ANCHOR + "\n" + variant.block().rstrip("\n"), 1)
    header = (
        f"# {variant.experiment_id}: generated throughput variant.\n"
        f"#\n"
        f"# GENERATED by scripts/sweep_p1c_throughput.py from {BASE_CONFIG.name}.\n"
        f"# Exactly one field differs from that parent: training.throughput.\n"
        f"# Resolution (224x224), batch size (64), precision (bfloat16), manifests,\n"
        f"# digests, split, seed and device policy are all identical.\n"
        f"#\n"
        f"# Isolates: {variant.rationale}.\n"
        f"#\n"
        f"# These options change kernel selection, not mathematics. The run is\n"
        f"# deterministic within this policy but is NOT bitwise comparable to a run\n"
        f"# under a different one, so no accuracy from this sweep is comparable\n"
        f"# across its points. Calibration only: throughput and device memory.\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{variant.experiment_id}.yaml"
    path.write_text(header + text, encoding="utf-8")
    return path


def run_point_isolated(config_path: Path) -> dict[str, object]:
    """Run one variant in its own interpreter, recording a failure as a result."""
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/calibrate_p1c_frame_loader.py"),
         "--config", str(config_path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    seconds = time.perf_counter() - started
    if completed.returncode == 0:
        return {"status": "COMPLETED", "exit_code": 0, "seconds": seconds}
    tail = (completed.stderr or completed.stdout or "").strip().splitlines()
    return {"status": "FAILED", "exit_code": completed.returncode, "seconds": seconds,
            "error": "\n".join(tail[-15:])[:1500]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=[v.key for v in VARIANTS],
                        help="variant keys to run; default is the whole grid")
    parser.add_argument("--config-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--configs-only", action="store_true",
                        help="write the configs and exit without occupying the GPU slice")
    parser.add_argument("--summary", type=Path,
                        default=REPO_ROOT / "results/p1c/p1c-video-sweep-010-summary.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_text = BASE_CONFIG.read_text(encoding="utf-8")
    selected = [v for v in VARIANTS if v.key in set(args.variants)]
    unknown = set(args.variants) - {v.key for v in VARIANTS}
    if unknown:
        raise SystemExit(f"unknown variant(s): {sorted(unknown)}")

    print(f"sweep: {len(selected)} variant(s): {[v.key for v in selected]}", flush=True)
    paths = [write_config(v, base_text=base_text, output_dir=args.config_dir) for v in selected]
    if args.configs_only:
        for path in paths:
            print(f"  wrote {path.relative_to(REPO_ROOT)}", flush=True)
        return 0

    results = []
    for index, (variant, config_path) in enumerate(zip(selected, paths, strict=True), start=1):
        print(f"\n===== [{index}/{len(selected)}] {variant.key} "
              f"(channels_last={variant.channels_last}, compile={variant.compile_mode}, "
              f"fused={variant.fused_optimizer}) =====", flush=True)
        outcome = run_point_isolated(config_path)
        outcome.update(variant=variant.key, channels_last=variant.channels_last,
                       compile=variant.compile_mode, fused_optimizer=variant.fused_optimizer,
                       config=config_path.name)
        results.append(outcome)
        print(f"----- {outcome['status']} in {outcome['seconds']:.1f}s", flush=True)
        if outcome["status"] != "COMPLETED":
            print(outcome.get("error", ""), flush=True)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(
        {"sweep_id": SWEEP_ID, "base_config": BASE_CONFIG.name, "points": results},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nsweep summary: {args.summary}", flush=True)

    completed = sum(1 for r in results if r["status"] == "COMPLETED")
    print(f"{completed}/{len(results)} variants completed", flush=True)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
