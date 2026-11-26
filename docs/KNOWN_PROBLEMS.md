# Known problems — SHAR

> Things that are **broken or surprising right now**, with the workaround.
>
> This is deliberately separate from `RISKS.md`. RISKS is about anticipated futures.
> This file is about the present: if you hit one of these, the answer is here.
>
> Delete an entry when it stops being true. Add one the moment something bites you —
> a problem rediscovered twice belongs in this file.

| # | Problem | Workaround | Found |
|---|---|---|---|
| K-01 | `pytest` is not installed in `.venv`, so `python -m pytest` fails outright. Docs and habits assume it exists. | Run the suite with `./.venv/bin/python -m unittest discover -s tests -p "test_*.py"`. Current baseline (2026-11-11): 1,007 tests, 6 skipped. | 2026-08-25 |
| K-02 | `/workspace/fast-dgx-storage/<account>` mounts on the DGX and looks usable, but every write fails with EACCES. | Do not target it. Durable artifacts go on `/home`; fast scratch is a pod-local `emptyDir`. Needs an admin change to become usable. | 2026-08-25 |
| K-03 | A Kyverno `require-gpu` ClusterPolicy rejects any pod not requesting a GPU, and the account's quota is exactly one `mig-1g.18gb`. Even a storage probe burns the only slice. | Request `nvidia.com/mig-1g.18gb: 1` in every pod spec, including non-GPU ones, and never expect two pods to coexist. Batch questions into a single pod. | 2026-08-25 |
| K-04 | D-34 records the official corpus as "all 320×240 at 30 fps". The geometry is right; the fps is not — 1,939 videos are 30.0, 10 are 25.0, and 1 is 29.97. | Frame index converts to a timestamp **per video only**, as `source_frame_index / reported_fps` using that video's own fps from `videos.csv`. Never treat stride 5 as a global 6 fps. Correction recorded in `EVIDENCE_REGISTER.md`; D-34 is dated and not rewritten. | 2026-08-25 |
| K-05 | `cv2.imencode`/`imwrite` always interpret their input as BGR and swap internally. Handing them an already-RGB array silently swaps red and blue — every structural check still passes. | Encode JPEG with Pillow (`Image.fromarray(arr, mode="RGB")`) after `cv2.cvtColor(..., COLOR_BGR2RGB)`. Covered by a pure-red test in `tests/test_p1c_video_extractor.py`. | 2026-08-25 |
| K-06 | `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` are container metadata and are frequently wrong for H.264. | Never use the reported count to decide how many frames to read — read to EOF and record the decoded count as ground truth. Both are stored as `reported_*` fields to keep the distinction visible. | 2026-08-25 |
| K-07 | Git write operations (`commit`, `checkout -b`) are blocked by the AI assistant's permission layer in this environment, not by project policy. | The owner runs them, e.g. by typing `! git commit -m "..."` in the session, or adds a Bash permission rule in settings. Committing locally is allowed and encouraged; pushing is not. | 2026-08-25 |
| K-08 | ~~Pod template had no scratch volume or staging step.~~ **RESOLVED 2026-08-25.** `pod-job.template.yaml` now mounts a node-local `emptyDir` at `$SCRATCH_MOUNT` (default `/scratch-emptydir`, `SCRATCH_SIZE_LIMIT` default 200Gi) and stages `$STAGE_FROM` into it before the job command, failing closed on a missing source or a size mismatch. | Set `STAGE_FROM=/path/to/blob` when invoking `run_pod_job.sh`; leave it unset for jobs that touch no bulk data. Gotcha worth keeping: `envsubst` rewrites bare `$VAR` as well as `${VAR}`, so the YAML placeholders use distinct `*_SUBST` names and the runner restricts `envsubst` to exactly those. Do not rename them to match the shell variables — that inlines values into the script and breaks on any path with a space or quote. | 2026-08-25 |
| K-09 | D-36 retired the D-27 owner-registration gate, but the **code still enforces it**: `_load_config` in `scripts/calibrate_p1c_frame_loader.py` refuses any config whose `experiment.status` is not `REGISTERED_CALIBRATION_READY`, and `configs/experiments/p1c/P1C-FRAME-GAP-CALIBRATION-005.yaml` still carries an "OWNER DECISION REQUIRED / CANNOT LAUNCH AS WRITTEN" header. | Either relax the status check to a warning or set the config's status directly — no human promotion ceremony is required any more. Not yet done because that path is paused under D-34 and the change was not worth making mid-extraction. | 2026-08-25 |
| K-10 | `<head-node>` is shared across every tenant using the GPUs and the VM. Heavy work there throttles other users and flags the account with administrators — its 128 cores are **not** a resource this project may draw on. | Run all compute in a pod tied to the account's `mig-1g.18gb` slice (D-39). Never propose the head node's core count as a speed option, even for CPU-only preprocessing. Note a pod cannot be shared and every pod consumes the only slice (K-03), so pod work is strictly serial. | 2026-08-25 |
| K-11 | A video stem is not a unique key: 1,950 files hold only 1,900 distinct videos, because all 50 `z_Normal_Videos_event` files are byte-identical duplicates of `Testing_Normal_Videos_Anomaly` files. | Key every corpus lookup by `(source_directory, stem)`, never the stem alone. Already done in `data/video_manifests.py` and `data/video_dataset.py`. | 2026-08-25 |
| K-12 | `--protocol anomaly_detection` refuses to build manifests, by design: its train/test label vocabularies differ (`Training_` vs `Testing_Normal_Videos_Anomaly`). | Use `--protocol action_recognition --fold N`, which yields one clean 14-label vocabulary. To use AD for classification, first register a label-mapping decision merging both Normal directories, then relax the check. The tool will not invent that mapping. | 2026-08-25 |
| K-13 | The pod memory quota is **16Gi**. A request above it is refused at admission (`exceeded quota: user-quota ... limited: limits.memory=16Gi`), not queued. | Pass `MEM=16Gi` or less to `run_pod_job.sh`. Combined with K-03 (one `mig-1g.18gb`) and CPU=8, that is the full envelope for any job. | 2026-08-25 |
| K-14 | `du -sb` on a directory includes the directory inode's own size, and NFS and xfs report it differently. Comparing `du -sb` across the two filesystems makes a byte-perfect copy look corrupt. | Compare the summed size and count of **regular files** (`find -type f -printf '%s\n'`), never directory totals. Fixed in `pod-job.template.yaml`; measured on 2026-08-25 when a correct 84 GiB stage reported a 28-byte "mismatch" that was entirely directory inodes. | 2026-08-25 |
| K-15 | Inside a pod, **file-backed page cache is charged to the container's memory cgroup**. Hashing the 84.3 GiB blob for its integrity check drove `memory.current` from 0.28 GiB to 15.99 GiB of a 16 GiB `memory.max`; staging the blob into scratch did the same, and both left the pod sitting at its ceiling before the job did any real work. Nothing in any process's RSS (1.3 GiB) showed it. | `_sha256_file` hints `POSIX_FADV_DONTNEED` every 64 MiB while reading, and the pod template drops the staged tree's cache after `cp` (`sync` first -- DONTNEED only drops *clean* pages). Shared helpers live in `core/pagecache.py`; any future full pass over bulk data needs the same treatment. **Correction:** this was originally recorded as the cause of the CALIBRATION-006 OOM. It was not -- it made the pod fragile and masked the real cause, which is K-16. Both fixes are kept: measured peak fell 15.99 -> 1.28 GiB for the hash and 16.00 -> 0.01 GiB for staging. | 2026-08-25 |
| K-16 | `build_p1c_frame_dataloader` sets `multiprocessing_context="spawn"`, so a Dataset is **pickled into every worker** rather than inherited copy-on-write. `VideoBlobReader` held a `np.memmap` over the 84.3 GiB blob, and `np.memmap` inherits `ndarray` pickling, which serialises the mapped bytes **by value** -- building the loader tried to push 84.3 GiB through a pickle buffer. This, not page cache, is what OOM-killed CALIBRATION-006: anon memory climbed ~1 GiB/s from "model ready" until the cgroup limit, with no batch ever produced. | `VideoBlobReader.__getstate__` drops both the memmap and the 85 MiB frame index; each is reopened lazily in the child, and the digest is deliberately not re-verified there (the parent already did, and re-hashing 84.3 GiB per worker is worse than the risk). `P1CVideoFrameDataset` also stores rows as parallel NumPy arrays plus a 476-entry per-video table instead of 391,014 dicts. Measured: the pickled dataset is **8.29 MiB**, built in 9 ms. Anything else that crosses into a worker under spawn needs checking the same way -- `pickle.dumps(dataset)` size is the test. | 2026-08-25 |
| K-17 | A pod sees the **whole node's** CPU count, not its cgroup allocation: `os.cpu_count()` reports 224 and torch sizes its thread pool to 112 while `cpu.max` grants 8 cores. Harmless for memory (measured: threads cost ~0 RSS) but it means every recorded `torch_threads` figure is a node property, not a run parameter, and oversubscribes the slice. | Not currently biting -- the calibration is GPU-bound with a 0.17% loader share, so thread oversubscription costs nothing measurable. Worth setting `OMP_NUM_THREADS`/`torch.set_num_threads` to the cgroup's CPU quota before any CPU-bound job, and worth remembering that it is *not* a licence to use 112 cores (D-39, K-10). | 2026-08-25 |
| K-18 | A pod's `/dev/shm` defaults to **64 MiB**, and PyTorch DataLoader workers return every batch through shared memory. The bound is roughly `batch_bytes x prefetch_factor x num_workers`, so a 256x3x64x64 float32 batch (12.6 MiB) across 6 workers prefetching 2 each wants ~151 MiB and the workers die with `RuntimeError: DataLoader worker ... killed by signal: Aborted`. The traceback names `signal_handling.py` and looks like a dataset bug; it is not. | `pod-job.template.yaml` mounts a `medium: Memory` emptyDir at `/dev/shm`, sized by `DSHM_SIZE_LIMIT` (default 4Gi, which covers 224x224 at batch 256: a 154 MiB batch, ~1.85 GiB in flight). Note this is tmpfs, so it is **real RAM inside the 16Gi limit and is not reclaimable** (unlike the page cache in K-15) -- size it for the actual in-flight batches rather than padding it. Measured 2026-08-25: at 64 MiB, 64x64 b64 and 112x112 b64 passed while 64x64 b256 and everything at 224x224 failed, including in a completely fresh process. | 2026-08-25 |
| K-19 | Running many calibration points in **one** interpreter to reuse the blob-verification memo also accumulates DataLoader workers: `persistent_workers=True` keeps 6 alive per loader, they are not released between points, and the 6th point of the 009 sweep OOM-killed the pod on host memory even though the same point succeeded alone. | Use `scripts/sweep_p1c_resolution_batch.py --isolate` for grids whose later points are memory-heavy, accepting ~112 s of re-verification per point; or run the expensive tail points individually. The in-process default is right for cheap grids, where hashing once instead of N times dominates. | 2026-08-25 |
| K-20 | Every video-blob config except `P1C-VIDEO-FRAME-BASELINE-011-AR-FOLD1` registers `data.dataset_roots` under the key `ucf_crime_official_videos`, but the manifests those configs read record `source_dataset: ucf_crime_official_video_frames` (`data/video_manifests.py:46`). 18 configs carry the mismatch. It has never fired because the video-blob loader path resolves pixels through the extract closure and never consults `dataset_roots`, and because `core.p1c_readiness` only walks manifests against `dataset_roots` for configs at `REGISTERED_FULL_RUN_READY` — a status none of them had until 2026-08-27. | Fixed in BASELINE-011 only. The other 18 are deliberately left alone: they are calibration configs whose `config_sha256` is recorded inside completed run artifacts, so editing them would invalidate the provenance of results already on disk for a field those runs never read. Fix the key in any *new* config, and fix an old one only if it is ever promoted to a full run — which would mean a new experiment ID anyway. | 2026-08-27 |
| K-21 | `P1C-VIDEO-FRAME-BASELINE-011-AR-FOLD1.yaml` (introduced in commit `efc5296`) was drafted without a `data.video_frames` block. In `_make_datasets`/`_datasets`, both `extract_closure_path` and `pack_closure_path` resolved to `None`, and the branch order fell through to `P1CFrameDataset` with `verify_digests: true`. The manifest's rows point at the original MP4s, so the run re-hashed a ~50 MB source video once per manifest row — 391,014 rows against a 97.7 GiB corpus, 39 TB of reads, 7h25m of GPU-slice time burned on 2026-08-27 before it was caught. Nothing refused; it just got very slow. | `data/pixel_source.refuse_if_blob_manifest_lacks_pixel_source` reads a manifest's first row (`data/video_manifests.SOURCE_DATASET` is written into every row, so one row settles the question) and refuses to build datasets if the manifest is blob-derived but neither `data.video_frames` nor `data.packed_frames` is registered. Both runners call it before branch selection and now `log_phase` the resolved `loader_mode` (`video_blob`/`packed_blob`/`per_file`) immediately before constructing datasets. Tests in `tests/test_pixel_source.py`. | 2026-08-27 |
| K-22 | Four more registered-but-unread (or unread-vs-inconsistent) config fields, same root cause as K-21 and the one `core/precision.py`'s docstring already names ("the precision field was previously validated and then ignored"): a config can say one thing while the code does another, and nothing checked the two agreed. (1) `fixed_protocol.augmentation` was registered in every P1C config (`name`, `rationale`) but validated and read by no code — a config registering anything but the no-op would be silently ignored. (2) `training.throughput` absent silently resolves to the fully-disabled policy (measured 1.745x slower) with no warning, for configs already at `REGISTERED_FULL_RUN_READY`. (4) `evaluation.primary_metric` and `training.selection_rule` (the field that actually drives checkpoint selection) were never checked against each other and could silently drift apart. | (1) `core/p1c_readiness.py` now fails closed unless `augmentation.name == "none"` plus a non-empty rationale, with an error naming that augmentation is not implemented and would otherwise be silently ignored. (2) `core/p1c_readiness.py` now requires `training.throughput` to be registered when `status_ready` (`REGISTERED_FULL_RUN_READY`); still optional for calibration configs. (4) `core/p1c_readiness.py` now refuses `evaluation.primary_metric != macro_f1` when `training.selection_rule == best_validation_macro_f1`. **(3) `loss.class_balance` was left alone, deliberately**: the runner hardcodes unweighted `nn.CrossEntropyLoss`, and the validator already forces `class_balance == "none"`, so this one already fails closed — that is the correct behaviour for the first (unweighted-control) research run, per owner decision 2026-08-27, and comments were added at both the validator and the runner's loss construction rather than "fixing" it into working. Tests in `tests/test_p1c_readiness.py`. | 2026-08-27 |
| K-23 | **A pod runs from the live working tree, so editing the repo mid-run can kill the run.** `scripts/dgx/_pod_common.sh` mounts the repository by `hostPath` and the runner imports from it directly. DataLoader workers respawn per epoch and re-import module source, so a dataset object constructed by the OLD `__init__` can find itself executing the NEW `__getitem__`. On 2026-08-28 the augmentation commit (`2c5f16e`, 07:17 UTC) landed while `P1C-VIDEO-SCREEN-013-LR3E5-AR-FOLD1` was running; at epoch 2 its workers raised `AttributeError: 'P1CVideoFrameDataset' object has no attribute '_augmentation_run_seed'` and the run died 34 minutes in, two epochs completed. Only the pixel blob is staged into the pod's emptyDir; the code is not. | Until pods run from an immutable snapshot: **do not commit or edit importable code while a pod is running.** Docs are safe (never imported). Check `kubectl get pods` before touching `core/`, `data/`, `models/`, or `scripts/`. The durable fix is to stage a `git archive` of the launch commit into the pod's emptyDir and run from that, which would also let `run_manifest.code_revision` record a real commit SHA instead of its current "local working tree; commit is intentionally not assumed". Failed attempt retained at `results/p1c/p1c-video-screen-013-lr3e5-ar-fold1-s0-20260828T065557Z.FAILED`. | 2026-08-28 |

## K-24 — training runs do not enable deterministic algorithms (2026-08-29)

`scripts/run_p1c_frame_baseline.py` calls `_set_seed` (Python/NumPy/torch RNGs) but never
`torch.use_deterministic_algorithms`, and `core/reproducibility.seed_everything` — which does — is
not called by any P1C run path. So every P1C training run executes on cuDNN's default, partly
non-deterministic kernels.

**This is not currently causing a wrong result**, and is recorded rather than fixed for that reason:
per-element jitter is ~1e-4 and washes out of metrics aggregated over 37,600 validation frames.
CONFIRM-015 seed 0 reproduced SCREEN-012's macro-F1 to four decimals across all 21 epochs on an
independent launch a day later, so reproducibility holds at the level the project reports.

**What it cost:** three rewrites of the B-1 cache verification and ~50 minutes of slice time, because
the check demanded bit-exactness that the producing pipeline never promised. The cache was correct
throughout.

**The rule it implies:** before writing any numerical equality check against this pipeline's output,
establish what determinism the producing code actually guarantees. Do not assume bit-exactness on GPU.

**If it is ever fixed**, note that enabling strict determinism may change throughput (some cuDNN
algorithms are excluded) and would therefore need re-measuring against D-41's registered policy — it
is not a free flag flip, which is the other reason it is recorded rather than done in passing.

## K-25 — `LinearSVC` is single-threaded and ~20x slower than `LogisticRegression` here (2026-09-01)

On the B-1 cache (391,014 x 2048 float32, 14 classes, standardized), measured in the B-9 sweep:

| probe | solver | C | fit time |
|---|---|---|---|
| `LogisticRegression` | lbfgs, multinomial, `max_iter=200` | 0.01 | **174 s** |
| `LinearSVC` | liblinear, `dual=False`, `max_iter=2000` | 0.01 | **3,333 s** |
| `LinearSVC` | liblinear, `dual=False` | 1e-4 | **4,044 s**, i.e. *slower* than C=0.01, not faster |

Two things bite. **liblinear is single-threaded** — it does not use BLAS for its coordinate/Newton
steps, so `OMP_NUM_THREADS=4` buys nothing and a "4-core" budget is really a 1-core budget. And
**lowering C did not speed it up**, which is the opposite of the usual intuition that stronger
regularisation converges sooner; at C=1e-4 the squared-hinge primal took longer than at C=0.01.

**Workaround.** Budget a `LinearSVC` point at roughly 1 h, not the ~3 min a logistic point costs,
and never queue an SVM C-sweep expecting logistic-like turnaround. If a hinge-loss probe is
genuinely wanted at sweep scale, `SGDClassifier(loss="hinge")` is the cheap substitute — a
different estimator, so it must be reported as one, not as `LinearSVC`.

**Consequence for B-9 as run:** the `linear_svm` arm covers two C values against the logistic arm's
eight, and **its optimum was never located** — the SVM scored 0.2880 at C=1e-4 against 0.2745 at
C=0.01, i.e. still improving at the low edge of the range. Every stop was deliberate and is recorded
in `results/p1c/p1c-probes-b9-ar-fold1/run-d-svm.log`: C=1 was killed ~10 min in, and C=0.001/C=0.1
were skipped, both because ~1 h per fit is not worth corroboration the arm had already supplied.
So the SVM figure is a **lower bound**, not that probe's best. B-9's conclusion rests on the logistic
arm, whose maximum is bracketed on both sides; and a better-swept SVM could only move a classical
probe *closer* to the trained head, which is the direction that reinforces D-43 rather than
threatening it.

## K-26 — the feature-cache verification tolerance is marginal for ResNet, and it halted folds 2-4 (2026-09-03)

`shar-folds234-cache-r2` built fold 2 completely (ResNet and DINOv2, both splits) and then
**failed on fold 3's ResNet train cache**: 480 of 512 sampled rows bit-identical, with two rows
just over the gate --

    row 215637: |delta| 2.930e-03 at element 1426 vs allowed 0.001 + 0.001*1.310 = 2.310e-03
    row 400397: |delta| 1.465e-03 at element 1079 vs allowed 0.001 + 0.001*0.4375 = 1.438e-03

Both overshoot by less than one part in 1e-3. At magnitude 1.31 one float16 ULP is ~9.8e-4, so
2.930e-03 is about **3 ULPs** -- the ordinary cuDNN convolution-algorithm variance K-24 already
records, not a wrong cache.

**Why it passed on fold 1 and fails here.** The gate is per element, `|a-b| <= atol + rtol*|b|`.
Fold 1's worst delta was *larger* (3.906e-3) but landed on an element with a large `|b|`, so its
allowance was larger too. Nothing about fold 3 is worse; the failure is where the deltas happened
to fall. The margin was always thin and fold 1 did not reveal it.

**Note which backbone this is.** DINOv2 verified **512/512 with 0 ULPs** on every split of folds 1
and 2. ViT features come from matmuls with a stable reduction order; ResNet's convolutions do not.
This is a ResNet-path problem only.

**NOT fixed, and deliberately not fixed by loosening the number.** "Raise the tolerance until it
passes" is the exact move the B-1 verification lesson warns against, and the honest options differ
in what they claim:

1. Express the gate in measured float16 ULPs rather than a hand-picked `atol` -- what
   `core/feature_cache_verify.py`'s docstring already argues is the meaningful unit near zero.
2. Enable `torch.use_deterministic_algorithms` for cache builds only (not training), making the
   ResNet path reproducible at some throughput cost -- this is K-24's open item.
3. Raise `atol` with a measured justification recorded here.

Option 1 or 2 is right; option 3 alone is not. Needs an owner decision before folds 2-4 resume.

**State on disk right now:** fold 2 complete for both backbones; fold 3 ResNet *train* built but
unverified (the build wrote the manifest before the verifier raised); fold 3 validation and all of
fold 4 not started. The job skips any cache with a manifest, so **fold 3's ResNet train cache must
be deleted before a rerun**, or it will be skipped while still unverified.

**Resolved 2026-09-03: option 1 implemented.** `core/feature_cache_verify.py` now gates on
`max_ulps` float16 steps (default 8) at `max(|b|, ulp_floor_magnitude)` (default 1.0) instead of
`atol + rtol*|b|`. 8 ULPs covers the largest measured real-magnitude disagreement (6 ULPs, above)
with modest headroom; the number is not meant to be tuned tighter, since this gate exists to catch
wiring bugs that move values by orders of magnitude more than a handful of ULPs, not to shave the
last bit off cuDNN jitter. `scripts/build_p1c_feature_cache.py` and
`scripts/verify_p1c_feature_cache.py` take `--max-ulps`/`--ulp-floor-magnitude` in place of
`--atol`/`--rtol`. Fold 3's unverified ResNet train cache was deleted 2026-09-03 before the rerun, so the
resume logic cannot skip it. Folds 1 and 2 were verified under the OLD atol/rtol gate and are
re-verified under this one in the same job, so all four folds are held to a single standard.

**Independently checked against the measured failures before the rerun** (allowance at
`max_ulps=8`, floor 1.0):

| case | \|b\| | delta | allowed | result |
|---|---|---|---|---|
| K-26 fold 3 row 215637 | 1.310 | 2.930e-03 | 7.812e-03 | passes |
| K-26 fold 3 row 400397 | 0.4375 | 1.465e-03 | 7.812e-03 | passes |
| fold 1 worst (passed before) | 4.0 | 3.906e-03 | 3.125e-02 | passes |
| near-zero jitter, 839 raw ULPs | 0.0 | 5.000e-05 | 7.812e-03 | passes -- the floor's whole purpose |
| gross: perturbed by 0.5 | 1.310 | 5.000e-01 | 7.812e-03 | **fails**, 64x over |
| gross: perturbed by 0.1 | 0.4375 | 1.000e-01 | 7.812e-03 | **fails**, 12.8x over |
| 9 ULPs at magnitude 1 | 1.0 | 8.789e-03 | 7.812e-03 | **fails** -- boundary is exact |

The gate accepts cuDNN jitter and still rejects a 0.1 perturbation by more than an order of
magnitude, so it has not been loosened into uselessness.

## K-27 — `verify_p1c_feature_cache.py` hardcoded ResNet50 and killed two fold-build jobs (2026-09-07)

B-2 added `backbone.family` dispatch (`_BACKBONE_LOADERS` / `_load_backbone`) to
`scripts/build_p1c_feature_cache.py` but **not** to `scripts/verify_p1c_feature_cache.py`, which
kept calling `_local_resnet50` directly. Verifying a DINOv2 cache therefore loaded ViT weights into
a torchvision ResNet50 and raised `local weights are not an exact torchvision ResNet50 state
dictionary`.

**Cost: two pod runs.** Both `shar-folds34-ulpgate` (2026-09-03) and `shar-folds34-ulpgate-r2`
(2026-09-07) died at the third re-verification -- fold 1 DINOv2 -- after completing fold 1's two
ResNet checks. The first died with its console log, and its disappearance was wrongly left open as
"errored, evicted, or deleted"; the second is explained because the job mirrored its output to
`results/p1c/pod-logs/folds34-ulpgate.live.log` on the NFS mount as it ran. **A pod's console log
dies with the pod; write the live log to /home for any job worth diagnosing.**

**Fixed.** The verifier now calls the builder's `_load_backbone`. Three regression tests in
`tests/test_dinov2_backbone.py` assert the registry covers exactly `core.p1c_readiness`'s
`BACKBONE_FAMILIES`, that an unregistered family is refused by name, and that the verifier source
contains no hardcoded `_local_resnet50`.

**The review lesson.** The B-2 agent's report said it added dispatch to the builder, and I checked
the builder. I never exercised the *verifier* against a DINOv2 cache, so a second call site kept a
single-backbone assumption the rest of the code had dropped. **After widening a registry, grep for
every call site rather than checking the one the report names.**

## K-28 — the feature caches were built with `cudnn.allow_tf32=True`; CPU recomputation disagrees by up to 4e-2 on some frames (2026-09-11)

`torch.backends.cudnn.allow_tf32` defaults to True. `core.feature_cache.build_feature_cache` runs the
backbone in float32 with no autocast, but on the GPU cuDNN executes DINOv2's patch-embedding Conv2d in
TF32 (10-bit mantissa). On CPU the same conv runs in full float32. That ~1e-3 error at the first layer
propagates through 12 transformer blocks and, for some inputs, amplifies to 1-4e-2 in the pooled vector.

Measured 2026-09-11 while smoke-testing the targeted Shooting-token extraction on CPU against the
registered DINOv2 caches: Shooting015 and Shooting002 matched to 2.6e-3 / 3.6e-3, but Shooting004 and
Shooting007 disagreed by 1.5e-2 / 4.2e-2 at the MEDIAN, with rows landing on the correct frames (nearest-
neighbour confirmed). The same extraction on the GPU matched the cache at median 2.05e-3, 94.6% of 14,415
rows within 5e-3 -- so the caches are correct and internally consistent (fold caches agree to 0.000 on
shared videos); the CPU check was the imprecise side.

**Second finding from the same run:** the GPU-vs-GPU comparison has a tail the 512-row ULP verification
did not sample. 8 of 14,415 rows (0.06%) exceed 2e-2, max 9.5e-2, all in videos where the cache was
built at batch 64 and the extraction ran at batch 32 -- the K-24 batch-composition mechanism. A 512-row
sample has ~25% odds of catching one. The gate is not wrong; the tail is fatter than the sample showed.

**Consequences.** (1) Any CPU-side check against these caches must budget ~5e-2, not fp16 tolerance,
or run on the same device. (2) Bit-level comparisons must state batch size, since it changes cuDNN's
algorithm choice. (3) Neither affects a LOVO logistic probe -- a 0.06% tail at a few percent in pooled
space does not move that result -- so no experiment is invalidated. Not fixed: pinning
`cudnn.allow_tf32=False` for cache builds would need every cache rebuilt to be comparable, ~4 h of
slice, for a precision the downstream heads never see (K-24, the caches are float16 anyway).

**Also recorded here because it cost slice time:** my first extraction launch decoded JPEGs in a bare
Python loop at 2.4 frames/s (loader-bound -- the SCREEN-014 / B-1 trap again, ~100 min projected vs the
10 estimated; killed at 7 min), and the second accumulated 5.7 GB of patch tokens in RAM before a
`np.concatenate` doubled it inside a 16 GiB pod (OOMKilled after writing the first artifact). Third
run streamed to a preallocated memmap. Measure throughput on the smoke, not by scaling from a
different job.

## K-29 — `transformers` 5.17 silently random-initialises VideoMAE's attention biases (2026-09-15)

`VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")` prints a load report
listing 36 keys MISSING and 36 UNEXPECTED, then returns a model anyway. The checkpoint stores
`encoder.layer.{i}.attention.attention.q_bias` and `.v_bias` -- VideoMAE's original
parameterisation, with the key bias fixed at zero -- while transformers 5.17's module expects
`query.bias`, `key.bias`, `value.bias`. Nothing maps them, so all 36 are freshly initialised.

Measured: the mis-loaded model's pooled features have **cosine 0.469** with a correct load on the
same input. Every downstream number would have been wrong, and nothing in the pipeline would have
noticed -- a cache verifies against a live recompute of the *same* model, so a consistently wrong
backbone verifies perfectly.

**Fix:** `models/videomae_backbone.py` loads the pinned safetensors file itself and remaps exactly
-- `q_bias -> query.bias`, `v_bias -> value.bias`, `key.bias = 0` per layer, `classifier.*`
dropped, `fc_norm` kept as the readout norm -- with `strict=True`, so a format drift fails loudly
instead of silently. Verified: layer-0 `|q_bias|` 0.34, `|k_bias|` exactly 0, strict load clean.

**General rule, third time this project has learned it (K-24, K-28, now this):** a library
"loading successfully" is not evidence the weights are right. Read the load report; if any key is
MISSING, the model is not the one you downloaded.
| K-30 | **`core.p1c_extraction_io.finalize_streamed_array` hashed its output with `Path.read_bytes()`, which OOM-killed the P3A patch-token cache build after the entire GPU pass had succeeded.** The helper was written for the 5.7 GB Shooting-token extraction, where reading the whole file fits; the P3A cache is **32.5 GB** and a 16 GiB pod cannot hold it. Pod `shar-patchcache-20260925093924` died at exit 137 / `OOMKilled` with 82,679 of 82,679 frames already written to disk, 47 minutes of slice spent. | **Fixed 2026-09-25.** `finalize_streamed_array` now calls the project's existing chunked `core.p1c_readiness.sha256_file` (1 MiB chunks) -- the helper already existed and simply was not being used, which is the rule-of-two failure underneath the memory one. Also added `scripts/build_p1c_patch_cache.py --finalize-only`, a recovery path that hashes, verifies and manifests an output root whose arrays are already complete, so a finalisation failure never costs the GPU pass again; it refuses a root whose row counts disagree or whose last row is all zeros. The P3A cache was recovered this way -- no re-run. **Lesson for any future streamed artifact: size the finalisation step, not just the write.** | 2026-09-25 |
| K-31 | **`scripts/run_p1c_head_comparison.py::_run_one` builds the head (`_build_head`) before any seeding, so the head's initial weights are not controlled by `--seeds`.** The seed reaches only the data-loader generator and the in-training `set_local_seed` (`core/p1c_head_training.py:248`), which runs after construction. Measured by the oracle-probe agent: two processes gave different initial weight sums (-11.99 vs -1.86); re-running fold 1 seed 0 of the headline head gave 0.5013 against the saved 0.5348. Per-run drift against the saved JSONs is 0.003-0.034; the mean of 12 re-runs is 0.4626 vs the registered 0.4599. | **Not fixed; the registered figures (D-46, D-53, D-54, REMEDIES_TRIED) stand as measured.** The headline's +/- already contains this init noise, so no conclusion changes, but the runs are not bit-reproducible from the recorded seed (rule 5). Any rerun that must reproduce a saved number needs `set_local_seed(seed)` before `_build_head`; the oracle probe does this in its own script. Fixing the shared runner would change what a seed means for future runs, so it is a deliberate decision, not a quiet edit. | 2026-10-07 |
| K-32 | **`.git/objects/pack` holds 366.1 GiB of stale `tmp_pack_*` garbage (701 files, dated 2026-10-30 to 2026-11-02); the real pack is 58 MiB.** `git count-objects -v` reports them as garbage and `git fsck --connectivity-only` passes without them, so the repository does not need them. Cause not established. They inflate `.git` to 368 GB, so a naive copy of the repo (998 GB) does not fit a 764 GB backup disk. | Exclude `.git/objects/pack/tmp_*` from any copy. Not deleted: removal is irreversible and needs the owner's OK, with no git process running. | 2026-11-06 |
| K-33 | **(Numbered K-32 on the `video-test` branch; renumbered at the 2026-11-25 branch unification, where both branches had used K-32.)** **The first demo's inference path (`core/demo_inference.py`, 2026-11-03) skipped the JPEG step the evaluated pipeline had.** Every cached feature came from the D-35 frame blob (JPEG quality 95), but the demo fed raw decoded frames to DINOv2. On fold 1's 56 validation videos this gave median frame cosine 0.998 (minimum 0.912) against the cache, and macro-F1 of 0.5257 / 0.5333 / 0.4931 against the recorded 0.5257 / 0.5162 / 0.5269 (predictions 100% / 98% / 93% identical). The earlier export check could not see it, because it scored the heads on cached features, never on decoded video. | **Fixed in `registered_image`:** frames now take the same JPEG round trip; the same check gives minimum cosine 0.99997 and exactly the recorded macro-F1 with 100% identical predictions (`results/demo/self_check_fold1.json`, `scripts/dgx/test_video.sh --self-check`). Recorded results are unaffected, since they all use the cache. Only predictions shown by the upload page before the fix, including the 2026-11-03 values in `results/p1c/pod-logs/demo-smoke.log`, came from the old path. The demo also treated a renamed protocol video as unseen footage; it now recognises one by SHA-256 (`results/demo/protocol_digests.json`). | 2026-11-04 |
