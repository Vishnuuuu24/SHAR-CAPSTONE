# DGX batch execution

Ephemeral-pod batch runner for the Presidency DGX cluster. No Jupyter, no
NodePort Service, no long-lived pod: one command runs against the repository on
the mounted home export, and the pod is deleted when it finishes.

Cluster facts this depends on are recorded in `docs/DGX_COMPUTE_EVIDENCE.md`.

## Storage facts

`/home` is a 66 TB local XFS disk on `<head-node>`, exported read-write
over NFS:

```text
/home    <internal-ip>/16(rw,no_subtree_check,no_root_squash,async)
```

Two consequences:

- The export is restricted to the **Bright cluster fabric `<internal-ip>/16`**. The
  campus-facing `<campus-subnet>` address of the headnode is not in the export list
  and cannot be used as `NFS_SERVER`.
- `no_root_squash` means the container's root user is root on the export. Files
  the pod writes under your home directory are owned as written; nothing is
  squashed to `nobody`.

There is no PVC and no StorageClass in the namespace, so a dynamically
provisioned volume is not an option. The pod takes `<home>`
as a `hostPath`, relying on the worker nodes mounting that export as Bright
Cluster configures by default.

`type: Directory` is deliberate. If a worker ever stops mounting the export, the
kubelet refuses to start the pod instead of creating an empty directory and
running the job against no data. There is no second storage path and no
automatic switch: the failure is loud, and the fix is a deliberate change here.

A Kyverno admission policy also confines `hostPath` volumes on this account to
an explicit allowlist:

```text
<home>
/workspace/fast-dgx-storage/<user>
/shared-ro
/shared-workspace
```

The bare `/home` export is not on that list and is rejected at admission
(`validate.kyverno.svc-fail`) before the scheduler ever sees the pod. `HOST_PATH`
must be one of the allowed paths, not the export root.

### Confirm the mount

Run this before anything else. It is cheap and deletes itself:

```bash
scripts/dgx/run_pod_job.sh mounttest \
  'ls -la <home> && df -hT <home> && id'
```

Your home directory should be listed and `df` should report the 66 TB
filesystem. If the pod fails to start with a mount error, the workers are not
mounting `/home`; stop and resolve that before running anything else.

### 2. Create the virtual environment on home storage

The NGC image ships its own PyTorch and its own Python (3.10.12), neither of
which is what the project pins (`requires-python==3.12.*`,
`requirements/p0c-lock.txt`). Running on the container's own interpreter/torch
is an unrecorded environment deviation and invalidates the calibration as
evidence. See `docs/DECISIONS.md` D-31 for the full record of why each step
below is shaped the way it is; this is the condensed, working sequence.

**2a. Install a standalone Python 3.12** (one-time; the headnode's own
Python 3.12 cannot build a venv here — `python3.12-venv`/`ensurepip` is
missing and there is no admin access to install it). A relocatable
`python-build-standalone` build needs no root:

```bash
scripts/dgx/run_pod_job.sh py312setup \
  'PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20260814/cpython-3.12.14%2B20260814-x86_64-unknown-linux-gnu-install_only.tar.gz" &&
   PY_SHA="3297691ae34f75fed81ac424e040145fccb0bafe8e581cd5cadbddfa1c0766c0" &&
   curl -fsSL -o "$HOME/cpython312.tar.gz" "$PY_URL" &&
   echo "$PY_SHA  $HOME/cpython312.tar.gz" | sha256sum -c - &&
   mkdir -p "$HOME/py312" &&
   tar -xzf "$HOME/cpython312.tar.gz" -C "$HOME/py312" --strip-components=1 &&
   rm -f "$HOME/cpython312.tar.gz" &&
   "$HOME/py312/bin/python3.12" --version'
```

**2b. Create the venv inside the repo** (`$REPO_DIR/.venv`, matching
`.gitignore` and what `scripts/check_environment.py` expects — not
`$HOME/venvs/...`). Use `--copies`, not the default symlinked interpreter,
and bootstrap pip as its own separate step: both the symlink and a
combined-with-`ensurepip` creation were unreliable across this NFS mount.

```bash
scripts/dgx/run_pod_job.sh venvsetup \
  'rm -rf "$REPO_DIR/.venv" &&
   "$HOME/py312/bin/python3.12" -m venv --copies --without-pip "$REPO_DIR/.venv" &&
   "$REPO_DIR/.venv/bin/python3.12" -m ensurepip --upgrade'
```

**2c. Install the pinned packages.** Always set `PIP_CONFIG_FILE=/dev/null`
(the NGC image's baked-in pip config points an extra index at
`pypi.ngc.nvidia.com`, which does not resolve from this account's pods) and
always pass `--index-url` explicitly — a bare `pip install -r
requirements/p0c-direct.txt` with no index silently resolves `torch` against
plain PyPI's default CUDA-13 build instead of the pinned CUDA-12.6 one, which
this account's driver (`nvidia-smi`, CUDA 12.8 max) cannot run.
`cu124` does not carry `torch==2.13.0`/`torchvision==0.28.0` at all; `cu126`
is the correct index for this driver generation.

```bash
scripts/dgx/run_pod_job.sh venvinstall \
  'VENV_PY="$REPO_DIR/.venv/bin/python3.12" &&
   export PIP_CONFIG_FILE=/dev/null &&
   "$VENV_PY" -m pip install --upgrade pip &&
   "$VENV_PY" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple &&
   "$VENV_PY" -m pip install -r requirements/p0c-direct.txt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple &&
   "$VENV_PY" -m pip check'
```

The resulting exact package set is recorded in
`requirements/p0c-lock-dgx-cu126.txt` — the Linux/CUDA companion to the
macOS lock, not a replacement for it.

**2d. Verify inside a pod before trusting any timing:**

```bash
scripts/dgx/run_pod_job.sh envcheck \
  '"$REPO_DIR/.venv/bin/python3.12" scripts/check_environment.py'
```

Read the printed JSON's `"verdict"` field. `PASS_WITHOUT_MPS` is the correct,
healthy outcome on this Linux/CUDA host (MPS never exists here); `BLOCKED`
means a real problem.

### 3. Point the runner at the cluster

Defaults are set at the top of `run_pod_job.sh` and overridable by environment
variable:

| Variable | Default | Meaning |
|---|---|---|
| `USER_HOME` | `<home>` | `$HOME` inside the pod, and the Kyverno-allowed hostPath |
| `HOST_PATH` | `${USER_HOME}` | path on the worker node to mount |
| `HOME_MOUNT` | `${USER_HOME}` | mount point inside the pod |
| `REPO_DIR` | `${USER_HOME}/SHAR` | repository location inside the pod |
| `CPU` / `MEM` | `8` / `16Gi` | must not exceed the namespace quota |

## Running a job

```bash
scripts/dgx/run_pod_job.sh <name> '<command>'
```

The runner applies the manifest, waits for scheduling, streams logs, waits for a
terminal phase, reports the container exit code, and deletes the pod from an
`EXIT INT TERM` trap, so the pod is reaped on success, failure, and Ctrl-C
alike. The script exits with the container's exit code, so a failed research
command fails the script rather than being masked by successful cleanup.

### CUDA calibration

```bash
scripts/dgx/run_pod_job.sh cal004 \
  '"$REPO_DIR/.venv/bin/python3.12" scripts/calibrate_p1c_frame_loader.py \
     --config configs/experiments/p1c/P1C-FRAME-GAP-CALIBRATION-004-CUDA.yaml'
```

## Long runs

`run_pod_job.sh` deletes the pod on exit (success, failure, or Ctrl-C), which
is wrong for a multi-day run: a dropped `kubectl logs -f` stream (API-server
blip, network hiccup, kubelet log rotation) must not be mistaken for job
completion, and "I want to stop watching" must not mean "delete the job".
For anything that will run longer than you intend to keep a terminal open,
launch and watch it as separate steps instead:

```bash
scripts/dgx/start_pod_job.sh <name> '<command>'   # launches, prints the pod name, exits 0
scripts/dgx/watch_pod_job.sh [pod-name]           # follows logs; Ctrl-C only stops watching
scripts/dgx/stop_pod_job.sh [pod-name]            # the only one of the three that deletes the pod
```

`watch_pod_job.sh` and `stop_pod_job.sh` take the pod name as an optional
argument; with none, they resolve the one Pending/Running pod named `shar-*`
and refuse to guess if there is more than one. All four scripts share their
cluster configuration and manifest rendering via `_pod_common.sh`.

## Notes

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set in the pod template. The
  `cuda_strict` policy calls `torch.use_deterministic_algorithms(True)` with
  `warn_only=False`, which raises a `RuntimeError` on CUDA >= 10.2 without it.
- `restartPolicy: Never` is required. Under the default `Always`, a completed
  container restarts and the pod never reaches `Succeeded`.
- The quota permits exactly one `1g.18gb` slice, so seeds run sequentially. See
  `docs/DGX_COMPUTE_EVIDENCE.md` §5.
- Results are written under `results/` inside `REPO_DIR`, which is on the
  persistent home export and therefore survives pod deletion.
