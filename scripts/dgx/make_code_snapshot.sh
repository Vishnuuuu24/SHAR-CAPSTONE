#!/usr/bin/env bash
# Freeze the committed code of the current checkout into a directory a pod can run from (the durable fix K-23 asks for).
#
# Usage:
#   scripts/dgx/make_code_snapshot.sh <snapshot-root> [<data-checkout>]
#
# Writes <snapshot-root>/<short-sha>/ holding `git archive HEAD`, then points the bulk, untracked
# directories (results, Datasets, .cache, .venv) at <data-checkout> (default: the main SHAR checkout),
# so repository-relative paths in the code resolve to the real caches while the code itself cannot
# change under a running pod. Prints the snapshot path and the commit it holds.
#
# A pod started from the snapshot records that commit as its code revision; commit before snapshotting.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <snapshot-root> [<data-checkout>]" >&2
    exit 2
fi

SNAPSHOT_ROOT="$1"
DATA_CHECKOUT="${2:-<repo>}"
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "$(git -C "${SOURCE}" status --porcelain --untracked-files=no)" ]]; then
    echo "refusing to snapshot: ${SOURCE} has uncommitted changes to tracked files" >&2
    exit 1
fi

SHA="$(git -C "${SOURCE}" rev-parse --short HEAD)"
TARGET="${SNAPSHOT_ROOT}/${SHA}"
if [[ -e "${TARGET}" ]]; then
    echo "${TARGET} ${SHA} (already exists)"
    exit 0
fi

mkdir -p "${TARGET}"
git -C "${SOURCE}" archive --format=tar HEAD | tar -x -C "${TARGET}"
rm -rf "${TARGET}/results"
for name in results Datasets .cache .venv; do
    ln -s "${DATA_CHECKOUT}/${name}" "${TARGET}/${name}"
done
echo "${TARGET} ${SHA}"
