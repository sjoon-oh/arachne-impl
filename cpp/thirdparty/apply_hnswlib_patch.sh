#!/usr/bin/env bash
# Resets cpp/thirdparty/hnswlib to the upstream commit our local patch
# (hnswlib.patch, in this same directory) was generated against, then
# applies it. See .gitmodules at the repo root for what's in that patch
# and why.
#
# Use this after `git submodule update` (which resets the submodule back
# to plain upstream) or after re-vendoring hnswlib from scratch, to bring
# the local uint8/int8/float16 SIMD additions back.
#
# WARNING: this discards any uncommitted changes and untracked files
# inside cpp/thirdparty/hnswlib (git reset --hard + git clean -fd) before
# applying the patch -- it assumes that submodule is meant to end up
# exactly at base commit + this patch, nothing else.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HNSWLIB_DIR="${SCRIPT_DIR}/hnswlib"
PATCH_FILE="${SCRIPT_DIR}/hnswlib.patch"
BASE_COMMIT="d9b3608c83d83b46c96e25088cb1d729b29dcfe9"

if [ ! -f "${PATCH_FILE}" ]; then
  echo "error: ${PATCH_FILE} not found" >&2
  exit 1
fi

if [ ! -d "${HNSWLIB_DIR}" ] || ! git -C "${HNSWLIB_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "error: ${HNSWLIB_DIR} is not a git checkout -- run 'git submodule update --init cpp/thirdparty/hnswlib' first" >&2
  exit 1
fi

echo "Fetching ${BASE_COMMIT} in case it isn't present locally..."
git -C "${HNSWLIB_DIR}" fetch origin "${BASE_COMMIT}" 2>/dev/null || true

echo "Resetting ${HNSWLIB_DIR} to ${BASE_COMMIT} (discarding local changes)..."
git -C "${HNSWLIB_DIR}" checkout "${BASE_COMMIT}" --detach
git -C "${HNSWLIB_DIR}" reset --hard "${BASE_COMMIT}"
git -C "${HNSWLIB_DIR}" clean -fd

echo "Applying ${PATCH_FILE}..."
git -C "${HNSWLIB_DIR}" apply --whitespace=nowarn "${PATCH_FILE}"

echo "Done: ${HNSWLIB_DIR} is now ${BASE_COMMIT} + hnswlib.patch."
