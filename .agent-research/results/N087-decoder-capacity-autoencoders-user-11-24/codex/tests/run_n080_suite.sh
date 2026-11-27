#!/usr/bin/env bash
# Invoke with the Python command (including apptainer arguments) as separate argv.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
original="${BSCCM_REPO:-$(cd "$here/../../../.." && pwd)}/.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/codex/tests"
mkdir -p /tmp/n087_codex
stage=$(mktemp -d /tmp/n087_codex/n080-tests.XXXXXX)
cp -R "$original" "$stage/tests"
ln -s "$here/train_ifc_sandbox.py" "$stage/train_ifc_sandbox.py"
export XDG_CACHE_HOME=/tmp/n087_codex/cache TORCH_HOME=/tmp/n087_codex/cache/torch
"$@" -m pytest -p no:cacheprovider --basetemp="$stage/output" -q "$stage/tests"
