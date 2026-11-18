#!/bin/bash
#SBATCH --job-name=n054
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --output=/hnvme/workspace/v181bb10-bsccm/logs/n054_%j.log
set -uo pipefail
PACKAGE="${N054_PACKAGE:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$PACKAGE/n054_trimodal.py" ]] || { echo "Set N054_PACKAGE to this package directory" >&2; exit 2; }
export TMPDIR=/tmp/codex_n054
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR"
SCRIPT=n054_trimodal.py
if [[ "${1:-}" == clinical ]]; then
    SCRIPT=n054_clinical.py
    shift
elif [[ "${1:-}" == select ]]; then
    SCRIPT=n054_select.py
    shift
fi
srun apptainer exec --nv \
    --bind /hnvme/workspace/v181bb10-bsccm \
    --bind "$PACKAGE:$PACKAGE:ro" \
    --bind /tmp/codex_n054 \
    /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif \
    python3 -I "$PACKAGE/$SCRIPT" "$@"
status=$?
if [[ "$status" -eq 99 ]]; then
    # Python saved ckpt.pt at an optimizer/inference boundary.
    scontrol requeue "$SLURM_JOB_ID"
fi
exit "$status"
