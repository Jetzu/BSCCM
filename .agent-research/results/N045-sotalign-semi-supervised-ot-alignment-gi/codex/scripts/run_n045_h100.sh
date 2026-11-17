#!/usr/bin/env bash
#SBATCH --job-name=n045-sotalign
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:29:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
# Add your cluster's --partition/--account/--constraint=h100 at submission.
# One protocol x seed per invocation. Re-submit the identical command to resume.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${N045_RUN_ROOT:?Set N045_RUN_ROOT to persistent writable storage OUTSIDE BSCCM and the data/source trees}"
PYTHON=${N045_PYTHON:-python}
PROTOCOL=${N045_PROTOCOL:-cross_b0to1}
SEED=${N045_SEED:-0}
DEVICE=${N045_DEVICE:-cuda}
THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$DEVICE" == cuda ]]; then
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; name=torch.cuda.get_device_name(0); assert "H100" in name, "Expected one H100, got " + name; print(name, flush=True)'
fi
OUT=${N045_RUN_ROOT}/${PROTOCOL}_seed${SEED}
args=("${SCRIPT_DIR}/n045_sotalign.py" --out "$OUT" --device "$DEVICE"
      --protocols "$PROTOCOL" --seeds "$SEED" --threads "$THREADS"
      --steps 1500 --batch-pairs 256 --batch-unpaired 256 --dim 128
      --n-unpaired 100000 --head-train 100000 --head-val 10000
      --eval-every 100 --checkpoint-every 50 --resume)
# Extra arguments allow alternate LF embeddings or a paired-negative-only control.
"$PYTHON" "${args[@]}" "$@" &
child=$!
forward_stop() { kill -USR1 "$child" 2>/dev/null || true; }
trap forward_stop USR1 TERM
status=0
# A trapped signal interrupts bash wait; wait again until the Python child has
# atomically saved and exited. Never requeue while a writer is still alive.
while true; do
    if wait "$child"; then status=0; break; else status=$?; fi
    if ! kill -0 "$child" 2>/dev/null; then break; fi
done
if [[ "$status" == 75 && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "N045 checkpoint saved; requesting requeue of ${SLURM_JOB_ID}"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
