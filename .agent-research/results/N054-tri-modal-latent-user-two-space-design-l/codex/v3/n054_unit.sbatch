#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --signal=B:USR1@900
#SBATCH -J n054v3
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/n054v3_%j.out
set -euo pipefail
: "${N054_UNIT:?Export N054_UNIT=lf, batch1, or batch0 before sbatch}"
case "$N054_UNIT" in lf|batch1|batch0) ;; *) exit 64;; esac
N054_CODEDIR=${N054_CODEDIR:-/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v3}
N054_RUNROOT=${N054_RUNROOT:-/hnvme/workspace/v181bb10-bsccm/runs}
N054_PREFIX=${N054_PREFIX:-n054v3}
export N054_UNIT N054_CODEDIR N054_RUNROOT N054_PREFIX
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$N054_RUNROOT/${N054_PREFIX}_${N054_UNIT}"
cd "$N054_CODEDIR"
# The house wrapper resubmits a cancelled link, forwards SIGTERM, and accepts
# Python's checkpointed exit 143. Parameters are environment variables so replay
# retains them. No srun and no dependence on Slurm REQUEUE behaviour.
exec "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
  apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$N054_CODEDIR/n054_trimodal.py" --stage all --unit "$N054_UNIT" \
  --out "$N054_RUNROOT/${N054_PREFIX}_${N054_UNIT}" \
  --lf-unit "$N054_RUNROOT/${N054_PREFIX}_lf"
