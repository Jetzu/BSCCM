#!/bin/bash
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-04:00:00
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH --signal=B:USR1@900
#SBATCH -J n074_fluor_aux
#SBATCH -o /hnvme/workspace/v181bb10-bsccm/n074_fluor_aux_%j.out
set -euo pipefail
cd "$HOME/BSCCM"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/tmp/n074_codex PYTHONPYCACHEPREFIX=/tmp/n074_codex/pycache XDG_CACHE_HOME=/tmp/n074_codex/cache
mkdir -p "$TMPDIR"
bash "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
  apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  /home/hpc/v181bb/v181bb10/BSCCM/scripts/pack_fluor_aux.py \
  --source /hnvme/workspace/v181bb10-bsccm \
  --out /hnvme/workspace/v181bb10-bsccm/fluor_aux64.npy --workers 48
