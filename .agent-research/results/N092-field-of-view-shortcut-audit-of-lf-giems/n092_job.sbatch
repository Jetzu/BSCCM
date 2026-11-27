#!/bin/bash
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-01:00:00
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH -J n092_audit
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
W=/hnvme/workspace/v181bb10-bsccm; R=/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N092-field-of-view-shortcut-audit-of-lf-giems
for p in fov_b0 fov_b1; do
  apptainer exec --bind $W $W/env/bsccm.sif env PYTHONPATH=$W/env/pylib_n045 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 python3 $R/n092_audit.py --protocols $p --threads 48 --out /home/hpc/v181bb/v181bb10/bsccm_runs/n092/$p --results-dir $R --resume
done
