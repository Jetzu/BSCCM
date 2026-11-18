#!/bin/bash
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-02:00:00
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH --signal=B:USR1@900
#SBATCH -J n064_ceiling
#SBATCH -o /hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling/slurm_%j.out
set -euo pipefail
cd $HOME/BSCCM
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Same container/bind/cwd pattern as scripts/cpu_job.sbatch, under the preemption chain wrapper.
bash "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
  apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N064-noise-ceiling-bracket-for-the-v10-goal-p/codex/n064_ceiling.py \
  --source /hnvme/workspace/v181bb10-bsccm \
  --out /hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling --workers 48
