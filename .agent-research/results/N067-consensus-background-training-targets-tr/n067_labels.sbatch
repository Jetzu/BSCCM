#!/bin/bash
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-02:00:00
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH --signal=B:USR1@900
#SBATCH -J n067_labels
#SBATCH -o /hnvme/workspace/v181bb10-bsccm/runs/n067_alllabels/slurm_%j.out
set -euo pipefail
cd $HOME/BSCCM
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
  apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N067-consensus-background-training-targets-tr/n067_alllabels.py \
  --all-cells --chunk-size 4096 \
  --source /hnvme/workspace/v181bb10-bsccm \
  --out /hnvme/workspace/v181bb10-bsccm/runs/n067_alllabels --workers 48
