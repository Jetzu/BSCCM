#!/bin/bash
# N045 SOTAlign: one protocol x seed per job, 1 H100 (preempt), resumable (--resume, atomic checkpoints, USR1 -> save).
#SBATCH --partition=preempt
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --signal=B:USR1@120
#SBATCH -J bsccm_n045
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
R=.agent-research/results/N045-sotalign-semi-supervised-ot-alignment-gi/codex
W=/hnvme/workspace/v181bb10-bsccm
exec ~/.claude/skills/work-log/bin/preempt-run.sh -- apptainer exec --nv --bind $W $W/env/bsccm.sif \
  env PYTHONPATH=$W/env/pylib_n045 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=16 \
  python3 $R/scripts/n045_sotalign.py --out /home/hpc/v181bb/v181bb10/bsccm_runs/n045/${N045_PROTOCOL}_seed0 --device cuda --protocols $N045_PROTOCOL --seeds 0 \
  --threads 16 --steps 1500 --batch-pairs 256 --batch-unpaired 256 --dim 128 --n-unpaired 100000 --head-train 100000 --head-val 10000 \
  --eval-every 100 --checkpoint-every 50 --resume
