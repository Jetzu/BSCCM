#!/bin/bash
# generic 1-GPU job: CMD = python args inside bsccm.sif. Preempt-safe: commands resume from <out>/ckpt.pt.
#SBATCH --partition=preempt
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --signal=B:USR1@900
#SBATCH -J bsccm_gpu
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
export WANDB_MODE=offline WANDB_DIR=/hnvme/workspace/v181bb10-bsccm/wandb WANDB_PROJECT=BSCCM
mkdir -p $WANDB_DIR
exec ~/.claude/skills/work-log/bin/preempt-run.sh -- \
  apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I $CMD
