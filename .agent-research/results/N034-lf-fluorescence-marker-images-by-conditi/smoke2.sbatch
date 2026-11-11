#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --time=0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -J n034_smoke2
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
export WANDB_MODE=offline WANDB_DIR=/hnvme/workspace/v181bb10-bsccm/wandb WANDB_PROJECT=BSCCM
RUN="apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I"

$RUN scripts/fm_train.py --out /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke2/cfm --mode cfm --max-train 6144 --epochs 1 --workers 15 --patch 2
$RUN scripts/fm_train.py --out /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke2/reg --mode reg --max-train 3072 --epochs 1 --workers 15 --patch 2
$RUN scripts/fm_eval.py --cfm /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke2/cfm/ckpt.pt --reg /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke2/reg/ckpt.pt --out /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke2/eval --per-stain 80 --K 3 --steps 4 --readout-per-stain 300
nvidia-smi --query-gpu=memory.used --format=csv
