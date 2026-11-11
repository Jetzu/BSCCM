#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --time=0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -J n034_smoke
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
export WANDB_MODE=offline WANDB_DIR=/hnvme/workspace/v181bb10-bsccm/wandb WANDB_PROJECT=BSCCM
RUN="apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I"
[ -e /hnvme/workspace/v181bb10-bsccm/fm_prep.npz ] || $RUN scripts/fm_prep.py
$RUN scripts/fm_train.py --out /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke/cfm --mode cfm --max-train 6144 --epochs 1 --workers 15
$RUN scripts/fm_train.py --out /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke/reg --mode reg --max-train 3072 --epochs 1 --workers 15
$RUN scripts/fm_eval.py --cfm /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke/cfm/ckpt.pt --reg /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke/reg/ckpt.pt --out /hnvme/workspace/v181bb10-bsccm/runs/n034_smoke/eval --per-stain 24 --K 2 --steps 4 --readout-per-stain 300
nvidia-smi --query-gpu=memory.used --format=csv
