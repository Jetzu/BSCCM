#!/bin/bash
# N096 A+B on CPU: linear decoders on frozen 2048-d features + correlation measurement (CPU: small linear models, memory-bound; GPU < 2x).
#SBATCH --partition=preempt_cpu
#SBATCH -c 48
#SBATCH -t 06:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
D=$(ls -d /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-*); cd $D
apptainer exec -B /hnvme /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I n096_bsccm.py
