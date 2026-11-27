#!/bin/bash
# N098 on CPU: logistic heads on frozen 2048-d features (sklearn lbfgs, multithreaded BLAS; GPU < 2x for this). 48 cores.
#SBATCH --partition=preempt_cpu
#SBATCH -c 48
#SBATCH -t 04:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
D=$(ls -d /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N098-*); cd $D
apptainer exec -B /hnvme /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I n098_lineage_heads.py
