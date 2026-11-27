#!/bin/bash
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-06:00:00
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH -J n093_router_fix
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex
apptainer exec -B /hnvme /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex/n093_class_match.py --seed 0 --teachers t0 t1 t2 --holdouts 1 0 --threads 48 --out /hnvme/workspace/v181bb10-bsccm/runs/n093_class_match_fix_s0
