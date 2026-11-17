#!/bin/bash
# generic short CPU analysis job: CMD env var = python args (run inside bsccm.sif). 48 cores = full NUMA fraction.
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-04:00:00
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH -J n051_build
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
bash .agent-research/results/N051-label-v10-per-marker-asinh-cofactor-pre/n051_build.sh
