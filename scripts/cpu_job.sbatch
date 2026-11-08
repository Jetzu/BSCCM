#!/bin/bash
# generic short CPU analysis job: CMD env var = python args (run inside bsccm.sif). 48 cores = full NUMA fraction.
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-02:00:00
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH -J bsccm_cpu
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I $CMD
