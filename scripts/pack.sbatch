#!/bin/bash
# BSCCM zarr-zip -> fixed-record .npy pack. CPU job: decode (blosc/zstd) is parallel over 256-cell blocks, resumable.
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-12:00:00
#SBATCH -c 48
#SBATCH --mem=160G
#SBATCH --signal=B:USR1@900
#SBATCH -J bsccm_pack
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
W=/hnvme/workspace/v181bb10-bsccm
exec ~/.claude/skills/work-log/bin/preempt-run.sh -- \
  apptainer exec --bind /hnvme/workspace/v181bb10-bsccm $W/env/bsccm.sif python3 -I $HOME/BSCCM/scripts/pack.py \
    $W/BSCCM.zarr.zip $W/bsccm_zip_index.npz $W/bsccm_zip_index_zarrmeta.json $W/pack 48
