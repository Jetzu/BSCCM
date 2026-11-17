#!/bin/bash
# N045 v2 (red-team fixes): one protocol x PBC variant per job, CPU (48 cores; GPU not faster, red-team M2), 4,500 steps (m3), resumable.
#SBATCH --partition=preempt_cpu
#SBATCH --time=0-03:00:00
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH --signal=B:USR1@120
#SBATCH -J bsccm_n045cpu
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
cd $HOME/BSCCM
R=.agent-research/results/N045-sotalign-semi-supervised-ot-alignment-gi/codex
W=/hnvme/workspace/v181bb10-bsccm
PBC=$W/emb/pbc_dinobloom_B.npy; [ "$N045_VARIANT" = recentred ] && PBC=$W/emb/pbc_dinobloom_B_recentred_${N045_TRAIN}.npy
exec ~/.claude/skills/work-log/bin/preempt-run.sh -- apptainer exec --bind $W $W/env/bsccm.sif \
  env PYTHONPATH=$W/env/pylib_n045 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 \
  python3 $R/scripts/n045_sotalign.py --out /home/hpc/v181bb/v181bb10/bsccm_runs/n045_v2/${N045_VARIANT}_${N045_PROTOCOL} --device cpu \
  --protocols $N045_PROTOCOL --seeds 0 --threads 48 --steps 4500 --batch-pairs 256 --batch-unpaired 256 --dim 128 \
  --n-unpaired 100000 --head-train 100000 --head-val 10000 --eval-every 100 --checkpoint-every 100 --pbc $PBC --resume
