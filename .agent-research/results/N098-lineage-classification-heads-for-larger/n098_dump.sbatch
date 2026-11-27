#!/bin/bash
# N098 ensemble arm: frozen-feature dumps of the E8 members' single-marker models (20 runs; inference only, ~2-4 min each).
# Placement: 1 GPU (CNN inference over ~150-280k cells per run; GPU >> CPU). Idempotent: runs with features.npz are skipped.
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -uo pipefail
R=/home/hpc/v181bb/v181bb10/BSCCM; D=$(ls -d $R/.agent-research/results/N098-*); W=/hnvme/workspace/v181bb10-bsccm; cd $R
PY="apptainer exec --nv -B /hnvme,/home/hpc $W/env/bsccm.sif python3"
export WANDB_MODE=disabled
for r in n048_v9_base n048_v9_zoom2 n053_cytoland n063_tpfov_s2 n063_tpfov_s3 n065_tpfov_z2 n065_tpfov_cyto n085_cnxs120_tpfov_s0; do
  for d in b10 b01; do echo "$W/runs/${r}_$d"; done; done > $D/dump_runs.txt
for r in n085_cnxs120_plain_s0_b10 n085_cnxs120_plain_s1_b10 n083_cnxs120_s0_b01 n084_cnxs120_s1_b01; do echo "$W/runs/$r"; done >> $D/dump_runs.txt
while read run; do
  [ -f $run/features.npz ] && { echo "skip $run"; continue; }
  cmd=$($PY -I $D/n098_dump_cmd.py $run) || { echo "CMDFAIL $run"; continue; }
  echo "RUN $cmd"; eval $PY $cmd || echo "DUMPFAIL $run"
done < $D/dump_runs.txt
