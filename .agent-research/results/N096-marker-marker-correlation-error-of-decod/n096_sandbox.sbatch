#!/bin/bash
# N096 C: sandbox correlation-matching arms (12 runs, resumable; each ~6-10 min on 1 H100). Placement: GPU (ResNet-50 image training).
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
R=/home/hpc/v181bb/v181bb10/BSCCM; D=$(ls -d $R/.agent-research/results/N096-*); C=$(ls -d $R/.agent-research/results/N087-*)/codex
W=/hnvme/workspace/v181bb10-bsccm; O=$W/runs/n096; PACK=$W/external/sbiad452_pack; mkdir -p $O
PY="apptainer exec --nv -B /hnvme $W/env/bsccm.sif python3 -IB"
for d in 0 1 2; do
  for lam in 1 10 100; do run=$O/sb_single_corr${lam}_d$d
    [ -f $run/eval_preds.npz ] || $PY $D/n096_train_ifc_sandbox.py --pack $PACK --regime single --test-donor $d --corr-match $lam --corr-source val --out $run; done
  run=$O/sb_full_corr10_d$d
  [ -f $run/eval_preds.npz ] || $PY $D/n096_train_ifc_sandbox.py --pack $PACK --regime full --test-donor $d --corr-match 10 --corr-source train --out $run
done
$PY $C/n087_sandbox_eval.py $O/sb_* $W/runs/n080_single_d* $W/runs/n080_full_d* --pack $PACK --out $D/sandbox_eval
