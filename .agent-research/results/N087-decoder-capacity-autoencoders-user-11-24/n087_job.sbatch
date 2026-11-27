#!/bin/bash
# N087 GPU steps (STEP=decoders | ae | sandbox | post). 1 x H100 preempt; every step is resumable or cheap to rerun.
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --signal=B:USR1@900
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out
set -euo pipefail
R=/home/hpc/v181bb/v181bb10/BSCCM; C=$(ls -d $R/.agent-research/results/N087-*)/codex; W=/hnvme/workspace/v181bb10-bsccm; O=$W/runs/n087
T=$W/targets_v5cx_v10.npz; PACK=$W/external/sbiad452_pack; mkdir -p $O
export WANDB_MODE=offline WANDB_DIR=$W/wandb WANDB_PROJECT=BSCCM
PY="apptainer exec --nv -B /hnvme $W/env/bsccm.sif python3 -IB"
declare -A protocol=([b10]=b1to0 [b01]=b0to1) excluded=([b10]=0 [b01]=1)
case $STEP in
decoders)
  for d in b10 b01; do [ -f $O/ssl_$d/features.npz ] || $PY $C/n087_extract.py --encoder ssl --ckpt $W/runs/n031_ssl_aug_ex${excluded[$d]}/encoder_final.pt --rows-from $W/runs/n051_v10_base_$d/features.npz --device cuda --out $O/ssl_$d/features.npz; done
  [ -f $O/ssl_b10/cluster.npz ] || $PY $C/n087_extract.py --encoder ssl --ckpt $W/runs/n031_ssl_aug_ex0/encoder_final.pt --rows-from $W/runs/n087_cluster_rows_b10.npz --device cuda --out $O/ssl_b10/cluster.npz
  $PY $C/n087_decoders.py finetuned=$W/runs/n051_v10_base_b10/features.npz finetuned=$W/runs/n051_v10_base_b01/features.npz ssl=$O/ssl_b10/features.npz ssl=$O/ssl_b01/features.npz --targets $T --out $O/decoders ;;
ae)   # DIM and D set by the submitter; plain autoencoder (no masked patches)
  exec ~/.claude/skills/work-log/bin/preempt-run.sh -- $PY $C/n087_image_ae.py --protocol ${protocol[$D]} --dim $DIM --mask-fraction 0 --device cuda --out $O/ae${DIM}_$D ;;
post)  # after all AE runs: extract AE features, decoders on them, clustering for every representation
  for k in 16 64 256; do for d in b10 b01; do run=$O/ae${k}_$d
    [ -f $run/features.npz ] || $PY $C/n087_extract.py --encoder ae --ckpt $run/encoder_final.pt --rows-from $W/runs/n051_v10_base_$d/features.npz --device cuda --out $run/features.npz; done
    [ -f $O/ae${k}_b10/cluster.npz ] || $PY $C/n087_extract.py --encoder ae --ckpt $O/ae${k}_b10/encoder_final.pt --rows-from $W/runs/n087_cluster_rows_b10.npz --device cuda --out $O/ae${k}_b10/cluster.npz; done
  $PY $C/n087_decoders.py ae16=$O/ae16_b10/features.npz ae16=$O/ae16_b01/features.npz ae64=$O/ae64_b10/features.npz ae64=$O/ae64_b01/features.npz ae256=$O/ae256_b10/features.npz ae256=$O/ae256_b01/features.npz --targets $T --out $O/decoders_ae
  $PY $C/n087_cluster_eval.py ssl=$O/ssl_b10/cluster.npz ae16=$O/ae16_b10/cluster.npz ae64=$O/ae64_b10/cluster.npz ae256=$O/ae256_b10/cluster.npz finetuned_dumprows=$W/runs/n051_v10_base_b10/features.npz --targets $T --out $O/cluster ;;
sandbox)  # marker prior: linear / ae, k = 2, 4, 6, every held-out donor; then evaluation
  for prior in linear ae; do for k in 2 4 6; do for d in 0 1 2; do
    run=$O/sandbox_${prior}${k}_d$d; [ -f $run/eval_preds.npz ] || $PY $C/train_ifc_sandbox.py --pack $PACK --regime single --test-donor $d --marker-prior $prior --prior-dim $k --out $run; done; done; done
  $PY $C/n087_sandbox_eval.py $O/sandbox_* $W/runs/n080_single_d* $W/runs/n080_full_d* --pack $PACK --out $O/sandbox_eval ;;
esac
