#!/usr/bin/env bash
# Run inside the GPU environment after integration. $1 is an output root.
set -euo pipefail
repo=/home/hpc/v181bb/v181bb10/BSCCM
workspace=/hnvme/workspace/v181bb10-bsccm
out_root=${1:?Usage: bash run_arms.sh OUTPUT_ROOT}
trainer=${TRAINER:-${repo}/scripts/train_lf.py}
for seed in 0 1; do
  for protocol in b1to0 b0to1; do
    exclude=0
    if [[ $protocol == b0to1 ]]; then exclude=1; fi
    ssl=${workspace}/runs/n031_ssl_aug_ex${exclude}/encoder_final.pt
    for arm in hlgauss twopart fov_re pair autofl0 panel; do
      case "$arm" in
        hlgauss) flags=(--head hlgauss --hl-mse .1); item=n060 ;;
        twopart) flags=(--head twopart --pos-q .98 --gate-w .5); item=n060 ;;
        fov_re) flags=(--fov-re .01 --re-shrink 1 --w-autofl 0); item=n061 ;;
        pair) flags=(--pair-loss .5 --fov-batch 64 --w-autofl 0); item=n061 ;;
        autofl0) flags=(--w-autofl 0); item=n061 ;;
        panel) flags=(--panel-head --re-shrink 1); item=n062 ;;
      esac
      python "$trainer" --out "${out_root}/${item}_${arm}_${protocol}_s${seed}" \
        --protocol "$protocol" --arch resnet50 --channels all --phase 1 \
        --init-ssl "$ssl" --epochs 10 --bs 256 --lr 5e-4 --w-cls .5 \
        --targets targets_v5cx_v10.npz --seed "$seed" "${flags[@]}"
    done
  done
done
