set -euo pipefail
cd $HOME/BSCCM
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I scripts/n036_detrend_diag.py .agent-research/results/N036-label-position-detrend-v7-cross-fit-the
