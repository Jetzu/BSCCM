set -euo pipefail
cd $HOME/BSCCM
RUN="apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif"
$RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit python3 -I scripts/make_targets_v4.py .agent-research/results/N036-label-position-detrend-v7-cross-fit-the
$RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit python3 -I scripts/make_targets_v5.py
$RUN python3 -I scripts/n036_detrend_diag.py .agent-research/results/N036-label-position-detrend-v7-cross-fit-the
