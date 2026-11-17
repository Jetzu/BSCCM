set -euo pipefail
cd $HOME/BSCCM
RUN="apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif"
O=.agent-research/results/N048-label-v9-asinh-cofactor-from-a-pre-regis
for cf in q50 50 150; do
  mkdir -p $O/cf$cf
  $RUN env BSCCM_CLEAN=1 BSCCM_COFACTOR=$cf python3 -I scripts/make_targets_v3.py
  $RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_COFACTOR=$cf python3 -I scripts/make_targets_v4.py $O/cf$cf
  $RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_COFACTOR=$cf python3 -I scripts/make_targets_v5.py
done
$RUN env OMP_NUM_THREADS=8 python3 -I $O/n048_qc.py $O
