set -euo pipefail
cd $HOME/BSCCM
RUN="apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif"
O=.agent-research/results/N051-label-v10-per-marker-asinh-cofactor-pre
for cf in 15 500; do mkdir -p $O/cf$cf
  $RUN env BSCCM_CLEAN=1 BSCCM_COFACTOR=$cf python3 -I scripts/make_targets_v3.py
  $RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_COFACTOR=$cf python3 -I scripts/make_targets_v4.py $O/cf$cf
  $RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_COFACTOR=$cf python3 -I scripts/make_targets_v5.py
done
$RUN env OMP_NUM_THREADS=8 python3 -I $O/n051_select.py $O | tee $O/n051_select.out
MAP=$(tail -1 $O/n051_select.out); echo "MAP=$MAP"; mkdir -p $O/v10
$RUN env BSCCM_CLEAN=1 BSCCM_COFACTOR="$MAP" BSCCM_TAG=_v10 python3 -I scripts/make_targets_v3.py
$RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_COFACTOR="$MAP" BSCCM_TAG=_v10 python3 -I scripts/make_targets_v4.py $O/v10
$RUN env BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_COFACTOR="$MAP" BSCCM_TAG=_v10 python3 -I scripts/make_targets_v5.py
