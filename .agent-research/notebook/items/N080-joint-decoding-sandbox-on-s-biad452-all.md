# N080 · Joint-decoding sandbox on S-BIAD452 (all 9 markers per cell, BF/SSC input): BSCCM-like single-marker training regime vs full joint supervision vs single+pooled, held-out-donor per-cell joint metrics

type: exp · date: 2026-11-23 · status: running
goal_effect: pending · scale: paradigm

## Goal
USER 11-23 priority: decode ALL markers of each cell jointly. Review N079: BSCCM cannot test joint per-cell decoding (pooled channels
mostly measure cell type: cell-type-only .730 vs model .737). S-BIAD452 measures 9 markers on EVERY cell (BF1/BF2/SSC label-free) ->
simulate BSCCM's single-stain regime and measure joint decoding against real joint truth on a held-out donor.

## Plan
Data: scripts/pack_sbiad452.py -> external/sbiad452_pack (lf.npy BF1/BF2/SSC at the BSCCM scale 1.75, labels.npz 9 markers asinh +
per-donor z). Trainer: Codex writes train_ifc_sandbox.py (results/N080/codex/brief.md); Claude red-teams before GPU.
Regimes: full (all 9 per cell), single (one random marker per cell, BSCCM-like), single_pooled (+10 % cells with pooled CD3+CD19+CD56 and
CD45 only); kNN-imputation baseline in the learned embedding; optional rank-2 covariance head. Folds: each donor held out once.
Metrics (pre-registered): per-marker r (macro), per-cell vector cosine/RMSE, cell-level (within true population) vs cluster-level r,
predicted-minus-true pair correlations (36 pairs), gating F, CD3+CD19+ double-positive rate. PRIMARY: joint-decoding gap =
macro within-population r (full) - (single); a gap >= .05 means single-stain training loses joint structure and joint methods/data are
needed; < .02 means per-marker training already decodes joint profiles in this modality. Cost ~1-2 GPU-h.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N080 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N080-slug/ + external absolute paths -->

## Next
