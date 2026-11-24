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
Codex build (codex/: train_ifc_sandbox.py, 31 CPU tests). Integrated as scripts/train_ifc_sandbox.py with Claude's fold-specific
preprocessing: label cofactor and per-run z from the TRAINING donor's REVEALED entries (validation runs revealed-only too: they select the
checkpoint; test-donor runs use their own truth for metrics only), input chan_scale from training cells. First version leaked hidden labels
into the cofactor -- Codex's leakage test caught it (3 failures); fixed; 31/31 pass (venv /tmp/n080_codex). Design kept: train 1 donor,
validate 1 (checkpoint selection), test 1, cyclic.
Claude red-team (agents_redteam.md; resumed after a usage limit, reviewed the integrated script): NO leakage into the trained model
(34/34 tests incl. 3 new: hidden CD3/CD19/CD8 of pooled cells changed -> identical weights in both pooled modes; test-donor images x7 ->
bit-identical); my fold-statistics block correct; masks/kNN bank/donor split/metric formulas correct. MAJOR (evaluation side only):
(1) test truth z-scored with its own run statistics (deliberate, fair between arms; absolute RMSE/gates shift) -> report both target
standardisations; (2) gates are fixed training quantiles (75th pct -> every marker 25 % positive; eos 15.5 % vs neutrophils 1.5 % on
donor 0) -- confirmed in code -> gate F1 / double positives / coarse populations NOT interpretable as written; (3) a fine-cell-type-mean
predictor reaches lymphocyte within-population r .67-.86 -> within-population r alone cannot show decoding beyond cell type.
Decision: trainer unchanged (training is clean); evaluation redone post-hoc from eval_preds: valley (bimodal) thresholds fitted on the
training donor + true population fractions, a fine-type-mean baseline arm with model-minus-baseline within fine types, both
standardisations. Minors noted (pooled weight ~4.5x per entry; single_pooled replaces 10 % of single labels; validation pooled scale;
rotation padding geometry).

## Evidence

2026-11-24 data recovery: S-BIAD452 sample2 (donor 2) zarr.zip files lack .zarray/.zattrs on the archive (chunk files only) -> packer
rebuilds the array metadata from a metadata-bearing part (chunks 100, blosc-lz4, vlen <u2) and infers each cell's (H, W) from its
flattened length by brightfield row smoothness (2084/2084 exact on sample0 cells with known shapes). Without object numbers donor 2
cannot be linked to IDEAS -> labels are IMAGE-DERIVED for all donors (masked background-subtracted sums; asinh r vs IDEAS on sample0
CD15 1.00, Siglec8 .977, CD14 .948, CD19 .762, CD3 .998, CD45 .997, CD4 .966, CD56 .937, CD8 .908; IDEAS kept as I_ideas). Earlier pack
failures: truncated FCS download (4 KB, re-fetched); all 11 zips CRC ok, uncompressed sizes = archive sizes.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N080 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N080-slug/ + external absolute paths -->

## Next
