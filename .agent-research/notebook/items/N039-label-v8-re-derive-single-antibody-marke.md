# N039 · Label v8: re-derive single-antibody marker labels with the authors' PER-CELL background (not background_lowess) + draw-1 own-dye QC; pre-registered label QC; GOAL recipe retrain scored on frozen v5 and on v8; proper in-distribution ceiling (r50 FOV on v5, data-matched)

type: exp · date: 2026-11-12 · status: open
goal_effect: pending · scale: module

## Goal
GOAL attempt + eval repair (N038 diagnosis rev. 2). The authors' unmixed values use a smoothed background (background_lowess);
in the dim dye channel shared by CD3/CD19/CD56 its error gives 34-49 % FOV-level label variance no LF model can predict (within-FOV
GOAL .7211 vs pooled .6527). v8 = single-antibody labels re-derived from total_raw - per-cell `background` (/shading), unmixed with
source-split-0-fitted spectra (dye spectrum from the authors' positive-gate examples, autofluorescence basis from same-draw
unstained cells), then the existing v3->v5 calibration and the N036 cross-fitted position detrend; grp/autofl targets unchanged.

## Plan
Stage 1 (CPU, 0 GPU-h) label QC, PRE-REGISTERED, checked before any model is scored on v8: per single target x draw (a) FOV-level
excess share <= .05; (b) CD19 positive mixture component 1-6 % with d' >= 3 in both draws; (c) r(v8, v5) >= .95 for CD16/CD45/CD14;
(d) draw-1 lymphocyte stratum: r(label, own dye) >= 0 for HLA-DR/CD19/CD14 (else flag/mask that stratum, as N023). Fail -> stop, record.
Stage 2 (GPU) (i) clean in-distribution ceiling: r50 SSL-init FOV split on v5 (and --max-train matched to b01's 128k), scored on the
GOAL cells; (ii) GOAL recipe (N031 inductive base + zoom2, both directions) trained on v8, scored on the FROZEN v5 GOAL and on v8.
Expected: v5 GOAL +0 to +.005 (eval-label cap unchanged); v8-scored ~.70-.72 = re-base, NOT progress; user decides the re-base.
Kill: QC fails -> v8 not used. Compute: ceiling 2 x ~.5 GPU-h + retrain ~1.3 GPU-h, 1 H100 each, preempt.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N039 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N039-slug/ + external absolute paths -->

## Next
