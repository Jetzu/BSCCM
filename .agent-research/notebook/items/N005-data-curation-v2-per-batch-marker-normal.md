# N005 · Data curation v2: per-batch marker normalisation vs unstained, correct columns, 3-class labels, batch-holdout split, LED flat-field

type: task · date: 2026-11-08 · status: done
goal_effect: n/a · scale: ? · value: - · why: infra

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- targets_v2.npz (scripts/make_targets_v2.py; build log results/N005-*/targets_v2_build.txt): 13 masked targets (8 single-antibody markers, 4 pooled 'all' groups calibrated vs same-batch unstained median/MAD, autofluorescence on all non-'all' cells incl. GS cells), per-(batch,target) z-scores, censored flags; cls3 lymph/gran/mono = 3,026 / 4,905 / 1,501 (b0 'all'); cls10 gates.
- Sanity: cls3 mean calibrated groups: lymph CD3/19/56 3.07 (gran 0.52, mono 1.02); gran CD16 4.59 (lymph -1.94); mono CD123/HLA-DR/CD14 11.96 (lymph 3.67) -> lineage-consistent.
- backgrounds_p50.npy (23 global flat fields; medians BF 3054, DF 244-520, DPC ~2000, LED119 394 -> ratio (raw-30)/(bg-30) stable for all channels).
- LF flat-field check (job 1275346, results/N005-*/led_norm_check.json): Spearman(background ring level, FOV radius) raw -> flat-fielded: BF -0.435 -> -0.040, DPC_Top -0.239 -> -0.028; LED119 ring CV 0.315 -> 0.031; slide scales 0.94-1.02.
- RULE: no rotation/flip augmentation on LF: DPC_L/R/T/B and DF_*_Bottom/_Right are direction-specific (rotation silently relabels illumination direction). Translations only.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N005 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N005-slug/ + external absolute paths -->

## Next
