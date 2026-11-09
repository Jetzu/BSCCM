# N013 · Targets v4: remove slide-position artefact from all fluorescence-derived targets (labels only), GS-LF pair suspect flags; rerun baselines on v4

type: task · date: 2026-11-09 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- targets_v4.npz (scripts/make_targets_v4.py, job 1282095; results/N013-*/targets_v4_detrend_summary.csv, targets_v4_detrend.json): variance of each target explained by the smooth position trend (median over slides, fitted on split-0 cells): autofl .83, CD19 .82, CD56 .44, CD123 .28, CD3 .18, grp CD123/HLA-DR/CD14 .18, grp CD3/19/56 .15, grp CD45 .13, HLA-DR .12, CD45 .08, grp CD16 .06, CD14 .05, CD16 .03. GS-LF pair suspects flagged: 567.
- Design rule (user 2026-11-09): clean LABELS only; inputs keep all real-data artefacts (no position/metadata at inference); robustness via augmentation (GS: stain, blur, gamma, tint, illumination gradient, per-channel chromatic blur).
- Baselines on v4 (jobs 1281410-14: fov s0, b1to0 s0/s1, b0to1 s0/s1) -> GOAL re-baseline (metric definition changes with the targets).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- v4 baselines (jobs 1281410-14; results/N013-*/n013_v4_final_evals.json, n013_v4_noise.csv): mean r 12 markers FOV .635 (v3 .707) | b1to0 .556/.561 (v3 .494) | b0to1 .551/.550 (v3 .471). FOV-vs-batch gap .22 -> .08. Held-out-batch CD19 -.05/-.07 -> .28/.32, CD56 .20/.18 -> .35/.33, CD3 .46/.49 -> .55/.59, grp CD3/19/56 .66/.61 -> .67/.71, grp CD45 .73/.59 -> .73/.69. Within-lymph CD3/19/56 .34 (v3 .28), CD45 .37 (.26). cls3 .95.
- => the v3 within-batch numbers for CD19/CD56/CD123 were spatial artefact the LF encoder learned (LF features encode position r .92); with detrended labels the LF teacher transfers to a new isolation. GOAL re-initialised on v4 (best .5545, noise .002).

## Outcome
<!-- on close: bin/effect.sh N013 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N013-slug/ + external absolute paths -->

## Next
