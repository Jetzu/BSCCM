# N051 · Label v10: per-marker asinh cofactor (pre-registered label-only rule protecting rare bright modes, e.g. CD123), then GOAL re-base on v10 (user decision 2026-11-17)

type: fix · date: 2026-11-17 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal
User decision 2026-11-17 (asked after N048): 'Per-marker scale first' -- build a label version with a per-marker asinh cofactor that keeps
the rare CD123+ cells (erased by the global cofactor 150 in v9), pre-registered, then re-base the GOAL on it.

## Plan
PRE-REGISTERED (fixed before any model is scored on v10). Candidate grid per single marker: cofactor in {q10 (per slide, = v5), q50
(per slide), 15, 50, 150, 500}; one cofactor per MARKER (same in both draws -- a dye/instrument property); autofl and pooled groups
unchanged; everything else = v5cx path (clean calibration, cross-fitted detrend, per-slide z).
Per marker, metrics on split-0 clean cells of each draw, label-only: (a) FOV excess share (vs 20-permutation null), mean over draws;
(b) tail contrast = (mean of the top 0.5 % - median) / (1.4826 MAD), mean over draws (protects rare bright populations).
RULE: choose the candidate with the lowest (a), subject to (b) >= 0.5 x the q10 value of (b) (no rare mode erased) and, for CD16/CD45/
CD14, r(v10, v5) >= .95 per draw; ties within .005 in (a) -> the larger cofactor (closer to flow-cytometry practice). If no candidate
passes, keep q10 for that marker.
Re-base (after the build): score N031 (trained v5), N048 (trained v9) and a v10-trained N031 recipe on v10; measure v10 seed noise from
the existing seed pairs (n023_v5 s0/s1, n026_zoom2 s0/s1) scored on v10; update GOAL.md (labels, best, noise, re-base note) as the
user decided. Compute: CPU build; retrain ~1.3 H100-h.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Build 1375686 (grid 15/500) crashed in selection (pandas .tail attribute clash); rerun 1375702 (n051_build2.sh). n051_metrics.csv,
  n051_select.out, n051_selection.json. Selection (pre-registered rule): CD123 500, CD14 500, CD16 150, CD19 500, CD3 500, CD45 150,
  CD56 500, HLA-DR 500 (500 failed the r >= .95 lineage check for CD16 .941 and CD45 .935). Split-0 FOV excess q10 -> chosen: CD3 .373 -> .037,
  CD56 .517 -> .212, CD19 .457 -> .347, CD123 .065 -> .025; tail contrast never fell (CD123 49.5 -> 76.3). -> targets_v5cx_v10.npz.
- Re-score (n051_rescore.py -> n051_rescore.csv, n051_noise_v10.json): on v10 N031 ens .6703, N040 students .6736, 10-model .6744, N048
  (trained v9) .6804; v10 seed noise from two seed pairs .0031/.0045 -> sd .0027.
- Retrain on v10 (N031 recipe) jobs 1375725-28 -> n051_retrain_goal.csv, n051_retrain_per_target.csv; plot .agent-research/plots/goal-v10-rebase_2026-11-17_2125.png.
  On v10: N031 (trained v5) .6703, N048 (trained cf150) .6804 (+.0101, 13/22, p .025), N051 (trained v10) .6788 (+.0085, p .11);
  N051 vs N048 -.0017 (ns). On frozen v5: N051 .6502, N048 .6522, N031 .6527. In-distribution reference on v10 (n039 FOV, v5-trained):
  .6862 (data-matched .6646).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
DONE (re-base, user decision): GOAL.md now scores on targets_v5cx_v10 (per-marker cofactor CD123/CD14/CD19/CD3/CD56/HLA-DR 500,
CD16/CD45 150). Re-scored reference N031 .6703; best .6804 = N048 (trained on cofactor-150 labels), +.0101 > 2 x noise on the new
yardstick -> the clean-label retrain is the first real improvement since N026. The v5-era target .67 is met by the re-base itself;
new target pending the user.
<!-- on close: bin/effect.sh N051 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N051-slug/ + external absolute paths -->

## Next
