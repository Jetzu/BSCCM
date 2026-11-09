# N016 · Simulated mosaic with ground truth (pseudo-GS = BF+DPC+phase, pseudo-pivot = darkfield subsets of 1-17 patterns): transitive error vs bounds, distillation vs oracle

type: exp · date: 2026-11-09 · status: done
goal_effect: n/a · scale: paradigm · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Subset features: frozen SSL xillum r50 encoder, 120k stained cells, A = BF+4 DPC+phase, B = 1/2/4/8/17 darkfield patterns (job 1284554).
- Run 1 (ridge alpha=10 on 2048-d) INVALID: b exploded (RMSE up to 25) from overfit/extrapolation -> .agent-research/results/N016-simulated-mosaic-with-ground-truth-pseud/run1_alpha10_invalid/.
- Run 2 (RidgeCV on PCA-256; job 1284986; .agent-research/results/N016-simulated-mosaic-with-ground-truth-pseud/sim_mosaic.{json,csv}): coverage of the no-assumption bound [(a-b)^2,(a+b)^2] 0.95 overall (0.92-1.00 per pivot); CIA estimate a^2+b^2 median relative error +.05..+.15 (slightly conservative). Mean r over 12 markers (teacher / distilled / direct oracle): df01 .374/.457/.465, df02 .466/.478/.465, df04 .493/.480/.465, df08 .527/.488/.465, df17 .540/.487/.465.
- => the transitive error bound is empirically valid, the CIA error estimate is accurate within ~5-15 %, and distillation through the pivot matches direct supervision in this mosaic (distilled >= teacher when the pivot is weak: regression of the teacher onto the richer modality averages out teacher noise). Caveat: A and B come from one encoder/instrument, so CIA violations are milder than for real GS.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N016 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N016-slug/ + external absolute paths -->

## Next
