# N098 · Lineage classification heads for larger T/B (all-lineage) separation (user 11-27): per-marker positive-vs-negative heads trained within each source slide vs the regression-output score

type: exp · date: 2026-11-27 · status: open
goal_effect: pending · scale: module

## Goal
User 11-27 (after fig14/fig15): "make like larger separations here it looks like CD3 and CD19 are not that well separable".
Secondary metric (lineage separability, not GOAL); run under STALL_OVERRIDE while review N095 is open.

## Plan
PRE-REGISTERED 2026-11-27 before results. Hypothesis: lineage separation is weak because MSE regression of rare, dim markers is
dominated by the negative bulk (CD19 r .36); positive-vs-negative classification heads (class-balanced) trained within each source
slide sharpen the positives. Frozen n051_v10_base features, both directions, CPU (n098_lineage_heads.py).
Baseline = the same model's regression outputs, score z(out a) - z(out b) (fig14 method). PRIMARY: T vs B AUROC on the test draw, b10
(39 B vs 89 T; b01 has only 10 B cells -> descriptive). Success = heads beat regression by >= .05 on b10 T vs B AND the mean over all
15 lineage pairs (both directions) rises by >= .02, with slide control |AUROC - .5| <= .1 for every pair. Kill: T vs B gain < .02.
AMENDMENT (user 11-27 "train a classifier head on this", before any N098 result was read): third arm "multiclass" = one 6-lineage multinomial head (n098_multiclass.py), same metric + slide control; descriptive unless its slide control stays within .5 +- .1.
If it succeeds: add the heads to fine-tuning (GPU, needs its own item) and re-score E8-style ensembles.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N098 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N098-slug/ + external absolute paths -->

## Next
