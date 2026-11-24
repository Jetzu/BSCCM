# N084 · Pre-registered ensemble confirmation: N053 GOAL-best ensemble + ConvNeXt-S member trained with a FRESH seed (b10 60-ep / b01 120-ep in-domain SSL encoders) vs the .6916 bar; MM3 reported

type: exp · date: 2026-11-24 · status: done
goal_effect: flat · scale: module · value: 0.6911

## Goal
N083: ConvNeXt-S in-domain SSL encoder replicates a multi-marker gain (MM3 +.0135, 2 seeds); descriptively (seed 0, already inspected)
the N053 ensemble + ConvNeXt-S reaches GOAL .6907 (bar .6916). Confirm with a fresh seed, pre-registered.

## Plan
Member: ConvNeXt-S seed 1 -- b10 = n083_cnxs60_s1 (existing seed-1 run on the 60-ep draw-1 SSL encoder, never scored in an ensemble),
b01 = n084_cnxs120_s1 (new seed-1 run on the 120-ep draw-0 SSL encoder, job in jobs.txt). Scorer fixed before the result: n084_score.py.
PRIMARY (pre-registered): GOAL of E = mean(N053 members + ConvNeXt-S seed 1) > .6916 -> improved GOAL (new best); else flat. MM3 reported.

## Changes

## Evidence
Job 1440201 n084_cnxs120_s1_b01 (exit 0, 16 min). Pre-registered scorer n084_score.py (committed before the result, c80bc21):
| set | b10 | b01 | GOAL | MM3 |
| control r50 | .6867 | .6592 | .6730 | .7563 |
| ConvNeXt-S seed 1 | .6862 | .6627 | .6744 | .7677 |
| N053 ensemble (GOAL best) | .7001 | .6723 | .6862 | .7734 |
| E = N053 + ConvNeXt-S seed 1 (PRIMARY) | .7030 | .6793 | .6911 | .7764 |
PRIMARY GOAL(E) .6911 < bar .6916 -> FLAT (by .0005). Consistent with the seed-0 descriptive .6907: ConvNeXt-S adds a reproducible
+.0045-.0049 to the GOAL-best ensemble (1.7-1.8 x noise), mostly on b01 (+.007) and the pooled panels (MM3 +.003).
Plot: plots/goal-r-n084_2026-11-24_1116.png

## Outcome
<!-- on close: bin/effect.sh N084 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N084-slug/ + external absolute paths -->

## Next
The registered GOAL best stays .6862 (N053). ConvNeXt-S is the best new ensemble member (+.005 twice) and the best single model for multi-marker decoding.
