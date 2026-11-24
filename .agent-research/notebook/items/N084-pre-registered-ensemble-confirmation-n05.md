# N084 · Pre-registered ensemble confirmation: N053 GOAL-best ensemble + ConvNeXt-S member trained with a FRESH seed (b10 60-ep / b01 120-ep in-domain SSL encoders) vs the .6916 bar; MM3 reported

type: exp · date: 2026-11-24 · status: running
goal_effect: pending · scale: module

## Goal
N083: ConvNeXt-S in-domain SSL encoder replicates a multi-marker gain (MM3 +.0135, 2 seeds); descriptively (seed 0, already inspected)
the N053 ensemble + ConvNeXt-S reaches GOAL .6907 (bar .6916). Confirm with a fresh seed, pre-registered.

## Plan
Member: ConvNeXt-S seed 1 -- b10 = n083_cnxs60_s1 (existing seed-1 run on the 60-ep draw-1 SSL encoder, never scored in an ensemble),
b01 = n084_cnxs120_s1 (new seed-1 run on the 120-ep draw-0 SSL encoder, job in jobs.txt). Scorer fixed before the result: n084_score.py.
PRIMARY (pre-registered): GOAL of E = mean(N053 members + ConvNeXt-S seed 1) > .6916 -> improved GOAL (new best); else flat. MM3 reported.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N084 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N084-slug/ + external absolute paths -->

## Next
