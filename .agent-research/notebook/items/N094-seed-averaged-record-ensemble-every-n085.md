# N094 · Seed-averaged record ensemble: every N085 E8 member averaged over 2 seeds (fresh seed 5 for the 8 single-seed members: N048 base/zoom2, N065 composites z2/cyto, N085 ConvNeXt-S composite); pre-registered E8x2 vs .7021

type: exp · date: 2026-11-27 · status: open
goal_effect: pending · scale: knob

## Goal

## Plan
Hypothesis: E8's single-seed members carry seed noise; averaging 2 seeds per member lowers ensemble variance (+.002-.005 expected).
New runs (fresh seed 5, identical recipes to the archived members, new output dirs *_seed5_*): n048_v9_base, n048_v9_zoom2, n065
tpfov/panel z2, n065 tpfov/panel cyto, n085_cnxs120 tpfov/panel; both directions = 16 runs, ~8 H100-h (preempt).
PRE-REGISTERED (written before any new run finished): E8x2 = equal-weight mean over the same 8 members, each member = mean of its two
seeds (existing seed + seed 5; members that already have 2 seeds keep them: N063 composite seeds 2+3, cnxs_plain s0+s1; Cytoland
uses n053_cytoland + n053_cytoland_s1). PRIMARY: GOAL(E8x2) > .7021 (= .6967 + 2 x .0027). Secondary: E8 with only the seed-5
versions (independent replicate of E8), MM3. Kill: E8x2 <= .6967.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N094 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N094-slug/ + external absolute paths -->

## Next
