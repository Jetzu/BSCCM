# N094 · Seed-averaged record ensemble: every N085 E8 member averaged over 2 seeds (fresh seed 5 for the 8 single-seed members: N048 base/zoom2, N065 composites z2/cyto, N085 ConvNeXt-S composite); pre-registered E8x2 vs .7021

type: exp · date: 2026-11-27 · status: done
goal_effect: flat · scale: knob · value: 0.6973

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
16 seed-5 fine-tunes, jobs 1468131-1468146 (results/N094-*/jobs.txt), all COMPLETED exit 0 (11-30 to 52 min each, ~10 H100-h),
outputs /hnvme/workspace/v181bb10-bsccm/runs/<member>_seed5_<b10|b01>. Pre-registered scorer n094_score.py (commit 5e98c51, before results):
| set | b10 | b01 | GOAL | MM3 |
|---|---|---|---|---|
| E8x2 (PRIMARY) | .7091 | .6855 | **.6973** | .7853 |
| E8 seed-5 replicate | .7101 | .6821 | .6961 | .7844 |
| E8 (N085 record, recomputed) | .7066 | .6869 | .6967 | .7856 |
The recomputed E8 row reproduces N085 exactly (scorer sanity). Plot: .agent-research/plots/goal_2026-11-27_1635.png

## Outcome
FLAT: E8x2 .6973 < bar .7021 (+.0006 over the record, inside noise). Useful by-product: an independent seed-5 replicate of the
whole record ensemble scores .6961 (-.0006), so the .6967 record replicates across a full re-draw of 8 member seeds; seed noise
is not what limits the ensemble (averaging 2 seeds per member adds nothing).

## Outputs
n094_score.py, n094_scores.csv, plot_goal.csv, jobs.txt; runs on /hnvme (seed-5 members, eval_preds.npz).

## Next
Ensemble seed variance is exhausted; stall 5/5 -> review.
