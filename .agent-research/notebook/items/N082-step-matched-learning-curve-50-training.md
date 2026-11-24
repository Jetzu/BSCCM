# N082 · Step-matched learning curve: 50% training labels at 10 vs 20 epochs vs 100% at 10 (separates label quantity from optimisation steps; K033 confound)

type: exp · date: 2026-11-23 · status: done
goal_effect: n/a · scale: knob · value: - · why: measurement

## Goal
Review N079 red-team (reproduced): the K033 learning curve doubled gradient steps together with labels (train_lf.py steps = epochs x
len(dtr)), so "the model is data-limited" is confounded with optimisation length. Decides whether new LABELS (external data) or more
OPTIMISATION is the lever.

## Plan
GOAL r50 recipe (n051_v10_base: SSL ex0/ex1, lr 5e-4, bs 256, w-cls .5), seed 0, both directions; --max-train = 50 % of the training rows
(b10 134,641 of 269,283; b01 63,973 of 127,946; fixed random subset, rng 0). Arms: half@10 epochs (half the steps), half@20 epochs
(step-matched to the full-data control), control = n051_v10_base seed 0 (100 % @ 10). 4 runs, ~1.5 GPU-h.
PRE-REGISTERED reading: D_data = GOAL(100@10) - GOAL(half@20) (labels at equal steps); D_steps = GOAL(half@20) - GOAL(half@10).
"Label-limited" if D_data >= .010 and D_data > D_steps; "optimisation-limited" if D_steps >= .010 and D_data < .005; else mixed.
Expected (K033): D_data + D_steps ~ .015-.02.

## Changes

## Evidence
Jobs 1430385-388 (exit 0). Scored with scripts/score_goal_arms.py (v10), seed 0 vs control n051_v10_base seed 0 (.6730).
| arm | b10 | b01 | GOAL |
| 100 % @ 10 ep (control) | .6867 | .6592 | .6730 |
| 50 % @ 20 ep (step-matched) | .6736 | .6414 | .6575 |
| 50 % @ 10 ep | .6541 | .6380 | .6460 |
D_data = .0155 (labels at equal steps), D_steps = .0115 -> PRE-REGISTERED reading: LABEL-LIMITED (D_data >= .010 and > D_steps), with
a real optimisation component. Per target: CD123 is step-sensitive (.391 @10 -> .530 @20 vs .543 full); CD19 is label-sensitive
(.312 step-matched vs .344 full); CD56/HLA-DR/grp:CD16 label-sensitive (.480/.460/.777 vs .506/.479/.801).
-> one doubling of labelled cells is worth ~+.015 GOAL at fixed steps: new labelled cells (external data, N081) are the lever; longer
training alone is not (N073 20 ep at full data was flat/worse).
Plot: plots/goal-r-n082_2026-11-23_1744.png

## Outcome
<!-- on close: bin/effect.sh N082 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N082-slug/ + external absolute paths -->

## Next
