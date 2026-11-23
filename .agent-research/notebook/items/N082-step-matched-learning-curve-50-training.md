# N082 · Step-matched learning curve: 50% training labels at 10 vs 20 epochs vs 100% at 10 (separates label quantity from optimisation steps; K033 confound)

type: exp · date: 2026-11-23 · status: running
goal_effect: pending · scale: knob

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
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N082 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N082-slug/ + external absolute paths -->

## Next
