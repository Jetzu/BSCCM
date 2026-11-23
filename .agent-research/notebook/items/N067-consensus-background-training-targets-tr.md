# N067 · Consensus-background training targets: train on the mean of v10 and local-corner-background labels (N064 reconstruction for all cells), evaluate on v10

type: exp · date: 2026-11-19 · status: done
goal_effect: worse · scale: module · value: 0.6624

## Goal
Review N066 diagnosis: the 6-way ensemble scores .6929 vs v10 labels but .6996 vs the consensus z(v10)+z(local-corner-background)
(n066_altlabels.csv) -> part of the label error is background-model error independent between estimators. Training labels carry the
same error; weak targets are label-starved (K033). Train on consensus targets, keep v10 as the evaluation label.

## Plan
1. n067_alllabels.py (= N064 Codex pipeline + --all-cells): local-corner-background ("full") labels for all masked GOAL cells, both
   draws; same frozen v10 curation/replay checks (CPU 48 cores, preempt_cpu; ~10 min).
2. n067_make_targets.py -> targets_v5cx_v10cons.npz. PRE-REGISTERED label-only rule: per (draw, target) consensus = mean of z(v10)
   and z(local) on that draw's split-0 cells, rescaled to v10 mean/sd, used only if r(v10, local) >= .5 on that draw's split-0 cells
   (expected: CD19 stays v10 in both draws).
3. r50 GOAL recipe identical to the control n051_v10_base{,_s1} (SSL init ex0/ex1, 10 ep, lr 5e-4, bs 256, w-cls .5), only --targets
   changed; seeds 0, 1; both directions. 4 runs x ~0.4 GPU-h = ~1.6 GPU-h, 1 x H100 each (preempt).
PRIMARY (pre-registered): mean GOAL (v10 labels, seeds 0+1) of n067_cons vs control .6706 (n051_v10_base + _s1); success if
delta > +.0054 (2 x noise). Expected +.003..+.010 (consensus gains concentrated on CD56/HLA-DR/groups). Kill: delta <= .0027.
If success: rebuild the ensemble members on consensus targets (separate item).
Cluster downtime 2026-11-19 04:00 -> 11-20 00:00: jobs queued with dependencies (jobs.txt).

## Changes

## Evidence
Scored with scripts/score_goal_arms.py on frozen v10 labels; control = n051_v10_base + _s1 (GOAL .6706: b10 .6810, b01 .6603). Jobs 1391804 (labels, 1:50 min, curation replay + pixel fidelity checks passed), 1391805 (targets), 1391806-809 (exit 0).
Rule (n067_rule.csv): consensus used for 20/22 (draw, target) cells; CD19 kept v10 in both draws (r v10-local .21/.23).
RESULT: cons GOAL .6624 (b10 .6667, b01 .6580) vs control .6706 -> delta -.0083 (paired, 2 seeds) = WORSE (> 2 x noise).
Per target (mean of 2 seeds x 2 directions): CD56 .504 -> .444 (the main loss), grp:CD45 .690 -> .671, HLA-DR .481 -> .467,
CD123 .539 -> .540, CD19 .320 -> .323, CD3 .768 -> .776. Training on the consensus moves predictions AWAY from what v10 measures on the
targets where the two background models disagree -> the local-corner background is not a cleaner version of the same quantity for
CD56/HLA-DR/grp:CD45 (consistent with the Codex red-team non-identifiability; the N066 consensus scoring gain was not usable).
Plot: plots/goal-r-n067_2026-11-23_1455.png

## Outcome
<!-- on close: bin/effect.sh N067 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
results/N067-*/: n067_alllabels.py, n067_labels.sbatch, n067_make_targets.py, n067_rule.csv, reconstruction/curation/pixel fidelity csv, goal_arms.csv, goal_summary.csv, plot_goal.csv; /hnvme/workspace/v181bb10-bsccm/{targets_v5cx_v10cons.npz,targets_v5cx_localbg.npz,runs/n067_*}

## Next
Label-version consensus is dead as a TRAINING signal. Separation (N075) tested the other way.
