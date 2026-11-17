# N046 · Noisy student round 2: teacher = N040 student ensemble predictions on the training draw; same recipe (pseudo w .3, phys aug .7)

type: exp · date: 2026-11-17 · status: done
goal_effect: flat · scale: knob · value: 0.6546

## Goal
N040 noisy student gave a consistent but sub-threshold GOAL gain (+.0037 ensemble, 16/22 items, p .001; single model +.0103). Round 2:
teacher = N040 student ensemble (base+zoom2) predictions on the training draw; does the gain compound past 2 x noise (.0054)?

## Plan
Same recipe as N040 (N031 inductive SSL init, pseudo w .3 on common-11 columns lacking a real label, phys aug .7, 10 ep, seed 0),
base + zoom2, both directions. Baseline = N031 ensemble .6527 (GOAL best); also vs N040 .6564. Expected +.002-.006 over N040;
improved if > .6527 + .0054 = .6581. Compute ~2 H100-h (4 dumps + 4 students, preempt).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Jobs (jobs.txt): dumps 1374709/10/13/14 (N040 students -> train_preds.npz), students 1374711/12/15/16 (afterok) -> runs/n046_student2_*.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- Results (scores_goal.csv, scores_per_target.csv; plot .agent-research/plots/n046-student-round2-goal-r_2026-11-17_1921.png): round-2 ensemble .6546
  (b10 .6628, b01 .6464) vs N031 .6527 (+.0018, 16/22, p .033) and vs round-1 N040 .6564 (-.0018, 7/22, p .012); round-2 base .6510,
  zoom2 .6450; 4-model N040+N046 .6557.

## Outcome
FLAT, and worse than round 1: re-distilling the student's own predictions does not compound (confirmation bias of self-training;
the pseudo labels carry the same draw-specific errors). The useful part of N040 is one round of dense pseudo labels.
<!-- on close: bin/effect.sh N046 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N046-slug/ + external absolute paths -->

## Next
