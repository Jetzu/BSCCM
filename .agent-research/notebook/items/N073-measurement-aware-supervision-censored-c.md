# N073 · Measurement-aware supervision: censored (clip-floor) labels as upper bounds + 20-epoch schedule on the GOAL r50 recipe

type: exp · date: 2026-11-19 · status: done
goal_effect: flat · scale: module · value: 0.6658

## Goal
Assumption breaker (review N066, agents/assumption_breaker.md; reproduced here): clip-floor ('censored') labels are treated as exact
values by the MSE although the true value is <= the label, and the censored fraction differs sharply between draws (CD3 .094/.427,
grp:CD16 .307/.082, grp:CD45 .095/.179, CD56 .087/.115). Held-out r of the 10-epoch control rises until epoch 8 and is flat at 8-10
(n051_v10_base b01 .6598/.6592, s1 .6616/.6613) under OneCycle annealing; the SSL recipe's length was never varied (only N014,
random init, ran 15 epochs).

## Plan
train_lf.py --censor-loss 1: on censored cells the target is replaced by the detached prediction wherever the prediction is already
below the label (zero loss/gradient), i.e. a one-sided squared loss = Tobit limit for an upper-bound label. Bundled with --epochs 20.
Arm cens20: GOAL r50 recipe (n051_v10_base: SSL ex0/ex1, lr 5e-4, bs 256, w-cls .5) + --censor-loss 1 --epochs 20; seeds 0,1; both
directions; 4 runs x ~0.8 GPU-h = ~3.2 GPU-h (1 x H100 each, preempt).
PRIMARY (pre-registered): mean GOAL (v10, seeds 0+1) vs control .6706 (n051_v10_base + _s1); success if delta > +.0054. Expected
+.000..+.006 (N059 monotone oracle caps tail/amplitude fixes near +.0045). Kill: <= .0027. If success: ablate (censor at 10 ep;
20 ep without censor) before ensembling.

## Changes

## Evidence
Scored with scripts/score_goal_arms.py on frozen v10 labels; control = n051_v10_base + _s1 (GOAL .6706: b10 .6810, b01 .6603). Jobs 1391827-830 (exit 0, ~46/32 min).
RESULT: cens20 GOAL .6658 (b10 .6719, b01 .6597) vs .6706 -> delta -.0048 = FLAT (within 2 x noise, negative). b10 loses -.009
(20 epochs on the larger draw-1 set likely overfits; CD123 .539 -> .520, grp:CD16 .803 -> .787); CD19 .320 -> .349 (+.03, the one
target where clip-floor handling plausibly helps). Bundled arm -> no ablation run (kill criterion met).
Plot: plots/goal-r-n073_2026-11-23_1455.png

## Outcome
<!-- on close: bin/effect.sh N073 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
results/N073-*/: jobs.txt, goal_arms.csv, goal_summary.csv, plot_goal.csv; runs/n073_*

## Next
Dead as bundled. Parked: censor loss alone at 10 epochs for CD19 only (small).
