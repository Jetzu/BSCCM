# N068 · Exact-symmetry augmentation: anti-diagonal reflection with LED channel swap (Bottom<->Right halves, DPC pairs) + 2-view TTA, LED119 dropped; vs all-channel and no119 controls

type: exp · date: 2026-11-19 · status: done
goal_effect: flat · scale: module · value: 0.6694

## Goal
Outsider idea 2 (review N066): the GOAL recipe trains with +-4 px translations only; rotations/flips were excluded because LED channels
are directional (bsccm/data.py:42). n066_d4_aniso: the anti-diagonal reflection maps each Bottom half-ring / DPC channel onto its
Right partner (r .72-.94 on real normalised cells; DF_80 near-isotropic .23); full rings are not sums of halves, so this 2-element
group is the only exact symmetry of the acquired set; LED119 has no partner -> marked absent. Doubles the effective training views on
the label-starved draw-0 targets and adds 2-view test-time averaging.

## Plan
train_lf.py: --diag-aug p (random anti-diagonal reflection + channel swap after normalisation/zoom), --diag-tta 1 (eval = mean of the
identity and reflected view), --drop-led LED119 (data 0 + SSL presence plane 0 in every sample; the SSL encoders were pretrained with
random channel dropout in this format; SSL init asserts channels=all). Self-test n068_diag_selftest.csv (involution, LED119 refusal,
mean anisotropic images of swapped channels match partners .72-.94 vs <= .19 for reflection without swap).
Arms (GOAL r50 recipe of n051_v10_base: SSL ex0/ex1, 10 ep, lr 5e-4, bs 256, w-cls .5; seeds 0,1; both directions):
 drop119 = --drop-led LED119; diag = --drop-led LED119 --diag-aug .5 --diag-tta 1. 8 runs, ~3.6 GPU-h, 1 x H100 each, preempt.
PRIMARY (pre-registered): mean GOAL (v10, seeds 0+1) of diag vs the all-channel control .6706 (n051_v10_base + _s1); success if
delta > +.0054. Secondary (attribution): diag - drop119 (augmentation+TTA), drop119 - .6706 (LED119 removal). Expected +.002..+.006.
Kill: delta <= .0027. If success: combine with N067 if that also passes, and rebuild ensemble members (separate item).
Queued during the cluster downtime (2026-11-19 04:00 -> 11-20 00:00); v1 with --channels no119 was cancelled before start (jobs.txt).

## Changes

## Evidence
Scored with scripts/score_goal_arms.py on frozen v10 labels; control = n051_v10_base + _s1 (GOAL .6706: b10 .6810, b01 .6603). Jobs 1391819-826 (exit 0; v1 1391811-818 cancelled before start).
RESULT: drop119 GOAL .6718 (+.0011), diag (drop119 + anti-diagonal aug p .5 + 2-view TTA) .6694 (-.0012) vs .6706 -> both FLAT.
Augmentation/TTA effect (diag - drop119) -.0024; LED119 removal ~0. Per target diag vs control within +-.02 (CD123 .539 -> .516,
CD19 .320 -> .306, CD3 .768 -> .777). The 2x symmetry augmentation does not add usable views for this model/budget.
Plot: plots/goal-r-n068_2026-11-23_1455.png

## Outcome
<!-- on close: bin/effect.sh N068 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
results/N068-*/: n066_diag_selftest.py, n068_diag_selftest.csv, n066_d4_aniso.csv, jobs.txt, goal_arms.csv, goal_summary.csv, plot_goal.csv; runs/n068_*

## Next
Dead: symmetry augmentation/TTA. LED119 carries ~nothing (drop is free).
