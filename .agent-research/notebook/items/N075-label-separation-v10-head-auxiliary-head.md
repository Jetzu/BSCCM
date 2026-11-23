# N075 · Label separation: v10 head + auxiliary head on the local-corner-background label version (shared encoder), eval on the v10 head; vs aggregation (N067) and control

type: exp · date: 2026-11-19 · status: done
goal_effect: flat · scale: module · value: 0.6708

## Goal
Review N066 follow-up (ideas.md #5; lit_labelnoise I1): with two noisy label versions, keeping them SEPARATE beats aggregating them when
noise is high or labels are scarce (Wei et al. ICML 2023, verified). N067 is the aggregation arm (consensus targets); this is the
separation arm: the scored v10 head is unchanged, a second linear head on the shared features fits the local-corner-background version.

## Plan
train_lf.py --aux-label targets_v5cx_localbg.npz --aux-w 1.0 (Claude, small: extra Linear head 'reg2', masked MSE on cells with a v10 label
and a finite local label; eval/prediction = v10 head only; flag off = legacy, verified: Codex N074 test suite 7/7 vs the pre-N074 harness
sha 88f70936 + a flag-on 2-epoch smoke with --censor-loss, verify/). Local labels: n067_make_targets.py also writes the local-bg version
alone (per-draw rescaled to v10 split-0 mean/sd; ALL 11 targets incl. CD19, no rule).
Arm sep: GOAL r50 recipe (n051_v10_base) + aux head; seeds 0,1; both directions; 4 runs x ~.45 GPU-h, afterok on the N067 target job.
PRIMARY (pre-registered): mean GOAL (v10, seeds 0+1) vs control .6706; success if delta > +.0054. Secondary: sep vs N067 cons (same
seeds), per target CD56/HLA-DR/grp. Expected +.003..+.008 (lit). Kill: <= .0027.

## Changes

## Evidence
Scored with scripts/score_goal_arms.py on frozen v10 labels; control = n051_v10_base + _s1 (GOAL .6706: b10 .6810, b01 .6603). Jobs 1392079-082 (exit 0).
RESULT: sep GOAL .6708 (b10 .6749, b01 .6667) vs .6706 -> +.0002 = FLAT. Direction split: b01 +.0064, b10 -.0061 (cancels).
vs aggregation N067 (.6624): separation avoids the consensus damage (CD56 .504 kept) but adds nothing; CD123 .539 -> .559.
Plot: plots/goal-r-n075_2026-11-23_1455.png

## Outcome
<!-- on close: bin/effect.sh N075 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
results/N075-*/: verify/ (flag-off equality + aux smoke), jobs.txt, goal_arms.csv, goal_summary.csv, plot_goal.csv; runs/n075_*

## Next
Dead: second label version as auxiliary head.
