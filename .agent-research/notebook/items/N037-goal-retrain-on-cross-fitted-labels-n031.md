# N037 · GOAL retrain on cross-fitted labels: N031 inductive base + zoom2 fine-tunes, both directions, trained on targets_v5cx (Giemsa-free fits + FOV-fold cross-fitted position detrend, N036) instead of v5; scored by GOAL (v5) and on v5cx

type: exp · date: 2026-11-12 · status: done
goal_effect: flat · scale: module · value: 0.6537

## Goal
GOAL attempt (label curation, module scale): does training on cleaner position-detrended labels move the GOAL? N036 found the v4/v5
detrend fitted in-sample on split-0 cells, absorbing split-0 FOV-composition noise in TRAINING labels (GOAL test cells are splits 1+2
only, so the test side was never affected). v5cx = Giemsa-free calibration fits + FOV-fold cross-fitted trend kept only where it lowers
out-of-fold variance.

## Plan
Recipe frozen = N031 (GOAL best .6527): inductive SSL(aug) encoders n031_ssl_aug_ex0/ex1, resnet50, 10 ep, bs 256, lr 5e-4, w_cls .5,
seed 0, base (zoom 1) + zoom2, dual-zoom, ensemble = mean of base + zoom2 predictions; ONLY change: --targets targets_v5cx.npz.
Score with N031's scorer on v5 (GOAL as defined) and on v5cx (n036_rescore-style). Baseline = N031 ensemble on the same label file:
v5 .6527, v5cx .6547. Expected: +0 to +.005 (training-label noise removal on split-0 cells only); flat if |diff| <= .004.
Compute: 4 x 1 H100 on preempt (~0.2-0.5 h each by N031 timings, ~1.5 GPU-h total, ResNet50 image training = GPU), wandb offline,
synced by the helma2 cron (whole WANDB_DIR).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Jobs results/N037-goal-retrain-on-cross-fitted-labels-n031/jobs.txt: 1312682 n037_base_b10, 1312683 n037_zoom2_b10, 1312684
  n037_base_b01, 1312685 n037_zoom2_b01 (runs/n037_*), submitted 2026-11-12; all COMPLETED (26, 28, 11, 14 min; ~1.3 GPU-h).
- Scores (n037_score.py -> n037_goal.csv, n037_per_target.csv; plot .agent-research/plots/goal-v5cx-retrain_2026-11-12_1736.png):
  GOAL common-11 (v5 labels, as defined), ensemble: N037 .6537 (b01 .6472, b10 .6602) vs N031 .6527 (b01 .6443, b10 .6612): +.0010.
  base .6429 vs .6425, zoom2 .6445 vs .6442. Scored on v5cx: ens .6555 vs .6547; on v5c .6541 vs .6532. Per target all within
  +-.006 except CD19 b01 .417 vs .395.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
FLAT. Training on cross-fitted, Giemsa-free-calibrated labels changes the GOAL by +.0010 (< 2 x noise .004; b01 +.003, b10 -.001),
on every label version. The split-0 in-sample detrend (N036 / F6) did not limit the models: the label-position flaw is real but
not a GOAL bottleneck. v5cx becomes the default training label only if the N036 held-out diagnostics pass; GOAL definition unchanged.
<!-- on close: bin/effect.sh N037 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N037-slug/ + external absolute paths -->

## Next
