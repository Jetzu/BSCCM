# N090 · Feature-preserving fine-tuning of the in-domain SSL ResNet-50 (N088 A-P1/C-P2): encoder LR x0.1 / x0.3, then LP-FT and WiSE-FT weight interpolation; pre-registered vs control .6706

type: exp · date: 2026-11-24 · status: done
goal_effect: worse · scale: knob · value: 0.6617

## Goal

## Plan
Hypothesis (N088 lit A/C): full-LR fine-tuning distorts the in-domain SSL features that generalise across draws; a gentler update keeps
them (Kumar et al. 2022 LP-FT; Wortsman et al. 2022 WiSE-FT). In-house: Cytoland encoder LR x0.1 .6671 vs x1.0 .6464 (N053).
Stage 1 (now): n051_v10_base recipe (SSL ResNet-50, v10 labels, 10 ep, lr 5e-4, bs 256) with --enc-lr-mult 0.1 and 0.3, seeds 0 and 1,
both directions (8 runs, ~3 H100-h, preempt). Stage 2 (Codex code): LP-FT (heads first with the encoder frozen) and WiSE-FT (interpolate
SSL and fine-tuned encoder weights, alpha 0.25/0.5/0.75, evaluation only).
PRE-REGISTERED: primary = 2-seed mean GOAL per arm vs control n051_v10_base seeds 0/1 (.6706); improved if > .6760 (+2 x .0027).
Secondary: MM3; as a 4th member of the N053 ensemble (descriptive). Expected +.000-.008, mostly on b01. Kill: both arms <= .6706.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Jobs 1442638-45 (1 x H100/H200 preempt, 15-32 min each); 1442643 failed on a container mount timeout (squashfuse, infra) ->
  resubmitted as 1443460 (completed). Scored with scripts/score_goal_arms.py -> goal_arms.csv, goal_summary.csv.
- PRIMARY (2-seed mean vs control n051_v10_base seeds 0/1 .6706): enc-lr x0.1 .6454 (b10 .6606 / b01 .6302, -.0252); x0.3 .6617
  (b10 .6784 / b01 .6451, -.0089). Both beyond 2 x noise below the control -> WORSE; the closer to the SSL weights, the worse (mostly b01).
  Plot: .agent-research/plots/goal-direction-r-v10_2026-11-24_1942.png (plot_goal.csv).
- Reading: unlike Cytoland (pretrained on another task: x0.1 .6671 > x1.0 .6464), the in-domain SSL encoders must move far from their
  pretrained weights; the literature's LP-FT/WiSE-FT motivation (distortion of good OOD features) does not hold here. Stage 2 (LP-FT,
  WiSE-FT interpolation toward the SSL weights) dropped by the pre-registered kill criterion (both arms <= control).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
WORSE: gentle fine-tuning hurts the in-domain SSL ResNet-50 (x0.3 -.009, x0.1 -.025); full-LR fine-tuning stays.
<!-- on close: bin/effect.sh N090 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N090-slug/ + external absolute paths -->

## Next
