# N085 · Stronger members: ConvNeXt-S with 120-ep in-domain SSL on both draws + composite heads (panel for pooled groups, twopart/fov-re singles); pre-registered union ensemble of all screened members vs .6916

type: exp · date: 2026-11-24 · status: running
goal_effect: improved · scale: module · value: 0.6967

## Goal
User 2026-11-24 "make it better and go on". Combine the ingredients that each showed a gain: ConvNeXt-S in-domain SSL (MM3 +.0135 over 2
seeds; +.005 to the GOAL ensemble twice; 120-ep SSL fixed the small draw), the N063 composite heads (+.0106 single r50), the N065
composite members, the N053 members.

## Plan
New: ConvNeXt-S 120-ep SSL on draw 1 (ex0, ~5 GPU-h); ConvNeXt-S 120-ep composite fine-tunes (tpfov + panel, seed 0) both directions;
ConvNeXt-S 120-ep plain b10 seeds 0+1. b10 runs are submitted by hand when the SSL finishes (no afterok on preemptible chains).
Scorer pre-registered BEFORE any new member exists: n085_score.py (E8 = 8 equal-weight members, all single-model-screened).
PRIMARY: GOAL(E8) > .6916. Secondary: MM3, E6 (no ConvNeXt), each ConvNeXt member alone. ~10 GPU-h total.

## Changes

## Evidence
- b10 members (draw-1 120-ep ConvNeXt-S SSL, n085_ssl_cnxs_120_ex0, chain 1440484 -> 1441407 -> 1441972, preempted twice, resumed):
  1442517/18 plain s0/s1, 1442519 tpfov s0, 1442520 panel s0 (28 min each, 1 x H100 preempt); b01 members 1440485/86 + N083/N084 runs.
- PRE-REGISTERED scorer n085_score.py -> n085_scores.csv: E8 (PRIMARY) b10 .7066 / b01 .6869 = GOAL .6967 (MM3 .7856) > bar .6916 ->
  IMPROVED; E6 (no ConvNeXt-S) .6929; N053 .6862; cnxs_plain alone .6751; cnxs_comp alone .6690 (MM3 .7810).
  Plot: .agent-research/plots/goal-direction-r-v10_2026-11-24_1917.png (plot_goal.csv).
- Codex red-team (codex_rt/): NOTHING BREAKS the registered claim: canonical recompute b10 .706573 / b01 .686887 = GOAL .696730; all 15
  prediction files per direction share identical ordered eval rows (13,113 / 26,322); all 30 fine-tunes used the correct source draw; all
  4 SSL checkpoints excluded every held-out row; registration commit 499bca4 (11:39) precedes the first new fine-tune (11:54).
  Leave-one-out GOAL all > .6916 (.69536-.69764). MAJOR LIMITATION (reproduced from the items): the 6 older members were selected after
  their test-draw scores were known (E6 = .6929 was N065's descriptive secondary; N053 picked Cytoland among 4 post hoc); the two new
  ConvNeXt-S members add +.0038 over E6 (< 2 x noise); the 120-ep choice followed the 60-ep result. Only a third, untouched draw could
  test generalisation (BSCCM has two). MINOR: LED calibration statistics sample both draws (shared by every run since N005).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
IMPROVED by the pre-registered rule: GOAL .6967 (E8) > .6916; new GOAL record (was .6862, N053). Caveat: the increment from members that
were not chosen with test-draw knowledge is +.0038 (< 2 x noise); report the record with this selection caveat.
<!-- on close: bin/effect.sh N085 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N085-slug/ + external absolute paths -->

## Next
