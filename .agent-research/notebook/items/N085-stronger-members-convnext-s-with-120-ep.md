# N085 · Stronger members: ConvNeXt-S with 120-ep in-domain SSL on both draws + composite heads (panel for pooled groups, twopart/fov-re singles); pre-registered union ensemble of all screened members vs .6916

type: exp · date: 2026-11-24 · status: running
goal_effect: pending · scale: module

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
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N085 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N085-slug/ + external absolute paths -->

## Next
