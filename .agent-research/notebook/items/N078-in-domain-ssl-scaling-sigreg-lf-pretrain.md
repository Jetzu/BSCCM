# N078 · In-domain SSL scaling: SIGReg LF pretraining 4x longer (r50, 160 ep) and a larger encoder (ConvNeXt-S, 60 ep), then the GOAL fine-tune; under-trained SSL (loss still falling at ep 40)

type: exp · date: 2026-11-23 · status: running
goal_effect: pending · scale: module

## Goal
User 2026-11-23: maximise marker recovery from the LF images; other encoders may be fully fine-tuned. N053: external pretrained encoders
(DINOv3, DinoBloom, Cytoland; all weights trained) do not beat the in-domain SIGReg-SSL r50 as single models. That SSL (N031 config:
arm aug, r50, 40 ep, bs 512, lr 2e-3, lam .05, rows all of the training draw) was cheap (1.9 h ex0 / .6 h ex1) and stopped while its
loss was still falling under cosine decay (ep20->30 -23 %, 30->35 -8 %; ex1 = draw-0 encoder for the weak b01 direction ends higher,
.278 vs .184). Hypothesis: the in-domain encoder is under-trained / under-sized.

## Plan
Arms (identical SSL recipe otherwise; held-out draw excluded = inductive): r50_160 = resnet50 160 epochs; cnxs_60 = convnext_small
(49.5 M params, 48-ch stem) 60 epochs. Then the GOAL fine-tune recipe of n051_v10_base (10 ep, lr 5e-4, bs 256, w-cls .5) from each
encoder, seed 0, both directions (screen). Cost ~19 GPU-h (SSL r50 ~10, ConvNeXt-S ~7.5, fine-tunes ~1.8), 1 x H100 each, preempt.
PRIMARY (pre-registered): GOAL(seed 0) vs control seed 0 (n051_v10_base: .6706 two-seed mean; seed-0 value from score_goal_arms);
success if > +.0054. Also MM3 (multi-marker metric, user priority) vs control. Expected +.000..+.010. If a winner: 2nd seed,
then ensemble + N076 final multi-marker stage.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N078 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N078-slug/ + external absolute paths -->

## Next
