# N078 · In-domain SSL scaling: SIGReg LF pretraining 4x longer (r50, 160 ep) and a larger encoder (ConvNeXt-S, 60 ep), then the GOAL fine-tune; under-trained SSL (loss still falling at ep 40)

type: exp · date: 2026-11-23 · status: done
goal_effect: flat · scale: module · value: 0.6707

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
SSL: ex1 (draw 0) r50 160 ep 1:49 h, ConvNeXt-S 60 ep 1:11 h; ex0 (draw 1) jobs preempted and resumed by preempt-run (encoders complete:
160/160, 60/60 epochs). b10 fine-tunes stuck on afterok of the preempted first links -> cancelled, resubmitted 1438937/938.
RESULT (seed 0, v10, score_goal_arms + n078_mm3_ens.csv):
| arm | b10 | b01 | GOAL | MM3 |
| control n051_v10_base | .6867 | .6592 | .6730 | .7563 |
| r50 SSL 160 ep | .6817 | .6597 | .6707 (-.0022) | .7563 |
| ConvNeXt-S SSL 60 ep | .6941 | .6444 | .6693 (-.0037) | .7735 (+.0172) |
| N053 ensemble (GOAL best) | .7001 | .6723 | .6862 | .7734 |
| N053 ens + ConvNeXt-S | .7045 | .6728 | .6886 | .7773 |
PRIMARY (GOAL > +.0054): both arms FLAT. 4x longer r50 SSL changes nothing (SSL length is not the lever). ConvNeXt-S: b10 +.0074 (SSL on
the 275k-cell draw) but b01 -.0148 (138k-cell draw; CD123 .499, HLA-DR .440, CD19 .311); pooled panels up on both (grp:CD45 .716 vs .689,
grp:CD16 .816 vs .801, grp:CD3/19/56 .789 vs .780) -> MM3 +.017 single model (= the 3-member ensemble's MM3), secondary metric, 1 seed.
Plots: plots/goal-r-n078_2026-11-24_0758.png, plots/mm3-r-n078_2026-11-24_0758.png

## Outcome
<!-- on close: bin/effect.sh N078 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N078-slug/ + external absolute paths -->

## Next
Follow-up (pre-register MM3 as primary): ConvNeXt-S replication seed 1 + longer draw-0 SSL (the small draw is where it loses); as ensemble member.
