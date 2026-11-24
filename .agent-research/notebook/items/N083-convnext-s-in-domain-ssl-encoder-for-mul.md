# N083 · ConvNeXt-S in-domain SSL encoder for multi-marker decoding: seed replication + 2x longer draw-0 SSL; MM3 pre-registered primary (user priority), GOAL secondary

type: exp · date: 2026-11-24 · status: running
goal_effect: pending · scale: module

## Goal
N078 secondary signal: ConvNeXt-S (49.5 M) with in-domain SIGReg SSL raised the multi-marker score MM3 (3 pooled panels) from .7563 to
.7735 as a single model (= the 3-member N053 ensemble), b10 +.0074, but b01 -.0148 (draw-0 SSL on only 138k cells, 60 ep). User priority
2026-11-23: joint multi-marker decoding. Replicate and fix the small-draw weakness.

## Plan
(a) seed-1 GOAL fine-tunes on the N078 ConvNeXt-S encoders (both directions; same recipe: 10 ep, lr 5e-4, bs 256, w-cls .5);
(b) draw-0 ConvNeXt-S SSL 120 ep (2x N078) -> b01 fine-tune seed 0 (queued after the SSL finishes; NO afterok on a preemptible chain).
PRIMARY (pre-registered): MM3 of ConvNeXt-S averaged over seeds 0+1 (both directions) minus control MM3 (n051_v10_base seeds 0+1,
.7577) > +.007 (2 x MM3 seed sd .0035). SECONDARY: GOAL; b01 of the 120-ep encoder vs the 60-ep encoder; as ensemble member (descriptive).
Cost ~4 GPU-h.

## Changes

## Evidence
Jobs: 1439071/072 seed-1 fine-tunes (exit 0); 1439073 draw-0 SSL 120 ep (running). Scored with scripts/score_goal_arms.py (v10).
PRIMARY (pre-registered, MM3 over seeds 0+1, paired): ConvNeXt-S .7735 / .7688 vs control .7563 / .7590 -> +.0172 / +.0098, mean +.0135
> bar +.007 = REPLICATED multi-marker gain (user priority). GOAL secondary: .6681 vs .6706 (-.0025; b10 .6901 vs .6810 +.009, b01 .6460
vs .6603 -.014): the small-draw (b01) weakness also replicates -> the 120-ep draw-0 SSL arm tests it (pending).

## Outcome
<!-- on close: bin/effect.sh N083 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N083-slug/ + external absolute paths -->

## Next
