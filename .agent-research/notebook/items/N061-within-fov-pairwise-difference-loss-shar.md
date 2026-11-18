# N061 · Within-FOV pairwise-difference loss (shared FOV label offsets cancel) + co-stained autofluorescence supervision removed from the shared encoder, GOAL r50 recipe, 2 seeds

type: exp · date: 2026-11-18 · status: open
goal_effect: pending · scale: module

## Goal
N059 review: FOV-level label error still costs ~.04 (within-FOV .726) and slide-specific pooled-panel readouts (grp:CD45 dim rep0 slide) are averaged into the shared head; co-stained autofl supervision may steer marker features (draw-dependent autofl slope, K029).

## Plan
Writer Codex (results/N060-*/codex/brief.md, one brief for N060-N062), Claude red-team + integration into scripts/train_lf.py.
GOAL recipe otherwise unchanged (SSL r50, base view, v10 labels, 10 epochs). 2 seeds x 2 directions per arm (per-item seed noise up
to .124 on CD123, N059 AB s8). PRE-REGISTERED bar: arm single-model GOAL (mean of 2 seeds) > r50 base .6730 + .0054, OR the arm as a
3rd member of N048 (base+zoom2) > .6862 + .0054 (= .6916). Report CD123-excluded GOAL alongside (CD123 carried 59 % of N053's gain).
Placement GPU 1 x H100 (r50 fine-tune).
Arms: (i) --fov-re 1 --w-autofl 0 (mixed-effects crowd layer: per-FOV offsets + per slide x class affine in the TRAINING loss
only; lit_crossbatch idea 1); (ii) --pair-loss + --w-autofl 0 (within-FOV pair differences; Codex idea 1); (iii) --w-autofl 0 alone
(ablation; Codex idea 3). Expected +.003-.007. Kill: no arm > base by .0054 and CD19/CD56 not up >= .01.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N061 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N061-slug/ + external absolute paths -->

## Next
