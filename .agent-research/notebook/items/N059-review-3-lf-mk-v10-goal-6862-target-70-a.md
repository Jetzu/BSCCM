# N059 · Review 3: LF->MK v10 GOAL (.6862, target .70) after 4 flat attempts (N054-N057); ensembling saturated, cross-draw best = in-distribution reference

type: review · date: 2026-11-18 · status: done
goal_effect: pending · scale: ?

## Goal
Review 3 of the LF->MK v10 GOAL (best .6862 N053, target .70, noise .0027) after 4 flat attempts N054-N057; opened proactively at
4/5 because ensembling saturated (K039) and the cross-draw best equals the in-distribution single-model reference.

## Plan
Step 1: declined.md, attempts.md ("All k attempts assumed that the remaining LF->MK error is predictor error that a better/more
diverse image model can remove, with the v10 labels of the held-out draw treated as a fixed, reachable truth.").
Step 2: diagnosis.md (n059_diag.py -> n059_diag.csv): draw shift ~0 on average (ens .6862 = in-dist .6862); oracle per-slide
affine +.003 (calibration not the gap); within-FOV .726 (+.04 FOV-level label error); Pearson .686 vs Spearman .531 (tail-driven);
deficits vs in-distribution concentrated in HLA-DR (-.04/-.14) and pooled grp:CD45 (-.10/-.05), grp:CD16 (b10 -.03); CD19/CD56
weak for both models (label-limited). Hypothesis: reachable gap = draw-specific readout of HLA-DR + pooled groups.
Step 3 agents: lit_crossbatch, outsider, assumption_breaker (Claude), codex_ideas, codex_diag_redteam (copies in agents/);
lit_noisy_ceiling pending. Every red-team / assumption-breaker finding used below was reproduced first-hand (diagnosis.md corrections).
Step 4 (ideas.md): ensembling closed (greedy test-selected peak .6922, r_inf .693); remaining shift sits in the held-out labels
(draw-1 HLA-DR tracks autofl r -.606; grp:CD45 deficit on one dim slide) -> inductive upside <= ~.01; few-shot new-draw readout
.7042 needs >= 200-500 labelled new-draw cells (protocol change -> user decision); weak markers are rare-positive problems.
Portfolio: N060 (module) rare-positive heads; N061 (module) training-only nuisance modelling (crowd layer / pair loss / no autofl);
N062 (paradigm) panel-matrix latent head. Parked: spatial descriptors, split-half reliability ceiling, learning curve, readout hedge,
per-target stacker (TODO.md). ~9 GPU-h total.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N059 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N059-slug/ + external absolute paths -->

## Next
