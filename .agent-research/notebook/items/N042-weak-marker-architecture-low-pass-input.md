# N042 · Weak-marker architecture: low-pass input probe (image- vs label-limited), then separate BF/DPC, DF-low, DF-high stems with late fusion + private late blocks for CD19/CD56/CD123/HLA-DR as an extra ensemble member

type: exp · date: 2026-11-12 · status: open
goal_effect: pending · scale: module

## Goal
Module: is the weak-target signal image-limited? If yes, give it a representation that keeps darkfield angular detail
and private capacity (Codex ideas 1+2; literature: resolution is the lever with controlled evidence).

## Plan
Stage 1 probe (falsifier): N031 base recipe on low-pass-filtered inputs (Gaussian sigma 1.5 px, all LEDs), both directions.
Image-limited if >= 2 weak targets drop >= .05 while lineage markers drop < .02; else label/count-limited -> stop stage 2.
Stage 2: new member = separate stems for BF+DPC+phase / DF-low / DF-high (GroupNorm), 32x32 grid kept to late fusion, shared trunk +
private last blocks + heads for CD19/CD56/CD123/HLA-DR; other targets from the reference. Blend weight chosen on training-draw
split-1 FOVs only. Score frozen v5 GOAL (member alone and blended with N031). Expected +.004-.015 if stage 1 says image-limited.
Compute: stage 1 ~.6 GPU-h, stage 2 ~2.5 GPU-h.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N042 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N042-slug/ + external absolute paths -->

## Next
