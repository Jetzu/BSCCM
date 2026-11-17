# N050 · Heterogeneous ensemble member: different LF backbone (ConvNeXt-T) trained on real + N040 pseudo labels to decorrelate the shared errors of the ResNet-50 family (member residual r .97)

type: exp · date: 2026-11-17 · status: open
goal_effect: pending · scale: module

## Goal
Review N047 / Codex red-team: the 10 GOAL-family members share encoder, recipe and labels (residual r .95-.97, n_eff ~1), so ensembling
adds little. A member with a different inductive bias (ConvNeXt-T, no shared SSL init) may make different errors.

## Plan
convnext_tiny (timm, in_chans 24, random init), N040 recipe otherwise (pseudo labels from the N031 ensemble on the training draw, w .3,
phys aug .7, 10 ep, bs 256, lr 5e-4, seed 0), base view, both directions. Score: alone, and added to the N040 student ensemble and to
the 10-model ensemble (equal weights fixed in advance); residual correlation with the family. Expected: alone below r50 (no SSL);
ensemble +.002-.006 if residual r < .9. Kill: residual r >= .95 or ensemble gain < +.002. ~1.5 H100-h, preempt.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N050 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N050-slug/ + external absolute paths -->

## Next
