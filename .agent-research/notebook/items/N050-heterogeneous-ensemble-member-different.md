# N050 · Heterogeneous ensemble member: different LF backbone (ConvNeXt-T) trained on real + N040 pseudo labels to decorrelate the shared errors of the ResNet-50 family (member residual r .97)

type: exp · date: 2026-11-17 · status: done
goal_effect: flat · scale: module · value: 0.6533

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
- Jobs 1375356/7 (convnext_tiny, random init, N040 recipe; 32 min b10). n050_score.py -> n050_per_target.csv, n050_residual_corr.csv;
  plot .agent-research/plots/n050-convnext-member-goal-r_2026-11-17_2024.png.
  ConvNeXt alone .6135 (b10 .6362, b01 .5908); residual r with the 10-model mean .939 (b10) / .963 (b01); N040 students .6564 ->
  + ConvNeXt .6533; 10-model .6571 -> .6570.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
FLAT / KILLED (pre-registered: residual r >= .95 in b01, ensemble gain < +.002). A different architecture without the shared SSL init still
makes the same errors (residual r .94-.96) -> the shared error is label/readout-driven, not an artefact of one model family
(supports the N047 diagnosis; answers the Codex red-team's diversity objection).
<!-- on close: bin/effect.sh N050 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N050-slug/ + external absolute paths -->

## Next
