# N012 · Skeptic control: is the within-slide GS->autofl signal spatial confounding? position-only predictors, FOV-grouped and region-blocked CV

type: analysis · date: 2026-11-09 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- spatial_control.json (job 1281023): autofl on GS cells is predicted by POSITION ALONE (FOV centre x/y/z + position in FOV, GBM): b0 slide r .72 (cell-random CV) / .70 (FOV-grouped) / .59 (4x4 region-blocked); b1 .61/.59/.57. GS DinoBloom ridge .62/.62/.56 (b0), .62/.62/.56 (b1); GS+position .74-.78. LF sup ridge b0 .61/.76/.74, b1 .82/.82/.78.
- => the measured autofluorescence is dominated by a slide-position artefact (consistent with the flawed fluorescence shading correction found in N003); GS images also encode position (stain thickness/focus gradients) -> the within-slide GS->autofl 0.62 and part of the cross-slide collapse are spatial artefacts, not biology. Autofl must be spatially detrended before it is used as ground truth (N012b running).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- N012b (job 1281048, .agent-research/results/N012-skeptic-control-is-the-within-slide-gs-a/spatial_detrend.json): POSITION-ONLY (region-blocked GBM) r, median over slides: CD19 .84, autofl .84, CD56 .53, CD123 .49, HLA-DR .30, grp CD45 .23, CD3 .19, CD45 .16, CD14 .13, grp CD16 .10, CD16 .03. => CD19/CD56/CD123 measurements are dominated by slide-position artefacts (exactly the markers that collapse under batch hold-out, N006); CD16/CD14/CD45 nearly artefact-free (transfer well).
- Features encode position: LF sup features predict FOV x/y with r .92 (both GS slides); GS DinoBloom .42 (b0) / .27 (b1).
- Detrended autofl on GS cells (cross-fitted GBM residual): GS ridge within-slide (blocked) .57 / .68, CROSS-slide .55 (b0->b1) / .47 (b1->b0); LF within .56 / .76, cross .47 / .24. => once the artefact is removed, the Wright-image signal transfers across the two stained slides (gap .57/.68 -> .55/.47 instead of .62 -> .44).

## Outcome
<!-- on close: bin/effect.sh N012 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N012-slug/ + external absolute paths -->

## Next
