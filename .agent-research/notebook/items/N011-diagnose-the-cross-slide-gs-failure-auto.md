# N011 · Diagnose the cross-slide GS failure: autofl comparability per slide, GS vs LF side, within- vs cross-slide penalty, domain separability

type: analysis · date: 2026-11-08 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- results/N011-diagnose-the-cross-slide-gs-failure-auto/diag_slide_shift.json: autofl on GS cells -- direct GS ridge (frozen DinoBloom) within-slide CV .62 (both slides) vs cross-slide .44-.45; LF teacher on GS slide b0 .47 / b1 .61, pooled .32 (slide offsets); direct LF on GS cells within .61/.82, cross b0->b1 .59, b1->b0 .22; slide classifier acc GS .92, LF .99 (both modalities carry strong slide fingerprints).
- Linear distillation (ridge GS -> teacher prediction): b0->b1 .23, b1->b0 .38.
- Cell-type-only oracle for autofl on GS cells: 3-class LF-pred class means .20-.38 (cross-slide means) ; 12 morphology clusters .21-.26 cross-slide (.32-.45 in-sample) -> the cross-slide student (.28-.31) is at the cell-type level; within-slide direct (.62) carries within-type signal that does not transfer across the two slides.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N011 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N011-slug/ + external absolute paths -->

## Next
