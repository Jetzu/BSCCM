# N040 · Dense label completion (noisy student): label-count learning curve, then N031-ensemble pseudo-labels for all 11 targets on every training-draw cell and an augmented student on real + down-weighted pseudo labels

type: exp · date: 2026-11-12 · status: open
goal_effect: pending · scale: paradigm

## Goal
Paradigm bet: each training cell carries 1 of 11 labels (single-antibody slides) and unstained slides none; supervision for the
rare populations (B/NK/basophil/pDC) is ~10^2-10^3 positives per draw (DeepIFC saturates ~3,200 per type). Dense pseudo-labels from
the current ensemble on every training-draw cell + a noisy (augmented) student = semi-supervised, a new training signal.

## Plan
Stage 1 learning curve (falsifier): N031 recipe base model with --max-train 50 % and 25 % of the training draw, both directions,
seed 0; weak-target (CD19/CD56/CD123/HLA-DR) mean r vs 100 %. Kill the bet if 50 % -> 100 % gains < .01 on weak targets.
Stage 2 (only if stage 1 passes): teacher = N031 ensemble predictions on ALL training-draw non-Giemsa cells (never test-draw images);
student = same recipe, loss = real-label MSE + w * pseudo-label MSE (w .3, pseudo for unobserved targets only), train augmentation
(translation + flips where legal + gain/noise), 1 seed both directions; score frozen v5 GOAL vs N031 (single and ensemble).
Expected +.003-.012. Compute: stage 1 ~1 GPU-h, stage 2 ~2 GPU-h.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N040 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N040-slug/ + external absolute paths -->

## Next
