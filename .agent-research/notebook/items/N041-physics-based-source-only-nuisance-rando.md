# N041 · Physics-based source-only nuisance randomisation (per-LED-family gain/offset, shot noise, angle-proportional defocus) + prediction-consistency loss, bundled with tail05 rare-population weighting: does invariance make the in-distribution CD123/HLA-DR gain transfer to the new draw?

type: exp · date: 2026-11-12 · status: open
goal_effect: pending · scale: module

## Goal
Module: the only inductive shift family with measured OOD gains in the literature (targeted/physics augmentation) applied to
the LED stack, bundled with tail05 rare-population weighting to test N038 revision 1 (tail05 lifts CD123 in-distribution .42->.65 but
not across draws).

## Plan
Arms (base model, both directions, seed 0, N031 inductive SSL init, v5 labels): (a) physics aug + consistency; (b) (a) + tail05
(tail_weight .5, focal 10-gate head as N022). Physics aug in the raw-count domain before GPUNorm: per-illumination-family gain
(BF/DPC/DF-low/DF-high, log-normal sd .1), additive offset, Poisson-matched shot noise, angle-proportional sub-pixel shifts per LED
(defocus +-1 um equivalent), with an unperturbed fraction .3; consistency = MSE between predictions of two augmented views (w .1).
Baseline = n031_base (single) per direction; also blend each arm with n031 zoom2. Source-only: perturbation ranges from training-draw
border statistics, no test-draw images. Expected +.003-.010; kill if both arms within +-.0054 of n031 base on the frozen GOAL.
Compute ~1.5 GPU-h.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N041 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N041-slug/ + external absolute paths -->

## Next
