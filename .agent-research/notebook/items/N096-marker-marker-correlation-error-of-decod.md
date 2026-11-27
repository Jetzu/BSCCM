# N096 · Marker-marker correlation error of decoded predictions (BSCCM co-measured pairs + ImageStream 36 pairs) and a correlation-matching decoder loss

type: exp · date: 2026-11-27 · status: open
goal_effect: pending · scale: module

## Goal
User 11-27: "On the decoding of the markers can we please look at the correlation error between the different markers try this out".
(1) Measure how wrong the predicted marker-marker correlations are; (2) try decoders/losses that fix them; (3) see what that costs
or gives on GOAL. Run under STALL_OVERRIDE (user request) in parallel with review N095.

## Plan
PRE-REGISTERED 2026-11-27 before any N096 result.
Co-measured pairs. BSCCM: only the pooled-panel targets are measured on the same cell (grp:CD3/CD19/CD56, grp:CD45, grp:CD16 on
both draws, + grp:CD123/HLA-DR/CD14 on draw 0) and each single:X with the autofl background target (8 pairs). Single markers are
never co-measured with each other in BSCCM. ImageStream sandbox (S-BIAD452): all 9 markers per cell -> 36 pairs.
Correlation error of a pair = r(pred_a, pred_b) - r(label_a, label_b) on the test cells where both are measured; summary =
mean |error| over pairs (BSCCM: per direction, then mean of b10/b01; sandbox: the registered N080/N087 36-pair mean_abs_bias).
A. Measurement (descriptive, no bar): N085 record E8, its members and the n051 single model on BSCCM, overall and within the
   gated cell types (draw-0 pooled cells only: cls3). Expectation: predictions over-state between-type correlations (|r_pred| >
   |r_true|) because all outputs are driven by the same cell-type features.
B. BSCCM frozen n051_v10_base features (2048-d, both directions): linear decoder trained with MSE (control) vs MSE + lambda x
   squared error between the batch correlation of predictions and the SOURCE-draw label correlation over co-measured pairs
   (lambda grid 1, 10, 100); plus a training-free post-hoc recolouring of the control (whiten predicted, recolour to the
   source-draw label correlation; pooled block ZCA, single:X anchored on autofl). lambda for the primary arm = the largest
   lambda whose inner source-draw validation mean r is within .005 of lambda=0. PRIMARY B: correlation error of the selected
   arm vs control on the held-out draw; "fixed" = error reduced by >= 50 % with GOAL(common-11) drop <= .005. Secondary: per-target
   r, the recolouring arm, the lambda curve.
C. Sandbox, same recipe as N080 single/full (15 ep, resnet50): single + corr loss to the validation donor's joint label
   correlation (same external joint data as the N087 marker prior), lambda 1, 10, 100; full + corr loss to the training donor's
   correlation, lambda 10. 3 held-out donors each (12 runs, ~1.5 H100-h). PRIMARY C: 36-pair error of the best-on-validation
   single+corr arm vs N080 single (.229 mean; full .189): success = mean <= .21 AND lower than single on 3/3 donors (same bar as
   the N087 marker prior). Secondary: macro r (loss > .02 = cost), within-type pairs (CD4-CD8 in T cells, N080 key pairs).
Kill: B and C both miss their bars -> correlation matching on the predicted means does not fix co-expression; record and stop.
GOAL effect: B is a frozen-feature probe (not a GOAL record attempt); an improvement claim would need a pre-registered E8
ensemble re-score > .7021 as a follow-up.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N096 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N096-slug/ + external absolute paths -->

## Next
