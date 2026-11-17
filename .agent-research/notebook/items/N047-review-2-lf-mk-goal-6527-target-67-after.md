# N047 · Review 2: LF->MK GOAL (.6527, target .67) after 5 more flat attempts (N039-N042, N046; N040 noisy student +.0037 sub-threshold)

type: review · date: 2026-11-17 · status: done
goal_effect: pending · scale: ?

## Goal
Review #2 since progress (stall 5/5 after N039-N042, N046). Files results/N047-review-2-lf-mk-goal-6527-target-67-after/: attempts.md, declined.md, diagnosis.md (2 revisions),
ideas.md, d1_big_ensemble.py/.csv, d2_cofactor_check.py/.csv, agents/ (outsider, lit_group_noise, assumption_breaker + ab/,
codex_ideas, codex_diag_redteam), codex/. Shared assumption: all 5 attempts changed how the predictor is TRAINED while the frozen
pooled-v5 score was assumed reachable.

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- d1: 10-model ensemble .6571 (+.0043, flat); Codex red-team: members share errors (residual r .95-.97, n_eff ~1) -> weak saturation test.
- Assumption breaker (reproduced first-hand, d2_cofactor_check.py, independent code): the per-slide q10 asinh cofactor of v3-v5 single
  labels inflates negative-cell noise for dim dyes; fixed cofactor 50: CD3 r .53 -> .78 / .60 -> .77, CD56 .26 -> .37 / .30 -> .46 for the
  SAME N031 predictions; FOV excess CD3 .49 -> .12; other markers unchanged; with v5 detrend/z (agents/ab/a6) N031 .6828, 10-model .6870.
- FOV offsets are not image-visible (LF FOV features R^2 <= .03) -> FOV-context head refuted. Draw-1 gap = draw-specific readout
  (oracle 13-head remap closes 91 %; autofl slope flips sign between draws; every zero-label correction negative).
- Literature: unseen-group offsets are predicted as 0 by mixed models; within-group r is the standard report; self-training gains
  scale with teacher-student misfit (explains N046).
- User 2026-11-17: dataset notes (labelling efficiency 60-99 %, cross-reactivity, single stains do not test other markers); lineage
  priors and diagnosis/population metrics 'do not matter for now'.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
Bottleneck: label curation (our own asinh cofactor) + draw-specific label readout; predictor family saturated against frozen v5 (shared
errors). Portfolio: N048 label v9 cofactor fix + retrain (module), N049 noise-aware positive/negative labels with explicit FP/FN rates
(paradigm), N050 heterogeneous backbone member (module). Re-scoping for the user: v9 re-base (curation fix, N023 precedent);
k-shot isolation calibration as a protocol variant; .67 never confirmed.
<!-- on close: bin/effect.sh N047 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N047-slug/ + external absolute paths -->

## Next
