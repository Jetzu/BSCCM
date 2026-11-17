# N048 · Label v9: asinh cofactor from a pre-registered label-only rule (negative-control spread) instead of the per-slide q10 that inflates CD3/CD56 background noise; label QC; retrain GOAL recipe; score frozen v5 and v9

type: exp · date: 2026-11-17 · status: open
goal_effect: pending · scale: module

## Goal
Fix a label-curation error found in review N047: v3-v5 single-antibody labels use asinh(x / q10) with q10 = the slide's 10th percentile
of the authors' unmixed value; for dim dyes q10 is tiny (CD3 .06-.84, CD56 .17-1.59) and asinh amplifies negative-cell noise (FOV excess
CD3 .49/.34, CD56 .65/.62). Labels only; inputs untouched (user rule: clean labels).

## Plan
Candidates (fixed BEFORE scoring any model on them): cofactor c in {q10 (= v5), per-slide split-0 median q50, fixed 50, fixed 150 (flow-
cytometry convention)}, same for all 8 single markers; everything else as v5 (clean calibration, cross-fitted detrend = v5cx path, per-
slide z). SELECTION RULE (label-only, pre-registered): the candidate with the lowest mean training-split (split 0) FOV-excess share over
the 8 single markers x 2 draws, subject to QC: r(v9, v5) >= .95 for CD16, CD45, CD14 (lineage markers must not change), and no
single-marker FOV excess increasing by > .05 vs v5. Disclosure: the review's assumption breaker already scored saved predictions on c=50
and q50 (re-base numbers seen), so v9-scored GOAL values are a re-base for the USER to accept, not progress.
GOAL attempt: retrain the N031 recipe (base + zoom2, both directions, inductive init) on v9; score on FROZEN v5 (the GOAL) and on v9.
Expected: frozen v5 0 to +.004 (cleaner CD3/CD56 training labels); v9-scored ~.68 (re-base). Kill: QC fails -> stop.
Compute: label build CPU (preempt_cpu 48 cores); retrain ~1.3 H100-h.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N048 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N048-slug/ + external absolute paths -->

## Next
