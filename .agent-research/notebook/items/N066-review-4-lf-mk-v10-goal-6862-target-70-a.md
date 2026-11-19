# N066 · Review 4: LF->MK v10 GOAL (.6862, target .70) after 5 flat attempts (N060-N065: heads/losses, composite, composite ensemble)

type: review · date: 2026-11-18 · status: done
goal_effect: pending · scale: n/a

## Goal
Project review 4 (review #2 without progress): 5 flat attempts N060-N065 (heads/losses, composite, composite ensemble); GOAL best .6862
(N053), descriptive 6-way .6929, target .70, bar .6916.

## Plan
Shared assumption (attempts.md): "All 5 attempts assumed the remaining gap is a modelling problem of the readout (head, loss, ensemble
member) on a FIXED training set -- same labelled cells, one training draw, the same v10 targets as training signal and independent
outputs -- and that a better function of the image of the same cells, trained the same way, can close it."

## Changes
Diagnosis (diagnosis.md, corrected after the Codex red-team, all corrections reproduced): see K043/K044. Agents (6 reports in agents/):
outsider, assumption breaker (+ ab/ scripts), lit_labelfree, lit_labelnoise, Codex ideas, Codex diagnosis red-team. Top lit claims
verified first-hand (ideas.md). Harness: train_lf.py gained --diag-aug/--diag-tta/--drop-led (N068) and --censor-loss (N073), self-tested
(N068/n068_diag_selftest.csv); uncommitted until their items close.

## Evidence
Portfolio (ideas.md): N067 consensus training targets (module), N068 exact-symmetry augmentation + TTA (module), N073 censored labels +
20 epochs (module), N074 dense privileged fluorescence supervision (paradigm, Codex write). ~11.4 GPU-h total (< 25). Queued behind the
cluster downtime 2026-11-19 04:00 -> 11-20 00:00. Rejected/parked ideas -> notebook/TODO.md "Parked by review N066".
Re-scoping candidates for the user (review #2 without progress): (a) accept ~.69-.70 as the inductive ceiling on these labels;
(b) few-shot new-draw calibration protocol (.7042 with >= 200-500 labelled new-draw cells; N059); (c) per-slide-averaged r (removes
replicate-slide label offsets; a re-base).

## Outcome
<!-- on close: bin/effect.sh N066 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/: attempts.md, declined.md, diagnosis.md, ideas.md, agents/, codex/{ideas,diag_rt}/, n066_*.py/.csv

## Next
