# N049 · Noise-aware positive/negative marker labels: per-marker two-component mixture with explicit labelling efficiency and false-positive rate -> P(positive) targets + probabilistic head trained jointly with the intensity head; GOAL r + AUROC

type: exp · date: 2026-11-17 · status: open
goal_effect: pending · scale: paradigm

## Goal
User's dataset notes (2026-11-17): antibody labelling efficiency 60-99 %, cross-reactive false positives, SNR/threshold-dependent false
positives/negatives; single stains do not test other markers. Current labels are continuous intensities trained with MSE; positive/
negative populations and FP/FN rates are not modelled. (User: lineage priors and diagnosis metrics do not matter for now.)

## Plan
Per single marker x training draw (split 0, clean): two-component mixture on the curated label (negative / positive), giving P(pos|y);
explicit noise model: observed-positive probability = eta * P(true pos) + phi * P(true neg) with labelling efficiency eta and false-
positive rate phi as per-marker parameters (fit by EM on the training draw; priors eta in [.6, .99], phi small) -> de-noised soft target
P(true pos | y). Add a probabilistic head (one logit per single marker) trained with BCE on those soft targets alongside the existing
intensity MSE (masked; missing markers never negative); prediction for the GOAL = intensity head (and, pre-registered alternative, the
expected value under the mixture). Score frozen v5 GOAL + per-marker AUROC vs soft labels. Expected +0-.005; kill if both directions
within +-.0054 and AUROC unchanged. ~1.5 H100-h. Code: Codex writer (codex/brief.md), Claude red-team.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N049 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N049-slug/ + external absolute paths -->

## Next
