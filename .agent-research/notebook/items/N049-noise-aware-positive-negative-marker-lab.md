# N049 · Noise-aware positive/negative marker labels: per-marker two-component mixture with explicit labelling efficiency and false-positive rate -> P(positive) targets + probabilistic head trained jointly with the intensity head; GOAL r + AUROC

type: exp · date: 2026-11-17 · status: done
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
- Codex writer (codex/report.md, implementation.patch, scripts/, tests/ 7 pass, diagnostics/): mixture + noise layer by MAP EM. Codex's
  own caveat: eta/phi are NOT identifiable from two emissions (only q = pi*eta + (1-pi)*phi is), so they reproduce the priors; on v5
  labels several fits are degenerate (draw-2 CD19/CD123 one component; draw-1 CD123/CD19 and draw-2 HLA-DR pi = 0).
- Refit on the N048 cofactor-150 labels (v9fit/): separation d' (draw 1 / draw 2) CD3 2.9/2.9 -> 7.6/7.4 (q .09/.12), CD56 2.7/3.0 ->
  5.8/5.5 (q .02/.05), CD19 4.4/0.1 -> 5.2/8.8 (q .03/.01), HLA-DR 4.7/5.0 -> 6.5/5.6, but CD123 21.7/0.1 -> 0.0/0.0 (the fixed cofactor
  150 erases the rare bright CD123+ cells; a per-marker cofactor would be better -- noted, not changed post hoc in N048).
- Revised plan (before any scoring): train on targets_v5cx_cf150 + soft positive targets (v9fit) and compare with N048 (same labels,
  no positive head) -> isolates the head; both scored on frozen v5 and on v9. Claude red-team of the trainer patch running.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- Claude red-team (redteam_claude.md): code LAND WITH FIXES (trainer based on current file, defaults/resume safe, BCE masking and row
  alignment exact, tests 7/7), but MAJOR 1 reproduced: with eta/phi/pi unidentifiable the fitter pins phi near its prior (~2 %), which
  absorbs rare positives: draw-2 CD19 and HLA-DR targets are exactly 0 for every cell, draw-1 CD56 max .054, CD19/HLA-DR max .31;
  CD123 fit stuck in a local optimum. m2: head creation shifts the torch RNG (not seed-paired with baselines).

## Outcome
FALSIFIED AT THE FITTING STAGE (no GPU run). Labelling efficiency and false-positive rates are not identifiable from these data
(two-emission mixture identifies only the observed-positive fraction; no isotype/unstained control exists through the same marker
readout -- the authors' unmixed value is defined only on its own antibody's slide); the 'noise layer' reduces to prior-driven label
smoothing that erases rare positive populations. By-product kept: on cofactor-150 labels the positive populations are clearly
separable (CD3 d' 7.4-7.6, CD56 5.5-5.8, CD19 5.2-8.8) -> supports N048; CD123 needs its own cofactor. Code not landed (results/ only).
<!-- on close: bin/effect.sh N049 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N049-slug/ + external absolute paths -->

## Next
