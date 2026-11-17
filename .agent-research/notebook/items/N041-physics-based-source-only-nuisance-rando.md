# N041 · Physics-based source-only nuisance randomisation (per-LED-family gain/offset, shot noise, angle-proportional defocus) + prediction-consistency loss, bundled with tail05 rare-population weighting: does invariance make the in-distribution CD123/HLA-DR gain transfer to the new draw?

type: exp · date: 2026-11-12 · status: done
goal_effect: flat · scale: module · value: 0.6531

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
- Code: Codex writer (codex/brief.md) implementing opt-in --phys-aug/--consist in train_lf.py; Claude integrates + GPU smoke.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- Code: Codex writer (codex/), Claude red-team (redteam_claude.md: LAND WITH FIXES), fixes by Claude: gain jitter sized from the
  training draw (family MIN of 2 x robust sd of normalised border level x s_c; log-sd .006-.012 vs planned .1 = 4-24x too large),
  tail weights as N022, stored calibration reused on resume. GPU smoke 1313717 OK.
- Runs (jobs 1374319/20/23/24, 1 H100, 14-35 min): GOAL (scores_goal.csv; plot .agent-research/plots/n041-phys-aug-goal-r_2026-11-17_1839.png):
  phys .6450 vs N031 base .6425 (+.0025, 15/22, p .046); phys+tail .6447 (+.0022, 12/22, p .70; CD123 +.031 but CD19 -.033);
  blended with N031 zoom2: .6531 / .6525 vs N031 ensemble .6527.

## Outcome
FLAT. Physics-based instrument augmentation gives at most +.0025 on a single model and nothing on the ensemble; rare-population
weighting still does not transfer across draws (CD123 up, CD19 down). Consistent with N039: the draw shift is mostly label-side.
<!-- on close: bin/effect.sh N041 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N041-slug/ + external absolute paths -->

## Next
