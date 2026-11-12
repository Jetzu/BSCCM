# N038 · Review: LF->MK common-11 batch-held-out GOAL (.6527, target .67) after 5 flat attempts (N027-N037 label/encoder/ensemble family)

type: review · date: 2026-11-12 · status: done
goal_effect: pending · scale: ?

## Goal
Stall breaker after 5 flat attempts (stall 5/5 since N023). Files: results/N038-review-lf-mk-common-11-batch-held-out-go/attempts.md, declined.md, diagnosis.md, ideas.md, agents/ (4
Claude reports + ab/ analyses), codex/ideas, codex/diag_rt.
Shared assumption: all 5 attempts assumed the per-cell LF->marker predictor is near what the data allow, so the gap had to be closed
AROUND it (labels, Giemsa student, output form). Only N037 measured the GOAL at all.

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Diagnosis (my own, then two red-team revisions, all reproduced first-hand):
  d1/d1b: same-arch FOV vs held-out on identical cells .658 vs .642 -> my claimed ".67 sits at the in-distribution ceiling" was
  WITHDRAWN after the Codex diagnosis red-team (confounded: v4 vs v5 training labels, cohort sizes, transductive SSL); d1c reproduces
  the counterexample N022 tail05 FOV .6800 (CD123 .42 -> .65 / .53 -> .68) while tail05 trained b1to0 scores .6306 vs base .6343.
  d2 (reproducing the assumption breaker): within-FOV GOAL .7211 vs pooled .6527; CD19/CD56/CD3 labels carry FOV-level excess
  .34-.49, predictions ~0; cause = authors' smoothed background_lowess in the shared dim dye channel (agents/ab/e_bgmodel).
  d3: seed sd on v5 .0027 (GOAL.md noise updated from .002).
- Agents (>= 4): agents/outsider.md, agents/lit_labelfree_subsets.md, agents/lit_labelnoise_shift.md, agents/assumption_breaker.md
  (+ agents/ab/ scripts/outputs copied from /tmp), codex/ideas/ideas_codex.md, codex/diag_rt/diag_redteam.md. Top 3 literature claims
  verified by a separate web-fetch pass (BSCCM paper CD19 35 % / pooled channel "too noisy" / isolation variation; DeepIFC held-out
  CD19 .61, CD56 .53; Pinkard 2405.20559 GMM heads, no numeric marker results).
- Scale audit (diag red-team): N030 is a knob, not a module (recorded label left; noted here).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
Bottleneck: the EVAL LABELS (FOV-level background-estimation error in CD19/CD56/CD3; draw-1 autofluorescence-driven HLA-DR/CD19/CD14
strata) plus a smaller draw-specific generalisation gap for rare populations (CD123, HLA-DR). Portfolio (ideas.md): N039 label v8 +
clean ceiling (module), N040 dense label completion / noisy student (paradigm), N041 physics nuisance randomisation + tail05
(module), N042 weak-marker architecture gated by a low-pass probe (module); ~10 GPU-h total. Parked: TODO.md. User decision requested
(not blocking): re-base the GOAL on v8 labels if N039's pre-registered QC passes.
<!-- on close: bin/effect.sh N038 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N038-slug/ + external absolute paths -->

## Next
