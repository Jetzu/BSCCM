# N088 · Literature review (user 11-24): label-free -> marker prediction, autoencoder / low-dim / projection representations of cell images, sparse multi-target decoding + marker priors, cross-modal image-molecular alignment; pick and test the best new approaches

type: research · date: 2026-11-24 · status: done
goal_effect: n/a · scale: ? · value: void · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Three parallel Claude research agents (context.md shared): lit_A_labelfree_representations.md (35 refs), lit_B_sparse_multitarget.md
  (35 refs), lit_C_alignment_shift.md (34 refs); each with ranked, costed proposals. synthesis.md = merged portfolio.
- Key findings: AE latents cluster cell identity worse than SSL (every microscopy comparison); pre-projector > post-projector features;
  distribution alignment fails under label shift (Zhao 2019); possible field-of-view shortcut in within-slide LF<->Giemsa retrieval
  (CAPTIONS fig8 caveat added, audit pending); feature-preserving fine-tuning is the top untested GOAL lever (-> N090); missing-marker
  completion from the measured marker (B-P1) and an equal-label-count control (running in N087).
- README for the GitHub repo: Codex writer (readme/round1/), Claude fact-check (readme/factcheck_claude.md: 13 errors, 8 overclaims,
  no private data), Codex fix round (readme/), integration in N089.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
DONE: literature review + synthesis; experiments continue in N087 / N090 and the synthesis portfolio.
<!-- on close: bin/effect.sh N088 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N088-slug/ + external absolute paths -->

## Next
