# N093 · Class-conditional Giemsa -> marker matching (user 11-24, N087 arm F): Giemsa cell-type classifier -> class-mean marker profiles (F1) and within-class OT to marker profiles (F2), scored in the N056/N058 clinical harness vs output and latent distillation

type: exp · date: 2026-11-27 · status: done
goal_effect: worse · scale: paradigm · value: void

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Round 1 (Codex writer, 8 fixture tests; CPU job 1467846, round1_results/): F1 class-mean MLL23_15 .516 / primary16 .581 / PBC10 .974 /
  all25 .699; F2 within-class OT .515 / .575 / .931 / .682 (3 teachers x 2 holdouts) vs N058 T .568 / .540 / .841 / .677 and H_matched
  .681 / .612 / .841 / .745. Same-cell agreement with the LF teacher ~0 for every marker; autofl r .01-.07.
- Claude red-team (redteam_claude.md): metrics/harness parity exact, no leakage; MAJOR: the PBC-anchored GS classifier routes 97-99 % of
  BSCCM Giemsa cells to granulocyte (REPRODUCED first-hand from t0_paired.npz: holdout0 99.2 %, holdout1 96.9 %) = PBC -> BSCCM domain
  collapse (cf. N008); F2's rank coordinate encodes domain (AUROC 1.00 PBC vs BSCCM). PBC10 .974 = the classifier's own domain.
  -> not a fair test of the idea; fix round: F1-oracle (route by the cell's own LF class, upper bound) and F1-bsccm (GS classifier
  trained on BSCCM training-slide Giemsa cells with LF-teacher class labels).
- Fix round (Codex, 11 fixture tests; Claude red-team round 2: NOTHING BREAKS, oracle never reaches clinical scoring; 'upper bound' is
  not strict -> reported as reference route). CPU job 1467983, fix_results/ (3 teachers x 2 holdouts): per-marker agreement with the LF
  teacher on the same held-out cells (mean of 12): F1 -.02, F2 .04, F1-bsccm .62 (CD45 .89, CD16 .87, CD14 .76, CD3 .66, HLA-DR .54,
  CD56 .43, CD123 .28, CD19 .19), oracle routing .64 / .66 (hard / soft); autofl r (holdout 0 / 1): F1-bsccm .465 / .261, oracle
  .49 / .22-.25 vs distillation TT6c .601 / .715 and hub .549 / .751 (N054); clinical F1-bsccm MLL23_15 .494, primary16 .569, PBC10
  .959, all25 .680 vs T .568 / .540 / .841 / .677 and H_matched .681 / .612 / .841 / .745.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
WORSE than distillation (not a GOAL attempt): class-conditional matching captures lineage-level marker signal when the Giemsa classifier
is trained on BSCCM Giemsa cells (agreement .62, ~ the routing reference .64-.66) but loses all within-class variation (CD19 .19, autofl
~half of distillation, MLL23 fine contrasts at chance .49); a PBC-trained classifier collapses on BSCCM images (97-99 % granulocyte).
<!-- on close: bin/effect.sh N093 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N093-slug/ + external absolute paths -->

## Next
