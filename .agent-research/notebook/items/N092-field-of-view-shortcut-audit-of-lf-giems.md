# N092 · Field-of-view shortcut audit of LF<->Giemsa retrieval (N088 lit C-P1): within-FOV identification, FOV-level retrieval, and a same-FOV-different-cell positive-pair control for the N045 within-slide SigLIP R@10 .38-.40

type: analysis · date: 2026-11-27 · status: done
goal_effect: n/a · scale: ? · value: void · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Code: Claude subagent wrote n092_audit.py (imports the unmodified N045 runner; smoke tests, metric toys exact, reproduces N045 v2
  R@10 .3994/.3842 from the archived checkpoints). CPU job 1466796 failed (output-path guard) -> 1466821 (48 cores, ok).
- metrics_fov_b{0,1}_seed0.csv (two rows per arm = best / last checkpoint): SigLIP true pairs R@10 .399/.403 (b0), .384/.361 (b1);
  R@10 with all other same-field cells removed .403/.405, .386/.363 (unchanged); within-field partner identification .840/.846,
  .799/.771 (chance .233 / .282); field-level hit@10 .44 / .41-.43. CONTROL siglip_samefov_shuffled (positives = a different cell of
  the same field): R@10 .029/.022 (b0), .028/.033 (b1) vs chance .010/.016; within-field .24-.31 = chance. siglip_pairs_multifov (true
  pairs on the control's cells) .33-.40. CCA .18-.20, ridge .14-.15.
- Reading: the field-of-view shortcut hypothesis (N088 lit C) is REFUTED: field identity alone gives near-chance retrieval; the LF<->GS
  embedding learns cell identity within a slide, and the cross-slide collapse (.02-.04) is the slide shift. Captions (fig8),
  make_alignment notes and README updated (caveat removed, N092 result added).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
DONE: no field-of-view shortcut; within-slide LF<->Giemsa retrieval is genuine cell identity.
<!-- on close: bin/effect.sh N092 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N092-slug/ + external absolute paths -->

## Next
