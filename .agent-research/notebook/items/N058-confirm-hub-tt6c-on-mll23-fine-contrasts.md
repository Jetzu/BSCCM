# N058 · Confirm hub > TT6c on MLL23 fine contrasts at matched anchoring: hub student with T-matched anchor gradient share, 3 seeds x 3 teachers vs TT6c 7 seeds (pre-registered seed-floor rule)

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: module · value: void

## Goal
N056 (K040): hub students beat TT6c on MLL23 in every seed (+.14) but within the registered seed floor, and the share-matched hub
(1 seed) kept only part of the gain. Decide whether the hub TARGET (latent distillation) beats TT6c at EQUAL anchoring strength.
Complements N057 (helma2 session: TT6c with stronger anchoring).

## Plan
Code = N056 v2 copy (code/) with 4 Claude edits: H_matched seeds 0-2 allowed (hub_student argparse + guard, eval seed guard,
sbatch case), sensitivity flag off so the registered seed-floor rule applies to H_matched; run root runs/n058_main.
Arms: H_matched (anchor weight set so the anchor-CE gradient share on the shared ViT = T's, per N056) s0-s2 x t0-t2 (9 trainings);
T = existing TT6c anchored students t0 s0-s2, t1/t2 s0-s1 re-evaluated in this root.
PRE-REGISTERED primary: MLL23_15 mean over teachers, paired by teacher, H_matched - T; claim "hub target better" only if the
paired mean difference > max within-teacher seed range of either arm (N056 rule); secondary primary16, all25, PBC10.
Prediction (Claude): +.05-.08 on MLL23, likely still below the floor (.07-.15). GPU 1 x H100 per job; ~3.5 GPU-h.

## Changes

## Evidence
- First submission 1381991-99 failed at contract (code dir lacked report.md / pbc_excluded.json; nothing trained); copied from N056
  v2 + N058 addendum; trainings 1382031-39 (9 x ~9 min), evals 1382178-93 (16; submitted once, guarded by a marker file).
- Aggregator requires the H arm -> registered rule computed by n058_decide.py (same rule: teacher-paired mean difference vs max
  within-teacher seed range of either arm) -> n058_per_run.csv, n058_decision.csv; plot plots/n058-mll23-fine-auroc_2026-11-18_1425.png.
- RESULT (raw students, mean over 3 teachers): MLL23_15 H_matched .681 vs T .568, diff +.113 > floor .074 -> REGISTERED DECISION
  "H_matched higher" (per teacher +.061/+.109/+.169; every H_matched seed .642-.739 > every T seed .499-.613). Secondary within the
  floor: primary16 +.072 (floor .108), all25 +.068 (floor .069); PBC10 equal (.841/.841).
- Context N057 (helma2 session): TT6c with 4x / 16x anchor weight gains only +.04-.06 on MLL23 (within floor) -> at EQUAL anchoring
  the hub target (latent distillation of the LF hub) is the larger lever for fine clinical markers; anchoring strength adds a smaller part.

## Outcome
CONFIRMED (pre-registered primary): hub students transfer fine markers to MLL23 better than TT6c at matched anchoring (+.11
AUROC). Recommended Giemsa student going forward: hub latent distillation + PBC class anchoring. Not a GOAL attempt -> effect void.
Feeds review N059 (opened by the helma2 session).

## Outputs
<!-- files in .agent-research/results/N058-slug/ + external absolute paths -->

## Next
