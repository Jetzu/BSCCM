# N057 · Anchor-strength sweep for the TT6c Giemsa student (PBC class-anchor CE weight .5 -> 2 and 8; 3 teacher runs x 2 seeds) scored with the N056 clinical harness and its seed-range rule

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: module · value: void

## Goal
N056 (K040): the hub student transfers better to clinical smears but its advantage tracks a ~4x larger anchor-CE gradient share;
the share-matched hub is close to TT6c. Test the lever directly on TT6c: does a stronger PBC class anchor raise non-anchor
clinical contrasts?

## Plan
TT6c recipe unchanged except --anchor-ce 2.0 and 8.0 (baseline .5 = the N056 T arm), same 3 teacher runs and seeds as N056 T
(t0 s0-s2, t1/t2 s0-s1; 14 trainings), evaluated with the N056 v2 harness copied to code_a2 / code_a8 (only change:
t_override.json selects the new checkpoints and the expected anchor weight; Claude edit in n056_hub_student.pairs()).
PRE-REGISTERED: primary endpoint = primary16 (non-anchor contrasts) mean over teachers, paired by teacher vs N056 T; claim only if
the paired difference exceeds the max within-teacher seed range of either arm (N056 rule). Secondary: MLL23_15, all25, PBC10,
BSCCM same-cell autofl sanity (must not drop > .03). Prediction (Claude): anchor 2 raises primary16 toward H (.63), anchor 8
starts to hurt BSCCM autofl. Placement GPU (DinoBloom-B fine-tune); ~3.8 GPU-h (14 x ~10 min train + 14 x ~4 min eval).

## Changes

## Evidence
- Trainings 1381969-82 (14, train_gs_bind.py --anchor-ce 2.0 / 8.0, else the TT6c recipe); evals 1382145-58 (first submission
  1382119-32 cancelled by me: code copies lacked pbc_excluded.json/report.md read by the seal check; see COORDINATION.md).
- n057_analysis.py -> n057_units.csv, n057_summary.csv, n057_paired.csv (raw students; anchor .5 = N056 T units in runs/n056v2_main):
| anchor | primary16 | MLL23_15 | PBC10 | all25 | BSCCM autofl slide 2 / 19 |
| .5 (TT6c) | .540 | .568 | .842 | .678 | .570 / .851 |
| 2 | .564 | .604 | .848 | .701 | .570 / .860 |
| 8 | .548 | .627 | .812 | .701 | .567 / .866 |
  Paired by teacher (floor = max within-teacher seed range): anchor 2 - .5: primary16 +.024 (+.044/+.022/+.005; floor .108),
  MLL23 +.036 (all 3 teachers up; floor .086), all25 +.024 (floor .070); anchor 8 - .5: MLL23 +.059 (+.044/+.087/+.045; floor
  .074), PBC10 -.030, primary16 +.008 -> all "no difference" by the pre-registered rule. BSCCM same-cell autofl unchanged.
- Plots: plots/n057-clinical-primary16_2026-11-18_1419.png, n057-clinical-mll23_2026-11-18_1419.png.

## Outcome
FLAT (void, not a GOAL attempt). Stronger anchoring lifts TT6c on MLL23 consistently in direction (+.04-.06, every teacher) without
hurting BSCCM, but stays inside the seed floor and far below the hub (MLL23 .704; share-matched hub .640, N056). So anchoring
explains only part of the hub's MLL23 advantage; the latent target may contribute -> decided by N058 (helma6: share-matched hub,
3 seeds x 3 teachers). Seed noise (.07-.11) dominates all clinical comparisons: more seeds or a larger clinical eval set needed.

## Outputs
<!-- files in .agent-research/results/N057-slug/ + external absolute paths -->

## Next

2026-11-23: run dirs packed (inode cap) -> /hnvme/workspace/v181bb10-bsccm/runs/archive_n054v3_n057_runs.tar (tar -xf to restore).
