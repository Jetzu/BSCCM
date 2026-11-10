# N025 · Giemsa chain without the trade-off: two-teacher student (v4-r18 markers + v5/v6 autofl), v6 labels (autofl from unstained slides only) r18 teacher; plus v6 GOAL runs (SSL-init r50)

type: exp · date: 2026-11-10 · status: open
goal_effect: pending · scale: module

## Goal
Remove the N024 trade-off (v5 teacher: Giemsa autofl truth up, MLL23 clinical fine contrasts down .62 -> .54). Diagnosis (this item): on the same Giemsa cells the v4 and v5 r18 teachers' autofl outputs correlate -.44, while their CD16 outputs correlate .87 and BSCCM lineage CD16 contrasts are unchanged (teacher gran-lym 2.0 vs 1.7; students ~1.9 both) -> the clinical loss is not a lost CD16 lineage signal; the autofl target changed meaning. Also test whether autofl from UNSTAINED slides only (v6) is a cleaner target, and whether dropping stained-slide autofl changes the LF->MK GOAL.

## Plan
Arms (anchored students: PBC half A anchors, anchor-ce .5, 3000 steps, fov holdout, 3 seeds; cross-slide: 2500 steps, holdout batch0/batch1, 1 seed):
- TT5: markers from the v4 r18 teacher (n013), autofl from the v5 r18 teacher (n024) [train_gs_bind --aux-feats/--aux-targets/--aux-names].
- v6: r18 teacher trained on targets_v6 (= v5, autofl only on unstained slides: 63,067 cells, 47,355 non-GS; 312,590 stained-slide autofl labels dropped).
- TT6: v4 r18 markers + v6 r18 autofl.
- GOAL: SSL-init r50 on v6, b1to0 + b0to1, seeds 0/1 (compare with N023 v5 .6404).
Metrics: Giemsa autofl truth WITHIN-SLIDE r; MLL23 mean AUROC over 12 pre-registered fine contrasts; PBC-B. References: N024 (v4 r18 .383/.689 within, .622 MLL23; v5 r18 .510/.867, .536). Success = within-slide autofl >= v5 and MLL23 >= v4 within seed spread.
Jobs (results/N025-*/jobs.txt): v6 teacher 1294463, probe 1294464; students/applies/GOAL 1294456-1294482. ~4.5 GPU-h, 1 H100 each, preempt.

## Changes
- scripts/train_gs_bind.py: --aux-feats/--aux-targets/--aux-names (two-teacher distillation; bound ingredient a skips aux targets). scripts/make_targets_v6.py -> targets_v6.npz.

## Evidence
- Side analysis, transitive GS->MK estimates (N017 method, r_est = r(S,T) * r(T,M); scripts/gs_mk_estimates.py -> gs_mk_estimates_v4_v5.csv; cross-slide students n017_gsbind_bound_* (v4 r18) vs n024_gsx_* (v5 r18), 1 seed each): v5 chain repairs the weak b1->b0 direction: CD45 .44 -> .62, CD14 .67 -> .77, grp CD3/19/56 .43 -> .70, grp CD45 .40 -> .76, grp CD16 .55 -> .70, CD3 .40 -> .47; b0->b1 similar (CD16 .79 -> .82, CD14 .73 -> .76, CD3 .48 -> .53); CD19/CD123/CD56 stay ~.04-.25.
- Real check of the estimator on the only GS truth (autofl): product estimate vs measured student r: v4 .27 vs .65 (b0->b1), .08 vs .32 (b1->b0); v5 .70 vs .78, .28 vs .59 -> the CIA product UNDER-estimates in 4/4 cases (the student corrects part of the teacher's error), so the marker estimates are conservative on this evidence.
- Giemsa students, 3 seeds each (n025_metrics.csv; plots .agent-research/plots/n025-gs-autofl-within-b0_2026-11-10_0807.png, .agent-research/plots/n025-gs-autofl-within-b1_2026-11-10_0807.png, .agent-research/plots/n025-mll23-fine-mean_2026-11-10_0807.png). Within-slide autofl truth r (b0 slide / b1 slide) | MLL23 mean over 12 fine contrasts:
  v4 r18 .383 / .689 | .622 ; v5 r18 .510 / .867 | .536 ; v6 r18 .579 / .846 | .527 ; TT5 (v4 markers + v5 autofl) .506 / .865 | .619 ; TT6 (v4 markers + v6 autofl) .578 / .855 | .611 (sd .009/.006/.008).
  Per contrast TT6 vs v4 vs v5: eos<segneut CD16 .74/.78/.55, myelocyte .76/.82/.64, hairy HLA-DR .85/.86/.75, plasma .61/.64/.47 (transfer_multiseed.csv). LGL CD56/CD16 and basophil CD123 still fail in every arm.
  Cross-slide (1 seed): b1->b0 v4 .32, v5 .585, TT5 .584, v6 .626; b0->b1 .649 / .780 / .767 / .762.
- => the clinical loss of the v5/v6 single teachers comes from their marker heads / feature space (TT arms keep the v4 r18 features + markers and recover clinical transfer), not from the autofl target; autofl from unstained slides only (v6) is the cleanest autofl teacher (+.07 on the harder b0 slide).
- GOAL (goal_runs_v6.json, goal_common11.csv; plot .agent-research/plots/goal-common11-v6_2026-11-10_0807.png): SSL-init r50 on v6 .6407 / .6372 = .639 vs v5 .6423 / .6384 = .6404 -> flat; dropping stained-slide autofl does not change marker prediction.
- Controls: all r18 teachers include_gs 0; every ridge row (primary and aux) requires hist_row < 0 (train_gs_bind.py:35,44); student configs identical across arms (3000 steps, anchor-ce .5, PBC half A, fov holdout, aug 1). Within-slide r is invariant to the per-slide affine label transforms of v5/v6.

## Outcome
FLAT on the GOAL (.639 vs .6404). Giemsa chain: TRADE-OFF REMOVED -- recommended chain = TT6 (markers from the v4 r18 teacher, autofl from the v6 r18 teacher): same-cell autofl truth .578 / .855 (v4 .383 / .689) at clinical parity (MLL23 .611 vs .622, within seed spread). Red-team pending at writing.
<!-- on close: bin/effect.sh N025 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N025-slug/ + external absolute paths -->

## Next
Re-run the transitive GS->MK estimates with the TT6 chain (cross-slide students); NK/basophil need a different signal; deliverables refresh.
