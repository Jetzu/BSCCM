# N024 · v5-label r18 supervised teacher for the Giemsa chain: anchored GS students (3 seeds) on PBC/MLL23 + cross-slide GS autofl truth, vs the v4 r18 chain (N021)

type: exp · date: 2026-11-09 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal
N022 showed the r18 supervised teacher distils into the Giemsa student far better than r50 SSL teachers (MLL23 fine .63 vs .50). N023 v5 labels fixed held-out autofl transfer (LF b1to0 autofl -.08 -> +.28). Does a v5-label r18 teacher improve the Giemsa chain (clinical fine contrasts; cross-slide GS autofl truth)?

## Plan
Teacher: train_lf r18, FOV, 8 ep, lr 1e-3, targets_v5 (= N013 r18 config, v4 -> v5) -> probe (all feats, 512-d). Students (train_gs_bind --targets targets_v5.npz): anchored (PBC half A, anchor-ce .5, 3000 steps, fov holdout) seeds 0-2 -> apply PBC half B + MLL23; reference = v4 r18 anchored students N021 s0/s1 + new s2 (same teacher as N021). Cross-slide (no anchors, 2500 steps): holdout batch0 / batch1 vs N013/N017 v4 r18 (.32/.32/.32 and .66/.65). GS autofl truth Pearson is within one slide -> invariant to v5 per-slide autofl rescaling, so the comparison is fair. Placement: 1 H100 each (preempt), ~2 GPU-h total. Kill: MLL23 fine mean and cross-slide autofl both within seed spread of the v4 reference.
Jobs: teacher 1288965, probe 1288966, anchored s0-2 1288967/69/71 (+ apply 68/70/72), ref s2 1288973 (+74), cross-slide batch0 1288975, batch1 1288976.

## Changes
- scripts/train_gs_bind.py: --targets (teacher-ridge labels + eval truth; default v4). scripts/aggregate_transfer.py reused.

## Evidence
- v5 r18 teacher (runs/n024_r18_v5_fov, job 1288965): FOV mean r .6405 (12, v5 labels; v4 r18 N013 .635), LF autofl r .528.
- Giemsa autofl truth (n024_metrics.csv). WITHIN-SLIDE Pearson (immune to v5 per-slide rescaling; pooled two-slide r is NOT: v4-teacher students .25 -> .48 merely by re-scoring on v5 truth, so pooled numbers are not used): anchored fov-holdout students, 3 seeds each, batch-0 slide .383 +- .010 (v4 teacher) -> .510 +- .011 (v5), batch-1 slide .689 +- .018 -> .867 +- .006 (plots .agent-research/plots/n024-gs-autofl-within-slide-b0_2026-11-09_2140.png, .agent-research/plots/n024-gs-autofl-within-slide-b1_2026-11-09_2140.png). Cross-slide (single held-out slide): b1->b0 .320 +- .002 (n=3) -> .585 (n=1, job 1288975), b0->b1 .654 (n=2) -> .780 (n=1, job 1288976); teacher .305 -> .451, .559 -> .812.
- Clinical transfer (transfer_multiseed.csv; ref = N021 s0/s1 + new s2, job 1288973): MLL23 mean AUROC over 12 pre-registered fine contrasts .622 +- .017 (v4 teacher) -> .536 +- .038 (v5) (plot .agent-research/plots/n024-mll23-fine-mean-auroc_2026-11-09_2140.png): eos<segneut CD16 .78 -> .55, myelocyte<segneut CD16 .82 -> .64, hairy>lym HLA-DR .86 -> .75, plasma<lym CD45 .64 -> .47; PBC-B held-out neutrophil>IG CD16 .89 -> .67. Mechanism: v5 students give segmented neutrophils lower CD16 (single:CD16 .56-.64 vs .69-.81). NOT a distillation loss-share effect: teacher-output variance share on GS cells autofl .01 -> .10 but CD16 .04 -> .10 also rises (checked from test_preds.npz).

## Outcome
TRADE-OFF (not a GOAL attempt: GOAL = LF->MK). Repaired labels make the Giemsa chain much better on its only same-cell truth (autofl, within-slide +.13 / +.18, cross-slide +.27 / +.13) but worse on clinical fine contrasts (MLL23 .62 -> .54). Keep both chains: v5 teacher for BSCCM-internal transitive GS->MK estimates (N017-style), v4 r18 chain for clinical transfer until the CD16 loss is understood. Cross-slide v5 = 1 seed; no cross-model red-team yet -> [partial].
<!-- on close: bin/effect.sh N024 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N024-slug/ + external absolute paths -->

## Next
Understand the clinical CD16 loss (teacher CD16 outputs on GS cells v4 vs v5; per-slide grp:CD16 normalisation of batch-1 'all'?); multi-teacher student (v4 markers + v5 autofl); re-run N017 GS->MK estimates with the v5 chain.
