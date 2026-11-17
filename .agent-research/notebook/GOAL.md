# Goal
goal: LF pivot that predicts surface markers in a NEW blood isolation (prerequisite for every transitive GS->MK claim)
metric: LF->MK common-11, RE-BASED 2026-11-17 (N051, user decision 'per-marker scale first'): batch-held-out Pearson r, mean over the 11 targets (all except grp:CD123/HLA-DR/CD14), averaged over b1to0 and b0to1, on targets_v5cx_v10.npz (per-marker asinh cofactor chosen by a pre-registered label-only rule; clean calibration; cross-fitted detrend). Legacy v5 score reported alongside (N031 .6527 on v5 = .6703 on v10; re-base, NOT progress)
direction: higher
target: 0.67
target_source: v5-era target .67 (Claude proposal N023, never confirmed) is met by the re-base itself; NEW TARGET PENDING USER (2026-11-17); in-distribution reference on v10 (v5-trained FOV model that saw both draws) .6862, data-matched .6646
best: 0.6804
best_item: N048
noise: 0.0027
noise_source: .agent-research/results/N051-label-v10-per-marker-asinh-cofactor-pre/n051_noise_v10.json (two seed pairs on v10: .0031/.0045 -> sd .0027)
constraints: SSL pretraining must EXCLUDE the held-out batch (inductive; N031: inductive ensemble .6527 vs transductive .6512 -- caveat resolved)
stall_limit: 5
stall_since: N023

rebase_history: N023 v4->v5 (common-11); N051 v5->v10 (per-marker cofactor, 2026-11-17). Previous metric line: LF->MK common-11 (re-based N023): batch-held-out Pearson r, mean over 11 targets = all except grp:CD123/HLA-DR/CD14 (batch-1 pooled assay failed QC: below CD16 crosstalk; best = N014 model re-scored, not a new model), averaged over b1to0 and b0to1, targets_v5 labels; scripts/rescore_targets.py r_mean_common (= train_lf.py r_mean_markers_valid on v5). Legacy 12-target v4 metric stays reported alongside and is UNRESOLVED (best .5742, N014); grp:CD123/HLA-DR/CD14 reported separately
