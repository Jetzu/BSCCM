# Goal
goal: LF pivot that predicts surface markers in a NEW blood isolation (prerequisite for every transitive GS->MK claim)
metric: LF->MK common-11 (re-based N023): batch-held-out Pearson r, mean over 11 targets = all except grp:CD123/HLA-DR/CD14 (batch-1 pooled assay failed QC: below CD16 crosstalk; best = N014 model re-scored, not a new model), averaged over b1to0 and b0to1, targets_v5 labels; scripts/rescore_targets.py r_mean_common (= train_lf.py r_mean_markers_valid on v5). Legacy 12-target v4 metric stays reported alongside and is UNRESOLVED (best .5742, N014); grp:CD123/HLA-DR/CD14 reported separately
direction: higher
target: 0.67
target_source: Claude proposal 2026-11-09 (N023): close half of the FOV-vs-batch gap on common-11 (FOV ~.71, batch .634); user to confirm
best: 0.6404
best_item: N023
noise: 0.002
noise_source: .agent-research/results/N023-label-qc-per-batch-dye-amplitude-single/rescore_v5_noise_n013.json (N013 seed0 .6119 vs seed1 .6136)
constraints: (none stated)
stall_limit: 5
stall_since: N023
