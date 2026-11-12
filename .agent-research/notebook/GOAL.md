# Goal
goal: LF pivot that predicts surface markers in a NEW blood isolation (prerequisite for every transitive GS->MK claim)
metric: LF->MK common-11 (re-based N023): batch-held-out Pearson r, mean over 11 targets = all except grp:CD123/HLA-DR/CD14 (batch-1 pooled assay failed QC: below CD16 crosstalk; best = N014 model re-scored, not a new model), averaged over b1to0 and b0to1, targets_v5 labels; scripts/rescore_targets.py r_mean_common (= train_lf.py r_mean_markers_valid on v5). Legacy 12-target v4 metric stays reported alongside and is UNRESOLVED (best .5742, N014); grp:CD123/HLA-DR/CD14 reported separately
direction: higher
target: 0.67
target_source: Claude proposal 2026-11-09 (N023): close half of the FOV-vs-batch gap on common-11 (FOV ~.71, batch .634); user to confirm
best: 0.6527
best_item: N031
noise: 0.0027
noise_source: .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/d3_seed_noise.json (N038: two v5 seed pairs, n023_v5 diff .0038, n026_zoom2 diff .0037 -> sd .0027; was .002 from one v4 pair, N023)
constraints: SSL pretraining must EXCLUDE the held-out batch (inductive; N031: inductive ensemble .6527 vs transductive .6512 -- caveat resolved)
stall_limit: 5
stall_since: N023
