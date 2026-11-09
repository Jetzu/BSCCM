# Goal
goal: LF pivot that predicts surface markers in a NEW blood isolation (prerequisite for every transitive GS->MK claim)
metric: batch-held-out LF->MK Pearson r, mean over 12 markers, averaged over b1to0 and b0to1 (train_lf.py protocol; targets_v4 = position-detrended labels)
direction: higher
target: 0.62
target_source: Claude proposal 2026-11-09 after the v4 re-baseline (user to confirm)
best: 0.5545
best_item: N013
noise: 0.002
noise_source: .agent-research/results/N013-targets-v4-remove-slide-position-artefac/n013_v4_noise.csv (seed0 .5535 vs seed1 .5556)
constraints: (none stated)
stall_limit: 5
stall_since: N013
