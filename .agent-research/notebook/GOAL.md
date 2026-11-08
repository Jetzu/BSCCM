# Goal
goal: LF pivot that predicts surface markers in a NEW blood isolation (prerequisite for every transitive GS->MK claim)
metric: batch-held-out LF->MK Pearson r, mean over 12 markers, averaged over b1to0 and b0to1 (train_lf.py protocol / probe on frozen encoders; targets_v3)
direction: higher
target: 0.55
target_source: Claude proposal 2026-11-08 (user to confirm); FOV-split 0.707, published cell-level split 0.726 (4 groups)
best: 0.4835
best_item: N009
noise: 0.002
noise_source: .agent-research/results/N006-*/noise_seed_reruns.csv (seed 0 vs 1: 0.4825 vs 0.4845)
constraints: (none stated)
stall_limit: 5
stall_since: N009
