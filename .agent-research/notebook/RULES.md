# Rules
- Truth = position-detrended labels only (targets_v4 or later); never raw autofl/CD19/CD56/CD123 as ground truth. Inputs keep all real artefacts: no position/metadata inputs, robustness via augmentation (user ruling 2026-11-09) [permanent] (N012, N013, N014, N015, N017, N018)
- External/clinical evals: adaptation and evaluation on disjoint data (PBC half A vs half B; MLL23 never used in training or alignment); contrasts pre-registered before results; anchored or in-domain contrasts count as partly circular (N019, N020, N021, N022)
- A label or metric re-base is NOT progress: always report the same model re-scored on the new labels next to the old number, and count an effect only against the best re-scored on the SAME labels (N023, N039, N047, N048, N051)
- Effects below 2 x seed noise (.0027-.0054 on the GOAL) are flat; back sub-threshold gains with paired per-item counts/Wilcoxon, never headline them (N037, N040, N041, N046, N050)
