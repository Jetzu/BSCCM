# Declined / refuted directions (do not propose again)
User rulings (MEMORY + chat):
- Lineage priors (marker exclusivity) and diagnosis / %-positive-per-sample metrics: "don't matter for now" (2026-11-17).
- No additional data beyond the public BSCCM release (BSCCM-coherent subset download allowed, judged low value); public clinical smear
  sets (PBC, MLL23) OK for Giemsa work only.
- GOAL = v10 labels (per-marker asinh cofactor, user 2026-11-17), target .70 (user 2026-11-18). Re-basing labels again is NOT progress.
- SSL pretraining must exclude the held-out draw (inductive). No transductive use of held-out-draw images for training.
- No Modal / external GPU; Helma only. >25 GPU-h per run/sweep needs user approval.
Refuted / flat in KNOWLEDGE (with item):
- v8 per-cell-background labels: changed labels, failed QC (N039). Noise-aware FP/FN positive labels: not identifiable (N049).
- Physics-based instrument augmentation + consistency (N041) flat; low-pass input costs <= .011 -> not image-limited (N042).
- Noisy student: round 1 +.0037 (sub-threshold), round 2 worse (N040, N046). ConvNeXt-T member: errors r .94-.96 with ResNets (N050).
- Test-time AdaBN hurts; DANN harmful (N026, N021). FOV-context head flat (N037). SOTAlign/SigLIP shared spaces: little marker info (N045, N054).
- Pretrained ViTs (DINOv3 S/B, DinoBloom-B) worse than SSL r50 as single models (N053). Ensembling saturated: +.002 per extra member (N055).
- Residual / plain flow matching does not improve new-draw calibration (N044, N034).

Added at review N066 (2026-11-18), flat since N059:
- Output-head / loss variants on the r50 recipe: HL-Gauss (worse .6591), two-part gate x amplitude (.6756 single, flat), within-FOV pair
  loss (.6742), per-FOV random-effect offsets (.6740), Pearson loss (.6726), autofl supervision removed (.6713), panel-matrix latent head
  (.6704), FOV-batch sampler (.6640) -- all within noise of the control .6706 as single models (N060-N062).
- Composite of those heads: best single-model recipe (+.0106 vs fresh controls, N063), but as ensemble member .6902 and as a full
  composite ensemble .6899 (N065) -> head/loss work on fixed labels is exhausted at the ensemble level.
- Ensemble asymptote ~.693 (N059 greedy test-selected .6922; N065 union 6-way .6929 descriptive): more members are closed.
- Unpaired GW alignment of Giemsa and LF spaces fails (N054); domain alignment of clinical smears not the lever (N056).
- Label-side: pixel noise is negligible (r_max .9999, N064); local-background re-derivation changes dim labels (K041) -> eval labels stay v10.
- Open user decision (not declined, not granted): few-shot new-draw calibration (.7042 with >= 200-500 labelled new-draw cells/target).
