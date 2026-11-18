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
