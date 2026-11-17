# Declined / refuted directions (BSCCM) -- checked MEMORY.md feedback + project memory + KNOWLEDGE Refuted, 2026-11-12

User rulings (memory bsccm_project + session):
- NO additional data acquisition (public smear sets OK; no public same-cell Romanowsky+marker dataset exists).
- Data curation is "really important": clean LABELS only; INPUTS keep real artefacts (augment instead of cleaning inputs).
- Slide position carries no biology (labels detrended for position; never use position as a predictive feature for markers).
- Never feed fluorescence to LF models (LF = 23 LED + DPC/phase only); Giemsa cells excluded from LF training.
- SSL pretraining must EXCLUDE the held-out batch (inductive; GOAL constraint).
- GPU via SLURM preempt only; > 25 GPU-h (sum of a portfolio) needs user approval; no CPU training tests.
- Generic skeptic rule (feedback_skeptic_controls_before_claims): baselines + leakage controls + hard subsets before any claim.
Refuted (KNOWLEDGE): DANN / adversarial invariance (flat or harmful everywhere: N010, N014, N020, N021); AdaBN / test-time BN for a
new isolation (-.005..-.011, N026); stain normalisation for GS; Procrustes shared-latent head transfer (N009); no-maxpool stem (hurts);
shared-BN dual-scale (collapses); re-unmixing batch-1 pooled CD123/HLA-DR/CD14 (ill-conditioned, N023 parked).
Measured as flat: label detrend cross-fit retrain (N037 +.001); fine-grained tail/focal teacher (in-distribution only, N022).

## Added at review 2 (N047, 2026-11-17)
- Refuted/flat since N038: cross-fitted label detrend retrain (N037 +.001); per-cell-background labels v8 failed pre-registered QC
  (N039; v8b = dim-dye-only variant needs a NEW pre-registration and the user's go); low-pass probe -> weak targets not image-limited
  (N042; detail-preserving architecture killed); physics-based instrument augmentation (N041, +.0025 single, flat ensemble); noisy
  student round 2 (N046, worse than round 1); semi-supervised OT alignment SOTAlign for Giemsa<->LF (N045 v1: below pairs-only on
  retrieval, far below distillation on markers).
- GOAL target .67 is a Claude proposal, never confirmed by the user; GOAL labels (v5) are frozen unless the user re-bases (N023 precedent).
