# Declined / refuted directions (from MEMORY feedback + KNOWLEDGE Refuted), BSCCM
- No held-out draw/donor data in training (inductive only; SSL must exclude the held-out batch). No transductive tricks.
- No hand-coded lineage/exclusivity rules (user 11-17); a LEARNED co-expression prior from external joint data is OK (11-24, N087).
- No diagnosis/clinical metric as the goal; no lineage prior (11-17).
- Labels: clean LABELS only, inputs keep their artefacts (augment instead); position carries no biology (11-09).
- Per-marker label scale fixed by the v10 rule (user 11-17 re-base); further label re-definitions need user OK and are a re-base, not progress.
- Multi-marker first; external datasets allowed; full encoder fine-tuning OK (11-23).
- No Modal; GPU >25 h needs approval; preempt queue.
- Refuted/flat already (do not repeat): stain normalisation/DANN for GS; Procrustes alignment; unanchored cross-slide students;
  more ensemble members of the same families (N055, N091, N094); seed averaging (N094); decoders on frozen features (N087);
  feature-preserving fine-tuning (N090); ViT/DINOv3/DinoBloom encoders (N053); flow-matching marker images/decoders (N034, N044, N087);
  unsupervised domain alignment for clinical smears (N056); FOV shortcut (N092 refuted it); per-cell background labels v8 failed QC.
