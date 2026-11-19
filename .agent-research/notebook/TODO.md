# TODO

## Next
- K006: head transfer was never re-tested on detrended autofl truth (only distillation, K011).
- K014: LF information map only on the FOV split; batch-held-out map not run.

## Unverified claims

## Unresolvable

## Parked by review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md)
- E slide-replicate nuisance subspace (project between-slide directions out of frozen features), < .5 GPU-h.
- F correlation-direct / HL-Gauss / mixture head for weak markers (likely flat vs noisy eval labels).
- G photon summaries as auxiliary label-side targets (CD123/HLA-DR first), ~.8 GPU-h.
- H reconstructive SSL (masked LF detail / held-out LED prediction), 2-3.5 GPU-h; SSL encoders: check the hist_row<0 filter in train_ssl_lf.py row selection before reuse (Codex).
- I synthetic-aperture probe on the 4,304 batch-1 single-LED cells (hardware ceiling).
- K down-weight the dim batch-1 'all' replicate-0 slide for grp targets.

## Parked from review N059 (2026-11-18)
- Fluorescence spatial descriptors (membrane ring/core, within-cell quantiles) as privileged auxiliary targets (Codex idea 5; 2-3 d build).
- Split-half (pixel checkerboard) reliability of raw fluorescence through the v10 pipeline -> per-marker r_max ceiling (outsider M2; CPU hours).
- Learning curve 1/8..1 of the source draw, in-dist + cross-draw (outsider M3; 6-10 GPU-h) -- run if N062 suggests label starvation.
- Simulated readout randomisation hedge: CPU test on saved predictions first; GPU only if >= +.0054 (lit_crossbatch idea 3).
- Per-target ensemble weights / cross-target stacker fitted on source OOF predictions (outsider 4; cross-target part may be a declined lineage prior).
- USER DECISION: few-shot new-draw readout protocol (.7042 with >= 200-500 labelled new-draw cells per target; AB s3).
- Noise-ceiling bracket on GOAL cells (lit_noisy_ceiling idea 1): unstained-cell noise floor + pixel split-half (Spearman-Brown) + FOV residual variance; pre-registered: optimistic ceiling <= .72 -> report .70 as near-ceiling; CD19/CD56 ceilings >= .75 -> "label-limited" refuted.
- [done N059, killed] Neighbour-bleed test: mean partial r .023 (n059_neighbour.csv).

## Parked by review N066 (2026-11-19; results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/ideas.md)
- AFTER N067 labels exist: one regression head per label version (v10 / local-bg; Wei et al. ICML 2023 separation > aggregation at high noise), 1 GPU-h.
- AFTER N067: precision-weighted targets from 3 background models via triple collocation (third = annulus/plane background); CPU check first.
- AFTER N067: residual correlation of v10- vs consensus-trained members on CD56/HLA-DR/grp (<= .90 -> label-diverse ensemble), 0 GPU-h.
- Rescore frozen members on held-out-draw splits 0+1 (10x cells; GOAL-level FOV-bootstrap SD .0089 vs seed noise .0027) -> needs an eval-only mode in train_lf.py; .5 GPU-h. Diagnostic of split-2 selection bias.
- b01 recipe audit: class losses and the binary grp:CD123/HLA-DR/CD14 target are active only when training on draw 0 -> arms w_cls 0 / drop binary target (1.2 GPU-h).
- Morphology micro-cohort supervision (Codex N066 idea 3; needs LF embeddings of all training cells).
- Assay-conditioned forward model of the 6 raw fluorescence channels (Codex N066 idea 5; 3-5 d).
- Raw-space negative-population (dsb/ADTnorm-like) background correction of TRAINING labels (lit_labelnoise I3).
- FOV/cell reliability weights from background-model disagreement (Codex 2, lit I4): CPU check that disagreement predicts |residual| first.
- Extra pooled labels from single-antibody slides (full_model_unmixed on CD45/CD16 slides; assumption breaker I3; capped, single heads carry the info).
