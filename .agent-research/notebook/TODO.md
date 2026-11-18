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
