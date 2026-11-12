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
