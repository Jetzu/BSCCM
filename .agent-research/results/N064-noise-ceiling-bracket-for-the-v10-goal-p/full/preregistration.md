# N064: v10 pixel-noise ceiling bracket

## Pre-registration (written before computation, 2026-11-18)

GOAL is the equal-weight mean Pearson correlation over 11 targets (eight single markers and three pooled groups, excluding grp:CD123/HLA-DR/CD14), on held-out-draw split-2 cells in both directions. Best reported score: 0.6862; desired score: 0.70.

Registered reads: optimistic mean r_max <= 0.72 means 0.70 is near-ceiling. CD19/CD56 r_max >= 0.75 refutes the pixel-noise version of “label-limited” for those targets. Evaluate each marker and direction separately; an overall refutation requires both markers in both directions. A result above 0.72 alone does not establish reachability.

For disjoint checkerboard foreground and background pixel halves, report r, rho = 2r/(1+r), and r_max = sqrt(rho). Undefined/negative reliability is reported explicitly, never silently averaged away. Equal weight targets within direction, then equal weight directions. A matched training-draw sample is a diagnostic, not part of GOAL.

Bias: pixel split-half captures photon/pixel noise only. Shared background/FOV/stain-lot error, spatially correlated pixels, spectral-model error and common calibration can inflate agreement. The resulting reliability is an optimistic UPPER bound, not a measurement of all label error or a guarantee of image predictability. Local-background reconstruction differs from the authors’ smoothed background and is audited against released intensities. The Spearman–Brown interpretation assumes equivalent halves and independent errors; clipping/asinh and checkerboard spatial structure make this approximate.

Unstained spread is a diagnostic containing autofluorescence and biological variation, not pure instrumental variance. FOV residual share is a descriptive variance decomposition of y minus the specified three-model ensemble; it also includes model error and FOV cell-composition differences, and is not a causal label-error estimate.

