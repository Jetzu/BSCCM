# N064: v10 pixel-noise ceiling bracket

## Pre-registration (written before computation, 2026-11-18)

GOAL is the equal-weight mean Pearson correlation over 11 targets (eight single markers and three pooled groups, excluding grp:CD123/HLA-DR/CD14), on held-out-draw split-2 cells in both directions. Best reported score: 0.6862; desired score: 0.70.

Registered reads: optimistic mean r_max <= 0.72 means 0.70 is near-ceiling. CD19/CD56 r_max >= 0.75 refutes the pixel-noise version of “label-limited” for those targets. Evaluate each marker and direction separately; an overall refutation requires both markers in both directions. A result above 0.72 alone does not establish reachability.

For disjoint checkerboard foreground and background pixel halves, report r, rho = 2r/(1+r), and r_max = sqrt(rho). Undefined/negative reliability is reported explicitly, never silently averaged away. Equal weight targets within direction, then equal weight directions. A matched training-draw sample is a diagnostic, not part of GOAL.

Bias: pixel split-half captures photon/pixel noise only. Shared background/FOV/stain-lot error, spatially correlated pixels, spectral-model error and common calibration can inflate agreement. The resulting reliability is an optimistic UPPER bound, not a measurement of all label error or a guarantee of image predictability. Local-background reconstruction differs from the authors’ smoothed background and is audited against released intensities. The Spearman–Brown interpretation assumes equivalent halves and independent errors; clipping/asinh and checkerboard spatial structure make this approximate.

Unstained spread is a diagnostic containing autofluorescence and biological variation, not pure instrumental variance. FOV residual share is a descriptive variance decomposition of y minus the specified three-model ensemble; it also includes model error and FOV cell-composition differences, and is not a causal label-error estimate.


## Results: FULL DATA

Directions b10 = train draw 1, evaluate draw 0; b01 = train draw 0, evaluate draw 1. Matched samples are clean split-0 cells in the training draw, matched without replacement by antibody condition to evaluation counts (shortages use all available cells). Smoke samples are balanced across cohort × antibody condition, capped globally.

Each half uses its own foreground mean and corner-background mean, then the identical shading, unmixing, clipping, cofactors and frozen v10 curation. The xfit correction Y_v3 − Y_v4 is reused; no detrending is skipped or refitted. v5 slide medians/IQR and batch restandardisation are replayed from clean split-0 frozen Y_v4. Both halves share calibration and trend corrections, which can increase conditional agreement.

Single-marker maps are the N039 clipped-affine recovery. Pooled maps use the [authors’ fixed gate spectra](https://github.com/Waller-Lab/BSCCM/blob/64de2e0f681a09be3f2a91ae4ead27628bab353d/bsccm/fluorescence_processing/demixing_util.py) and nonnegative active-set optimisation, with nuisance KKT offsets recovered on clean split-0 cells. This approximates the released iterative solver; full-data registered reads are withheld if any GOAL map has r < .99 on calibration or split-2 validation. No alternate unmixing is selected after evaluation.

Pixel aggregates reproduce the [authors’ disk/corner geometry](https://github.com/Waller-Lab/BSCCM/blob/64de2e0f681a09be3f2a91ae4ead27628bab353d/bsccm/fluorescence_processing/compute_fluorescence.ipynb) within 5.99189e-05 counts. Minimum GOAL unmixing fidelity r = 0.998071; maximum v10 curation replay error = 8.1e-07. Reconstructed full-label agreement with v10 is separately reported in reconstruction_fidelity.csv: local backgrounds can change labels despite high unmixing fidelity. Independent local backgrounds also add pixel variance compared with the authors’ smoothed backgrounds, which can lower this photon-noise estimate. Thus the upper-bound interpretation for total label reliability is heuristic, conditional on equivalent-half assumptions.

**Estimand limitation:** independent local corner backgrounds replace the authors’ LOWESS backgrounds. High agreement of these halves is not sufficient to establish reliability of actual v10 labels. Read the full-label reconstruction audit below before interpreting the registered pixel-noise reads; no reachability claim is established.

| direction | target | r_reconstructed_vs_v10 | reconstruction_error_sd | half_difference_sd |
| --- | --- | --- | --- | --- |
| b01 | single:CD45 | 0.97454 | 0.22384 | 0.01000 |
| b10 | single:CD45 | 0.99710 | 0.07991 | 0.00323 |
| b01 | single:CD123 | 0.91576 | 0.45689 | 0.02265 |
| b10 | single:CD123 | 0.86799 | 0.35034 | 0.01210 |
| b01 | single:CD19 | 0.24293 | 0.87063 | 0.03280 |
| b10 | single:CD19 | 0.39542 | 0.82193 | 0.03691 |
| b01 | single:CD56 | 0.75691 | 0.59229 | 0.05011 |
| b10 | single:CD56 | 0.66556 | 0.73497 | 0.06168 |
| b01 | single:CD14 | 0.89050 | 0.45533 | 0.03405 |
| b10 | single:CD14 | 0.97942 | 0.21316 | 0.01113 |
| b01 | single:CD16 | 0.99631 | 0.09702 | 0.00313 |
| b10 | single:CD16 | 0.99890 | 0.06142 | 0.00234 |
| b01 | single:HLA-DR | 0.76190 | 0.68903 | 0.03874 |
| b10 | single:HLA-DR | 0.98387 | 0.17754 | 0.00615 |
| b01 | single:CD3 | 0.98708 | 0.17013 | 0.01376 |
| b10 | single:CD3 | 0.97954 | 0.24280 | 0.02877 |
| b01 | grp:CD3/CD19/CD56 | 0.78970 | 0.62656 | 0.01998 |
| b10 | grp:CD3/CD19/CD56 | 0.93219 | 0.34461 | 0.02348 |
| b01 | grp:CD45 | 0.68260 | 0.86529 | 0.05594 |
| b10 | grp:CD45 | 0.94869 | 0.31198 | 0.03110 |
| b01 | grp:CD16 | 0.91565 | 0.38180 | 0.03809 |
| b10 | grp:CD16 | 0.96383 | 0.26720 | 0.03415 |

| direction | role | target | n | split_half_r | rho | r_max |
| --- | --- | --- | --- | --- | --- | --- |
| b01 | eval | single:CD45 | 2232 | 0.99995 | 0.99997 | 0.99999 |
| b01 | train_matched | single:CD45 | 2232 | 1.00000 | 1.00000 | 1.00000 |
| b10 | eval | single:CD45 | 987 | 0.99999 | 1.00000 | 1.00000 |
| b10 | train_matched | single:CD45 | 987 | 0.99994 | 0.99997 | 0.99998 |
| b01 | eval | single:CD123 | 2973 | 0.99980 | 0.99990 | 0.99995 |
| b01 | train_matched | single:CD123 | 2973 | 0.99992 | 0.99996 | 0.99998 |
| b10 | eval | single:CD123 | 1820 | 0.99985 | 0.99993 | 0.99996 |
| b10 | train_matched | single:CD123 | 1820 | 0.99964 | 0.99982 | 0.99991 |
| b01 | eval | single:CD19 | 722 | 0.99933 | 0.99966 | 0.99983 |
| b01 | train_matched | single:CD19 | 722 | 0.99916 | 0.99958 | 0.99979 |
| b10 | eval | single:CD19 | 1299 | 0.99911 | 0.99956 | 0.99978 |
| b10 | train_matched | single:CD19 | 1299 | 0.99941 | 0.99970 | 0.99985 |
| b01 | eval | single:CD56 | 2112 | 0.99829 | 0.99915 | 0.99957 |
| b01 | train_matched | single:CD56 | 2112 | 0.99725 | 0.99862 | 0.99931 |
| b10 | eval | single:CD56 | 1500 | 0.99779 | 0.99889 | 0.99945 |
| b10 | train_matched | single:CD56 | 1500 | 0.99813 | 0.99906 | 0.99953 |
| b01 | eval | single:CD14 | 2168 | 0.99933 | 0.99967 | 0.99983 |
| b01 | train_matched | single:CD14 | 2168 | 0.99994 | 0.99997 | 0.99998 |
| b10 | eval | single:CD14 | 1771 | 0.99994 | 0.99997 | 0.99998 |
| b10 | train_matched | single:CD14 | 1771 | 0.99918 | 0.99959 | 0.99980 |
| b01 | eval | single:CD16 | 2911 | 1.00000 | 1.00000 | 1.00000 |
| b01 | train_matched | single:CD16 | 2911 | 1.00000 | 1.00000 | 1.00000 |
| b10 | eval | single:CD16 | 1006 | 1.00000 | 1.00000 | 1.00000 |
| b10 | train_matched | single:CD16 | 1006 | 1.00000 | 1.00000 | 1.00000 |
| b01 | eval | single:HLA-DR | 3418 | 0.99923 | 0.99961 | 0.99981 |
| b01 | train_matched | single:HLA-DR | 3418 | 0.99998 | 0.99999 | 1.00000 |
| b10 | eval | single:HLA-DR | 2274 | 0.99998 | 0.99999 | 1.00000 |
| b10 | train_matched | single:HLA-DR | 2274 | 0.99895 | 0.99948 | 0.99974 |
| b01 | eval | single:CD3 | 2968 | 0.99991 | 0.99996 | 0.99998 |
| b01 | train_matched | single:CD3 | 2968 | 0.99958 | 0.99979 | 0.99989 |
| b10 | eval | single:CD3 | 629 | 0.99968 | 0.99984 | 0.99992 |
| b10 | train_matched | single:CD3 | 629 | 0.99990 | 0.99995 | 0.99998 |
| b01 | eval | grp:CD3/CD19/CD56 | 2807 | 0.99977 | 0.99988 | 0.99994 |
| b01 | train_matched | grp:CD3/CD19/CD56 | 2807 | 0.99966 | 0.99983 | 0.99991 |
| b10 | eval | grp:CD3/CD19/CD56 | 1191 | 0.99968 | 0.99984 | 0.99992 |
| b10 | train_matched | grp:CD3/CD19/CD56 | 1191 | 0.99979 | 0.99990 | 0.99995 |
| b01 | eval | grp:CD45 | 2807 | 0.99879 | 0.99940 | 0.99970 |
| b01 | train_matched | grp:CD45 | 2807 | 0.99946 | 0.99973 | 0.99987 |
| b10 | eval | grp:CD45 | 1191 | 0.99947 | 0.99973 | 0.99987 |
| b10 | train_matched | grp:CD45 | 1191 | 0.99889 | 0.99944 | 0.99972 |
| b01 | eval | grp:CD16 | 2807 | 0.99903 | 0.99951 | 0.99976 |
| b01 | train_matched | grp:CD16 | 2807 | 0.99942 | 0.99971 | 0.99985 |
| b10 | eval | grp:CD16 | 1191 | 0.99940 | 0.99970 | 0.99985 |
| b10 | train_matched | grp:CD16 | 1191 | 0.99803 | 0.99901 | 0.99951 |

| direction | role | n_targets | mean_r_max | all_targets_defined |
| --- | --- | --- | --- | --- |
| b01 | eval | 11 | 0.99985 | True |
| b01 | train_matched | 11 | 0.99987 | True |
| b10 | eval | 11 | 0.99988 | True |
| b10 | train_matched | 11 | 0.99981 | True |

Equal-direction GOAL-style mean r_max = 0.999867. Registered read valid: True. Near-ceiling read (<= .72): False. CD19/CD56 read (all four >= .75): True.

## FOV residual share

This aggregate diagnostic uses **all GOAL evaluation cells**, including in pixel smoke mode. Ensemble = arithmetic mean of n048_v9_base, n048_v9_zoom2 and n053_cytoland, joined by global G rather than array position. Labels are exactly v10 Y. The primary share is cell-weighted Var(E[residual|FOV])/Var(residual), using all measured cells, including singleton FOVs. Unweighted variance of FOV means, singleton counts, finite-sample null share and an ANOVA component sensitivity are in fov_error_share.csv. Sparse FOV means inflate the primary share; it must not be read as proven technical error. Predictions were trained with their original target scale; any scale mismatch is retained, with no evaluation-fitted rescaling.

| direction | target | n | n_fov | ensemble_r | fov_share | finite_n_null_share | anova_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b10 | single:CD45 | 987 | 238 | 0.72857 | 0.24813 | 0.24037 | 0.01028 |
| b10 | single:CD123 | 1820 | 299 | 0.64954 | 0.26304 | 0.16383 | 0.11909 |
| b10 | single:CD19 | 1299 | 301 | 0.34233 | 0.68498 | 0.23112 | 0.59118 |
| b10 | single:CD56 | 1500 | 299 | 0.48390 | 0.49180 | 0.19880 | 0.36662 |
| b10 | single:CD14 | 1771 | 268 | 0.84405 | 0.24106 | 0.15085 | 0.10666 |
| b10 | single:CD16 | 1006 | 206 | 0.90699 | 0.24837 | 0.20398 | 0.05609 |
| b10 | single:HLA-DR | 2274 | 363 | 0.52557 | 0.17135 | 0.15926 | 0.01442 |
| b10 | single:CD3 | 629 | 110 | 0.80874 | 0.31364 | 0.17357 | 0.17090 |
| b10 | grp:CD3/CD19/CD56 | 1191 | 252 | 0.80601 | 0.29450 | 0.21092 | 0.10635 |
| b10 | grp:CD45 | 1191 | 252 | 0.75256 | 0.34155 | 0.21092 | 0.16617 |
| b10 | grp:CD16 | 1191 | 252 | 0.85270 | 0.26322 | 0.21092 | 0.06656 |
| b01 | single:CD45 | 2232 | 282 | 0.82432 | 0.16217 | 0.12595 | 0.04158 |
| b01 | single:CD123 | 2973 | 388 | 0.52790 | 0.22932 | 0.13022 | 0.11425 |
| b01 | single:CD19 | 722 | 82 | 0.32766 | 0.53843 | 0.11234 | 0.48334 |
| b01 | single:CD56 | 2112 | 317 | 0.56372 | 0.44175 | 0.14969 | 0.34432 |
| b01 | single:CD14 | 2168 | 333 | 0.82909 | 0.43680 | 0.15321 | 0.33563 |
| b01 | single:CD16 | 2911 | 371 | 0.88953 | 0.18508 | 0.12715 | 0.06655 |
| b01 | single:HLA-DR | 3418 | 396 | 0.43962 | 0.28731 | 0.11560 | 0.19459 |
| b01 | single:CD3 | 2968 | 341 | 0.76435 | 0.17777 | 0.11459 | 0.07158 |
| b01 | grp:CD3/CD19/CD56 | 2807 | 632 | 0.78827 | 0.37158 | 0.22488 | 0.18952 |
| b01 | grp:CD45 | 2807 | 632 | 0.65407 | 0.42300 | 0.22488 | 0.25593 |
| b01 | grp:CD16 | 2807 | 632 | 0.78656 | 0.34102 | 0.22488 | 0.15006 |

## Unstained controls

No GOAL marker has masked-in labels on unstained cells in the targets file. Therefore a GOAL-scale unstained noise floor is unavailable; masked zero fills are not measurements. The available autofl spread is in unstained_spread.csv. Released raw pooled outputs on unstained cells are in unstained_raw_pooled.csv as an explicitly separate raw-scale diagnostic. These spreads include autofluorescence and biological variation. Aggregate diagnostics use all available control rows even in smoke mode; the pixel cap remains enforced.

| draw | split | target | n | sd | iqr |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | autofl | 13095 | 0.87862 | 0.99989 |
| 0 | 2 | autofl | 1650 | 0.93781 | 1.06192 |
| 1 | 0 | autofl | 37251 | 1.03228 | 0.81776 |
| 1 | 2 | autofl | 4620 | 0.97228 | 0.75871 |

Runtime: 69576 unique pixel cells, 8.26 s pixel stage, 9.82 s total, 48 workers. See runtime.json, manifest.json, cell_plan.csv, pixel_fidelity.csv, unmixing_fidelity.csv, curation_replay.csv and reconstruction_fidelity.csv. Completed chunks resume after preemption; changed code, inputs or selection require a fresh output directory.
