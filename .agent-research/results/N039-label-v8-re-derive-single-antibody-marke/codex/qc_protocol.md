N039 stage-1 QC protocol, fixed before computing v8 QC

Candidate: the recovered, floor-clipped six-channel affine map applied to per-cell background, followed by the unchanged clean v3 calibration, v4 xfit position detrend, and v5 masking/normalization. No QC-driven retuning. No LF images/features or model prediction files are accessed.

Primary cohort: all finite measured single-antibody rows with hist_row < 0, separately by marker and draw, across splits 0/1/2 (matching N038 FOV diagnostic). Report split-2 FOV/correlation sensitivity too. Every fitted quantity (map, floor, q10, position model, normalization, GMM, proxy stratum thresholds) uses same-draw clean split-0 only.

(a) Keep FOVs with >=3 cohort cells. Cell-weighted between-FOV share = variance of the FOV means broadcast to cells / total cell variance. Null = mean of 20 global cell-value permutations holding FOV memberships fixed. Excess = (observed - null)/(1 - null), without clipping negative values. Independent deterministic seed 3900 + 2*marker_index + draw. PASS iff excess <=0.05, per marker/draw.

(b) CD19 only: 2-component 1D GaussianMixture, full covariance, n_init=3, random_state=0, reg_covar=1e-6, on final v8 Y, clean split-0. Positive component = larger mean. Fraction = its fitted weight. d' = (mu_pos-mu_neg)/sqrt(var_neg), matching N038's e_bgmodel.py definition. PASS iff 0.01<=fraction<=0.06 AND d'>=3 in each draw. Mixture components are descriptive; no threshold/initialization search.

(c) Pearson r(final v8 Y, frozen targets_v5.npz Y) on identical finite measured clean rows. PASS iff r>=0.95 for CD16/CD45/CD14, per draw. Also report correlation with targets_v5cx.npz (clean xfit) to distinguish raw change from detrending.

(d) Own dye = asinh(XL[channel]/10); channel 500-550 for CD19, 426-446 for HLA-DR/CD14. Pearson r with final v8 Y must be >=0 in draw-1 lymphocytes. The authors' lymphocyte gates have zero coverage on the single-antibody slides, and true same-cell CD16/CD14 labels are missing on the other antibody slides. No clarification was received before QC. Report primary d as FAIL/unassessable rather than assert these cells are verified lymphocytes; provide a clearly labeled proxy diagnostic using the intersection of <=split-0 medians of released CD16_full_model_unmixed and CD123/HLA-DR/CD14_full_model_unmixed on each tested slide. This is LF-independent but represents crosstalk/group signal where antibodies are absent, so cannot validate the intended biological stratum. No proxy cutoffs selected from QC results.

Overall: all required tests must PASS. Any FAIL or unassessable required check stops stage 2. No masking, new candidate, or model scoring after QC.
