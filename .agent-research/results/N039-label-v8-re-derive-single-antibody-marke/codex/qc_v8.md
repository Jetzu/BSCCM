# N039 pre-registered v8 QC

**Overall: FAIL — stop; v8 is not approved for stage-2 training/scoring.**

Candidate: local-background raw → clean v3 → v4 FOV xfit → v5 mask/normalization (`targets_v8cx.npz`). No model predictions, LF images, or LF features were loaded. Thresholds were not tuned.

Primary cohort is all clean measured single-antibody cells across splits. Map/calibration, GMM, and proxy thresholds fit clean split-0 of the same draw. See `qc_protocol.md` for formulas and fixed seeds; CSV includes split-2 sensitivities.

| Target | Draw | n | (a) FOV excess | (a) | (b) CD19 fraction / d′ | (b) | r(v8,v5) | (c) | (d) |
|---|---:|---:|---:|---|---:|---|---:|---|---|
| CD45 | 0 | 10951 | 0.0129 | PASS | — | N/A | 0.9717 | PASS | N/A |
| CD45 | 1 | 22033 | 0.0046 | PASS | — | N/A | 0.9750 | PASS | N/A |
| CD123 | 0 | 18224 | -0.0014 | PASS | — | N/A | 0.9693 | N/A | N/A |
| CD123 | 1 | 30890 | -0.0089 | PASS | — | N/A | 0.9465 | N/A | N/A |
| CD19 | 0 | 12768 | -0.0006 | PASS | 4.26% / 17.309 | PASS | 0.5255 | N/A | N/A |
| CD19 | 1 | 7736 | 0.0175 | PASS | 2.84% / 0.016 | FAIL | 0.6887 | N/A | FAIL |
| CD56 | 0 | 14212 | 0.0380 | PASS | — | N/A | 0.5597 | N/A | N/A |
| CD56 | 1 | 23141 | 0.0096 | PASS | — | N/A | 0.4806 | N/A | N/A |
| CD14 | 0 | 18442 | 0.0596 | FAIL | — | N/A | 0.9854 | PASS | N/A |
| CD14 | 1 | 25223 | 0.0057 | PASS | — | N/A | 0.9248 | FAIL | FAIL |
| CD16 | 0 | 9397 | 0.0185 | PASS | — | N/A | 0.9769 | PASS | N/A |
| CD16 | 1 | 27857 | 0.0052 | PASS | — | N/A | 0.9905 | PASS | N/A |
| HLA-DR | 0 | 20881 | 0.0237 | PASS | — | N/A | 0.9877 | N/A | N/A |
| HLA-DR | 1 | 32965 | 0.0111 | PASS | — | N/A | 0.8848 | N/A | FAIL |
| CD3 | 0 | 6538 | 0.0056 | PASS | — | N/A | 0.6955 | N/A | N/A |
| CD3 | 1 | 31332 | 0.0026 | PASS | — | N/A | 0.6788 | N/A | N/A |

(a) excess ≤0.05 (20-permutation null; FOVs ≥3 cells). (b) CD19 positive weight 1–6% AND d′≥3 in both draws, d′=(μpos−μneg)/σneg. (c) r≥0.95 for CD16/CD45/CD14 against frozen v5; other correlations are descriptive.

(d) Authors’ lymphocyte gates cover zero cells on these single-antibody slides. True same-cell CD16/CD14 labels are absent. **Required check FAIL/unassessable**; a fluorescence-only proxy below is diagnostic and does not establish a lymphocyte-specific PASS.

The proxy intersects values ≤same-slide clean split-0 medians for released `CD16_full_model_unmixed` and `CD123/HLA-DR/CD14_full_model_unmixed`. These can be crosstalk on slides lacking those antibodies; the latter is a pooled group. Own dye is asinh(XL/10): 500-550 (CD19), 426-446 (HLA-DR/CD14).

| Draw-1 target | Proxy n | r(v8, own dye) | Sign test | r(v5, own dye) | Primary (d) |
|---|---:|---:|---|---:|---|
| CD19 | 2080 | -0.1407 | FAIL | -0.0675 | FAIL |
| CD14 | 6512 | 0.4505 | PASS | 0.3429 | FAIL |
| HLA-DR | 4776 | 0.6624 | PASS | 0.4366 | FAIL |

Mixture weights are descriptive two-Gaussian fits, not externally validated B-cell frequencies. FOV excess measures clustering of the label; it cannot distinguish background error from actual FOV cell-composition differences. No masking, retuning, alternative candidate selection, or model scoring follows a failure.
