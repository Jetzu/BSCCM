# N039 stage 1: local-background v8

**Decision: FAIL; stop before stage 2. No LF model was trained or scored on v8; no prediction files were loaded.**

Recovered the authors’ single-antibody unmixing separately for eight markers × two draws, using only same-draw split-0 cells with hist_row<0. XS=(total_raw−background_lowess)/shading; XL=(total_raw−background)/shading over the six specified channels. The released outputs have floors and nonnegative-unmixing boundary regimes. I fitted a six-channel affine map on cells above both marker and autofluor floors, and restored the split-0 marker floor when assessing fidelity on **all** clean split-0 cells. The v8 raw label applies that exact clipped map to XL. The autofluor column only identifies interior calibration cells; its released values are never an input to the new raw prediction.

Plain OLS tests are also in `unmixing_fidelity.csv`: six channels alone have minimum r=0.978174 (CD19 draw 0), and adding released autofluor gives r≥0.999579. The selected clipped six-channel maps all exceed the required 0.99 fidelity. The spectral least-squares fallback is implemented and documented but was not used.

| Marker | Draw 0 r | Residual SD | Draw 1 r | Residual SD |
|---|---:|---:|---:|---:|
| CD45 | 1.000000000 | 1.01447e-05 | 0.999999996 | 0.00179807 |
| CD123 | 0.999987103 | 0.0624545 | 1.000000000 | 2.85853e-06 |
| CD19 | 0.998070731 | 2.52153 | 0.999999748 | 0.0272037 |
| CD56 | 0.999973905 | 0.119342 | 1.000000000 | 1.80976e-05 |
| CD14 | 1.000000000 | 4.79954e-06 | 1.000000000 | 4.91748e-06 |
| CD16 | 1.000000000 | 1.35792e-05 | 0.999999376 | 0.114919 |
| HLA-DR | 0.999998820 | 0.0411041 | 1.000000000 | 5.71535e-06 |
| CD3 | 1.000000000 | 1.25426e-05 | 1.000000000 | 1.75711e-05 |

Residual SD is in released raw marker units. Largest residual is CD19 draw 0: 2.522 against target SD 40.611. An independent application of the already-fitted maps to split-2 cells has minimum r=0.998937 (`unmixing_validation.csv`); this is raw-map validation, not LF-model scoring.

Ran the existing v3 asinh/q10/z → v4 xfit → v5 mask/per-slide-z code in a scratch copy with `BSCCM_RAW=v8 BSCCM_CLEAN=1 BSCCM_DETREND=xfit`; final name `targets_v8cx.npz`. The unified patches add a separate target output directory, raw-column overlay, and v8 final naming while retaining default paths. For v8, the auxiliary Giemsa pair-suspect flag is copied from the frozen clean xfit target file; its unrelated Giemsa regression is not refitted in the delivered pipeline.

| Required QC | Result |
|---|---|
| (a) FOV excess ≤0.05, each target/draw | **FAIL**: CD14 draw 0 = 0.05960; 15/16 pairs PASS. CD19/CD56/CD3 are all ≤0.03804. |
| (b) CD19 positive fraction 1–6%, d′≥3, both draws | **FAIL**: draw 0 = 4.26%, d′=17.309 (PASS); draw 1 = 2.84%, d′=0.016 (FAIL). |
| (c) r(v8,frozen v5)≥0.95 for CD16/CD45/CD14 | **FAIL**: CD14 draw 1 = 0.92476; other five pairs PASS (0.97171–0.99046). |
| (d) Draw-1 lymphocyte own-dye sign | **FAIL/unassessable**: no authors’ lymphocyte-gate coverage and no same-cell CD16/CD14 measurements on these single-antibody slides. Diagnostic proxy CD19 r=−0.1407 also fails. |

The primary FOV test uses all clean measured rows, FOVs with ≥3 cells, and 20 fixed-seed permutations, matching the N038 diagnostic. GMM fits and proxy thresholds use only clean split-0. Criterion (c) uses the frozen `targets_v5.npz`; CSV also reports the clean xfit baseline and split-2 sensitivity. Detailed per-pair PASS/FAIL values and the exact fixed protocol are in `qc_v8.md`, `qc_v8.csv`, and `qc_protocol.md`.

Caveats: faithful reproduction of the authors’ smoothed-background map does not establish that the same spectra/subtraction are biologically valid after replacing the background. In particular, negative spectral weights can preserve autofluorescence-related artifacts. Draw-1 CD19’s two-Gaussian fit separates variance rather than positive/negative means; its d′ fails despite an in-range weight. The diagnostic low-CD16/pooled-CD14 proxy uses released full-model values on slides where these antibodies are absent; it cannot establish a verified lymphocyte stratum. FOV excess also includes genuine cell-composition clustering. None of these caveats changes the registered thresholds or the FAIL decision. No masking, alternate candidate, or criterion tuning followed QC.

Execution note: the first scratch v4 run executed its inherited auxiliary Giemsa-pair audit regression. I corrected the v8 patch to reuse the frozen flag and rebuilt the candidate. No Giemsa rows entered fluorescence-label calibration or detrending.

Validation: the 412,941×13 candidate has finite measured labels; group/autofl Y and Y_v3 are bitwise unchanged against clean xfit/v3 baselines. Masks, splits, FOVs, batch IDs, histology IDs, class annotations, and the frozen pair-suspect flag are unchanged (`validation.json`). Source data and repo files were not edited; LF inputs were not read or altered. Exact requested package versions were used on CPU.

Reproduce with the commands in `README.md`. The rejected raw/intermediate/final target files remain in `/tmp/codex_N039/targets/`; their SHA-256 hashes are recorded in `build_manifest.json`. Deliverable code, diffs, fitted maps, build logs and QC are in this output directory. Do not promote the rejected candidate for training or scoring.
