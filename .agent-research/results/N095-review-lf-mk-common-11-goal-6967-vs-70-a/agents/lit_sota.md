# N095 lit/SOTA: label-free -> marker prediction across donors/batches (read-only, 2026-11-27)

Fetch status: arXiv/bioRxiv/EuropePMC worked; PMC, Nature, Springer, AACR blocked (403/captcha). Items marked [abstract/snippet] were seen only as abstract or search text.

## 1. Who does it, how evaluated, what number
| system | input -> target | split | number |
|---|---|---|---|
| Nazir & Behera, ICPR 2026, https://arxiv.org/html/2605.14717 | BSCCMNIST 28x28 DPC -> 4 marker groups | **split not stated**; test 1,418 cells | r: "CD45 \| 0.670 \| 0.430 \| 0.799 \| 0.723"; "CD16 \| ... \| 0.819"; "CD123/HLA-DR/CD14 \| ... \| 0.339"; overall .7263 |
| Pinkard et al. 2024 (info-driven design), https://arxiv.org/html/2405.20559 | BSCCM LED -> 8 markers, GMM heads | 15 random splits | NLL only, no r: "A composite loss function summed the negative log likelihood across protein markers, with missing values masked out." |
| DeepIFC (Timonen, Cytometry A 2023), https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full | ImageStream BF+DF -> fluorescence images | "all images from two samples into a test dataset" (held-out samples, 6 donors) | "CD45 (r=0.90), CD14 (r=0.88) and CD3 (r=0.87)"; "CD56 (r=0.53, AUROC=0.805), CD19 (r=0.61, AUROC=0.800) and CD8 (r=0.41, AUROC=0.728)" |
| Nassar et al. 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6767740/ | IFC BF/DF -> WBC class | subject-wise CV, 13 donors, "13 iterations" | ~97 % F1 main types; T vs B ~78 % F1 |
| Lippeveld 2020 (doi 10.1002/cyto.a.23920 [abstract]); Ryu 2021 RI tomography (https://pmc.ncbi.nlm.nih.gov/articles/PMC10521749) | stain-free IFC / 3D RI -> WBC class | stratified 5-fold, 4 samples; "randomly shuffled the entire dataset", 10 donors | bal. acc .778 classical vs .703 DL; 96.7 % (cell-level splits) |
| Walsh-lab CNN, https://www.biorxiv.org/content/10.1101/737346v1.full-text | autofluorescence -> T activation | "a leave-one-patient-out scheme", 6 donors | held-out donor 98.83 %; "Having training data from five donors instead of four likely contributes" |

**Take-aways.** (a) No published work reports per-cell Pearson across BSCCM batches. The only BSCCM r values (Nazir) come from an unspecified, likely random split and sit at our cross-draw level (CD45 .80, HLA-DR group .34), so .70 cross-draw is at or above published SOTA. (b) Strong cross-donor numbers (DeepIFC CD45 .90) come from ImageStream with compensated, background-normalised targets. DeepIFC's weak markers are blamed on rare positives and "morphological features ... difficult to distinguish". (c) Cell-level splits (Ryu, Lippeveld) are optimistic. The lever the donor-held-out papers name is **more training donors**, not a better encoder.

## 2. How they handle the three issues
**(1) Label batch shift.** Every high-accuracy system calibrates on the label side, per batch, using controls from that batch:
- DeepIFC: "Compensated images were created by applying a compensation matrix ... in IDEAS"; per-image background "quantile ... q was set to 0.6".
- Nassar: FITC-only-control compensation; models "would only be expected to achieve high accuracy" on identically prepared samples.
- CytoNorm (Van Gassen 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC7078957): "An identical control sample ... is included in each of the batches". Batch effects are cell-type-specific: "some technical variations affected different cell types at different rates" (within-type corr .99/.97 vs across types .79/.86). Clustered per-type quantile alignment cut EMD by 0.62 (±0.43) vs "only 0.21 (± 1.62)" for global quantile normalisation.
- BSCCM itself (https://arxiv.org/html/2402.06191v1): the background term is "identical for cells from the same physical microscope slide" (no FOV term); "From slide-to-slide and batch-to-batch, there are variations in the average intensity"; "the fluorophores/filters chosen were not ideal for multi-channel imaging"; bleed-through "in many cases substantial"; the 4-spectrum model "didn't explicitly account for autofluorescence". This matches the diagnosis (FOV offsets, sign-flipping autofluorescence-label correlations, N096).

**(2) Noisy/censored targets.** None of the image papers models censoring. Tools elsewhere: Tobit-likelihood NN loss, where "the tobit model yielded better results" than censored MSE/MAE, and NNs "output extreme values for the censored interval" (Dănăilă & Buiu 2024, https://link.springer.com/article/10.1007/s10044-024-01216-9 [abstract]). Tobit works well when "the proportion of nondetects was less than 30%" (BMC Immunol 2008, https://pmc.ncbi.nlm.nih.gov/articles/PMC2592244/ [snippet]). Nazir trains with "Smooth L1 plus β(1 − Pearson)" on targets "Z-scored using training set statistics". Cluster-level label noise: random-effects DNNs (LMMNN, NeurIPS 2021, https://papers.nips.cc/paper_files/paper/2021/file/d35b05a832e2bb91f110d54e34e2da79-Paper.pdf [abstract]); ARMED (https://arxiv.org/abs/2202.11783v3) reports "generalization to unseen clusters (up to 9% vs. conventional models)". DeepIFC label noise: dead cells "6.6%–42.0%" per sample, a CD56 reagent switch mid-study.

**(3) Cross-donor evaluation.** Leave-donor-out is best practice (Nassar, Walsh). Cytoland (Liu 2025, doi 10.1101/2024.05.31.596901 [abstract]: "training protocols ... robust to variations in imaging parameters") gets its robustness from input augmentation plus SSL. That is the input-side fix we already exhausted (N053/N090/N091).

## 3. Limits (consensus)
In order: (1) per-batch target calibration, (2) few training donors/batches (BSCCM has 2), (3) markers without a morphological correlate (DeepIFC CD8 .41, Nazir HLA-DR group .34), (4) rare positives. Our diagnosis is #1, and no paper fixes it from the image side.

## 4. Ideas (respecting declined.md)
**I1 Train-draw compensation of training labels (learned spillover/autofl regression from the training draw's own single-antibody/unstained slides; v10 test labels unchanged).**
- Mechanism: remove the draw-0-specific autofl leakage (HLA-DR~autofl +.37 in draw 0 vs -.21 in draw 1) from the targets, so the model stops learning a component that anti-transfers.
- Why it moves the bottleneck: it attacks the label-shift part that is inductively reachable. This is the standard per-batch compensation (DeepIFC, Nassar).
- Expected gain: HLA-DR b01 (.45) is the main beneficiary. It may cost grp:CD45, whose draw-1 labels carry autofl .60. Net GOAL +0 to +.004, high variance.
- Cheapest falsifier: 0 GPU-h. Refit ridge heads on frozen E8 features with compensated training labels, score b01 on v10 (CPU, under 1 h).
- Build cost: about 0.5 day.

**I2 FOV/slide random-effect label model (LMMNN/ARMED style).**
- Mechanism: a learned per-training-FOV intercept (and scale) with a Gaussian prior, dropped at test.
- Why it moves the bottleneck: FOV-level label variance is about .07 r (within-FOV .72 vs pooled .65). The loss then rewards within-FOV ranking instead of fitting FOV offsets.
- Expected gain: +0 to +.003. N092 found no FOV shortcut, so the gain is mostly lower target variance. There is a confounding risk if FOV offsets track composition.
- Cheapest falsifier: 0 GPU-h. Ridge with FOV dummy columns on frozen E8 features (CPU, 10 min). Escalate to an encoder run (about 4 GPU-h) only if it gains at least +.002.
- Build cost: about 0.5 day.

**I3 Censoring-aware Tobit head + clipped-mean readout.**
- Mechanism: clip-floor labels are treated as "<= floor". The model predicts E[max(latent, floor)], which matches how the clipped labels are scored.
- Why it moves the bottleneck: the censored fractions differ by draw (grp:CD16 train .307 vs test .082; CD3 .094/.427), and MSE treats censored values as exact.
- Expected gain: +.002 to +.006 on 2-3 targets, at most +.001 GOAL. Bundle with I1/I2.
- Cheapest falsifier: 0 GPU-h, Tobit linear head on frozen features.
- Build cost: 2 h.

**I4 Re-base, NEEDS USER RULING. Per-draw re-unmixing of BOTH draws.**
- Mechanism: shared spectra plus an explicit autofluorescence component and a per-FOV background term. Optionally add a CytoNorm-style per-cell-type quantile alignment to a reference.
- Why it moves the bottleneck: this is the literature's actual fix for test-label shift. K043's alt-bg labels already showed .6996 vs .6722.
- Expected gain: largest of the four, plausibly above +.0054, but a re-base.
- Cheapest falsifier: 0 GPU-h, re-unmix the 15 % subset and re-score frozen E8 (CPU, a few hours).
- Build cost: 1-2 days.

Not recommended: more encoder/augmentation/SSL work (exhausted). Per-slide standardisation of test predictions is transductive and needs a ruling.
