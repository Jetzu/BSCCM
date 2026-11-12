# N038 literature: is the dim-marker plateau label noise or image information, and what fixes draw shift?

Agent: read-only literature agent (Claude), 2026-11-12. Sources were read with WebSearch, WebFetch and the bioRxiv API.
Quotes are verbatim as returned by the fetch tool. Lines marked **[snippet]** come from search-engine abstract text only;
the full text was not fetched (PMC and PubMed returned reCAPTCHA or cookie walls). Lines marked **[derived]** are my
own arithmetic or reasoning, not a paper's claim. Nothing in the repo was modified except this file.

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Neighbouring fields settle "label or image" with a noise ceiling: r_max = sqrt(reliability of the target).**
   Neuroscience uses CCnorm / CChalf. Label-free microscopy uses Ounkomol's Cmax, estimated from blank-control image noise.
   Psychometrics uses Spearman disattenuation. When replicates are missing, they build the reliability from (a) a split
   of the measurement units, (b) a physical noise model, or (c) a mixture model of positive and negative populations.
   I found **no paper that does a split-half over spectral channels for unmixed markers**. That would be our own
   construction from standard pieces; see Idea 1.
2. **The BSCCM authors say outright that some labels are noisy.** Quote: "The CD123/CD14/HLA-DR channel was deemed too
   noisy be useful". Quote: "For the cells in batch 1 that were stained with all antibodies, the amount CD19 antibody used
   was 35 percent". In other words, there is a draw-specific label definition for grp:CD3/CD19/CD56.
3. **Comparable label-free work hits the same per-marker plateau.**
   - DeepIFC: brightfield and darkfield imaging flow cytometry, held-out donors. CD19 r=0.61, CD56 r=0.53, CD8 r=0.41,
     against CD45 0.90 and CD14 0.88.
   - ROSIE (H&E to 50 proteins): Pearson 0.285 overall. The authors **re-scoped to the "top 24 biomarkers"**.
   - Nassar 2019: B vs T lymphocytes only reach F1 78%.
   - Nazir & Behera (May 2026) on BSCCM itself: grp:CD3/19/56 r .768, grp:CD45 .799, grp:CD16 .819,
     grp:CD123/HLA-DR/CD14 .339. These match our ranges.
   - Pinkard et al. (NeurIPS 2025) on the same 8 BSCCM markers used **per-marker Gaussian-mixture output heads (NLL)**,
     not MSE.
   - What these groups did instead of pushing regression: thresholded classification and AUROC, balanced training sets,
     mixture outputs, or dropping the markers that cannot be predicted.
4. **Training on noisy labels cannot beat a ceiling set by noisy evaluation labels.**
   - Noise2Noise: "the estimate remains unchanged if we replace the targets with random numbers whose expectations match
     the targets". Zero-mean label noise only adds estimator variance ("divided by the number of training samples").
   - So label cleaning or robust losses can at most reduce variance. That fits N037 being flat.
   - Histogram / HL-Gauss losses "degrade more gracefully" under noisy targets and help optimisation.
   - **I found no paper showing that a noise-aware loss raises correlation against the *noisy* test labels.** Every
     noisy-label regression paper I found scores against clean labels.
5. **Draw shift.** Domain-generalisation algorithms (CORAL, IRM, GroupDRO, DANN) do not beat ERM on RxRx1. AdaBN, TENT
   and ARM-BN **collapse under label shift**: 0.229 vs 0.894 for the composition-aware CS-ARM-BN (Apr 2026).
   - That collapse is a plausible mechanism for our AdaBN failure if the two draws differ in cell composition.
   - **Only targeted, physics- or stain-specific augmentation closes shift without test-domain images.** Gao et al. 2023:
     +3.2 to 15.2 pp OOD. Tellez 2019: "the best performing method did not use any normalization". BigAug: unseen-domain
     degradation 39% to 11%.
   - Context-token methods (ContextViT), AdaBN, TENT, Harmony and TVN all need test-domain images or controls.
6. **Part of our "draw shift" may be label shift, not image shift.**
   - BSCCM: "There is presumably some degree of biological variation between these two isolations". The CD19 dose
     differs in batch 1.
   - Monocyte HLA-DR and lymphocyte subset readouts depend on pre-analytical handling: Döcke 2005, Lao 2024.
   - [derived] D1b has *negative* gaps, where the in-distribution model is worse than the held-out model:
     - single:CD123, draw 0: -.147
     - grp:CD3/19/56, draw 1: -.106
     - grp:CD16, draw 1: -.088

     A pure image shift does not predict negative gaps. They point to label definitions that disagree between the draws.

---------------------------------------------------------------------------------------------------------------------

## 1. How neighbouring fields establish label-noise vs image-information ceilings

### 1.1 The noise ceiling (what every field uses)

| Field / source | Method | Verbatim | Number |
|---|---|---|---|
| Psychometrics: Spearman 1904, via Rasch Measurement Transactions https://rasch.org/rmt/rmt101g.htm | Disattenuation | "Rxy = rxy / sqrt (rxx ryy)"; "Disattenuated values greater than 1.00 indicate that measurement error is not randomly distributed." | n/a |
| Label-free microscopy: Ounkomol et al. 2018 Nat Methods https://pmc.ncbi.nlm.nih.gov/articles/PMC6212323 | Cmax from target-image SNR; noise variance from blank controls | Controls were images "taken with identical imaging conditions but contained no fluorescent labels"; "Model performance for each structure is well-bounded by this limit." | The rendered formula lost its root sign. [derived] corr(S, S+N) = sqrt(SNR/(1+SNR)). |
| Neuroscience: Schoppe et al. 2016 Front Comput Neurosci https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00010/full | CCnorm; upper bound from split-half of trials (CChalf) | [snippet] "the upper bound can be approximated by looking at the similarity between one half of the trials and the other half of the trials" | needs repeated trials |
| Genomics: Basenji, Kelley 2018 Genome Res https://genome.cshlp.org/content/28/5/739/F2.expansion.html | Replicate concordance as ceiling | [snippet] "both the mean and median Basenji prediction accuracy exceed the replicate-to-replicate accuracy" | Caveat: replicate-vs-replicate is a lower bound on the ceiling, because one replicate is noisier than the expectation. |

[derived] For a deterministic predictor f(x) and a label y = t + e with independent noise e:
corr(f, y) = corr(f, t) · sqrt(rel_y), where rel_y = Var(t)/Var(y).
- **The GOAL is scored on v5 labels, so every target is capped at sqrt(rel_v5).**
- Correlated errors (position, autofluorescence, background) that the image *can* see (K007: LF features encode
  position, r .92) count as "signal" in this formula. That makes the ceiling optimistic for biology, but it is the right
  ceiling for the metric as defined.

### 1.2 Without replicates: three estimators used in cytometry and single-cell protein work

**(a) Measurement-model or physical noise (flow and spectral cytometry).**
- Nguyen et al. 2013 Cytometry A (spillover spreading matrix), https://zendy.io/title/10.1002/cyto.a.22251. [snippet]
  "After compensation, the measurement errors arising from multiple fluorescences spilling into each detector become
  evident by the spreading of nominally negative distributions." The SSM is "highly reproducible; its non-trivial values
  show a CV of less than 30%".
- Novo, Grégori & Rajwa 2013 Cytometry A, https://zendy.io/title/10.1002/cyto.a.22272. [snippet, paraphrased by the
  search tool] Modelling Poisson noise in signal formation gives better unmixing, "especially for dim populations", with
  a weighted-least-squares approximation.
- **Recent: AutoSpectral (Burton ... Liston, bioRxiv 2025-10-27)**, https://www.biorxiv.org/content/10.1101/2025.10.27.684855v1.
  Abstract:
  - "An additional source of errors is heterogeneity of cellular autofluorescence, which can affect both the unmixing
    matrix and misalign signals when the matrix is applied to individual cells in complex cell mixtures."
  - "improving incorrectly assigned cell positions by up to 9000-fold, reduces spread, particularly in samples with
    variable autofluorescence".
- Relevance [derived]: BSCCM unmixes single-antibody slides with one global autofluorescence spectrum. Quote: "the
  antibodies spectrum and the ever-present autofluoescence spectrum forming a two-spectra mixing matrix". AutoSpectral
  identifies exactly this autofluorescence heterogeneity as the main spread source for dim markers.

**(b) Background/foreground mixture models (CITE-seq protein).**
- totalVI (Gayoso 2021), https://docs.scvi-tools.org/en/1.4.3/_sources/user_guide/models/totalvi.md:
  - "$\beta_{nt}$ represents a background protein signal due to ambient antibodies or non-specific antibody binding"
  - $\pi_n$ is the "Probability of background for each protein"
  - initialised by a procedure that "fits a two-component Gaussian mixture model for each cell"
  - the denoised value is "the expected value of $(1 − \pi_{nt})\beta_{nt}\alpha_n$". That is a foreground-weighted
    label, i.e. label smoothing through population gating.
- dsb (Mulè, Martins, Tsang; Nat Commun 2022), https://www.biorxiv.org/content/10.1101/2020.02.24.963603v1. Abstract:
  "two major noise sources: protein-specific noise from unbound antibodies and cell-specific noise captured by the shared
  variance of isotype controls and background protein counts."
- Bimodality index (Wang et al. 2009 Cancer Inform.), https://pmc.ncbi.nlm.nih.gov/articles/PMC2730180. [snippet]
  BI = δ·sqrt(π(1−π)), with δ = standardised distance between the component means and π = mixing proportion.
  "a bimodality index of 1.1 or greater is likely to indicate a useful bimodal pattern".
- Stain index (Maecker et al. 2004 Cytometry A), via https://www.nature.com/articles/nmeth.f.229.pdf. [snippet]
  SI = D/W, "where D is the difference between positive and negative populations and W is equal to 2 s.d. of the
  negative population". So δ ≈ 2·SI when the two populations have equal spread.

**(c) Mixture ceiling for a perfect population classifier [derived].**
For a two-Gaussian label with equal component variance, the correlation between the true population indicator and the
continuous label is r_pb = BI / sqrt(1 + BI²).

| π (fraction positive) | δ (separation in negative-SD units) | BI | r_pb |
|---|---|---|---|
| 0.05 | 2.5 | .55 | .48 |
| 0.05 | 4 | .87 | .66 |
| 0.10 | 3 | .90 | .67 |

**A model that identifies every B cell perfectly still scores only about .5-.67 on single:CD19, unless the
within-population label variation is itself image-predictable.** K004 already measured within-class r of .08-.45. Our
in-distribution CD19 is .28/.47.

### 1.3 What the BSCCM paper says about its own labels (arXiv 2402.06191, https://arxiv.org/html/2402.06191v1)

- "The CD123/CD14/HLA-DR channel was deemed too noisy be useful."
- Spectra were combined for signals "with insufficient signal-to-noise to be individually distinguished into one".
- "For the cells in batch 1 that were stained with all antibodies, the amount CD19 antibody used was 35 percent" (S1.5).
- The antibody spectrum was estimated by "Taking the difference of the means of the antibody-positive and
  antibody-negative populations". Regularisation was validated by visual inspection.
- "It is unclear if all of these 10 classes are related to biological variability."
- "The batch index captures the two biological replicates (i.e. distinct cell isolations) on two different dates."
  "There is presumably some degree of biological variation between these two isolations."
  "The vast majority of cells/slides did not have a second replicate."
- "From slide-to-slide and batch-to-batch, there are variations in the average intensity of images in a given channel."
- Imaging: 22 multi-LED patterns plus 1 single-LED, 128×128 px, 12-bit, 20×/0.5 NA, "pixel_size_um": 0.166.

### 1.4 What comparable label-free cell work found, and what it did instead

| Work | Setup | Verbatim / number | What they did instead |
|---|---|---|---|
| **DeepIFC** (Timonen et al., Cytometry A 2023), https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full | IFC brightfield and darkfield to fluorescence images; test = "all images from two samples" (two of six donors, 200,000 cells) | "CD56 (r=0.53, AUROC=0.805)", "CD19 (r=0.61, AUROC=0.800) and CD8 (r=0.41, AUROC=0.728) were less accurately reconstructed"; "likely due to the relatively small numbers of cells ... exhibiting these markers". CD45 0.90, CD14 0.88, CD3 0.87. | Thresholded positivity and AUROC. **Balanced training** (50% target type): "recall of NK, cytotoxic T and B cells increased from 25% to 67%, 13% to 68% and 5% to 54%". Antibody lot changed mid-study: "1st three samples eBioscience, next four samples Miltenyi Biotec". |
| **ROSIE** (Wu et al., Nat Commun 2025), https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full | H&E to 50 CODEX proteins, MSE on the centre-8×8 mean | "a Pearson R of 0.285, a Spearman R of 0.352"; held-out studies 0.241 vs 0.319; "this data imbalance is one reason why our performance on certain biomarkers is poor"; oversampling gave no significant improvement | **Re-scoped**: "we focus our phenotyping analysis on using the top 24 biomarkers, where performance is the most robust". Excluded markers by dynamic range. |
| **Nassar et al. 2019** Cytometry A, https://cronfa.swan.ac.uk/Record/cronfa50355 | Unstained IFC, 85 donors | "two subtypes of WBCs, B and T lymphocytes, were distinguished from each other with an average F1-score of 78%" | Classification of gated classes |
| **Lippeveld et al. 2020** Cytometry A, https://www.biorxiv.org/content/10.1101/680975.full.pdf | Stain-free IFC, 8 WBC classes | [snippet] "0.778 and 0.703 balanced accuracy for classical ML and DL" | Classification; "fine-grained classification remains difficult" [snippet] |
| **Nazir & Behera, arXiv 2605.14717 (May 2026, BSCCM)**, https://arxiv.org/html/2605.14717v1 | DPC 4×28×28; joint classification + regression; split not stated; 1,418 test cells | Pearson r: CD3/CD19/CD56 0.768, CD45 0.799, CD16 0.819, CD123/HLA-DR/CD14 0.339; "These markers reflect functional activation states rather than stable morphological phenotypes." | Loss "ℒreg = SmoothL1(ŷreg, yreg) + β(1 − PearsonCorr(ŷreg, yreg))" plus a focal classification term |
| **Pinkard et al., NeurIPS 2025 (BSCCM)**, https://arxiv.org/html/2405.20559v5 | LED-array information estimates vs protein prediction | "eight protein markers (CD3, CD19, CD56, CD123, HLA-DR, CD14, CD16, and CD45)"; "These networks output parameters for Gaussian mixture models corresponding to each protein marker"; "we simulated lower photon counts by adding synthetic Poisson noise". Information estimates "correlated with protein prediction accuracy". | **Mixture-density heads** with NLL; information-theoretic image-side ceiling |
| **Yoon et al. 2017** Sci Rep, https://www.biorxiv.org/content/10.1101/107805v1 | 3D refractive-index tomography (higher-information input) | "label-free identification of three lymphocyte cell types (B, CD4+ T, and CD8+ T lymphocytes) with high specificity and sensitivity" | More image information (3D RI) rather than a better model |

[derived] On a held-out-donor protocol, DeepIFC's CD19 .61 and CD56 .53 sit above our held-out CD19 .28-.37 and
CD56 .36-.37. Their labels are direct per-channel IFC fluorescence, not 6-channel NMF unmixing followed by a position
detrend, and their inputs include darkfield (scatter). Their magnification was not checked. So an image-limited CD19
ceiling near .3 is **not** what this literature suggests. The most likely gap is label reliability, especially for draw 0, where N035 found CD19 only
r .32 predictable from the cell's own fluorescence integrals.

---------------------------------------------------------------------------------------------------------------------

## 2. Training under noisy continuous or bimodal labels

**Theory: what label noise does to an L2 learner.** Noise2Noise (Lehtinen et al. 2018), https://ar5iv.labs.arxiv.org/html/1803.04189:
- "the estimate remains unchanged if we replace the targets with random numbers whose expectations match the targets"
- "we can, in principle, corrupt the training targets of a neural network with zero-mean noise without changing what the
  network learns"
- Variance: "the variance is the average variance of the corruptions in the targets, divided by the number of training
  samples"

[derived] With roughly 10^5 cells per draw, the variance term is small. Cleaner training labels or robust losses
therefore help mainly through optimisation, imbalance, or non-zero-mean / heavy-tailed noise. Retraining on cleaner
labels while still scoring on noisy v5 labels cannot raise the sqrt(rel_v5) cap. This is consistent with N037
(+.001, flat) and N022/K020 (tail/focal helps in-distribution only).

**Regression as classification (histogram / HL-Gauss).**
- Imani & White, ICML 2018, https://icml.cc/virtual/2018/poster/2092. [snippet] The loss is "easier to optimize, with
  better behaved gradients, resulting in improved generalization". The follow-up, Imani et al., JMLR 2026
  (https://jmlr.org/papers/v27/24-0260.html), attributes the gain to optimisation, not representation.
- Farebrother et al., ICML 2024, https://arxiv.org/html/2403.03950v1:
  - "Classification is less prone to overfitting to noisy targets than regression"
  - HL-Gauss "degrades more gracefully than MSE as the noise scale increases"
  - "we set σ/ς = 0.75 for our experiments" (smoothing width relative to bin width)
  - Evidence comes from RL returns, not from correlation against noisy test labels.

**Imbalanced regression (rare positives).**
- DIR (Yang et al., ICML 2021), https://ar5iv.labs.arxiv.org/html/2102.09554. IMDB-WIKI-DIR MAE, vanilla vs best LDS+FDS:
  all 8.06 to 7.65, few-shot 26.33 to 22.19, many-shot 7.23 to 7.06.
- Balanced MSE (Ren et al., CVPR 2022), https://arxiv.org/abs/2203.16427: [snippet] "accommodate the imbalanced training
  label distribution".
- [derived] Pearson over all cells is dominated by the many-shot bulk. Re-weighting toward rare positives helps r only if
  it improves ranking across the population boundary. DeepIFC's recall gain from balancing is a thresholding effect and
  does not show r gains.

**Noisy-label regression.**
- ConFrag (2025), https://arxiv.org/html/2502.17771v1: "The vast amount of noisy label learning research has focused more
  on classification than regression." Under 30% Gaussian noise, vanilla vs ConFrag MRAE: AFAD-B 28.77 vs 17.49,
  IMDB-Clean-B 26.89 vs 15.70. These are measured against **clean** labels.
- **Evidence gap:** I found no study that reports a noise-aware loss raising correlation against *noisy* test labels.
  [derived] If the noise is independent of the image, a gain on clean labels shows up on noisy labels shrunk by
  sqrt(rel). It is never larger.

**Mixture and population-gated targets used in practice:**
- totalVI's foreground-weighted denoised protein (section 1.2b).
- Pinkard 2025's per-marker GMM heads on BSCCM.
- DeepIFC / Nassar / Lippeveld: gated classes with AUROC or F1.
- Nazir & Behera 2026: SmoothL1 + (1 − Pearson).

---------------------------------------------------------------------------------------------------------------------

## 3. Donor, batch and isolation shift in microscopy-based single-cell prediction: what closes it without test images

### 3.1 What does not work, or needs test-domain data

- **WILDS RxRx1** (Koh et al. 2021), https://ar5iv.labs.arxiv.org/html/2012.07421:
  - "We study the shift induced by batch effects on a variant of the RxRx1 dataset"
  - ERM in-distribution 39.8 vs OOD 29.9; CORAL 28.4, IRM 8.2, Group DRO 23.0
  - "these algorithms did not improve over empirical risk minimization (ERM)"
- **Interventional style transfer** (Pernice ... Caicedo, CVPR-W 2023), https://arxiv.org/html/2306.11890v1:
  - "our naive baseline shows almost complete collapse across all three levels of generalization on GRID"
  - Baseline macro-F1 0.1877 / 0.1381 / 0.1254 vs IST 0.5839 / 0.5350 / 0.3673
  - IST trains only on training domains, but "During testing, we also pass test images through our IST-model by
    randomly pairing them with a training image".
- **CS-ARM-BN** (Sanchez-Fernandez ... Klambauer, arXiv 2604.20824, Apr 2026), https://arxiv.org/html/2604.20824v1:
  - "The accuracy of standard ResNets drops from 0.939±0.005, on the training domain, to 0.862±0.060"
  - "Foundation models, even after Typical Variation Normalization, fail to close this gap."
  - BN statistics under label shift "entangle (i) batch-specific nuisance variation with (ii) class-mixture-specific
    feature shifts"
  - Strongest label shift (α=0.01): AdaBN 0.229, ARM-BN 0.228, CS-ARM-BN 0.894. CS-ARM-BN needs **negative-control
    samples from the new batch**.
- **ContextViT** (Bao & Karaletsos 2023), https://ar5iv.labs.arxiv.org/html/2305.19402:
  - The context network "can indeed also be used during test time on a previous unknown distribution" and needs
    unlabeled test images of the group.
  - Camelyon17 OOD 93.8 to 97.5. **With test batch size 1 it still gives 97.0 vs 93.8**, i.e. per-image context is
    nearly as good.
  - JUMP-CP results are in-distribution only.
- **Cell Painting batch correction** (Arevalo et al., Nat Commun 2024), https://www.biorxiv.org/content/10.1101/2023.09.15.558001v1:
  "we found that Harmony, a mixture-model based method, consistently outperformed the other tested methods". These are
  post-hoc corrections of feature profiles that use all batches (transductive), at well level.
- **SIT-ADDA** (Yang ... Lin, arXiv 2511.12006, Nov 2025), https://arxiv.org/abs/2511.12006: adapts only the earliest conv
  layers by adversarial alignment, choosing depth "without target labels". It still needs target images.

### 3.2 What closes shift without test-domain images

- **Targeted augmentation** (Gao, Sagawa, Koh, Hashimoto, Liang, ICML 2023), https://arxiv.org/abs/2302.11861:
  - generic augmentations "fail to randomize domain-dependent features"
  - domain-invariant ones "randomize all domain-dependent features, both perform poorly OOD"
  - "targeted augmentations set new states-of-the-art for OOD performance by 3.2-15.2 percentage points" (Camelyon17,
    iWildCam, BirdCalls)
- **Stain augmentation vs normalisation** (Tellez et al., MedIA 2019), https://ar5iv.labs.arxiv.org/html/1902.06543:
  - "stain color augmentation improved classification performance drastically"
  - "the best performing method did not use any normalization"
  - "all stain color normalization techniques obtained a poor performance when no color augmentation was used"
  - [derived by the fetch tool from Table 1, 9 external sets] mean AUC: no colour augmentation ≈0.67-0.72, HED-light
    ≈0.95.
- **BigAug / deep stacked transformation** (Zhang et al., IEEE TMI 2020), https://arxiv.org/abs/1906.03347. [snippet]
  "DST models on unseen datasets degrade only by 11% (Dice score change), compared to the conventional augmentation
  (degrading 39%) and CycleGAN-based domain adaptation method (degrading 25%)".
- **Physics-based noise simulation on BSCCM itself** (Pinkard 2025): "we simulated lower photon counts by adding synthetic
  Poisson noise to the original images". The same machinery can serve as photon-budget augmentation.

### 3.3 Isolation shift may sit in the labels (pre-analytical biology), not the images

- BSCCM: "There is presumably some degree of biological variation between these two isolations." The batch-1 CD19 dose
  was 35% (quotes in section 1.3).
- Monocyte HLA-DR, Döcke et al. 2005 Clin Chem (PMID 16214828), https://www.pubmed.gov/16214828/. [snippet, paraphrased]
  With EDTA, ice and staining within 4 h the drift is ~4%. Inter-laboratory CVs were 15-25%, partly due to transport.
- Storage effects, Lao et al. 2024 Pathology, https://aminer.org/pub/65b73faa939a5f40824e026d. [snippet] "24-48 h
  storage/transport can affect the percentage of CD3+, CD4+ T cells, CD8+ T cells, B cells, NK cells and HLADR+ T cells".
- [derived] Our shift is concentrated in HLA-DR (+.05/.15) and CD19 (.00/.10). Those are exactly the readouts the
  pre-analytical literature flags.
- [derived] D1b also shows negative gaps (single:CD123 d0 −.147; grp:CD3/19/56 d1 −.106; grp:CD16 d1 −.088), meaning
  training on both draws *hurt* in-draw prediction. That is the signature of draw-inconsistent label definitions.
  [unverified] It should be reproduced before anyone acts on it.

---------------------------------------------------------------------------------------------------------------------

## 4. Ideas for N038

GPU cost basis (N037 item): one GOAL-recipe ResNet-50 run takes 11-28 min on 1 H100. Base + zoom2 in both directions is
about 1.3 GPU-h. A base-only, 1-seed screen in both directions is about 0.8 GPU-h.

### Idea 1: Label-reliability ceiling per marker per draw (decides whether .67 on v5 labels is reachable)

- **Mechanism.** Three no-replicate estimators on the v5 label pipeline. Each is an *upper* bound on reliability, so
  together they are conservative.
  - (a) **Spectral split-half.** On single-antibody slides the 2-species unmixing (fluor + autofluorescence) is
    over-determined with 6 channels. Unmix twice from disjoint channel triplets, apply the identical background, detrend
    and asinh/z steps, and correlate the halves. Report rel = 2r/(1+r) (Spearman-Brown). This is my construction; I found
    no published precedent.
  - (b) **Poisson/WLS error propagation** of each cell's fluorescence counts through the unmixing (Novo 2013 / SSM
    logic). Report rel = 1 − E[σ²_cell]/Var(label).
  - (c) **Two-component mixture fit (BI) giving the r_pb ceiling** for a perfect population classifier, plus the share
    of within-population variance.
  - Output: r_model / sqrt(rel), i.e. CCnorm, for CD19, CD56, CD123 and HLA-DR in each draw.
- **Why it moves the bottleneck.** It turns diagnosis hypotheses (a) image-limited and (b) label-limited into a
  measurement, and chooses between two programs: an image program (resolution, context, model) or a re-scoping program.
- **Expected outcome [derived].** N035 found CD19 labels only r .32/.56 explainable by the cell's own photons.
  Section 1.2c shows the r_pb ceiling is about .5-.67 for a rare positive. So I expect sqrt(rel) for CD19 on draw 0 near
  .4-.6 and normalised r of .5-.7. In that case the four dim markers cannot each gain +.10 on v5, and .67 is not
  reachable on v5 labels.
- **Cheapest falsifier.** 0 GPU-h, about 2-4 CPU-h. If sqrt(rel) ≥ .85 for the dim markers, labels are not the cap and
  the image/model is.
- **Build cost.** About 1 day. It needs the per-channel fluorescence integrals already used in N035 and the v5 detrend
  code.

### Idea 2: Population-aware output head for the dim markers (MDN / HL-Gauss + Pearson term)

- **Mechanism.**
  - Replace the scalar MSE head for CD19, CD56, CD123 and HLA-DR with either:
    - a 2-component Gaussian mixture-density head (as Pinkard 2025 used on these very markers), predicting the
      posterior mean; or
    - an HL-Gauss histogram head (σ/bin = 0.75).
  - Add an auxiliary population head trained on GMM-gated soft labels (totalVI-style foreground probability).
  - Optionally add the (1 − Pearson) term used by Nazir & Behera.
  - Score as usual against v5 continuous labels.
- **Why it moves the bottleneck.** For bimodal or rare markers Pearson is dominated by the between-population boundary.
  MSE gradients are dominated by within-population autofluorescence noise, and rare positives get shrunk. Histogram and
  mixture losses optimise better and overfit noisy targets less (Imani & White; Farebrother).
- **Expected gain [derived].** Small. Noise2Noise says MSE already targets E[y|x]. I estimate +.00 to +.04 on the dim
  markers in-distribution, i.e. +.000 to +.015 on the 11-target mean. It transfers to held-out only if the head separates
  populations more robustly. This is worth running only if Idea 1 shows headroom (normalised r ≤ .7).
- **Cheapest falsifier.** About 0.8 GPU-h: base recipe, both directions, 1 seed, MDN arm only. Kill if the mean Δr over
  the dim markers is < +.01 held-out.
- **Build cost.** About 0.5 day: head + loss in train_lf.py, plus a GMM gating script on CPU.

### Idea 3: Physics-targeted LED augmentation (inductive draw-shift fix)

- **Mechanism.** Randomise only the acquisition factors that BSCCM documents as varying slide-to-slide and
  batch-to-batch, keeping morphology intact:
  - per-LED-channel multiplicative gain and offset jitter ("variations in the average intensity of images in a given
    channel");
  - Poisson re-noising at random lower photon counts (Pinkard's simulation);
  - defocus rendered as angle-proportional lateral shifts of each LED image. [derived from geometric optics: lateral
    shift ≈ Δz·tanθ_LED; this is the basis of LED-array light-field refocusing, Tian & Waller, Optica 2015.]
- **Why it moves the bottleneck.** Targeted augmentation is the one inductive family with large measured OOD gains
  (Gao +3.2-15.2 pp; Tellez AUC ~.70 to ~.95; BigAug degradation 39% to 11%). DANN, AdaBN and DG algorithms did not move
  RxRx1 or our project.
- **Expected gain [derived].** It closes only the acquisition part of the ~.016 gap. If HLA-DR and CD19 shift is
  pre-analytical or label-side (section 3.3), the gain is about 0. A realistic guess is +.000 to +.008 on the GOAL.
- **Cheapest pre-check.** 0 GPU-h, minutes of CPU. Compare per-LED mean intensity, background and a focus metric
  between draws against the within-draw slide spread. If the draws do not differ beyond slide noise, skip the GPU arm.
- **GPU falsifier.** About 0.8 GPU-h base-only in both directions (about 1.3 GPU-h with zoom2).
- **Build cost.** About 1 day in the data loader. LED angles come from BSCCM metadata.

### Idea 4: Attribute the draw shift to label shift vs image shift before more shift work

- **Mechanism.**
  - Use Idea 1 per draw. If held-out r ratios track sqrt(rel_testdraw), and in-distribution r differs between draws in
    the same way (CD19 .28 vs .47), the "shift" is a reliability difference.
  - Fit per-draw GMMs to get the positive fraction (composition / label shift).
  - Reproduce the D1b negative gaps (CD123 d0, grp d1) with a label-swap control: train on both draws with draw-specific
    label standardisation vs pooled.
  - As a **transductive control only**, run composition-balanced BN statistics on the test draw (CS-ARM-BN idea: reweight
    test cells to the training composition by predicted population). This explains the AdaBN failure if label shift is
    the cause.
- **Why it moves the bottleneck.** Shift methods are pointless for the part of the gap that is label-side.
  CS-ARM-BN shows that AdaBN collapses from label shift alone (0.229 vs 0.894).
- **Expected outcome.** I expect HLA-DR and CD19 shift to be mostly label- or composition-side. The negative gaps in
  D1b already point that way.
- **Cheapest falsifier.** CPU for the reliability and composition parts; about 0.5 GPU-h for the BN control.
- **Build cost.** About 0.5 day on top of Idea 1.

### Idea 5 (conditional on Idea 1; user decision): re-base like the neighbouring fields did

- If sqrt(rel_v5) caps the dim markers, the precedents are:
  - report normalised r (CCnorm; neuroscience);
  - restrict the headline to reliably measured markers (ROSIE's "top 24 biomarkers"); or
  - re-unmix the single-antibody labels with per-cell matched autofluorescence (AutoSpectral) and re-base the GOAL on
    the new labels, as was done for v4 to v5.
- Training on cleaner labels while scoring on v5 is predicted to be flat (Noise2Noise; N037).
- Cost: CPU only, 1-2 days for the AF-matched re-unmixing. **Changing the metric needs user approval**; this is not an
  experiment to run autonomously.

---------------------------------------------------------------------------------------------------------------------

## 5. Gaps / unverified

- No primary source found for spectral-channel split-half reliability of unmixed cytometry labels (Idea 1a is our own
  construction).
- No paper found that shows noise-aware losses raising Pearson against *noisy* test labels.
- Döcke 2005, Lao 2024, Novo 2013, Nguyen 2013, Wang 2009, Lippeveld 2020 and BigAug are quoted from search abstracts
  only. PMC and PubMed blocked the fetches.
- ROSIE per-marker values exist only in its figures. DeepIFC per-donor numbers are not reported.
- The Ounkomol Cmax formula lost its square root in the PMC rendering; the sqrt form is derived.
- The D1b negative-gap interpretation is [unverified]. It comes from our own CSVs and needs reproduction.

## Sources

https://arxiv.org/html/2402.06191v1 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6212323 · https://rasch.org/rmt/rmt101g.htm ·
https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00010/full ·
https://genome.cshlp.org/content/28/5/739/F2.expansion.html · https://zendy.io/title/10.1002/cyto.a.22251 ·
https://zendy.io/title/10.1002/cyto.a.22272 · https://www.biorxiv.org/content/10.1101/2025.10.27.684855v1 ·
https://docs.scvi-tools.org/en/1.4.3/_sources/user_guide/models/totalvi.md · https://www.biorxiv.org/content/10.1101/2020.02.24.963603v1 ·
https://pmc.ncbi.nlm.nih.gov/articles/PMC2730180 · https://www.nature.com/articles/nmeth.f.229.pdf ·
https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full · https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full ·
https://cronfa.swan.ac.uk/Record/cronfa50355 · https://www.biorxiv.org/content/10.1101/680975.full.pdf ·
https://arxiv.org/html/2605.14717v1 · https://arxiv.org/html/2405.20559v5 · https://www.biorxiv.org/content/10.1101/107805v1 ·
https://ar5iv.labs.arxiv.org/html/1803.04189 · https://icml.cc/virtual/2018/poster/2092 · https://jmlr.org/papers/v27/24-0260.html ·
https://arxiv.org/html/2403.03950v1 · https://ar5iv.labs.arxiv.org/html/2102.09554 · https://arxiv.org/abs/2203.16427 ·
https://arxiv.org/html/2502.17771v1 · https://ar5iv.labs.arxiv.org/html/2012.07421 · https://arxiv.org/html/2306.11890v1 ·
https://arxiv.org/html/2604.20824v1 · https://ar5iv.labs.arxiv.org/html/2305.19402 · https://www.biorxiv.org/content/10.1101/2023.09.15.558001v1 ·
https://arxiv.org/abs/2511.12006 · https://arxiv.org/abs/2302.11861 · https://ar5iv.labs.arxiv.org/html/1902.06543 ·
https://arxiv.org/abs/1906.03347 · https://www.pubmed.gov/16214828/ · https://aminer.org/pub/65b73faa939a5f40824e026d
