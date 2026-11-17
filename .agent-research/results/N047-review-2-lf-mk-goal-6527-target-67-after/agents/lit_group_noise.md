# N047 literature: group-level (FOV / batch) additive offsets in regression TEST labels -- what moves correlation with them?

Agent: read-only literature agent (Claude), 2026-11-17. Single question: when the regression test labels carry group-level
(batch / field-of-view) additive offsets that the input cannot explain, and only one training domain is available, which
methods (from any field) raise the per-sample prediction's correlation with those noisy labels, or legitimately change the
evaluation, and what evidence do they have?

Access: web worked (WebSearch + WebFetch; arXiv/ar5iv, bioRxiv, NeurIPS/ICLR proceedings, BMJ, GitHub raw, rdrr.io).
pmc.ncbi.nlm.nih.gov returned reCAPTCHA walls (ARMED was read from ar5iv instead; others from bioRxiv/publisher).
The fetch tool caps verbatim quotes at ~125 characters, so quotes are exact FRAGMENTS. Marks: **[snippet]** = search-engine
summary only, full text not read; **[derived]** = my arithmetic/reasoning, not a paper's claim; **[BSCCM]** = project numbers
from the context files. Not repeated from N038 lit reports: Spearman disattenuation, Ounkomol Cmax, CCnorm, Noise2Noise, DIR,
Balanced MSE, ConFrag, WILDS RxRx1, ContextViT, Arevalo Cell Painting batch correction, targeted augmentation, dsb/totalVI,
DeepIFC, ROSIE (all in N038 agents/lit_*.md). Nothing in the repo was modified except this file. A throw-away arithmetic script
lives in the scratchpad (not the repo).

----------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Every mixed-model framework gives the same answer for a NEW group: predict the fixed effect, set the group effect to its
   prior mean (0).** LMMNN: when the test set "holds levels unseen before ... it is customary to use" the marginal prediction.
   MBNet (speech MOS with per-judge bias): "the bias net is discarded and only the mean net is used". So, with no group-level
   covariates, no training-side trick can make a per-cell predictor carry the FOV offsets of a never-seen FOV. Pooled r is then
   capped near r_within x sqrt(1 - phi), phi = offset share of label variance [derived]. On the N031 predictions this
   ICC-corrected pooled mean is **.7036** and the within-FOV mean **.7211**, versus pooled **.6527** [derived from BSCCM N038
   d2_within_fov.csv].
2. **The only predictor-side lever on the frozen pooled GOAL is a group-level covariate that predicts the offset** (the
   "contextual"/Mundlak effect; ARMED's Z-predictor; ICRM's in-context unlabelled examples). Evidence that this helps on unseen
   batches is modest: ARMED live-cell melanoma, unseen batches (days), AUROC **.789 vs .773** conventional; MCI unseen sites
   **.837 vs .806**; on AD-MRI it lost to the fixed-effect-only DA net (.645 vs .652). **[derived] On BSCCM, explaining just
   q = 10 % of the FOV-offset variance from LF context would lift the GOAL from .6527 to ~.6695 (q = 5 %: .6614; 20 %: .6847).**
   So the .67 target equals "the LF images predict ~10 % of the authors' background-estimation error". That is falsifier (b) of
   the N047 diagnosis, and a CPU-only test (Idea 1).
3. **Neighbouring fields remove group offsets from the LABELS by default, before training or scoring.** Examples:
   - Multiplexed IF: slide-mean division performed best (Harris 2022).
   - RESTORE: per-tissue negative-control cells set the background.
   - 7-UP CODEX targets are z-scored "across all cell expression values in the sample".
   - Cell Painting/L1000 profiles are "standardized ... per plate".
   - Qlib's default label processor is the cross-sectional z-score `CSZScoreNorm` on `label`.
   - Hidden-factor removal from expression labels gave "a threefold larger collection of cis eQTLs" (Stegle 2010).
   The BSCCM v5 GOAL is unusual in scoring pooled raw intensities across FOVs.
4. **Evaluation practice for repeated measures is settled: report the within-group correlation, not the naive pooled one.**
   - Bland & Altman 1995 (BMJ): within-subject r = -0.51 vs naive pooled r = -0.07.
   - rmcorr (Bakdash & Marusich 2017) is the standard implementation, with CIs.
   - The project's "within-FOV r" (demean label and prediction per FOV) **is** this estimator.
   - Spatial transcriptomics benchmarks compute PCC per image and then average.
   - Dip & Zhang (NeurIPS 2026 workshop) show that pooling across slides is a design choice, not a neutral one: a predictor
     that only knows each slide's mean expression recovers 59-99 % of the reported pooled score. There pooling inflates the
     score; in BSCCM it deflates it.
5. **Self-training plateaus after round 1 for a known reason: the gain is bounded by the student-teacher misfit.**
   - Charikar et al. (NeurIPS 2024): "Gain ... ≈ Misfit between the weak and strong model". A same-architecture round-2 student
     trained to match its teacher has near-zero misfit, so near-zero gain.
   - Noisy Student's later rounds added only +0.5 and +0.3 pp, and only with a much LARGER student and, for round 3, a doubled
     unlabelled ratio.
   - Theory: iterated self-distillation trades denoising against "signal forgetting". Mobahi 2020 shows under-fitting; a 2026
     arXiv paper shows a U-shape, with CIFAR-10 error 47.7 → 43.6 → 45.0 %.
   - In BSCCM, same-draw pseudo-labels cannot carry either draw-2 label definitions or FOV offsets. They can only raise
     within-FOV r, whose pooled effect is scaled by sqrt(1 - phi) on the dim targets.
6. **No 2025-2026 paper was found that studies round-by-round self-training for REGRESSION under domain shift.**
   - The nearest work: NeurIPS 2025 heteroscedastic pseudo-label weighting (no shift); arXiv 2602.14029 (theory, linear
     regression + CIFAR); pseudo-label KRR under covariate shift (uses target inputs, which the inductive constraint excludes).
   - Test-time adaptation for regression (SSA, ICLR 2025) also needs target images. The project already refuted AdaBN for this
     shift.

----------------------------------------------------------------------------------------------------------------------------

## 1. Why a per-sample predictor cannot learn a new group's offset (mixed-model view)

| Source | Claim (verbatim fragment) | Number / relevance |
|---|---|---|
| Simchoni & Rosset, "Integrating Random Effects in Deep Neural Networks", JMLR 24 (2023), https://ar5iv.labs.arxiv.org/html/2206.03314 | For a random-intercept model whose test set "holds levels unseen before", "In this case it is customary to use" the marginal distribution (predict without the random part). Abstract: mixed models "avoid overfitted parameter estimates and ultimately yield better predictive performance." | No unseen-cluster results reported separately. Their gains are for SEEN levels. |
| MBNet / LDNet (Huang et al., "LDNet: unified listener dependent modeling in MOS prediction", arXiv 2110.09103), https://ar5iv.labs.arxiv.org/html/2110.09103 | MBNet's BiasNet predicts the listener bias, defined as "the difference between the mean score and listener score". At test "the bias net is discarded and only the mean net is used to make the prediction". LDNet-ML "relies on a learned virtual mean listener for fast prediction". | BVCC utterance-level LCC: MBNet (MeanNet, self-impl.) .757; LDNet-ML (mean listener) .795; system level .854 → .885. Modelling the per-rater offset in TRAINING helps (repeated ratings per item). At test the offset is still set to the mean. |
| SUREAL (Li & Bampis, arXiv 1611.01715), https://ar5iv.labs.arxiv.org/html/1611.01715 | Subject parameters "represent the bias (i.e., mean) and inconsistency (i.e., variance) of subject"; "The confidence intervals for the quality scores recovered by MLE is generally tighter, compared to the ones by MOS." | Figures only, no numeric table read. Netflix's standard for de-biasing rater offsets in LABELS. |

**[derived] What this implies for BSCCM.** Write the test label as y_ij = s_ij + b_j + e_ij (b_j = FOV offset, independent of the
cell image). The best a crop-only model can do is predict E[y | crop] = E[s | crop], because E[b_j | crop] = 0 (BSCCM already
measures prediction FOV excess ≈ 0). Then:
- pooled r ≈ r_within x sqrt(1 - phi), with phi = var(b)/var(y);
- corrected for phi per target, the N031 ensemble reaches **.7036**, within-FOV **.7211**, pooled **.6527**.

The pooled number moves only through three channels:
- (a) raising r_within. Predictor work does this, but the pooled effect is damped by sqrt(1 - phi) ≈ .71-.81 on CD19/CD56/CD3.
- (b) predicting part of b_j from information that is not in the crop (Section 2).
- (c) changing the label or the metric (Sections 3-4).

----------------------------------------------------------------------------------------------------------------------------

## 2. Predicting the group effect from group-level context (the only predictor-side lever on pooled r)

| Source | Mechanism | Evidence |
|---|---|---|
| **ARMED** (Nguyen, Treacher, Montillo; IEEE TPAMI 45:8081, 2023; arXiv 2202.11783), https://ar5iv.labs.arxiv.org/html/2202.11783 | A "Z-predictor": "train this classifier to predict Z from X on the data from seen clusters"; its softmax gives "a weighted combination of seen clusters that are most similar to each unseen cluster sample", whose random effects are applied. | AUROC on UNSEEN clusters: MCI conversion, unseen sites: ARMED .837 vs conventional .806, LMMNN .811, MeNet .750, DA-only .723. AD-MRI: DA-CNN .652 > ARMED .645 > conventional .603. **Live-cell melanoma, cluster = imaging day/batch** (13 training batches, 11 held-out): ARMED-AEC .789 vs conventional .773 vs DA-AEC .759. Random cluster assignment instead of the Z-predictor lowered performance. |
| **ICRM, "Context is Environment"** (Gupta, Jegelka, Lopez-Paz, Ahuja; ICLR 2024), https://arxiv.org/abs/2309.09888 | Attend to "unlabeled examples as they arrive" so the model can "zoom-in on the test environment risk minimizer". | Abstract only: "significant out-of-distribution performance improvements" (no number read). The group = FOV analogue: the other LF crops of the same FOV are context. |
| **Within-between (Mundlak) random-effects models** (Bell, Fairbrother & Jones, Quality & Quantity 53:1051, 2019), https://eprints.whiterose.ac.uk/id/eprint/134123/ | Separate level-1 "within" effects from level-2 "between" or "contextual" effects, which "should not be assumed to be the same". | **[snippet]**. Methodological: group means of covariates enter as level-2 predictors of the group intercept. |
| **PEER / VBQTL hidden factors** (Stegle et al., PLoS Comput Biol 2010), https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000770 | Latent factors "need to be inferred from the expression profiles" (the LABELS) and removed before association. | "a threefold larger collection of cis eQTLs than previously found". HapMap CEU: 1051 vs 382 genes with an eQTL. Pooled populations: 2696 vs 1045. Here the factors were estimated from labels, so this is a label re-base, not a predictor fix. |

**[derived] Size of the effect if part of the FOV offset were image-visible.** Use the N038 per-target values
(r_pooled, label_fov_excess as phi; 11 targets x 2 directions = 22 entries). Assume an offset predictor h_j explains a fraction
q of var(b) and is uncorrelated with s. The best linear combination then has R² = r_p² + q·phi per target, which gives these
GOAL means:

| q (share of offset variance explained) | 0 | .05 | .10 | .20 | .30 |
|---|---|---|---|---|---|
| GOAL mean, all targets | .6527 | .6614 | .6695 | .6847 | .6987 |
| only CD19/CD56/CD3 get the correction | .6527 | -- | .6664 | .6785 | .6896 |

The diagnosis says crop predictions carry ~0 FOV-level variance. So q > 0 needs information OUTSIDE the crop: FOV cell density,
debris/lysis residue, FOV-wide LF background or focus, the other cells' predictions. Whether any of these tracks the authors'
`background_lowess` error is unknown. The cause sits in the fluorescence channel, so my prior that q ≥ .10 is low (~20-30 %).

----------------------------------------------------------------------------------------------------------------------------

## 3. Structured (group-level) label noise in imaging and genomics: what those fields do to the labels

| Field / source | What they do with group offsets in the TARGET | Evidence |
|---|---|---|
| Multiplexed IF, Harris et al. Bioinformatics 2022 (bioRxiv 2021.07.16.452359), https://www.biorxiv.org/content/10.1101/2021.07.16.452359v1 | Reports "clear slide-to-slide variation in the raw, unadjusted data"; compares ComBat, functional registration, mean division. | "dividing this data by its slide mean, and the functional data registration methods, perform the best under our proposed evaluation framework" (abstract; full PDF not parseable here). |
| RESTORE (Chang et al., Commun Biol 2020), https://www.biorxiv.org/content/10.1101/792770v1 | "we automatically identify negative control cells for each marker within the same tissue" and "use their expression levels to infer background signal level"; profiles "normalized by the inferred background level of the negative control cells". | "RESTORE can remove unwanted variations effectively and shows robust performance" (no number in abstract). This is the per-group negative-population anchor, i.e. a cheaper relative of the per-cell-background v8 re-derivation that failed QC. |
| CODEX imputation, 7-UP (Wu et al., PNAS Nexus 2023), https://www.biorxiv.org/content/10.1101/2022.06.03.494624v1.full | Targets z-scored per sample: "the mean and standard deviation across all cell expression values in the sample". | Patchwise PCC .534 (UPMC-HNC), .489 on external Stanford-CRC. Offsets are removed per sample BEFORE training and scoring. |
| Cell Painting + L1000, Haghighi et al. Nat Methods 2022 (bioRxiv 2021.09.08.459417), https://www.biorxiv.org/content/10.1101/2021.09.08.459417v1.full-text | "We standardized replicate-level profiles per plate to have zero mean and unit variance before averaging"; also "batch effect corrections to improve alignment are another target for future machine learning research". | Cross-modal prediction R² per gene against a shuffled null. Per-plate centring of BOTH modalities is the default. |
| H&E → mIF, MIPHEI-ViT (arXiv 2505.10294, 2025), https://arxiv.org/html/2505.10294v2 | Counter-example: one global per-marker percentile normalisation "across the training set", and "Pearson correlation is computed over the full evaluation set rather than averaged per tile". | Reports problems from "low-quality tiles with high auto-fluorescence" but does not quantify slide-level offsets. Pooled evaluation is used, but nobody measured what it costs. |
| Quant finance, Qlib default handler, https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/data/handler.py | `_DEFAULT_LEARN_PROCESSORS = [{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}]` | Labels (next-day returns, which carry day-level market offsets the features cannot explain) are z-scored per date for TRAINING by default. |
| Numerai data docs, https://docs.numer.ai/numerai-tournament/data | "Each era groups observations from one point in time." The docs advise treating each era as one data point, and metrics are computed per era (paraphrase of docs). | Scoring is per-era correlation. Offsets across eras never enter the score. |

**Cell Painting labels specifically:** I found no paper that treats batch effects in Cell Painting PREDICTION TARGETS as label
noise beyond the per-plate standardisation above. Arevalo 2024 (already in N038) is about input/profile batch correction.

**[derived] Pattern.** Imaging, genomics and finance all remove the group-level component from the target, by default:
- **training:** CSZScoreNorm, per-sample z-score;
- **scoring:** per-era correlation, per-image PCC;
- **association:** PEER.

Some anchor the removal on known-negative units (RESTORE; dsb in N038). None tries to predict an unexplainable group offset from
the inputs.

----------------------------------------------------------------------------------------------------------------------------

## 4. Evaluation practice: within-group vs pooled correlation

| Source | Statement | Number |
|---|---|---|
| Bland & Altman, BMJ 310:446 (1995), https://www.bmj.com/content/310/6977/446 | Within-subject correlation by regression with subject as a factor ("also known as analysis of covariance"), "equivalent to fitting parallel lines through each subject's data". Analysing repeated observations "as if the data were a simple sample" can be "highly misleading". | Within-subject r = -0.51 vs naive r = -0.07 (P = 0.7). |
| rmcorr (Bakdash & Marusich, Front Psychol 2017), https://pmc.ncbi.nlm.nih.gov/articles/PMC5383908 ; R package https://cran.r-project.org/web/packages/rmcorr/ | "Repeated measures correlation (rmcorr) is a statistical technique for determining the common within-individual association"; "Rmcorr is calculated using a form of ANCOVA". | Examples: rmcorr .96 vs averaged-data .13. Raz 2005: r_rm(71) = -.70 vs between-person -.36 to -.40. Gilden 2010: r_rm(32) = -.41 vs pooled r(42) = +.38. Bootstrap CI is in the package. |
| Nakagawa & Schielzeth, Methods Ecol Evol 4:133 (2013), via `performance::r2_nakagawa` docs https://rdrr.io/pkg/performance/man/r2_nakagawa.html | Marginal R² "considers only the variance of the fixed effects"; conditional R² "takes both the fixed and random effects into account". | Standard way to report fixed vs group variance. Marginal R² of y ~ pred + (1·FOV) is still deflated by the FOV variance in its denominator, so on its own it is no fix. |
| ST benchmark, "Benchmarking the translational potential of spatial gene expression prediction from histology", Nat Commun 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11814321/ | "Metrics under this category of evaluation were first calculated at an image level"; "correlation was measured for each gene per image), and then averaged over each patient". | Best PCC .28 (EGNv2, HER2+/cSCC); "all methods consistently exhibited average correlations below 0.25 for ST data". Caveat: best CV fold chosen on the test set. |
| Dip & Zhang, "Measuring What Histology Can Predict: Leakage-Stratified Evaluation for Spatial Transcriptomics", NeurIPS 2026 workshop, https://neurips.cc/virtual/2026/172559 (also /177178) | Pooling across slides: "a predictor that is told each slide's average expression" recovers most of the score; patient-level splits "buy almost nothing". They also "estimate how reliably each gene is measured in the first place". | 360 Visium slides, 12 organs, ~830k spots. The slide-mean-only baseline recovers **59-99 %** of the pooled score (59-87 % excluding two-study organs). Study holdout costs 1.3-6.7x; 58-77 % of the drop survives fake regrouping. |
| McConnell, Nonchev, Koelzer, Rätsch, NeurIPS 2026 workshop, https://neurips.cc/virtual/2026/169428 | Representations "are low-dimensional and strongly structured by slide identity"; recommends "diagnosing slide-specific structure before applying such supervision". | Qualitative. Slide structure limits cross-modal transfer. |

**[derived] Mapping to BSCCM.**
1. The project's within-FOV r (`d2_within_fov.py`: label and prediction demeaned per FOV, FOVs with ≥ 3 cells) is the
   Bland-Altman / rmcorr estimator. It can be reported under that name, with rmcorr's degrees of freedom (N - k - 1) and
   bootstrap CI.
2. A complete report has three numbers per target: pooled, within-FOV (rmcorr) and between-FOV (FOV-mean prediction vs FOV-mean
   label, the REWB decomposition). The between-FOV r is exactly falsifier (b) of the diagnosis. It should be ~0 now and becomes
   positive only if predictions start tracking the offsets.
3. Dip & Zhang show the opposite failure in ST. There, slide means are predictable from tissue type, so pooling INFLATES the
   score. Here offsets are unpredictable, so pooling DEFLATES it. Both say pooled r mixes two components that must be reported
   separately.

----------------------------------------------------------------------------------------------------------------------------

## 5. Self-training / pseudo-labelling for regression under shift, and why it plateaus after one round

| Source | Finding (verbatim fragment) | Number |
|---|---|---|
| Noisy Student (Xie et al., CVPR 2020), https://ar5iv.labs.arxiv.org/html/1911.04252 | Iterations (Table 7): B7 → L2, then L2 → L2, then L2 → L2 with a doubled unlabelled:labelled ratio. Same-model teacher/student: "a consistent improvement of around 0.8% for all model sizes". Noise "plays an important role in enabling the student model to perform better than the teacher". | ImageNet top-1 **87.6 → 88.1 → 88.4 %** (batch ratio 14:1, 14:1, **28:1**). Noise ablation (B5 teacher, 130M images): full 85.1, no augmentation 84.6, no noise 84.3, supervised 84.0. The round-1 jump came with a larger student. Later rounds add ≤ .5 pp. |
| Charikar, Pabbaraju, Shiragur, NeurIPS 2024, https://arxiv.org/html/2405.15116v2 | Thm 1: d(f_sw∘h_s, f*∘h*) ≤ d(f_w∘h_w, f*∘h*) - d(f_sw∘h_s, f_w∘h_w), under convexity and realizability; "Gain in accuracy in weak-to-strong generalization ≈ Misfit between the weak and strong model". | Squared loss. The gain is bounded by how much the student DISAGREES with its teacher. |
| Mobahi, Farajtabar, Bartlett, NeurIPS 2020, https://arxiv.org/abs/2002.05715 | "self-distillation iterations modify regularization by progressively limiting the number of basis functions"; "while a few rounds of self-distillation may reduce over-fitting, further rounds may lead to under-fitting". | Theory plus experiments. Explains a round 2 that is worse than round 1. |
| "Why Self-Training Helps and Hurts: Denoising vs. Signal Forgetting", arXiv 2602.14029 (Feb 2026), https://arxiv.org/html/2602.14029 | "a systematic component that grows with iteration due to progressive signal forgetting", against a stochastic part that shrinks, so the risk is U-shaped. In the theory each round draws a FRESH feature matrix. | CIFAR-10 ResNet-50, label noise .4: error 47.7 → 43.6 % (iteration 2) → 45.0 %. Under very high noise (.8) the error falls monotonically. |
| Pareek, Du, Oh, NeurIPS 2024, https://proceedings.neurips.cc/paper_files/paper/2024/hash/0eb1ac7551ddbae575415aa5183a88be-Abstract-Conference.html | Multi-step self-distillation can reduce excess risk "by a factor of $d$" in linear regression. | UCI regression: MSE down "by up to $47$%". Conditions not read; abstract only. The counterpoint to Mobahi: multi-step CAN help in linear models. |
| Ildiz et al., ICLR 2025, https://arxiv.org/abs/2410.18837v2 | "W2S training, with the surrogate as the weak model, can provably outperform training with strong labels" but "is unable to improve the data scaling law". | A constant-factor gain only. Distillation cannot substitute for more (or differently defined) labels. |
| Gerstgrasser et al., ICML 2024, https://arxiv.org/abs/2404.01413 | Replacing real data with model outputs each generation → collapse; accumulating real + synthetic → bounded test error. | **[snippet]** The linear-model theory gives a finite bound when data accumulate. |
| Sun et al., "Semi-Supervised Regression with Heteroscedastic Pseudo-Labels", NeurIPS 2025, https://arxiv.org/abs/2510.15266v1 | "naive pseudo-labeling can lead to error accumulation and overfitting to incorrect labels"; "SSR involves continuous outputs with heteroscedastic noise". | Abstract has no numbers. No domain shift studied. |
| K. Wang, "Pseudo-Labeling for KRR under Covariate Shift" (v4, Jul 2025), https://arxiv.org/abs/2302.10160v4 | Pseudo-labels on TARGET inputs are used to select among source-trained candidates. Minimax-optimal up to log factors. | **[snippet]** Needs target-domain inputs, which conflicts with the GOAL's inductive constraint. |
| Black-box assisted regression (arXiv 2606.25743, 2026), https://arxiv.org/html/2606.25743 | "holdout selection to revert to f0 when the learned correction is not supported by validation data". | Phase transition between trusting a prior predictor and the labels. A safety rule, not a gain. |
| Adachi et al., "Test-time Adaptation for Regression by Subspace Alignment", ICLR 2025, https://arxiv.org/html/2410.03263v2 | **[snippet]** Naive feature alignment is ineffective for regression. SSA aligns the output-relevant PCA subspace. | Needs target images at test time. The project already refuted AdaBN for this shift (N026). |

**[derived] Why N040 helped once (+.0037 ensemble) and N046 did not.**
1. N040's teacher was a 5-model ensemble and its students were single models. Ensemble → single-model distillation has real
   misfit (the single model cannot match the ensemble), so a gain is possible.
2. N046 used N040 students as teachers for same-architecture students. The misfit between consecutive same-family students is
   small, so Charikar's bound allows ≈ 0 gain. Mobahi and arXiv 2602.14029 predict a slight loss from signal forgetting. Both
   match "worse than round 1".
3. Neither round can learn draw-2 label definitions or FOV offsets, because pseudo-labels come from the training draw.
4. Their pooled effect on CD19/CD56/CD3 is damped by sqrt(1 - phi) ≈ .71-.81.

----------------------------------------------------------------------------------------------------------------------------

## 6. Ideas

### Idea 1 -- FOV-context offset probe (contextual random effect; ARMED Z-predictor / ICRM analogue). Diagnostic first.

- **Mechanism:**
  - On the TRAINING draw only, build per-FOV features from LF data alone. Allowed features: number of cells in the FOV;
    per-LED percentiles of crop-background pixels; DPC/phase background statistics; a focus metric; count of debris/non-cell
    objects; mean model prediction over the OTHER cells of the FOV.
  - Not allowed: no fluorescence (user ruling) and no slide position (user ruling).
  - Regress the per-FOV offset target on these features. Target = FOV mean of the v5 label minus the slide mean, computed from
    labels only and not from any model trained on the other draw, to avoid cross-direction leakage.
  - Use FOV-grouped CV. Then apply the frozen regressor to held-out-draw FOVs and add c·h_j to every cell prediction, with c
    fitted on the training draw.
- **Why it moves the bottleneck:** The diagnosis puts the cap in FOV-level label offsets. A mixed-model reading says a new
  group's offset is predictable only from group-level covariates (LMMNN marginal rule; ARMED Z-predictor; Mundlak). This is the
  only change that can move the frozen pooled GOAL without touching labels, and it IS falsifier (b) of the N047 diagnosis.
- **Expected gain (reasoned):** GOAL ≈ .6527 + {.009, .017, .032} for q = {.05, .10, .20} (table in Section 2).
  - q ≈ .10 reaches the .67 target.
  - My prior for q ≥ .10 is low (~20-30 %). The offset comes from fluorescence background estimation, and crop predictions
    already carry ~0 FOV variance.
  - ARMED's analogous unseen-batch gain was +.016 AUROC.
- **Legitimacy:** This predicts a measurement artefact, not biology. It must be reported separately and approved by the user
  before it counts toward the GOAL, in the spirit of the "position carries no biology" ruling.
- **Cheapest falsifier (0 GPU-h, CPU ≤ 1 h):**
  - Kill if the FOV-grouped CV R² of the offset regressor on the training draw is < .05 for CD19/CD56/CD3, or does not exceed a
    shuffled-feature null.
  - Kill if the cross-draw transfer R² (fit draw 0 → score draw 1 and the reverse) is ≤ 0.
  - A linear/HGB model on ~10 scalar features per FOV is enough. A deep set-model (ICRM-style) is only worth building if the
    linear probe gives q > .05.
- **Build cost:** ~0.5 day CPU. It needs per-FOV aggregates of crop statistics and cell counts, which the existing `fov` index
  in targets_v5.npz supports.

### Idea 2 -- FOV-anchored v5 labels ("v5a") + within-FOV (rmcorr) co-primary. A minimal re-base for the user to decide on.

- **Mechanism:**
  - Subtract a per-FOV anchor from v5 labels. The anchor is a robust FOV-level background: for each marker, the FOV median, or
    a lower quantile, over all cells. Most cells are negative for CD19/CD56/CD3, which is RESTORE's negative-control logic.
  - A variant uses the lower mixture component per FOV.
  - Within-FOV ordering stays identical to v5. That is the property v8 lacked: v8 changed labels to r .48-.70 with v5 and
    failed QC.
  - Score the existing predictions on v5a (pooled) and with rmcorr on v5, and report pooled, within and between r side by side.
- **Why it moves the bottleneck:** It removes only the component the diagnosis identified (FOV offsets) and leaves the cell-level
  label untouched. The precedents are standard practice:
  - per-slide/per-sample/per-plate target normalisation: Harris 2022, 7-UP, Haghighi 2022, Qlib CSZScoreNorm;
  - per-group scoring: Bland-Altman, Numerai eras, ST per-image PCC;
  - the 2026 ST evaluation paper showing pooled scores mix slide-level and within-slide components.
- **Expected gain (reasoned):** On existing N031 predictions the re-based score should land between the ICC-corrected **.704**
  and within-FOV **.721** [derived], i.e. above .67 without a new model. This is a RE-BASE, not progress, so it needs the
  user's go and a new pre-registration (N023 precedent). The legacy pooled v5 GOAL stays reported.
- **Cheapest falsifier (0 GPU-h, CPU minutes):** pre-register these QC checks:
  1. The anchored labels' FOV excess for CD19/CD56/CD3 must fall to ≤ .05 above the permutation null (now .34-.49).
  2. Lineage targets (CD16/CD14/CD45, FOV excess ≤ .1) must change pooled r by < .005.
  3. Anchor reliability: split-half anchors (two random halves of each FOV's cells) must correlate > .8. If not, FOVs hold too
     few cells and the anchor injects noise, so fall back to rmcorr-only reporting.
  4. The anchor must not correlate with FOV composition, i.e. with the FOV-mean prediction (|r| < .1). Otherwise it subtracts
     real biology.
- **Build cost:** a few hours CPU. Labels are a deterministic transform of v5.

### Idea 3 -- Misfit-gated self-training: heterogeneous students + label accumulation, stopped at round ≤ 2. Only if predictor work continues.

- **Mechanism:**
  - Before any GPU run, measure the student-teacher misfit (MSE between student and teacher predictions on training-draw
    cells) for N031→N040 and N040→N046 from the saved predictions.
  - If round-2 misfit ≪ round-1 misfit, Charikar's bound explains N046. The next student must then DISAGREE more while staying
    accurate. Options:
    - a different input view (e.g., DPC/phase-only vs LED-brightfield-only students teaching each other, COREG-style
      co-training [snippet: Zhou & Li, IJCAI 2005, https://ijcai.org/proceedings/2005/66]);
    - a different architecture family;
    - a larger student (the Noisy Student recipe).
  - Keep the ground-truth labels in every round (accumulate, never replace: Gerstgrasser 2024) and stop at round 2 (Mobahi;
    arXiv 2602.14029 U-shape).
- **Why it (only partly) moves the bottleneck:** It raises within-FOV r only. On CD19/CD56/CD3 the pooled gain is scaled by
  sqrt(1 - phi) ≈ .71-.81. It cannot reach draw-2 label definitions or FOV offsets.
- **Expected gain (reasoned):** ≤ +.003-.006 at ensemble level. N040 gave +.0037 with a high-misfit ensemble → single-model
  setup, and N047 shows single-model gains get absorbed by ensembling. This is likely inside the ±.0054 flat zone, so it is not
  a route to .67 on its own.
- **Cheapest falsifier:**
  - 0 GPU-h: the misfit audit on existing predictions. Kill the line if N046's misfit ≥ N040's, because then misfit theory does
    not explain the failure.
  - Then ~1.5 GPU-h for one heterogeneous-view student. Kill it if its misfit to the teacher is not higher than N046's, or if
    the single-model pooled gain is < +.005.
- **Build cost:** ~0.5 day for the audit plus one config change for a view-restricted student.

### (Not recommended) Idea 4 -- within-FOV / mixed-effects training loss

- Train with an FOV-demeaned MSE, a per-FOV Pearson loss, or the LMMNN NLL with an FOV random intercept, so the gradient carries
  no offset noise.
- MBNet/LDNet show gains from modelling per-rater offsets in TRAINING (BVCC utterance LCC .757 → .795). Their labels have
  several ratings per item, though, and at test the offset is still set to the mean.
- The project's own N037 (cross-fitted label detrend, which removed part of the FOV variance, e.g. CD19 FOV share .54 → .34)
  was flat (+.001). The crop model already ignores the offsets (prediction FOV excess ≈ 0).
- Expected ≤ noise. Listed only so it is not re-proposed.

----------------------------------------------------------------------------------------------------------------------------

## 7. Gaps / unverified

- PMC full texts (ARMED, rmcorr, Harris, Nakagawa 2017) were behind reCAPTCHA or 403. ARMED numbers come from the arXiv/ar5iv
  version (the TPAMI version may differ slightly). rmcorr quotes came through a PMC fetch that succeeded once. The Harris
  numbers were not read (abstract only).
- Bell et al. 2019, Gerstgrasser 2024, K. Wang 2025, COREG and SSA are **[snippet]**-level.
- Pareek et al.: the conditions for multi-step gains were not read.
- Numerai: the per-era statement is a paraphrase of the docs. The `numerai_corr` source itself does no era grouping.
- Not found: a 2025-2026 study of round-by-round self-training for regression under domain shift; a Cell Painting paper treating
  batch effects in prediction TARGETS as label noise; any paper that predicts an unexplainable label-side group offset from input
  context in imaging regression. ARMED is the closest, and it is classification/AUROC.
- The q-gain table assumes the offset predictor is uncorrelated with the cell signal, and uses N038's label_fov_excess as phi.
  Both are approximations.

## Sources

https://ar5iv.labs.arxiv.org/html/2206.03314 · https://ar5iv.labs.arxiv.org/html/2110.09103 · https://ar5iv.labs.arxiv.org/html/1611.01715 ·
https://ar5iv.labs.arxiv.org/html/2202.11783 · https://arxiv.org/abs/2309.09888 · https://eprints.whiterose.ac.uk/id/eprint/134123/ ·
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000770 · https://www.biorxiv.org/content/10.1101/2021.07.16.452359v1 ·
https://www.biorxiv.org/content/10.1101/792770v1 · https://www.biorxiv.org/content/10.1101/2022.06.03.494624v1.full ·
https://www.biorxiv.org/content/10.1101/2021.09.08.459417v1.full-text · https://arxiv.org/html/2505.10294v2 ·
https://raw.githubusercontent.com/microsoft/qlib/main/qlib/contrib/data/handler.py · https://docs.numer.ai/numerai-tournament/data ·
https://www.bmj.com/content/310/6977/446 · https://pmc.ncbi.nlm.nih.gov/articles/PMC5383908 · https://cran.r-project.org/web/packages/rmcorr/ ·
https://rdrr.io/pkg/performance/man/r2_nakagawa.html · https://pmc.ncbi.nlm.nih.gov/articles/PMC11814321/ ·
https://neurips.cc/virtual/2026/172559 · https://neurips.cc/virtual/2026/177178 · https://neurips.cc/virtual/2026/169428 ·
https://ar5iv.labs.arxiv.org/html/1911.04252 · https://arxiv.org/html/2405.15116v2 · https://arxiv.org/abs/2002.05715 ·
https://arxiv.org/html/2602.14029 · https://proceedings.neurips.cc/paper_files/paper/2024/hash/0eb1ac7551ddbae575415aa5183a88be-Abstract-Conference.html ·
https://arxiv.org/abs/2410.18837v2 · https://arxiv.org/abs/2404.01413 · https://arxiv.org/abs/2510.15266v1 · https://arxiv.org/abs/2302.10160v4 ·
https://arxiv.org/html/2606.25743 · https://arxiv.org/html/2410.03263v2 · https://ijcai.org/proceedings/2005/66
