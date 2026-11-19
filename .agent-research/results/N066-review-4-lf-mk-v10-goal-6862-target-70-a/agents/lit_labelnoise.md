# N066 lit: structured (cluster-level, pipeline) error in regression targets that also sits in the evaluation labels

Agent: read-only literature researcher, 2026-11-19. Tools: WebSearch + WebFetch, both available.
Quote provenance: WebFetch passes pages through an extraction model that caps each quote at about 125 characters, so the quotes are
short verbatim fragments. Quotes tagged [snippet] come from the search engine's result text because the page itself was blocked
(RSC 403, PMC captcha, bioRxiv bot wall). Treat [snippet] quotes as lower confidence.
Context read: GOAL.md, N066 attempts.md / declined.md / diagnosis.md, KNOWLEDGE K028/K029/K031/K033/K041, IDEAS.md, INDEX (N067 =
plain consensus targets v10+local-bg, already queued).

## Bottom line

1. **Theory gives a clear ceiling. Denoised targets help only through variance.** With L2 loss, replacing training targets with
   any targets that have the same conditional mean leaves the optimum unchanged. Noise matters only through finite-sample variance
   (Noise2Noise). On the evaluation side, error in y attenuates the observed r by sqrt(reliability of y) (Spearman). A gain in
   r(pred, truth) obtained from cleaner TRAINING targets therefore carries over to r against NOISY v10 labels at about
   x sqrt(rel_v10), as long as the v10 error is independent of the prediction. The evaluation-side cap (lever (a)) cannot be moved
   inductively. That matches the N066 diagnosis.
2. **No direct quantified study fits our case**: train on denoised regression targets, score Pearson r against the noisy pipeline
   labels. The closest numbers come from MOS prediction (LDNet). There, training on each rater's labels with a mean-listener head
   beats training on the averaged label by +.026/+.029 SRCC on BVCC, scored against the noisy mean scores. It is -.017 at
   utterance level on VCC2018, so the result is mixed. Retinopathy (Krause): adjudicated labels "allow substantial improvements"
   (no number in the abstract). fMRI (GLMsingle): decoding accuracy roughly triples with denoised betas, but it is scored on
   denoised data.
3. **Multi-annotator work favours keeping label versions separate when noise is high or labels are scarce** (Wei et al.; Guan et
   al.; LDNet). That describes exactly the weak markers here (CD19 v10-local r .24-.40, CD56 .67-.76, K041). N067's plain average
   is the "aggregation" arm. The literature predicts that "separation" (multi-head) wins on those targets.
4. **Neighbouring fields correct cluster-level target error with a reference population, not with model residuals.** Examples:
   HCS B-score uses plate row/column median polish; JUMP has a well-position correction step and uses DMSO negative controls per
   plate; CITE-seq has dsb (empty droplets + isotype controls), totalVI (a per-protein background mixture) and ADTnorm
   (negative-peak landmark alignment per batch). The BSCCM analogue is the marker-negative population inside each FOV. It must be
   corrected in RAW intensity space, before asinh. A per-FOV intercept in label space (N061, flat) is mis-specified for additive
   background error.
5. **No paper on imaging-flow or LF-to-marker deep learning tests cross-donor or cross-draw per-cell marker regression.** Published
   work uses random or held-out-sample splits (DeepIFC, BSCCMNIST 2026) or LOOCV on 9 patients on one instrument (iCellCnn). The
   project's draw-held-out .69 has no published comparator. The newest BSCCM regression paper (May 2026) uses a random split and
   gets r .339 on grp:CD123/HLA-DR/CD14.

## (1) Consensus / multiple label versions / errors-in-variables

- **Noise2Noise** (Lehtinen et al., ICML 2018), https://ar5iv.labs.arxiv.org/html/1803.04189
  - "Consequently, the optimal network parameters θ of Equation (5) also remain unchanged," / "if input-conditioned target
    distributions p(y|x) are replaced with arbitrary distributions" / "that have the same conditional expected values."
  - "For finite data, the variance is the average variance of the corruptions in the targets," / "divided by the number of
    training samples"
  - For non-Gaussian (Monte Carlo) target noise: "noisy targets took approximately twice as long to converge." and "the gap
    between the two methods had not narrowed appreciably, leading us to believe that some quality difference will remain even in
    the limit."
  - Consequence for us: averaging K label pipelines whose errors correlate at rho cuts the label-noise variance term by a factor
    (1+(K-1)rho)/K. That is worth the same as more labels on that term only. K033 gives a rough scale: 50 -> 100 % labels gave
    weak-4 +.034 and GOAL +.015 on v5.
- **Measurement error in y only**, https://en.wikipedia.org/wiki/Correction_for_attenuation. Error in y "causes uncertainty in the
  estimated slope, but not bias". Formula r_x'y' = r_xy / sqrt(r_xx' r_yy').
- **Label-noise performance bounds.** Li et al. 2019/2021, https://arxiv.org/abs/1912.08141: "Such label noise puts a fundamental
  limit to the performance attainable by regression models." The R2 bound "depends only on the noise associated with the response
  variable". Crusius, Cipcigan & Biggin, Faraday Discuss. 2025, https://pubs.rsc.org/en/content/articlehtml/2025/fd/d4fd00091a
  [snippet]: "When incorporating non-zero model prediction noise σpred, which could arise from model bias, model variance, or noise
  in the training dataset, we also identify a realistic performance bound." They also report: "Four were identified to have
  reached or surpassed dataset performance limitations and thus, they may potentially be fitting noise." Caveat for us: these bounds
  need sigma_noise. Pipeline disagreement only lower-bounds the v10 error (its shared part is invisible). This agrees with K041:
  "no numeric ceiling for v10 is identifiable without a reference measurement".
- **Model each labeler separately.** Guan, Gulshan, Dai, Hinton, https://arxiv.org/abs/1703.08774. Standard approaches "do not make
  any use of potentially valuable information about which expert produced which label". Their method: "we propose modeling the
  experts individually and then learning averaging weights for combining them".
- **LDNet** (Huang et al., ICASSP 2022), https://ar5iv.labs.arxiv.org/html/2110.09103. "most previous works only used the mean score
  as the training target"; "LDNet-ML is the best way to utilize the mean scores". Table 1, MBNet (mean-trained) vs LDNet-ML:
  BVCC system SRCC .860 -> .886 and utterance .765 -> .794; VCC2018 system .957 -> .979 but utterance .630 -> .613. The evaluation
  labels are themselves noisy rater means, which is the closest analogue to our setting that I found.
- **Aggregate or separate?** Wei et al., https://arxiv.org/abs/2206.07181: "label separation is preferred over label aggregation
  when the noise rates are high" / "or the number of labelers/annotations is insufficient".
- **Adjudicated (denoised) labels.** Krause et al., Ophthalmology 2018, https://arxiv.org/abs/1710.01711v2: "We find that a small
  set of adjudicated DR grades allows substantial improvements in algorithm performance."
- **Pipeline-denoised responses** (GLMsingle, Prince et al., eLife 2022), https://pmc.ncbi.nlm.nih.gov/articles/PMC9708069/. Method:
  "data-derived nuisance regressors are identified and used to remove noise from" responses, with "a cross-validation procedure to
  determine the optimal number of nuisance regressors to add". Effect: "we observe that mean accuracy approximately triples in NSD
  and doubles in BOLD5000" (decoding scored on denoised data, so this is not a noisy-eval test). The design pattern is useful: a
  library of label versions plus a cross-validated choice of denoising strength.
- **Inverse-variance weighting when the labelling process yields per-label variance** (BIV, Mai, Khamies & Paull 2021),
  https://ar5iv.labs.arxiv.org/html/2107.04497. "In many situations however, the labelling process is able to estimate the variance
  of such distribution for each label". Result: "BIV was performing consistently better than L2 and at least better than cutoff in
  all the experiments we ran." Table 1 test loss: UTKF .46 vs L2 .56; BikeSharing .096 vs .122.
- **Last 12 months.** GRAIN (Guo, Wang & Yi, arXiv 2611.10550, 7 Nov 2026), https://arxiv.org/abs/2611.10550: "label noise exhibits
  heterogeneous magnitudes across instances". For large deviations it uses "ordinal structure among samples", and it claims to
  "consistently improves robustness and generalization under diverse noise conditions" (no numbers in the abstract). I found
  nothing else from the last 12 months on multi-version regression targets.

## (2) Cluster-correlated target error: HCS / Cell Painting / single-cell omics / mixed models

- **HCS plate-position effects (B-score)**, https://rdrr.io/bioc/cellHTS2/man/Bscore.html: "Using this method, a two-way median polish
  is fitted, on a per-plate basis, to account for row and column effects." The MAD scaling step: "The intention of such a further
  adjustment is to compensate for plate-to-plate variability in dynamic range."
- **JUMP profiling recipe**, https://raw.githubusercontent.com/broadinstitute/jump-profiling-recipe/main/DOCUMENTATION.md:
  "well_correct: Corrects for well position effects"; "cc_regress: Performs cell count regression"; an example step order is
  "profiles_outlier_wellpos_harmony". Arevalo et al. 2024 (preprint full text, https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10516049/fullTextXML) describes
  the data hierarchy: "with each readout originating from a region in a well within a multi-well plate" / "which in turn comes from
  an experimental batch at a particular laboratory". Layouts are designed "to account for position effects that may cause profiles
  to look similar". DMSO wells "are used for detecting and correcting plate to plate variations". In the hardest scenario, "none of
  the methods are able to adequately remove the batch effects". Note that these corrections act on profiles that serve as features
  or targets, and the reference is negative-control wells, never model residuals.
- **dsb for CITE-seq** (Mulè, Martins & Tsang, Nat Commun 2022), https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9018908/fullTextXML: "We discover that
  protein-specific noise originates from unbound antibodies encapsulated during droplet generation". The technical component
  "combines this value with the shared variation in values of isotype control proteins", and "the dsb technical component for each
  cell is then regressed out of the ambient-noise-corrected values". The CRAN vignette,
  https://cran.r-project.org/web/packages/dsb/vignettes/end_to_end_workflow.html, adds: "ADT reads in empty / background drops
  (outnumbering cell-containing droplets > 10-fold in all experiments) were highly concordant with ADT levels in unstained
  spike-in cells."
- **totalVI**, https://docs.scvi-tools.org/en/stable/user_guide/models/totalvi.html: the background term "represents a background
  protein signal due to ambient antibodies or non-specific antibody binding". Its prior parameters "are initialized through a
  procedure that fits a two-component Gaussian mixture model for each cell". It also offers "the `transform_batch` parameter that
  allows counterfactual prediction of expression in an unobserved batch".
- **ADTnorm** (Nat Commun 2025), https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11261982/fullTextXML: "ADTnorm normalizes protein expression by aligning the
  landmarks across datasets" (the landmarks are the negative and positive peaks). On the negative peak, normalization should keep
  its "essential role in cell-type identification rather than its artificial removal". It notes that "variable staining quality is
  a major source of batch artifacts across samples and laboratories" and reports that it "outperformed other tools in balancing
  cell-type separation with cross-study batch effect removal" against 14 methods.
- **Random-effects deep nets.** ARMED (Nguyen, Treacher & Montillo, TPAMI 2023), https://ar5iv.labs.arxiv.org/html/2202.11783: "The
  authors of LMMNN propose to use only the fixed effects of their model on unseen clusters". ARMED reports "generalization to unseen
  clusters (up to 9% vs. conventional models)" and warns "we suggest some caution when using our method for data with fewer than 4
  clusters." ARMED also uses a domain-adversarial term, and DANN was harmful here (N021). Our test FOVs are all unseen clusters,
  so a random-effects head can only help by cleaning the fixed-effect fit. That is the same lever as target denoising, which fits
  N061 coming out flat at .6740.
- **Synthesis for BSCCM.** Every field above estimates the cluster-level technical term from a population whose true signal is
  known to be about zero: negative controls, empty droplets, isotype controls, the negative peak. They correct in the measurement's
  native (additive) space. Our background error is additive in raw fluorescence and is then pushed through the per-marker asinh. In
  label space the error is therefore largest for negative and dim cells, so a constant per-FOV offset in label space (N061) cannot
  represent it.

## (3) Do denoised TRAINING targets raise r against NOISY evaluation labels?

- **Mechanism.** If the v10 error e is independent of the prediction, r(pred, v10) = r(pred, truth) x sqrt(rel_v10) (Spearman,
  above). Any training-side gain in r(pred, truth) passes through, shrunk by sqrt(rel). There is one exception:
  - **Artefact-fitting risk** (Northcutt, Athalye & Mueller 2021, https://ar5iv.labs.arxiv.org/html/2103.14749): "ResNet-18
    outperforms ResNet-50 if the prevalence of originally mislabeled test examples increases by just 6%", and "lower capacity
    models may be practically more useful than higher capacity models". Model rankings on noisy test labels can flip.
  - Applied to us: if part of the v10 background error is predictable from the image (crowding, bright neighbours), a v10-trained
    model scores that part and a consensus-trained model loses it. N066 instrument 4 found the FOV-level residual not predictable
    from LF context, so the risk is low. A v10 head kept in the model removes it entirely.
- **Noisy validation is still a valid ranking proxy** (classification; Chen et al., AAAI 2021, https://arxiv.org/abs/2012.04193)
  [snippet]: "The accuracy on hold-out noisy samples is an unbiased estimator of the accuracy on noisy distribution. Therefore, a
  noisy validation set is reliable."
- **Quantified cases:**
  - LDNet (above): +.026/+.029 SRCC against noisy mean labels; one setting at -.017.
  - Bovine satellite cells, https://arxiv.org/html/2410.13685: "background noise was reduced by subtracting the average of the
    background fluorescence scans", and "This pre-processing resulted in higher values for SSIM and Pearson Correlation Coefficient
    metrics". DAPI: "After denoising, the average Pearson correlation coefficient for DAPI increased to 0.212" (from .065). The
    paper does not say whether it evaluated against raw or denoised targets, so this is weak evidence.
  - I found no paper quantifying exactly "denoised training targets, Pearson against noisy pipeline labels" for per-cell regression.
- **Expected size for us.** The size depends on how much of the weak-target error is label-noise variance in the learned function.
  - Upper bound: the K033 doubling effect (+.034 on weak targets, +.015 GOAL on v5).
  - Realistic: a fraction of that, on about 5 of 11 targets.
  - **Direction caveat** (from K041 + diagnosis): v10 and local-bg disagree on HLA-DR and grp:CD45 mainly in draw 1 (.68-.76 vs
    .95-.98 in draw 0). Cleaner draw-1 training targets therefore help b10 (already .7036). The b01 deficit on those targets sits in
    the draw-1 EVALUATION labels, which lever (a) cannot fix. Only CD56 and CD19, dim in both draws, can gain in both directions.

## (4) Cross-batch generalisation in imaging-flow / LF-to-marker deep learning

- DeepIFC (Timonen et al., Cytometry A 2023), https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full.pdf [snippet]. The
  generalisation test is an unseen cell type, not a new batch: "We trained DeepIFC models on IFC data generated on peripheral blood
  mononuclear cells (PBMC), and evaluated the performance of these models also on data on red blood cells (RBC), a cell type not
  used in training." It reports positivity AUROCs "(CD45, AUROC=0.982; CD14, AUROC=0.979; CD3, AUROC=0.959)" on a withheld test
  set.
- iCellCnn (Otesteanu et al., Cell Rep Methods 2021), https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9017143/fullTextXML: "All the data in our study were
  acquired by using an in-house built IFC platform." Validation is "a leave-one-out-cross-validation (LOOCV) approach" over 9
  patients. On other instruments: "We assume that refining the model for data acquired from a different IFC might involve
  re-training of the iCellCnn."
- WBC classification under domain shift (Tsutsui, Su & Wen, ICASSP 2023), https://ar5iv.labs.arxiv.org/html/2303.01777: "batch
  normalization is the cause of poor generalization". GroupNorm "significantly improved ResNet accuracy (%) on the LISC dataset from
  23.11±3.04 to 73.07±2.07". Here AdaBN hurt (N026), and the diagnosis says the input is not the limit, so this is low priority.
- Nazir & Behera, arXiv 2605.14717 (14 May 2026), https://arxiv.org/html/2605.14717, using BSCCMNIST 28x28 DPC: "BSCCM test split
  (1,418 cells)." Targets "are Z-scored using training set statistics to reduce scale imbalance". Per-marker r: CD16 .819, CD45
  .799, grp:CD3/19/56 .768, grp:CD123/HLA-DR/CD14 .339. There is no draw split and no discussion of label or background error:
  "Certain markers simply lack strong morphological correlates."
- Virtual staining for HTS (Tonks et al. 2024), https://arxiv.org/html/2407.06979v3: test plates were held out ("three plates,
  excluded from any training or validation sets"), and models "trained on breast cell samples consistently show poor
  generalization."
- Verdict: no published benchmark tests per-cell marker regression across blood draws. Nothing here offers a better cross-batch
  recipe than what the project already tested.

## Ideas (respecting declined.md; N067 = plain v10+local average, already queued, is the reference arm)

**I1. Label-version multi-head ("separate, don't aggregate").**
- Mechanism: shared trunk with three heads, for v10, local-corner-bg (N064 reconstruction) and optionally an annulus-bg third
  pipeline. The registered prediction is the v10 head. A pre-registered alternative is the mean of the heads (LDNet's "mean
  listener" / Guan's learned weights).
- Why it moves the bottleneck: the trunk sees two partly independent realisations of the background error. This lowers the
  label-noise variance term (Noise2Noise) without forcing outputs onto the local-bg labels. Those labels are near noise for CD19
  (v10-local r .24-.40, where consensus lost in instrument 5). Wei et al. predict that separation beats aggregation for exactly
  these high-noise, label-starved targets. The v10 head also keeps any image-predictable v10 artefact (Northcutt risk).
- Expected gain: N067's gains on CD56/HLA-DR/grp without its CD19 loss. Single model about +.003-.008 GOAL, mostly CD56 and the b10
  direction (reasoned from the K033 doubling bound of +.015 and partial overlap). Alone it will probably not reach .70 on the
  registered score.
- Cheapest falsifier: 1 seed x 2 directions = 1 GPU-h on the composite recipe, compared paired per target with the N067 control.
  Kill if the v10 head is not better than both control and N067 on CD56 and the pooled groups.
- Build: about half a day (label tensors exist from N064/N067; add heads plus a masked loss).

**I2. Label-version bagging: ensemble members trained on different label realisations.**
- Mechanism: seed or architecture ensembles average optimisation noise. Every member fits the SAME label-noise realisation, so the
  noise-fitted part of the error is shared, which fits the member error correlation of r .94-.96 (N050) and the ~.693 asymptote
  (N055/N059). Members trained on v10, local-bg and consensus decorrelate that part. This is LDNet's "all-listeners" averaging and
  Guan's per-expert models plus averaging.
- Why it moves the bottleneck: it targets the ensemble ceiling itself, not the head.
- Declined check: this is not "more members" (declined), because only the label realisation changes. It counts as distinct only if
  the pre-check below passes.
- Cheapest falsifier: 0 GPU-h. On N067's saved predictions (CPU minutes), compute the residual correlation between the
  v10-trained and consensus-trained members, per weak target, against seed pairs (.94-.96). If it is not at or below about .90 on
  CD56/HLA-DR/grp, stop. If it passes, score a pre-registered 6-way + N067 + I1 member ensemble on v10 (no new training).
- Expected gain: +.002-.006 on the descriptive ensemble (.6929 -> about .695-.699). Build: about 2 h of scoring code.

**I3. Raw-space, negative-population background correction of TRAINING labels (dsb / ADTnorm analogue).**
- Mechanism: for each marker and FOV, find that marker's own negative peak with a per-FOV two-component mixture on the RAW,
  pre-asinh intensity (totalVI-style). No lineage or exclusivity prior is used (declined).
- Shrink the per-FOV negative-peak offset toward the slide value (random-effect shrinkage, cross-fitted). Subtract it in raw space,
  then re-apply the frozen v10 asinh cofactor. Use the result as the TRAINING target only; evaluation stays v10.
- Why it moves the bottleneck: background error is additive before asinh, so it is largest on dim cells in label space. N061's
  label-space per-FOV intercept could not represent that, and HCS/CITE-seq practice estimates such error from a reference
  population.
- Targets: the high FOV-excess ones (CD19 +.46/.43, CD56 +.29/.30, CD14 b01 +.29, grp:CD45).
- CPU pre-check (minutes, required): on the training draw, the correction must raise v10-local agreement and cut the FOV excess
  share toward the permutation null. If it does not, it is not removing background-model error; stop.
- GPU falsifier: 1 GPU-h (1 seed x 2 directions), as an arm alongside I1.
- Expected gain: similar size to I1 on CD19/CD56, possibly additive with it (I1 uses a second pipeline, I3 a reference population).
- Build: about 1 day (needs v10's pre-asinh intensities and the per-FOV mixture).

**I4. Per-FOV reliability weighting from pipeline disagreement (BIV).**
- Mechanism: weight_i,t = 1/(s_t^2 + v_FOV,t), where v_FOV,t is the within-FOV variance of z(v10) - z(local) on the training draw.
  Weights are clipped and normalised per batch, as in BIV.
- This refines the open IDEAS item "reliability-weighted loss per (batch, target)" down to FOV granularity, using a label-only
  variance estimate.
- Expected gain: small, +.001-.004. BIV's 18-21 % test-loss gains assume known variances, and ours are estimated.
- Falsifier: 1 GPU-h arm, bundled with I1/I3. Build: about 2 h.

Bundling (CLAUDE.md): run I1 + I3 + I4 as three arms of one attempt against the N067 and control runs. That is 3 GPU-h for a
1-seed screen, Helma preempt, far under the 25 GPU-h gate. Run I2's CPU pre-check on N067 outputs as soon as N067 lands. The
literature does not support a separate evaluation-ceiling estimate (Crusius/Li bounds need sigma_noise, and pipeline disagreement
only lower-bounds it, K041). Report it as a bracket, not a number.
