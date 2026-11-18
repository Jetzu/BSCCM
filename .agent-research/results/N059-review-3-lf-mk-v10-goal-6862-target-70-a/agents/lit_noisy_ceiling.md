# N059 literature agent: noise ceilings and raising r against NOISY held-out labels

Status: literature AVAILABLE (WebSearch + WebFetch worked; PMC pages were blocked by a captcha, so bioRxiv, arXiv and publisher mirrors
were used instead). Quotes marked **[snippet]** come from search-result text, not a fetched page. Treat them as weaker evidence.
Read-only pass, 2026-11-18. Context read: GOAL.md, attempts.md, declined.md, diagnosis.md, n059_diag.csv, scripts/train_lf.py (MSE
+ per-row weight + 3/10-class heads already present; tail-weight option exists; 8-view TTA used downstream), N048 item (~1.3 H100-h
per retrain).

## 0. One-line theory that frames everything below
If y = s(x) + e with e independent of the image x, then corr(f(x), y) = corr(f, s) * sqrt(Var s / Var y). Two consequences:
1. No predictor can beat sqrt(reliability). The Pearson-optimal predictor is E[y|x], which is the MSE-optimal predictor up to an affine
   map. "the noise ceiling for the Pearson correlation coefficient is the square root of the signal-to-total-variance ratio"
   (https://himjl.github.io/blog/2026/04/20/spearman-brown/, Apr 2026, eta* = sqrt(Var(E[Y|X]) / Var(Y))).
2. A method can raise r on NOISY labels in only three ways: (i) estimate E[y|x] better (loss shape, weighting, ensembling);
   (ii) weight the targets the way the metric does (mean of 11 r values, not the sum of MSEs); (iii) predict the part of the "noise"
   that actually depends on the image (artefacts visible in the input). A draw-specific readout that does not depend on x (the N059
   diagnosis for HLA-DR and the pooled groups) is out of reach of all three by construction.
   Interpretation trap: a model can correlate with one replicate MORE than replicate-to-replicate r, because the model's ceiling is
   sqrt(rel) while replicate-replicate r = rel. Basenji: "both the mean and median Basenji prediction accuracy exceed the replicate
   accuracy" **[snippet]** (https://genome.cshlp.org/content/28/5/739.abstract).

## (a) How neighbouring fields estimate the achievable ceiling

| Field / source | Method | Verbatim evidence |
|---|---|---|
| Label-free fluorescence prediction, Ounkomol 2018 (https://www.biorxiv.org/content/10.1101/289504v3.full) | C_max from a noise floor measured on UNLABELED cells under identical settings; T = S + N | "a corresponding estimate of noise was developed based upon image stacks taken of unlabeled, wild-type hIPSC cells"; "for which microscope settings were identical to those used during labeled acquisitions."; "the variance of the unlabeled image stacks is a lower bound on the variance of uncorrelated, random fluctuations"; "The best possible model would output exactly S, and thus the highest correlation between a model's output and the target is the correlation between T and S"; "Individual structure model performance is well-bounded by this limit" |
| Neural encoding, Schoppe 2016 (https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2016.00010/full) | CC_norm = CC / CC_max | "CC_max is the maximum correlation coefficient between the recorded firing rate y and the best prediction ŷ"; "For CC_norm the normalization factor is the inverse of CC_max." |
| Neural encoding, Pospisil & Bair 2021 (https://www.biorxiv.org/content/10.1101/2020.10.30.361253.full) | Unbiased r²_ER; with a single repeat (n = 1) the noise variance is ASSUMED (our case: one fluorescence readout per cell) | "Yet it is biased by trial-to-trial variability"; "For example if stimuli are shown only once (n = 1), trial-to-trial variability could be assumed"; "we find ~ 40% or less of some neural recordings do not pass even a liberal SNR criterion" |
| fMRI, Lage-Castellanos 2019 (https://www.biorxiv.org/content/10.1101/377101v1.full) | Split-half plus Spearman-Brown vs analytic/Monte-Carlo ceilings; known biases | "the observed correlations are adjusted using the Spearman-Brown correction."; "the split half estimator provides a pessimistic estimate of the noise ceiling due to the small amount of data"; "the parametric nature of the Monte Carlo estimator results in overly optimistic estimates" |
| Split-half estimator, 2026 note (https://himjl.github.io/blog/2026/04/20/spearman-brown/) | eta* = mean over B splits of sqrt(2 r_half / (1 + r_half)) | "The key assumption in this formula is that the test items are _parallel_."; "it can behave badly if split-half correlations are near zero or are negative." |
| fMRI datasets, NSD formula (CNeuroMod-THINGS, arXiv 2507.09024) | NC from the single-trial SNR | "We then calculate the noise ceiling as: (100 × ncsnr²)/(ncsnr² + 1/n), where n=3 is the number of repetitions per image." **[snippet]**; the page fetch returned 404 |
| Flow cytometry, Nguyen 2013 (https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1002/cyto.a.22251&resultType=core&format=json) | Spillover-spreading matrix: unmixing noise in channel B grows with signal in channel A | "We propose that the SSM is a critical component of QA/QC in evaluation of flow cytometer performance."; "the degree of SS had been predicted theoretically to increase with measurement error" **[snippet]**. Novo 2013: LS unmixing "may lead to unmixing artifacts, such as population distortion (spreading) and the presence of negative values"; a Poisson-noise formulation "leads to superior unmixing results, particularly for dim fluorescent populations" **[snippet]** |
| BSCCM itself (https://arxiv.org/html/2402.06191v1) | How the label noise arises: background from the crop corners; underdetermined 6-channel unmixing; few slide replicates | "First the raw fluorescence was measured in the foreground and background of each cell by summing all the pixels inside or outside (Fig. S3a) of a circle inscribed in the square crop."; "This size was chosen so that the pixels outside of it captured the local background intensity of that area of the slide/field of view, while the pixels inside contained the background plus fluorescence from the cell."; "The CD123/CD14/HLA-DR channel was deemed too noisy be useful."; "Cells also showed varying amounts of autofluorescence."; "with only six measurement channels"; "The vast majority of cells/slides did not have a second replicate." |
| Histology to expression, benchmark (https://www.biorxiv.org/content/10.1101/2023.12.12.571251v1.full) | No replicate ceiling is computed; noisy and low-expressed genes are filtered instead | "expression at the limits of detection, should not have correlation between predicted and measured intensity"; "some methods perform better when noisy genes are removed from the analyses"; "the most complex deep learning architectures involving more components were not superior in either SGE prediction" |
| Remote sensing, Burke et al. 2021 (https://arxiv.org/abs/2010.06988); Lobell et al. 2020 (https://econpapers.repec.org/RePEc:wly:ajagec:v:102:y:2020:i:1:p:202-219) | Score against a gold-standard subset; noisy labels understate skill | "highlighting how this noise often leads to incorrect assessment of models' predictive performance"; satellite yields "explain as much or more variation in yields based on (gold-standard) full-plot crop cuts" **[snippet]** |

Takeaway for BSCCM. The field standard is a two-sided bracket. The noise floor from unlabeled cells (Ounkomol) gives an OPTIMISTIC
ceiling, because noise that grows with intensity (spillover spreading) is under-counted. Split-half on few data is PESSIMISTIC
(Lage-Castellanos). None of the histology-to-expression papers checked here computes a replicate ceiling. Report r/r_max (CC_norm)
alongside raw r. That number is diagnostic, not progress.

## (b) What raised held-out r against NOISY labels? Controlled evidence only

| Method | Controlled evidence on noisy held-out labels? | Verbatim evidence |
|---|---|---|
| Per-target Pearson (or CCC) loss added to MSE | YES, weak (one dataset, overlapping sds); a second domain agrees; one counter-example | Spots-to-Pixels, Visium HD, Table 4 (https://arxiv.org/html/2503.01347): MSE only PCC@M "0.293±0.03", PCC only "0.319±0.08", MSE+PCC "0.325±0.02"; "We set λ to 0.5."; "The joint objective achieves the best overall balance, simultaneously reducing errors and improving correlation". Speech emotion (https://arxiv.org/abs/2003.10724): CCC loss gave "better performance than an error-based loss function with mean squared error (MSE) loss and mean absolute error (MAE), in terms of the averaged CCC score."; "The results are consistent with two input feature sets and two datasets." Counter-example, side-channel (https://eprint.iacr.org/2023/1681.pdf): "z-MSE achieves excellent performance and is the best-performing loss function regardless of the setting in most cases" **[snippet]** |
| Distributional soft-binned head (HL-Gauss) with an expectation readout | YES for RMSE on real (inherently noisy) test labels; YES under injected target noise in RL; LegNet won DREAM with it but no isolated ablation was found | Imani & White 2018 (https://ar5iv.labs.arxiv.org/html/1806.04613): CT test RMSE ℓ2 "29.512 (±3.622)" vs HL-Gaussian "19.980 (±2.169)"; Bike "4601.21 (±80.54)" vs "4182.06 (±57.35)"; Song Year "869.569 (±1.923)" vs "862.656 (±1.683)"; "this loss is easier to optimize, with better behaved gradients"; ℓ2+Noise / ℓ2+Softmax "do not achieve the same gains as HL-Gaussian". Farebrother 2024 (https://arxiv.org/html/2403.03950v1): "Classification is less prone to overfitting to noisy targets than regression."; "the performance of HL-Gauss degrades more gracefully than MSE as the noise scale increases." LegNet (https://www.biorxiv.org/content/10.1101/2022.12.22.521582v1.full): "LegNet was trained to predict not the single expression value but a vector of expression bin probabilities." |
| Tail / imbalance reweighting (LDS, FDS, inverse weighting), rank regularisers (RankSim, ConR) | NO robust gain in overall Pearson on noisy labels (STS-B-DIR, averaged human ratings) | Original DIR paper (https://ar5iv.labs.arxiv.org/html/2102.09554), All Pearson: Vanilla "74.2", Inv "72.8", Inv+LDS+FDS "76.0". Independent reproduction (https://arxiv.org/html/2509.23240), All Pearson: VANILLA "0.766", LDS "0.762", FDS "0.751", FDS+LDS "0.747", RankSim "0.756", CONR "0.735". In this project tail05 also helped in-distribution only (KNOWLEDGE, N022/N041) |
| Heteroscedastic Gaussian-NLL heads | NO: at best equal mean accuracy, sometimes worse | Seitzer 2022 (https://arxiv.org/abs/2203.09168): "very poor but stable parameter estimates"; "we identify the culprit to be the log-likelihood loss". Stirn 2023 (https://arxiv.org/abs/2212.09184): "mean estimates that are provably as accurate as those from its homoscedastic counterpart" |
| Inverse-variance sample weighting (BIV) | Gains measured on CLEAN test labels with SYNTHETIC noise only | Mai et al. 2021 (https://ar5iv.labs.arxiv.org/html/2107.04497): "measured the performance of the network using mean square error on ground-truth test data."; UTKF L2 "0.56±.05" vs BIV "0.46±.03"; Bike "0.122±.010" vs "0.096±.006" |
| Self-distillation | Raises single models toward the ensemble, not the ensemble itself (and it is close to noisy student, which is declined) | Sensorium 2022 entry (https://github.com/lRomul/sensorium): "In the second stage, the ensemble of models achieves nearly the same single-trial correlation"; "each fold model performs better by an average score of 0.007 than the corresponding model from the first stage." |
| Spatial context: local neighbours | YES in spatial transcriptomics (noisy spot labels), where neighbour bleed is a documented label artefact; slide-level context alone does not help | MIST (https://arxiv.org/html/2610.18769, Oct 2026): "Local context alone improves over the backbone (+0.016, p=0.02)"; "slide context does not help alone (−0.010, p=0.59) but adds +0.007 on top of local context"; "replacing kNN with random neighbors of the same size costs 0.031 PCC (p=8.4×10^-5, n=21)". SpotClean (https://www.nature.com/articles/s41467-022-30587-y.pdf): "messenger RNA bleed from nearby spots causing substantial contamination of UMI counts" **[snippet]** |
| Random effects for clustered data (mixed-model layer) | Tabular data only; no image or noisy-label evidence found | LMMNN, JMLR 2023 (https://www.jmlr.org/papers/v24/22-0501.html): "mixed models are able to avoid overfitted parameter estimates" **[snippet]** |
| Mixture of experts per population, TTA, foundation features | MoE: no controlled noisy-label evidence found in this pass. TTA (8-view) and foundation encoders (N053) are already in the project | none found |

## Ideas (none of these is in declined.md; GOAL labels stay v10, nothing is re-based)

### Idea 1: noise-ceiling bracket on the v10 GOAL cells (decides whether .70 is reachable). 0 GPU-h
- **Mechanism.** Compute three per-target estimates on the same cells and the same v10 transform:
  (i) Ounkomol-style floor: the variance of the v10-transformed readout on the unstained cells of the same draw, against the
  variance on stained GOAL cells, gives r_max = sqrt(1 - var_unstained / var_stained). This is optimistic, because noise grows with
  intensity (spillover spreading).
  (ii) Pixel split-half: recompute each cell's readout from checkerboard halves of the foreground pixels plus DISJOINT halves of the
  background pixels (alternate corners), and send both halves through the same unmixing, cofactor and detrend. Correlate the halves,
  apply Spearman-Brown 2r/(1+r) and take sqrt. This captures shot noise and background-estimation noise at the actual intensity.
  (iii) FOV component: a one-way random-effects model on the best-ensemble residuals within each held-out draw,
  tau² = Var(FOV-mean residual) - mean(sigma²_within / n_f), multiplied into (ii).
  Report r_max per target, the GOAL ceiling C = mean of r_max, and CC_norm = r / r_max.
- **Why it hits the diagnosed bottleneck.** The diagnosis calls CD19 and CD56 "label-limited" without a measurement, and its
  in-distribution reference is v5-trained (caveat). This bracket measures the claim directly.
- **Pre-registered reads.**
  (1) If optimistic C <= .72, then .70 needs about 40% or more of all remaining explainable headroom. Report .70 as near-ceiling to
  the user and stop predictor-side GOAL work.
  (2) If r_max for CD19 and CD56 is >= .75, "label-limited" is refuted and those targets (r .33-.56) become the main lever.
  (3) If HLA-DR and grp:CD45 have high r_max while cross-draw CC_norm stays low, that confirms the draw-readout gap.
- **Expected gain.** 0 on the metric; this is a measurement. Its value is in settling the stall.
- **Cheapest falsifier and cost.** (i) and (iii) take CPU minutes from existing labels and predictions. (ii) is about half a day of
  build, reusing the N039 v8 raw per-cell fluorescence extraction, then under 1 h on preempt_cpu (48 cores). 0 GPU-h.

### Idea 2: loss reshaping, an HL-Gauss soft-binned head plus a per-target batch-Pearson term. About 4 H100-h
- **Mechanism.** Three arms in one bundled attempt (1 seed, both directions, the r50 SSL base recipe, training on cofactor-150
  labels as in N048):
  B1: HL-Gauss head with expectation readout (bins over each target's training range; sigma set on a training-draw FOV split).
  B2: MSE + 0.5 * (1 - r_j), with r_j computed per target over the minibatch (lambda as in Spots-to-Pixels and MIST).
  B3: B1 + B2.
  The expectation readout still targets E[y|x], the Pearson-optimal predictor, while HL's bounded gradients fit less of the
  heavy-tailed label noise. The Pearson term weights targets the way the metric does: for calibrated standardised predictions,
  dr_j/dMSE_j = -1/(2 r_j), so CD19 (.34), CD56 and HLA-DR get about 1/r more weight than under summed MSE. It is also invariant to
  per-batch affine label shifts.
- **Why it hits the bottleneck.** The score rests on the bright tails (Pearson .686 vs Spearman .531), and the low-r targets are
  exactly where the weighting changes.
- **Expected gain.** Imani & White show large gains only where ℓ2 training is unstable (CT, -32% RMSE) and small ones elsewhere (Song
  Year, -0.8%). Spots-to-Pixels gained +.032 PCC, but in a sparse-gene regime with overlapping sds. My estimate: +.002 to +.008 per
  single model and +.001 to +.004 for the ensemble, unless the new loss decorrelates errors (the N050 architecture swap left error
  correlations at .94-.96). It will not close .014 alone. Run Idea 1 first: if CD19 and CD56 sit at their ceiling, the B2 weighting
  buys nothing.
- **Cheapest falsifier.** 3 arms x ~1.3 H100-h, about 4 H100-h, preempt, 1 H100 each. Kill if no arm beats the seed-matched base
  single model (.6730) by >= .0054 (2x noise) AND no arm adds >= .003 to the 3-member ensemble.
- **Build cost.** About 60 lines in train_lf.py (K x B logits, Gaussian-CDF target encoding, PCC term): half a day plus red-team.

### Idea 3: correct the label-geometry neighbour bias, an image-visible per-cell label error. 0 GPU-h falsifier
- **Mechanism.** BSCCM takes background from the crop pixels outside the inscribed circle. A bright neighbour reaching into the
  corners inflates the background and depresses the label. That error depends on the image, which is route (iii) in section 0:
  neighbour position comes from slide coordinates, and the neighbour's marker level comes from the model's own prediction for that
  neighbour. In spatial transcriptomics the analogous bleed is documented (SpotClean), and kNN context is worth +.016 PCC (MIST).
  Here the zoom2 member sees only the centre 64 px, and translation augmentation blurs the corner geometry for the base model.
- **Cheapest falsifier (CPU).** z_j = sum over neighbours of overlap(neighbour disc, background region) x yhat_neighbour,j, using
  ensemble inference only. Take the partial correlation of the residual (y - yhat) with z_j per target and draw. Kill if |partial r|
  < .05 for HLA-DR, grp:CD45, grp:CD16, CD19 and CD56. If it passes, fit beta_j on training-draw OOF predictions only and apply
  yhat' = yhat - beta_j z_j to the held-out draw. That stays inductive: held-out images are used at inference only.
- **Expected gain.** If the residual partial r is rho, the gain is about sqrt(r² + rho²(1 - r²)) - r at r = .686: +.001 at
  rho = .05, +.004 at rho = .1, and +.015 at rho = .2 (the rho needed, averaged over targets, to reach .70 on its own). The likely
  range is 0 to +.004, concentrated in crowded FOVs. This is a different mechanism from N037 and K029, which concern FOV-level
  offsets.
- **Build cost.** Half a day: per-cell slide coordinates already exist (K007), plus the crop geometry. Debris that is not segmented
  stays invisible to this correction.

### Idea 4 (lowest priority): a random-FOV-intercept (GLS) training loss. 1.3 H100-h as an optional arm B4
- **Mechanism.** FOV offsets are not image-visible (K029, R² <= .03), so the FOV-mean part of each training residual is label noise.
  Down-weight it by sigma²_w / (sigma²_w + n_f tau²), with tau² from Idea 1(iii) on the training draw only.
- **Expected gain.** <= +.004. Only estimator efficiency improves, because the test labels keep their FOV noise. N040, N046 and
  N049 also suggest that training-label noise is not what binds.
- **Cost.** A FOV-grouped sampler and loss, half a day.

### Do not pursue (the evidence says no, or it is already covered)
Tail or imbalance reweighting, RankSim and ConR (STS-B reproduction: none beats vanilla on overall Pearson). Gaussian-NLL
heteroscedastic heads (at best equal). BIV (evidence only on clean test labels). Self-distillation (raises single models, not the
ensemble; close to the declined noisy student). More ensemble members (N055).
