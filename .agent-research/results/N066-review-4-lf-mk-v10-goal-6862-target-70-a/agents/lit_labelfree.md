# N066 literature: how label-free -> marker systems reach their accuracy, and what that implies for the v10 GOAL

Agent: read-only literature agent (Claude), 2026-11-19. Scope: the question in the N066 brief.
- (1) Work that uses BSCCM itself, and any information bounds it reports.
- (2) Dense pixel-level supervision vs scalar regression for per-cell readouts.
- (3) How these systems handle label noise from fluorescence background subtraction.
- Plus the last 12 months.

Access: WebSearch and WebFetch worked. Full texts were downloaded with curl, converted locally and grepped:
- arXiv HTML;
- the ROSIE PDF (PMC mirror);
- the DeepIFC bioRxiv full text;
- the Raykar JMLR PDF;
- Europe PMC XML for Sengupta 2025;
- the Cheng 2021 PDF (NSF PAR).

Quotes are verbatim unless marked as a paraphrase.

Tags used below:
- [derived] marks my own inference.
- [BSCCM notebook] marks a project number.
- "via N038" marks a quote taken from the verified N038 literature report
  (`results/N038-review-lf-mk-common-11-batch-held-out-go/agents/lit_labelfree_subsets.md`). I did not re-fetch those.

---------------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **No published work gives per-marker BSCCM numbers or per-marker information bounds that we could use as a ceiling.**
   - **Information-Driven Design (Pinkard et al., NeurIPS 2025).** Setup: DenseNet121 (ImageNet init, one LED channel), 8 Gaussian-mixture heads, NLL loss, missing labels masked. It reports only that total image information (PixelCNN on 40x40 patches, an upper bound) tracks the protein-prediction NLL across brightfield, DPC and single-LED. There are no per-marker values in the text, and single-LED is flagged as inflated by "system-derived randomness".
   - **Nazir & Behera (ICPR 2026).** The only other BSCCM marker numbers. Input is 28x28 4-channel DPC; the split is unstated. CD3/19/56 group r .768; CD123/HLA-DR/CD14 group r .339. The model is multi-task (focal classification plus SmoothL1 + Pearson regression).
   - **2604.05211.** BSCCM-tiny only, no marker regression.
   - The project's own N064 bracket therefore remains the only marker-level ceiling for BSCCM.

2. **Dense pixel-level vs scalar: no controlled head-to-head exists for per-cell marker readouts.** I found none in blood, IFC or histology. The indirect evidence agrees on one point: dense targets help as a training signal for the encoder, not as the readout path.
   - The best published label-free dim-marker numbers come from a pixel-trained model: DeepIFC, per-marker Inception U-Net, pixel BCE, 60x IFC. CD19 r .61 (AUROC .800), CD56 r .53 (AUROC .805), CD8 r .41.
   - HEmIF-Bench (Nov 2026): generated outputs trail the generators' own encoder features by .093 macro-AUROC on single-cell classification.
   - Sengupta et al. (BOE 2025): virtual staining as a preprocessing step helps only low-capacity task networks, consistent with the data-processing inequality.
   - In-project N034: reading markers from a dense LF->fluorescence regressor or flow model was worse per cell than the scalar route on subtle markers [BSCCM notebook K027].
   - Auxiliary-target theory supports the encoder role. Supervised autoencoders "never harm" and can improve generalization. In silico labeling's multi-label pixel network "learned generalized features, enabling it to solve new problems with small training datasets".

3. **Background-subtraction label noise is handled locally and per image everywhere I looked.** Every pixel-trained system subtracts a per-image or per-crop background before training:
   - DeepIFC: per-image quantile q = 0.6;
   - Cheng 2021: per-image histogram mode;
   - MIPHEI: autofluorescence subtraction plus percentile normalisation, then GMM pseudo-labels;
   - ROSIE: patches below the 90th-percentile background are dropped from evaluation.

   None smooths the background across fields of view as BSCCM's per-slide LOWESS does. That smoothing is the source of the FOV-clustered error in the N066 diagnosis.

   None trains on several background models or models their errors. The statistical tools for that exist outside microscopy:
   - Raykar et al. 2010, learning from multiple noisy regression "annotators": train on the precision-weighted mean;
   - triple collocation: estimate each of three measurement systems' error variance without ground truth.

   Both map directly onto the diagnosis lever (b).

4. **[derived] Where lever (b) can act.** The label-model disagreement sits in draw 1:
   - HLA-DR v10-vs-local r .76 and grp:CD45 .68 in draw 1, vs .95-.98 in draw 0 [K041];
   - CD56 .67-.76 in both draws.

   Cleaner training targets therefore mostly help **b10** (train on draw 1). That direction is already .7036. They also help CD56 in both directions. The b01 deficit (grp:CD45, grp:CD16, HLA-DR on draw-1 evaluation labels) is evaluation-label error, lever (a), and no training-side method in this literature addresses it.

---------------------------------------------------------------------------------------------------------------------------------

## 1. Systems table: mechanism, training signal, scale, dim/rare-marker accuracy

| System (year) | Input / optics | Training signal | Scale | Dim/rare-marker numbers | Background / label handling |
|---|---|---|---|---|---|
| Info-Driven Design (Pinkard, NeurIPS 2025) | BSCCM, one LED pattern (BF / DPC / single LED), simulated photon noise | **scalar**: 8 GMM heads, NLL, masked missing | BSCCM, 15 (v5) or 11 (v1) splits | none in text (Fig. 2d / S20 only) | uses released labels |
| Nazir & Behera (ICPR 2026) | BSCCMNIST 28x28, 4 DPC | **scalar multi-task**: focal CE + SmoothL1 + Pearson term + aux consistency | 1,418 test cells; split unstated | CD3/19/56 .768; CD123/HLA-DR/CD14 .339; CD16 .819; CD45 .799 | z-scored targets; no noise handling |
| DeepIFC (Cytometry A 2023) | ImageStream 60x, BF + DF | **pixel**: BCE on fluorescence image, one U-Net per marker | 247k train cells, 7 samples / 6 donors; test = 2 samples | CD19 r .61 / AUROC .800; CD56 .53 / .805; CD8 .41 / .728; CD45 .90, CD14 .88, CD3 .87 | per-image quantile (q = 0.6) background normalisation; compensation matrix |
| ROSIE (bioRxiv 2024 / Nat Commun 2025) | H&E -> CODEX | **scalar per patch**: mean of centre 8x8 px, MSE, ConvNeXt-S | ~30M cells, >2000 samples | mean Pearson .285 over 50 markers (held-out studies) | eval excludes patches below the 90th-pct background |
| MIPHEI-ViT (arXiv 2505.10294) | H&E -> mIF | **pixel**: inverse-variance-weighted MSE, U-Net with ViT FM encoder | ORION CRC | cell F1: CD3e .57, CD20 .30, CD68 .36 | AF subtraction (manual lambda, b) + 99.9-pct normalisation; GMM pseudo-labels |
| HEmIF-Bench (arXiv 2611.19905, 15 Nov 2026) | benchmark of 5 frozen H&E->mIF generators | compares encoder features vs generated outputs | 8 cohorts | outputs trail encoder features by .093 macro-AUROC | n/a |
| Chlis et al. (NAR 2020) | IFC brightfield 32x32 | scalar CNN | mouse | CD34 .38, FcgR .50 (via N038) | n/a |
| Ghost cytometry iSGC (eLife 2021) | waveforms | classification | 5-part diff incl. basophils (CD123) | macro-F1 .911 / .904 intra / inter donor; basophil false negatives (via N038) | gates |

[derived] The pattern across systems:
- Lineage markers (CD45, CD14, CD16, CD3) reach r ~.8-.9 with any signal.
- Dim lymphoid markers sit at r .4-.6 even at 60x with one fluorophore per marker (DeepIFC).
- Pooled or activation markers (CD123/HLA-DR/CD14) reach .34 on BSCCM at 28x28.
- Our GOAL targets are already in or above those bands. No system closes them by architecture alone.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Q1: work on BSCCM itself

**[A] Pinkard, Kabuli, Markley, Chien, Jiao, Waller, "Information-Driven Design of Imaging Systems", NeurIPS 2025, arXiv
2405.20559.** URLs: https://arxiv.org/html/2405.20559v5 (S6.5.4) and https://arxiv.org/html/2405.20559v1.

Predictor (v5, S6.5.4):
- "The network uses a DenseNet121 [154] backbone pretrained on ImageNet, modified to accept single-channel input by averaging the first convolutional layer's RGB weights."
- "eight parallel fully-connected networks, each with two hidden layers. These networks output parameters for Gaussian mixture models corresponding to each protein marker"
- "A composite loss function summed the negative log likelihood across protein markers, with missing values masked out."
- Training budget: "Typical trainings took 1-2 days on NVIDIA TITAN Xp GPUs".

Metric and information estimate:
- "we selected the minimum information estimate across the 15 replicates (since our estimator provides an upper bound) and the best prediction performance (lowest negative log likelihood on test data)"
- v1 states the metric as "the average log likelihood across proteins as an overall performance metric".

Result:
- "Higher information estimates generally predicted better neural network classification performance, though single-LED showed inflated estimates due to optical imperfections"
- "single-LED illumination yielded less accurate protein predictions than differential phase contrast despite higher information estimates."

System information:
- v1: "Spatially coherent illumination, such as that from a single LED, encodes more information per photon because it is more sensitive to small changes in the object structure. For the same reason, it can introduce more system-derived randomness due to dust or optical imperfections".
- v5 S4.1: "single-LED illumination exhibited higher background information than differential phase contrast or brightfield illumination."

Task-specific information:
- "information estimates strongly predict reconstruction performance, they may correlate less strongly with specialized tasks that rely only on specific features of measurements".
- A task-specific decomposition I(Y;T) is proposed (S2.8), not applied to BSCCM.

What this means for us:
- No per-marker numbers or bits appear in the text; I could not read Fig. 2d / S20 as numbers.
- The information bound is total image information, not marker information. It cannot serve as a CD19/CD56 ceiling.
- [derived] The only reusable signal is the warning that the single-LED channel (our LED119 input) carries the most acquisition-specific ("system") information. That is a candidate carrier of draw-specific features.

**[B] Pinkard, Liu, Nyatigo, Fletcher, Waller, BSCCM dataset, arXiv 2402.06191.** https://arxiv.org/html/2402.06191v1

Label pipeline, the source of the diagnosed error:
- Local measurement: "the raw fluorescence was measured in the foreground and background of each cell by summing all the pixels inside or outside ... of a circle inscribed in the square crop".
- Slide-level smoothing: "there were also variations in background fluorescence between slides, likely due to antibodies in the chamber that were unbound to cells. The spatial pattern of the background for each slide was first visualized using a spatial histogram, and then a smoothed version of this was calculated using a locally weighted scatterplot smoothing (LOWESS)".
- Correction step: "the foreground of all cells was corrected for spatial variations by subtracting the smoother backg[round]".

Other caveats:
- "The CD123/CD14/HLA-DR channel was deemed too noisy be useful".
- The unmixing is "an underdetermined problem, with only six measurement channels".
- Artifact: "There was a small drop of oil on an internal lens element ... When imaging with certain low-NA single LEDs, this created a strongly visible artifact across the field of view."

**[C] Nazir & Behera, "Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning", arXiv 2605.14717 (May 2026, ICPR 2026).**
https://arxiv.org/html/2605.14717

Training signal and targets:
- Regression combines Smooth L1 with "a Pearson correlation alignment term scaled by β".
- "Protein expression values are Z-scored using training set statistics to reduce scale imbalance."

Results:
- Table 4 r: CD16 .819, CD45 .799, CD3/CD19/CD56 .768, CD123/HLA-DR/CD14 .339.
- Ablation r: regression-only .7012 vs full model .7263.
- Their explanation for the weak group: "These markers reflect functional activation states rather than stable morphological phenotypes" (via N038).

Caveats: split and batch handling are not stated; 28x28 resolution.

[derived] Their tricks are already covered in-project:
- multi-task classification (lineage head),
- Pearson loss (N061 flat).

**[D] Other BSCCM users in the last 12 months.**
- arXiv 2604.05211 (Apr 2026, dictionary learning on BSCCM-tiny): "lymphoid-vs-myeloid separation at ARI=0.575" (via N038). No marker regression.
- Searches for "BSCCM", "BSCCMNIST", "BSCCM-coherent" and "LED array leukocyte 2026" found nothing else. The dataset records show a May 2025 Dryad metadata fix and an Oct 2026 GitHub catalogue entry (AI-HHMI/mia-agentic-search PR #786). Neither has results.

---------------------------------------------------------------------------------------------------------------------------------

## 3. Q2: does dense pixel-level supervision beat scalar regression for per-cell readouts?

**Direct evidence: none.** No paper trains the same backbone on the same cells with (i) per-cell scalar targets vs (ii) fluorescence-image targets, and then scores per-cell r. The indirect evidence follows.

**[E] DeepIFC: pixel-trained, best dim-marker numbers in the label-free blood literature.** (Timonen et al., Cytometry A 2023)
https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full

Training setup:
- "a separate DeepIFC model is trained for each fluorescent channel in the input data".
- Loss: "binary cross-entropy (BCE) of the generated fluorescent images compared to the true (target) image".

Per-marker results:
- "accurately reconstruct fluorescent images of markers CD45 ( r =0.90), CD14 ( r =0.88) and CD3 ( r =0.87), thresholding the average intensity values to determine marker positivity".
- "the surface markers CD56 ( r =0.53, AUROC=0.805), CD19 ( r =0.61, AUROC=0.800) and CD8 ( r =0.41, AUROC=0.728) were less accurately reconstructed, likely due to the relatively small numbers of cells in the complete dataset exhibiting these markers".

Scale:
- "For most cell types, the saturation point in performance increase was reached at 3200 cells".
- Balanced training raised B-cell recall from 5 to 54 % (via N038).

Caveat: 60x, 0.33 um pixels, one dedicated fluorophore per marker. The paper has no scalar baseline.

**[F] ROSIE: scalar per-patch regression at very large scale.**
https://rcastoragev2.blob.core.windows.net/59344b34753bcaf0f93396986e80f4cd/nihpp-2024.11.10.622859v2.PMC11601356.pdf

Design:
- "given an input 128x128 pixel patch, it produces a prediction for the average expressions of the biomarker panel across the center 8x8 pixels".
- "instead of using difficult-to-train adversarial methods, we use a straightforward single MSE objective".
- "we find that ConvNext outperforms several ViT models despite its smaller size".

Result: "Pearson R correlation of 0.285, a Spearman R correlation of 0.352".

[derived] At ~30M cells, a scalar regression objective is competitive enough to be the chosen design.

**[G] HEmIF-Bench (arXiv 2611.19905, 15 Nov 2026), within the last 12 months.** https://arxiv.org/html/2611.19905

- Headline: "internal encoder features generally outperform generated outputs".
- Single-cell gap: "generated outputs trail internal encoder features by 0.093 mean macro-AUROC ( 0.721 to 0.627 ) in single-cell classification".
- Dimensionality control: encoder features projected by PCA to C channels "still outperforms generated outputs across all five generators".
- Scalar-patch vs pixel generators, mixed and confounded:
  - cross-organ NuCLS: "3-marker HEMIT at 0.562 and 11-marker HistoPlexer at 0.577 outperform both 23-marker GigaTIME at 0.559 and 50-marker ROSIE at 0.493";
  - on cohort-aligned Immunocto, ROSIE leads ("50-marker ROSIE reaches 0.757").

[derived] Dense training is useful for what it does to the encoder. Reading the readout off the generated image loses information.

**[H] Sengupta et al., "On the utility of virtual staining for downstream applications as it relates to task network capacity", Biomed Opt Express 16:4224 (Oct 2025).**
https://pmc.ncbi.nlm.nih.gov/articles/PMC12532334/ (text via Europe PMC XML)

- Low-capacity networks benefit: "Virtual staining was found to enhance the performance of low-capacity classifiers ... In contrast, for higher-capacity classifiers, virtual staining was observed to degrade performance."
- High-capacity networks do not: "when the task network's capacity was sufficiently high, virtual staining provided no improvement and, in some cases, even degraded performance."
- Their framing: "consistent with the data processing inequality (DPI) ... any transformation or image processing applied to the raw images cannot increase their task-relevant information content".

[derived] Virtual staining as a preprocessing step will not help an r50; this matches N034.

**[I] MIPHEI-ViT (arXiv 2505.10294): loss details that matter for dim markers.** https://arxiv.org/html/2505.10294v1

- Loss: "weighted Mean Square Error (MSE) loss ... Each marker's loss is scaled by the inverse of its standard deviation".
- Adversarial term: "the discriminator slightly reduces pixel-level metrics, and consistently cell-type classification accuracy".
- L1 for dim markers: "the use of the L1 loss, which may be less effective for markers with low intensity or high background proportion".
- Results: "F1 scores of 0.88 for Pan-CK, 0.57 for CD3e, 0.56 for SMA, 0.36 for CD68, and 0.30 for CD20".

**[J] Proper scoring for virtual staining (arXiv 2602.23305, Feb 2026).** https://arxiv.org/html/2602.23305

- On the GAN: "Pix2pixHD underestimates cell intensity".
- On the diffusion model: "cDDPM produces much more variability in samples compared to Pix2pixHD".
- Its proposal is to score per-cell feature posteriors with information gain.

[derived] Shrinkage of per-cell intensity is a known failure of image-translation readouts.

**[K] Auxiliary-target theory.**

- In silico labeling (Christiansen et al., Cell 2018), https://research.google/pubs/in-silico-labeling-predicting-fluorescent-labels-in-unlabeled-images/
  - "The network learned generalized features, enabling it to solve new problems with small training datasets."
  - "ISL predicts different labels in multiple cell types from independent laboratories."
- Supervised autoencoders (Le, Patterson, White, NeurIPS 2018), https://proceedings.neurips.cc/paper/2018/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html
  - "the supervised auto-encoder compared to the corresponding standard neural network never harms performance", and the same sentence says it can significantly improve generalization (paraphrase).

**In-project anchors [BSCCM notebook]:**
- K027 (N034/N035): an LF->fluorescence UNet. As MSE regression it scored new-draw readout r .599. As flow matching it was worse per cell on subtle markers (CD3 .25 vs .55, HLA-DR .36 vs .50).
- The real-image readout itself reached only .726 (new draw) against curated labels, and CD19 .49 in-batch. The pixel route has its own label-definition gap.
- N053: the Cytoland VSCyto2D encoder was the best ensemble diversifier (.6804 -> .6862; error correlation with the r50 family .91 vs .95-.99). It is a model pretrained for dense virtual staining.
- Codex idea 5 at N059 ("fluorescence spatial structure as privileged supervision") was proposed but never run.

**Answer to Q2 [derived].** Dense supervision should be used as an *auxiliary training signal on the shared encoder*, with the scalar v10 heads kept as the readout. The literature gives no case where integrating a predicted image beat a direct scalar head per cell. It does give three reasons to expect auxiliary value:
- consistent encoder-side value (HEmIF-Bench, ISL, supervised autoencoders, Cytoland in-project);
- far more supervision per labelled cell, which matters because K033 shows weak targets are label-starved;
- a target that can be made independent of the LOWESS background (see §4).

Contrastive LF<->fluorescence objectives: I found no label-free cell paper using them (searched twice). In-project shared-space work (SOTAlign/SigLIP, N045/N054) already found little marker information, and it is declined.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Q3: how fluorescence-background label noise is handled

| System | Background model for targets | Noise-aware training? |
|---|---|---|
| BSCCM labels | local corner background -> per-slide LOWESS -> subtracted; shading; NNMF unmixing with background row | n/a |
| DeepIFC | per-image quantile: "To normalize backgrounds in fluorescent images, each image Y is transformed with ... where Q is the quantile function and q was set to 0.6" | no |
| Cheng et al. Sci Adv 2021 | "we calculate the histogram of each fluorescence image and denote the mode value (i.e., the most frequent value) as the constant background. This background of each fluorescence image is subtracted; the negative values from the subtraction are clipped to zero." | no |
| MIPHEI | "We subtract the AF from each marker channel ... using λc and bc which were manually adjusted"; then 99.9-pct normalisation; GMM positivity pseudo-labels | no (inverse-sd loss weights) |
| ROSIE | "we exclude patches with a lower groundtruth expression value than the 90th percentile value of background noise for each biomarker" (evaluation only) | no |
| Ounkomol 2018 | ceiling C_max = sqrt(SNR/(1+SNR)) from target-image SNR (via N038) | ceiling only |

Sources:
- DeepIFC: https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full
- Cheng et al.: https://par.nsf.gov/servlets/purl/10210192
- MIPHEI: https://arxiv.org/html/2505.10294v1
- ROSIE: the PDF above.

**Statistical tools for several noisy versions of one continuous label:**

- **Raykar et al., "Learning From Crowds", JMLR 11:1297 (2010)**, https://jmlr.org/papers/volume11/raykar10a/raykar10a.pdf
  - Annotator model, §5.1: "For the jth annotator we will assume a Gaussian noise model with mean yi (the true unknown value) and inverse-variance (precision) τj".
  - The fitted target is the precision-weighted mean: "we are training a linear regression model with ŷ as the ground truth, which is a precision weighted mean of the response vectors from all the annotators."
  - Toy example: "only the model for annotator three (with highest precision) is close to the true regression model".
  - Their caveat on instance dependence: "We have assumed that τj does not depend on the instance xi ... It is known that some nodules are harder to measure than others."
- **Triple collocation (Stoffelen 1998; KNMI summary)**, https://www.knmi.nl/research/publications/triple-collocation
  - What it estimates: it can "simultaneously calculate linear calibration coefficients and measurement error variances".
  - Requirement: "three independent measurement systems are needed that must deliver simultaneous collocated measurements".
- **Noise2Noise (Lehtinen et al. 2018)**, https://arxiv.org/abs/1803.04189: "it is possible to learn to restore images by only looking at corrupted examples".
  - [derived] With zero-mean noise independent of the input, an MSE optimum on noisy targets equals the clean one. Noisy training labels then cost only variance (effective sample size), plus any noise component the network can fit from image features. Instrument 4 found draw-specific focus terms, so such a fittable component may exist within a draw.

**Answer to Q3 [derived].**
- The fields closest to us avoid smoothed cross-FOV backgrounds altogether and use local per-image backgrounds.
- Nobody models background-model error in training.
- For us, evaluation must stay v10. The available move is the crowd / triple-collocation treatment of several background models when building the *training* targets.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Mapping to the N066 diagnosis [derived]

- **Lever (b), training-label error, is supported by theory.**
  - Raykar and triple collocation say how to combine label versions optimally, rather than with the plain z-average used in instrument 5.
  - Noise2Noise says the gain is bounded by variance and shortcut reduction. Expect a small gain, not a jump.
- **Direction asymmetry.**
  - Draw-1 training labels are where v10 and local disagree (HLA-DR .76, grp:CD45 .68 [K041]). So b10 (train on draw 1) gains on grp:CD45 / HLA-DR.
  - b01 trains on draw 0, where the two agree at .95-.98. It gains only on CD56 (and CD14 / grp:CD3/19/56 where they disagree in draw 0).
  - The b01 deficit itself is evaluation-label error (instrument 2: replicate 0 vs 1, .578 vs .881). No training-side method from this literature can reach it.
- **CD19 caveat.** Local-bg CD19 is near noise (v10-local r .24-.40). An equal-weight consensus hurts it (.31/.29 < v10). A precision-weighted scheme must be allowed to put almost all weight on v10.
- **Dense auxiliary targets** with a per-crop background contain none of the LOWESS error by construction. They are a third label source that is independent of the background model. They also carry more supervision per scarce labelled cell, which K033 shows matters: 50 -> 100 % labels gives +.035 on weak targets.

---------------------------------------------------------------------------------------------------------------------------------

## 6. Ideas

All are inductive: training draw only, evaluation labels v10 frozen, no new data beyond the public release, no held-out-draw images. Bar = .6862 + 2 x .0027 = **.6916** (registered) or a pre-registered ensemble. Screen on 1 seed; one r50 GOAL run is ~0.5 GPU-h per direction. Ideas 1-3 can be bundled into one attempt of 5 arms x 2 directions ≈ **5-6 GPU-h**, well under 25.

### Idea 1: precision-weighted multi-background training targets (Raykar crowd regression with triple-collocation weights)

**Mechanism.**
- For every training cell and target, build K = 3 label versions from the released raw fluorescence aggregates, all through the identical v10 pipeline (unmixing, cofactor, curation):
  - (i) v10, LOWESS background;
  - (ii) N064 local corner background;
  - (iii) a third background that is as independent as possible, e.g. a per-crop low percentile of the border ring excluding neighbour pixels, or a robust median of corners within the FOV.
- In each training draw, per target, z-scale each version. Estimate each version's error variance by triple collocation, i.e. from the three pairwise covariances.
- Train on the precision-weighted mean (Raykar eq. 12). Optionally stratify by FOV density.
- Keep v10 for scoring.

**Why it moves the bottleneck.** It acts directly on lever (b). It also replaces instrument 5's ad hoc 50/50 z-average with weights that:
- protect CD19 (expected weight mostly on v10);
- shift weight toward local labels on CD56, and on HLA-DR / grp:CD45 in draw 1.

**Expected gain (reasoned).** Instrument 5 scored the ensemble against consensus labels; it is an upper-bound-flavoured view:
- CD56 +.03-.04;
- HLA-DR b01 +.023;
- grp targets +.005-.014.

Training-side gains are smaller:
- Noise2Noise says only variance and shortcut components are recoverable.
- K033's label-count elasticity (+.035 for 2x labels on weak targets), with an effective-sample gain of maybe 1.3-1.5x on 4-5 targets, gives +.01-.02 on those targets.

Net: **GOAL +.003-.008 single model**, mostly b10 and CD56. Could be 0.

**Cheapest falsifier.**
- (a) **0 GPU-h, CPU, about 1 h.** Run triple collocation on training-draw labels. If v10's estimated error variance on CD56 / HLA-DR / grp:CD45 is < 10 % of label variance in both draws, close lever (b) before any GPU spend.
- (b) One r50 GOAL run per direction on the weighted targets vs a same-seed control: **~2 GPU-h** including the control.
- Kill if Δ GOAL < +.0054 or the per-target gains do not land on CD56 / HLA-DR / grp.

**Build cost.** 0.5-1 day. The N064 pixel-aggregate replay plus `n066_altlabels.py` already produce (ii). (iii) is a new aggregate on the same packed data. `train_lf.py` takes a target npz.

### Idea 2: dense "privileged" auxiliary decoder that predicts each cell's fluorescence crop with a per-crop background

**Mechanism.**
- Add a light decoder (~3-5 M parameters) on the shared r50 encoder. It predicts the 6-channel fluorescence crop, downsampled to 64x64 and asinh-scaled.
- Subtract a per-crop background before training (DeepIFC-style quantile or border median, as in N034's image targets).
- Condition the decoder on the slide's antibody panel, as N034 did. Use a masked MSE weighted by inverse channel sd (MIPHEI); no GAN term (MIPHEI) and no L1 (it hurts dim markers).
- Train-time only. Inference and the v10 scalar heads are unchanged.
- Every cell in the training draw, on every slide type including unstained (autofluorescence), contributes pixel targets.

**Why it moves the bottleneck.**
- It is a training signal free of LOWESS error by construction.
- It shows the spatial signature of bound antibody (ring/caps) vs diffuse unbound-antibody background. The BSCCM paper attributes slide background to "antibodies in the chamber that were unbound to cells".
- It gives roughly 10^4 target values per labelled cell instead of 1-4 scalars, against the label starvation in K033.
- It is not a head or loss on the fixed labels (declined: N060-N065). It is a new source of supervision, the untested N059 Codex idea 5.

**Expected gain.**
- Literature supports encoder-side benefit (HEmIF-Bench, ISL, supervised autoencoders, Cytoland in-project) but gives no per-cell head-to-head.
- **+.002-.008 single model.** Possibly larger value as a decorrelated ensemble member (Cytoland-like, +.002-.004 at ensemble level).
- Risk: capacity competition on lineage markers. Supervised-autoencoder evidence says the auxiliary rarely hurts.

**Cheapest falsifier.**
- One seed, both directions, auxiliary weight fixed a priori: **~1.5 GPU-h** (decoder overhead about +40 %).
- Kill if Δ single-model < +.0054 vs a same-seed control, and its errors correlate at ≥ .94 with the N053 members (N050 ConvNeXt-T bar .94-.96; Cytoland .91).
- Cheaper pre-check, **≤ .1 GPU-h**: on frozen r50 features, fit a ridge probe to per-crop ring-vs-core contrast for stained CD56 / HLA-DR cells, blocked by FOV. If the probe finds no out-of-FOV signal beyond the scalar prediction, skip.

**Build cost.** 1.5-2 days. The crop packing exists from N034/N064, but N064 deleted its pixel_chunks, so it must be re-packed as one file because of the /hnvme inode cap. Then a decoder plus a data-loader change in `train_lf.py`.

### Idea 3: per-cell reliability weights from background-model disagreement (instance-dependent annotator precision)

**Mechanism.**
- Raykar's caveat ("some nodules are harder to measure than others") applies per cell.
- Weight each training cell's loss on weak / pooled targets by w = 1 / (s² + |bg_local - bg_lowess|² · k_t). The slope k_t is fit on training-draw out-of-fold residuals.
- Cells whose local corner background departs most from the LOWESS surface are where v10 is most likely wrong.
- Evaluation is unchanged.

**Why it moves the bottleneck.**
- Instrument 3 shows FOV-clustered residuals: CD19 +.46/+.43, CD56 +.29/+.30.
- Training on those cells teaches draw-specific FOV signatures, which is the shortcut part of lever (b).
- Down-weighting cuts both the variance and the shortcut component.

**Expected gain.** +.002-.005, overlapping Idea 1. Run it as an arm with Idea 1: weighted consensus vs consensus.

**Cheapest falsifier.**
- **0 GPU-h first.** On training-draw out-of-fold cells, check whether |bg_local - bg_lowess| predicts |v10 residual| of the current ensemble with r ≥ .1 for CD19 / CD56 / HLA-DR. If not, kill.
- Otherwise run as one arm, **~1 GPU-h** for both directions.

**Build cost.** 0.5 day, reusing N064 aggregates.

### Idea 4 (optional, off the diagnosed bottleneck): system-information hygiene for LED119

**Mechanism.** Drop LED119 (the single coherent LED), or flat-field it with its own background.
- Pinkard et al.: single-LED images carry the most "system-derived randomness" / "background information".
- The BSCCM paper notes an oil-drop artifact under low-NA single LEDs.

**Why.** It might carry draw-specific acquisition features into b01.

**Expected.** 0 to +.003.

**Falsifier.** One r50 run per direction without LED119, **~1 GPU-h**.

**Build.** Trivial. Check first: N066's `n066_diag_selftest.py` / `n066_d4_*` already exercise LED119, so this may already be in flight.

**Not proposed, already covered in-project:**
- balanced / tail sampling (DeepIFC-style; N022 tail weighting);
- Pearson loss (Nazir; N061);
- reading markers from virtual-stained images (Sengupta, HEmIF-Bench; N034);
- contrastive shared spaces (N045/N054, declined).

---------------------------------------------------------------------------------------------------------------------------------

## 7. Gaps and unverified points

- The Pinkard Fig. 2d / S20 values are figure-only. I could not extract numbers, so no BSCCM information bound per marker exists in text.
- No controlled dense-vs-scalar comparison was found in any domain. Idea 2's prior rests on indirect evidence.
- DeepIFC does not state its r formula in the text. It is most likely the correlation of per-image mean intensities ("thresholding the average intensity values").
- Nazir & Behera's split may be a random cell split, so their numbers may be in-distribution.
- I did not open the full Cytoland paper. A WebFetch summary said VSCyto2D uses MS-SSIM + L1 and FCMAE pretraining. Not used for any claim.
- Triple collocation assumes independent errors. v10 and local labels share the corner-pixel measurement, so their errors are partly correlated. The third background should use different pixels, and the estimate is then a bound, not an identity.

## Sources

- https://arxiv.org/html/2405.20559v5 ; https://arxiv.org/html/2405.20559v1 ; https://bair.berkeley.edu/blog/2026/01/10/information-driven-imaging/
- https://arxiv.org/html/2402.06191v1
- https://arxiv.org/html/2605.14717
- https://arxiv.org/pdf/2604.05211 (via N038)
- https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full
- https://rcastoragev2.blob.core.windows.net/59344b34753bcaf0f93396986e80f4cd/nihpp-2024.11.10.622859v2.PMC11601356.pdf (ROSIE)
- https://arxiv.org/html/2505.10294v1 (MIPHEI-ViT)
- https://arxiv.org/html/2611.19905 (HEmIF-Bench)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12532334/ (Sengupta et al. 2025)
- https://arxiv.org/html/2602.23305 (proper scoring rule for virtual staining)
- https://research.google/pubs/in-silico-labeling-predicting-fluorescent-labels-in-unlabeled-images/
- https://proceedings.neurips.cc/paper/2018/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html
- https://par.nsf.gov/servlets/purl/10210192 (Cheng et al. Sci Adv 2021)
- https://jmlr.org/papers/volume11/raykar10a/raykar10a.pdf
- https://www.knmi.nl/research/publications/triple-collocation
- https://arxiv.org/abs/1803.04189 (Noise2Noise)
- https://github.com/AI-HHMI/mia-agentic-search/pull/786 ; https://datadryad.org/dataset/doi:10.5061/dryad.sxksn038s
