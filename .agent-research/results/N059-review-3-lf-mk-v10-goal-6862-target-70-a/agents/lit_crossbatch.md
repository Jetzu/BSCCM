# N059 lit agent: cross-batch generalisation of label-free -> marker predictors, and batch-specific READOUT correction

Question: how do published label-free -> immunophenotype / fluorescence systems handle a NEW donor / draw / batch, what do they reach on
dim markers (CD19, CD56, HLA-DR, CD123), and which methods close a batch-specific readout difference of the fluorescence LABELS?
Constraints honoured: no data beyond the public BSCCM release; inductive (no held-out-draw labels or images in training); nothing from
declined.md. Web tools worked (WebSearch, WebFetch, curl of arXiv/Europe PMC/bioRxiv/GitHub). Quotes below were checked by grep against
the downloaded page text unless marked [fetch-summary].

## 1. Evidence (claim, source, verbatim quote)

### 1a. Label-free -> marker systems: how they split and what dim markers reach
| # | Claim | Source | Verbatim quote |
|---|---|---|---|
| L1 | DeepIFC (IFC brightfield+darkfield -> fluorescence images) holds out whole donors | https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full | "The same two donors as in the complete dataset experiment were assigned to the test set, and other four donors to the training set." |
| L2 | DeepIFC dim markers on held-out donors: CD56 .53, CD19 .61, CD8 .41; blamed on few positive cells | same | "fluorescent images for the surface markers CD56 ( r =0.53, AUROC=0.805), CD19 ( r =0.61, AUROC=0.800) and CD8 ( r =0.41, AUROC=0.728) were less accurately reconstructed, likely due to the relatively small numbers of cells in the complete dataset exhibiting these markers" |
| L3 | DeepIFC bright markers | same | "accurately reconstruct fluorescent images of markers CD45 ( r =0.90), CD14 ( r =0.88) and CD3 ( r =0.87)" |
| L4 | DeepIFC labels come from ONE controlled readout: single-colour compensation and isotype/unstained autofluorescence controls in the same run | same | "Single color controls were used for compensation. Isotype controls and unlabelled cells were used to determine the auto fluorescence and non-specific signal." |
| L5 | DeepIFC uses PBMC (Ficoll), not whole-blood WBC as in BSCCM, so CD19+/CD56+ prevalence and r are not directly comparable | same; BSCCM https://arxiv.org/html/2402.06191v1 | DeepIFC: "Human peripheral blood mononuclear cells (PBMC) were extracted from buffy coats ... using Ficoll-Pague™ Plus"; BSCCM: "12 mL of blood were drawn by venipuncture and added to 50mL tubes containing red blood cell lysis buffer" |
| L6 | Only other BSCCM marker-regression paper (2026, ICPR): BSCCMNIST 28x28, one test split, no cross-batch test; pooled CD123/HLA-DR/CD14 r .339, CD3/CD19/CD56 .768 | https://arxiv.org/abs/2605.14717 (HTML v1) | "Table 4 : Per-protein regression performance. Protein RMSE MAE Pearson r r CCC CD123/HLA-DR/CD14 0.945 0.507 0.339 0.137 CD3/CD19/CD56 0.646 0.415 0.768 0.713 CD45 0.670 0.430 0.799 0.723 CD16 0.595 0.463 0.819 0.781"; "Table 1 : BSCCM test split (1,418 cells)." |
| L7 | Same paper concedes the activation-marker group is not recovered | same | "the weak performance on activation markers (CD123/HLA-DR/CD14, r r =0.339) establishes that label-free morphology cannot replace molecular assays in all marker types" |
| L8 | Stain-free WBC classification (8 classes incl. B and CD56+ NKT) used cell-level stratified CV, not donor-level | https://www.biorxiv.org/content/10.1101/680975v1.full (Lippeveld et al., Cytometry A 2020) | "four machine learning classifiers were evaluated on stain-free imagery using stratified 5-fold cross-validation"; "both approaches did not always succeed in making reliable cell subtype classifications" |
| L9 | QPI leukocyte subtyping (B vs T, CD4 vs CD8) claims donor cross-validation (abstract only, no numbers) | https://api.crossref.org/works/10.1002/AISY.202000277 (AIRFIHA, Adv Intell Syst 2021) | "The performance of AIRFIHA in a randomly selected test set is validated and is cross‐validated across all blood donors." |
| L10 | In-silico labeling: a new condition is handled by fine-tuning on new data (not zero-shot) | https://www.research.google/blog/seeing-more-with-in-silico-labeling-of-microscopy-images/ [fetch-summary] | "may be learned from a _single image_" |

### 1b. The BSCCM readout itself (why draw-specific label definitions exist)
| # | Claim | Source | Verbatim quote |
|---|---|---|---|
| B1 | The two batches are distinct isolations with staining variation, not just biology | https://arxiv.org/html/2402.06191v1 | "The batch index captures the two biological replicates (i.e. distinct cell isolations) on two different dates. There is presumably some degree of biological variation between these two isolations, in addition to variation in antibody staining/processing/etc." |
| B2 | Documented dose error in batch-1 pooled slides | same | "For the cells in batch 1 that were stained with all antibodies, the amount CD19 antibody used was 35 percent of the amount used in other CD19 stains" |
| B3 | Pooled groups are spectral merges, an underdetermined unmixing | same | "The spectra for CD14, HLA-DR, and CD123 were combined into a single spectrum"; "CD3, CD19 and CD56 were also combined into a single channel, since the antibodies that targeted them all used the same fluorophore"; "this is an underdetermined problem, with only six measurement channels" |
| B4 | Documented crosstalk into the pooled lymphoid spectrum, attributed to unmodelled autofluorescence | same | "CD123, CD45 and CD16 all gave rise to signal in the CD3/CD19/CD56 spectra, which may be due in part due to the face that this model didn't explicitly account for autofluorescence." |
| B5 | Single-antibody labels come from a regularised 2-spectrum (antibody + autofluorescence) fit whose regulariser trades the two | same | "Over-regularizing separates the two, but collapses all autofluorescence values to 0." |
| B6 | Authors themselves dropped the HLA-DR group for cell typing | same | "The CD123/CD14/HLA-DR channel was deemed too noisy be useful." |

### 1c. Methods that correct a batch-specific readout (cytometry)
| # | Claim | Source | Verbatim quote |
|---|---|---|---|
| C1 | CytoNorm learns per-batch transforms from a control sample run in every batch | https://pmc.ncbi.nlm.nih.gov/articles/PMC7078957 | "This work proposes CytoNorm, a normalization algorithm to ensure internal consistency between clinical samples based on shared controls across various study batches." |
| C2 | Technical variation is cell-type specific -> per-cluster quantile splines | same | "here we demonstrate that technical sources of variation can impact cell types differently"; "calculates subset‐specific transformations on the control samples by computing their quantile distributions and aligning them with splines" |
| C3 | CytoNorm without controls uses the new batch's own aggregate (i.e. its marker distribution) | https://raw.githubusercontent.com/saeyslab/CytoNorm/master/vignettes/CytoNorm_without_controls.Rmd | "The key is to treat an aggregate of the batch as a proxy for a control sample of that batch." |
| C4 | cyCombine avoids controls but needs the new batch's marker data (ComBat per SOM cluster) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8971492 | "We developed cyCombine to be independent of technical replicates across batches"; "minimize the batch effects for groups of similar cells, one group at a time. The grouping of similar cells is done using a self-organizing map (SOM)" |
| C5 | gaussNorm aligns per-channel density peaks of each new sample (needs that sample's fluorescence) | https://rdrr.io/bioc/flowStats/man/gaussNorm.html | "normalizes a set of flow cytometry data samples by identifying and aligning the high density regions (landmarks or peaks) for each channel" |
| C6 | CytoVI: batch-conditional generative model; batch = observed covariate; needs panel overlap | https://www.biorxiv.org/content/10.1101/2025.09.07.674699v1.full ; https://docs.scvi-tools.org/en/stable/user_guide/models/cytovi.html | "categorical nuisance factors, representing experimental batches, utilized technologies or antibody panels applied. For simplicity we will describe the case of a one-hot encoded batch identifier"; limitation: "Requires at least partial feature overlap across datasets for effective integration." |
| C7 | Batch effects in cytometry come from isolation and staining conditions | CytoVI bioRxiv (above) | "batch effects attributed to differences in sample isolation protocols or antibody staining conditions" |
| C8 | Per-cell autofluorescence-aware unmixing (AutoSpectral 2025) | https://www.biorxiv.org/content/10.1101/2025.10.27.684855v1.full | "uses the per-cell residual signal following application of the unmixing matrix to calculate the optimal autofluorescence subtraction, being the profile which minimises residual"; "heterogeneity of cellular autofluorescence, which can affect both the unmixing matrix and misalign signals" |

### 1d. Label-noise-robust / batch-conditional heads
| # | Claim | Source | Verbatim quote |
|---|---|---|---|
| N1 | Crowd layer: per-annotator readout layers learned end-to-end; works for regression | https://arxiv.org/abs/1709.01779 | "a novel general-purpose crowd layer is proposed, which allows us to train deep neural networks end-to-end, directly from the noisy labels of multiple annotators, using only backpropagation. We empirically show that the proposed approach is able to internally capture the reliability and biases of different annotators and achieve new state-of-the-art results for various crowdsourced datasets across different settings, namely classification, regression and sequence labeling." |

## 2. Answer to the question

1. **No published label-free -> marker system tests transfer of a batch-specific label readout.** DeepIFC (L1-L4) holds out donors, but all
   donors share one panel, one compensation and one set of autofluorescence controls (L4). The label readout is identical across its
   split. The other BSCCM regression paper (L6) uses one random-style test split of BSCCMNIST with no batch hold-out. Lippeveld (L8) uses
   cell-level CV. AIRFIHA (L9) claims donor CV for classes, not intensities. I found no paper reporting batch-held-out Pearson on BSCCM.
2. **Dim-marker reference points.** On held-out donors with a single readout, DeepIFC reaches CD19 r .61 and CD56 .53 (AUROC .80/.81) on
   PBMC (L2, L5). BSCCM is lysed whole blood, with a granulocyte majority and fewer B/NK cells (L5). The project's CD19 .33-.34 and
   CD56 .48-.56 cross-draw numbers are therefore not obviously below the literature. HLA-DR and CD123 have no single-marker benchmark.
   The BSCCM authors and the 2026 paper both find the HLA-DR/CD14/CD123 group poor (B6, L6-L7; pooled r .339 even in-distribution).
3. **Every readout-correction method in cytometry needs data from the new batch.** CytoNorm needs a control sample in each batch (C1).
   CytoNorm-without-controls (C3), cyCombine (C4), gaussNorm (C5) and CytoVI (C6) all need the new batch's own marker distribution or
   batch label. AutoSpectral (C8) needs that batch's single-stain controls and raw spectra. None of these is inductive in the GOAL's
   sense. The closest inductive analogue was already tested: N047 c3 read the coefficient from the new draw's unstained control and got
   -.0017. That supports the N059 diagnosis: the HLA-DR/grp readout gap cannot be closed by an inductive LF model.
4. **The mechanism the literature points to fits the project's measurements.** The authors list the cause themselves: a regularised 2-spectrum
   fit that trades antibody against autofluorescence (B5), spectral merging with crosstalk (B3-B4), and draw-specific staining (B1-B2).
   CytoNorm's finding that technical variation hits cell types differently (C2) matches the class-specific autofl-slope flip (K029,
   N047 c2). Draw-1 stains are 2.5-3.9x dimmer while the unstained-cell artefact coefficient is the same in both draws (N047 c3: HLA-DR
   -.128/-.160). So the readout difference is a brightness-scaled artefact weight. Its direction cannot be known from one draw.
5. **What remains is training-side only.** Two levers can be tested without breaking the inductive constraint:
   - cleaner training labels: label-noise-robust or crowd-style heads (N1);
   - more indirect supervision for the label-limited CD19 and CD56 (L2 attributes their weakness to few positive cells).
   Expected gains are small, and the ideas below say so.

## 3. Ideas (all inductive, training-draw only, eval labels v10 frozen)

GOAL arithmetic: 11 targets x 2 directions = 22 cells. +.01 r on one cell = +.00045 GOAL. The gap is .0138, and 2 x noise is .0054.

### I1. Mixed-effects "crowd" training: FOV random intercepts plus slide x LF-class readout heads; the canonical head predicts
- *Mechanism.* During training only, model y_i = a_s,c + b_s,c * f_k(x_i) + u_FOV(i) + e:
  - f_k is the shared marker head;
  - (a, b) is an affine readout per slide s x LF 3-class c (the existing nuisance strata of N047 c2, not an exclusivity prior);
  - u_FOV is a shrunk random intercept with prior N(0, tau^2_k), where tau^2_k is set from the training draw's permutation excess (K028
    method).
  At test, output f_k (intercept 0, canonical head). This is a crowd layer (N1) with slides and FOVs as annotators. It is CytoNorm's
  cell-type-specific transform (C2) learned as a nuisance during training, not applied to the test draw.
- *Why it moves the bottleneck.*
  - CD19/CD56 are label-limited for both models (N059 point 5; label-count curve +.035-.05 weak-target r per label doubling, K033).
  - Their training labels still carry FOV-level error (within-FOV +.040, N059 point 3). Absorbing it raises the effective label count.
  - Slide heads stop the trunk from fitting slide-specific pooled-panel readouts: draw 1 has two 'all' slides, rep0 13,675 and rep1
    13,399 cells, and per N047 the test-draw grp:CD45 gap sits on the 5x-dimmer rep0 slide.
  - This is NOT the refuted FOV-context head (N037). That head tried to predict offsets at test. Here offsets are never predicted;
    they are partialled out of the training loss.
  - It differs from the parked within-FOV contrast loss (N047 Codex F). Shrinkage keeps the real between-FOV composition that
    centring destroys, the point the N047 red-team raised.
- *Expected gain.* Assume a FOV share of .2-.4 on CD19/CD56 training labels. Removing it gives roughly 1.3-1.7x effective labels, so
  +.015-.03 r on CD19/CD56 (4 cells), plus about +.005 on the grp cells via the slide heads. That is about +.003-.006 GOAL before
  ensemble absorption. On its own it may not clear 2 x noise; bundle it with I2.
- *Cheapest falsifier (0 GPU-h).* Run on CPU, 48 cores, under 2 h, using frozen SSL-r50 embeddings of the training draw (extracting them
  costs about 0.2 GPU-h if they are not cached):
  - ridge readout vs alternating ridge with FOV random intercepts and slide x class affine heads;
  - score held-out-draw v10 r in both directions.
  Kill if mean CD19/CD56 r gains less than +.01, or if any grp cell loses more than .01. GPU screen: base model, 2 directions, about
  0.7 GPU-h (N047 estimate); add zoom2 for about 1.3 GPU-h.
- *Build.* 1-1.5 days: FOV/slide index tensors already exist in targets (fov, batch), plus a nuisance-parameter table and a masked loss.

### I2. Unmixing-consistent heads: pooled 'all' slides supervise the single-marker heads through the training draw's own measurement model
- *Mechanism.* Today grp:CD3/CD19/CD56 and grp:CD123/HLA-DR/CD14 are separate free heads. Replace them with a fixed measurement layer:
  grp_hat = sum_k w_k * m_hat_k + sum_j g_j * m_hat_j. The first sum runs over the merged markers (B3). The second adds the
  documented crosstalk from CD123/CD45/CD16 (B4).
  - Fit w and g on the TRAINING draw only, by passing its single-antibody cells through the recovered 4-spectrum unmixing (N047 a4
    recovered the authors' maps at fidelity .98-1.0).
  - Set CD19's weight in batch-1 'all' slides from the published dose (B2: 35 %). This is protocol metadata, not labels.
  - Gradients from the 'all' cells (10,210 in draw 0; 27,074 in draw 1) then reach the CD19, CD56 and HLA-DR heads. The batch-0 HLA-DR
    group passed QC (K019).
  - This is the measurement-model counterpart of CytoVI's panel masking (C6).
- *Why it moves the bottleneck.* CD19 and CD56 are label-limited (L2; K033 curve). Single-antibody training cells are only 12,768/7,736
  (CD19) and 14,212/23,141 (CD56) per draw. The 'all' cells add indirect sum-constrained labels for exactly those heads.
- *Expected gain.* An 'all' cell carries perhaps 1/3-1/5 of a direct CD19 label, because the sum is dominated by CD3. That is about
  +15-25 % effective labels, or roughly +.01 r per weak cell, so +.002-.003 GOAL. Most of it would come in b01, where draw-0 'all' has
  the full CD19 dose. This is NOT the refuted "35 % dose" correction (N047 b2): that was a post-hoc subtraction on predictions, while this
  is extra training signal.
- *Cheapest falsifier (0 GPU-h).* On the same frozen embeddings, run joint least squares (single-label rows plus sum-constraint rows on
  'all' cells) against independent ridge. Score held-out r on CD19, CD56 and HLA-DR. Kill if the weak-target mean gains less than +.008.
  GPU screen: base, 2 directions, about 0.7 GPU-h. Bundle with I1 as one attempt: about 1.4 GPU-h for base+zoom2.
- *Build.* 1 day: fitting the measurement weights on CPU plus one output layer.

### I3. Physics-simulated readout randomisation with an expected-readout hedge (the decisive test of the "readout ceiling")
- *Mechanism.*
  1. On training-draw cells, rescale the antibody component of the raw 6-channel signal by a brightness factor s, then re-run the
     recovered authors' unmixing, the v10 asinh and the per-slide z. The prior is pre-registered as log-uniform [1/4, 4], justified by the
     documented 35 % dose (B2) and generic lot/isolation variation (C7). The draw-1 brightness ratio must NOT be used.
  2. Train an s-conditioned head (FiLM on log s), or fit post hoc the artefact weight kappa(s).
  3. At test, predict E_s[f(x; s)].
  This is label-side domain randomisation. It is distinct from N041, which randomised the input instrument.
- *Why it moves the bottleneck.* The diagnosed HLA-DR/grp gap is a brightness-scaled autofluorescence artefact: the unstained coefficient is
  draw-invariant and the stained-cell weight scales with 1/brightness (N047 c3; B5). Simulating s reproduces the whole family of
  readouts from one draw, so the model hedges instead of hard-coding the training draw's ratio.
- *Expected gain.* 0 to +.004 GOAL, stated honestly:
  - the oracle for this family is +.0134 (N047 c2, v5 units), but it needs the DIRECTION of the brightness change;
  - a prior centred on the training draw has none, so any gain comes only from the concavity of r in kappa;
  - N047's four zero-label corrections were all -.002 to -.045.
  Its real value is diagnostic. If the simulation reproduces draw-1's readout but the hedge still fails, the "inductive ceiling" claim
  becomes [verified], and .70 needs a protocol decision.
- *Cheapest falsifier (0 GPU-h, CPU about 1 h).*
  (a) Mechanism check: does simulated s = 1/3 on draw-0 HLA-DR/CD14/CD19 reproduce draw-1's within-class autofl slope sign and size
      (+.05 -> about -.6 for HLA-DR, N047 c2)? Draw-1 values are used only as a yes/no check after the prior is frozen. If not, kill.
  (b) Apply kappa_hedge post hoc to the saved N048/N053 ensemble predictions and score the GOAL. Proceed to a GPU run (about 0.7 GPU-h)
      only if the gain is at least +.0054.
- *Build.* 0.5 day CPU (the recovered unmixing maps exist), plus 0.5 day for the FiLM head if (b) passes.

### Not proposed (outside the constraint), for the user only
The field's standard fix is control-anchored calibration (C1-C5, L4), and in BSCCM the matching controls are the new draw's unstained and
single-stain slides. Its simplest form was already -.0017 (N047 c3). The only lever measured to close the readout gap is labelled cells
from the new isolation: k-shot calibration, +.016-.026 at 500-1000 cells per marker (N047 I2). Both break "inductive" as written, so
either needs a user ruling as a protocol variant, not a model change.

## 4. Bottom line
The literature has no inductive fix for a batch-specific fluorescence readout. Every cytometry method uses the new batch's controls or its
marker distribution, and every label-free predictor that "generalises to new donors" keeps the readout fixed. I1+I2 bundled are the best
inductive bet, at about +.004-.008 (borderline vs 2 x noise .0054), aimed at the label-limited CD19/CD56. I3 is mainly the decisive test of
the readout ceiling. Reaching .70 (+.0138) inductively is unlikely on this evidence. If I1-I3 fail, record the ceiling and take the
protocol question (k-shot or control-anchored calibration) to the user.
