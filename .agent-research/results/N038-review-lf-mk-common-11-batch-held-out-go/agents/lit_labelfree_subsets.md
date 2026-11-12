# Literature: label-free single-cell imaging -> leukocyte SUBSETS and dim surface markers (N038 review)

Agent: read-only literature agent (Claude), 2026-11-12. Question: how do label-free (brightfield / LED-array / phase / QPI /
darkfield / holographic / tomographic) single-cell systems predict B (CD19), NK (CD56), basophil/pDC (CD123), HLA-DR and T (CD3)
cells per cell, how accurate are they, and which mechanism makes the difference.

Access: web worked. arXiv, bioRxiv (rate-limited, retried), Europe PMC REST API (abstracts + OA full text; used because
pmc.ncbi.nlm.nih.gov / pubmed / nature.com returned bot walls), publisher PDFs (text extracted locally). Every number below comes
from the cited source; quotes are verbatim (fragments where the fetch tool truncated long sentences). Project-internal numbers are
marked [BSCCM notebook].

---------------------------------------------------------------------------------------------------------------------------------

## 1. Bottom line

1. **No label-free system gets dim lymphoid markers near lineage-marker quality.** The best directly comparable result is
   DeepIFC (60x imaging flow cytometry, brightfield + darkfield, 6 donors, test = 2 held-out samples): CD45 r .90, CD14 .88,
   CD3 .87 but **CD19 r .61, CD56 r .53, CD8 r .41**. Mouse IFC-seq: label-free CD34 r .38, FcgR r .50. Our in-distribution
   CD19 .28/.47 and CD56 .41/.39 sit below that band. Our lineage markers (.85-.9) sit at it.
2. **Optical resolution/NA is the mechanism with the cleanest controlled evidence.** On donor-held-out DHM, CD4-vs-CD8 accuracy
   rises from 59.4 % (20x/0.4 NA) to 69.3 % (60x/0.8) and 76.2 % (100x/0.9). In deep-UV, blurring 300 nm -> 600 nm "significantly
   reduces" CD4/CD8 subtyping. BSCCM images at 20x/0.5 NA, i.e. the low end of that range.
3. **3D refractive index beats 2D** for lymphoid subsets: 96.7 % (3D RI tomogram) vs 87.9 % (2D MIP) vs 80.1 % (2D phase) vs
   76.6 % (2D amplitude) on B/T/monocyte/myelocyte. The caveat is a random cell split. Separately, RI-derived dry mass and protein
   density beat shape-only features for B/CD4/CD8.
4. **Training-set size per rare subset matters a lot.** DeepIFC with balanced data: recall of NK 25 -> 67 %, CD8 13 -> 68 %,
   B 5 -> 54 %, saturating around 3,200 cells per type. BSCCM per-marker labels come only from single-stain slides: 12,768 / 7,736
   CD19-stained cells in batch 0 / 1 [BSCCM notebook, N001 audit]. If B cells are a few % of leukocytes, that is O(10^2-10^3)
   positives per draw, roughly 10x below that saturation point.
5. **Bigger or self-supervised models and fancier heads are not where the gains are.** Engineered features beat deep learning
   on stain-free IFC (balanced accuracy .778 vs .703). A frozen MoCo-v2 SSL encoder comes within ~2 % of supervised. On BSCCM
   itself, a 2026 multi-task CNN+ViT gets grp:CD3/19/56 r .768 but CD123/HLA-DR/CD14 only r .339.
6. **Spectral / metabolic / temporal contrast** separates some subsets that static morphology cannot: autofluorescence lifetime
   for T activation and CD4/CD8 at 97 %; dynamic deep-UV for CD4/CD8 at ~90 %, where static images failed. It is not a universal
   fix: in PBMC, metabolic imaging finds NK cells (recall 74 %, AUC .91) but "Cytotoxic T cells, helper T cells, and B cells were
   not strongly resolved". These modalities are also outside the BSCCM input rules (no fluorescence into LF models, no new
   acquisition).
7. **The BSCCM paper itself** reports no prediction baselines. It flags label-side limits that bear directly on our four
   drag targets:
   - the CD123/CD14/HLA-DR channel was "deemed too noisy";
   - CD3, CD19 and CD56 share one fluorophore (BV510);
   - batch-1 all-antibody CD19 was stained at "35 percent of the amount";
   - unmixing is "underdetermined".
8. **Context (neighbouring cells, wider field) has no published evidence either way** for leukocyte subsets. I found none.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Evidence table

| # | Source (year) | Modality, optics | Task / targets | Split | Key number(s) |
|---|---|---|---|---|---|
| 1 | Pinkard et al., BSCCM (2024) | LED array, 20x/0.5 NA, 22 patterns, 128 px, 0.166 um/px | dataset; no baselines | 2 isolations (batch 0/1) | 412,941 cells; CD123/CD14/HLA-DR channel "too noisy" |
| 2 | Pinkard et al., Information-driven design (NeurIPS 2025) | BSCCM single patterns (BF, DPC, single LED) | 8 markers, DenseNet121 + GMM heads, NLL | n/a | information estimate tracks protein-prediction accuracy across 3 patterns (no numbers in text) |
| 3 | Nazir & Behera (arXiv 2605.14717, May 2026) | BSCCMNIST 28x28, 4-ch DPC | WBC class + 4 marker groups | 1,418 test cells, split unstated | r: CD16 .819, CD45 .799, CD3/19/56 .768, CD123/HLA-DR/CD14 .339 |
| 4 | Timonen et al., DeepIFC (Cytometry A 2023) | ImageStream 60x, BF + DF | virtual fluorescence, 7 markers | 2 held-out samples (6 donors) | r: CD45 .90, CD14 .88, CD3 .87, CD19 .61, CD56 .53, CD8 .41; balanced data: B recall 5 -> 54 % |
| 5 | Lippeveld et al. (Cytometry A 2020) | ImageStream 60x, BF + DF | 8 WBC classes incl. CD4/CD8 T, B, NKT | stratified 5-fold (2 samples) | bal. acc. .778 (features) vs .703 (DL); DL recall CD4 .454, CD8 .299, B .632 |
| 6 | Nassar et al. (Cytometry A 2019) | IFC BF + DF | 4-part diff + B vs T | leave-one-donor-out (13 donors) | B vs T F1 ~78 %; main types F1 ~97 % |
| 7 | Ugawa et al., iSGC (eLife 2021) | ghost cytometry waveforms (+FSC/BSC) | 5-part diff incl. basophils (CD123) | intra- vs inter-donor | macro-F1 .911 vs .904; CD4/CD8 "challenging" |
| 8 | Gupta et al. (bioRxiv 2021 / Optics Continuum 2023) | off-axis DHM at 3 magnifications | CD4 vs CD8 T | train/test from different donors | test acc. 59.4 % (20x/0.4) / 69.3 % (60x/0.8) / 76.2 % (100x/0.9) |
| 9 | Yoon et al. (Sci Rep 2017) | 3D RI tomography | B / CD4 / CD8 (mouse, sorted) | random 70/30, n = 356 | B vs T test 89.8 %; CD4 vs CD8 84.4 %; 3-class 75.9 % |
| 10 | Ryu et al. (BME Frontiers 2021) | 3D RI tomography, 110 nm lateral | B / T / monocyte / myelocyte (bone marrow) | random 7:1:2 | 3D 96.7 % vs MIP 87.9 % vs 2D phase 80.1 % vs amplitude 76.6 % |
| 11 | Deep-UV T-cell (BME Frontiers, Feb 2026) | 255 nm deep-UV, ~300 nm res., static + 500-frame dynamics | CD4 vs CD8 (activated) | 5-fold CV | static: no separation; dynamic ~90 %; blur to 600 nm "significantly reduces" accuracy |
| 12 | OMI in PBMC (Biophotonics Discovery, Jul 2026) | NAD(P)H/FAD lifetime | monocytes, NK, B, CD4, CD8 in PBMC | 3 donors | NK recall 73.6 %, AUC .91; monocytes 96 %; B/CD4/CD8 "not strongly resolved" |
| 13 | Walsh et al. (Nat Biomed Eng 2021) | autofluorescence lifetime | T activation and CD4/CD8 | (T-cell cultures) | 97-99 % activation; 97 % activation x subtype |
| 14 | Cell-MICS (J Biophotonics 2026; arXiv 2506.14449) | 2-photon autofluorescence (+Dodt) | 6 isolated mouse immune types | grouped 5-fold | multi-class F1 .689; B cells confused with CD8 T |
| 15 | Chlis et al. (NAR 2020) | IFC BF, 32x32 px | label-free CD34, FcgR (mouse) | 10 % held out | r .38 +- .16 (CD34), .50 +- .016 (FcgR) |
| 16 | Dannhauser et al. (Lab Chip 2021) | in-flow light scattering | NK CD56bright vs CD56dim | sorted subclasses | >= 86 % accuracy |
| 17 | Muthumbi et al. (BOE 2019) | LED array, learned illumination | malaria-infected RBC | -- | +5-10 % accuracy from learned illumination |
| 18 | Cheng et al. (Sci Adv 2021) | LED-array reflectance, 4 oblique DF + drDPC | virtual IF (HeLa) | -- | drDPC channels +~9 % actin prediction |
| 19 | Ounkomol et al. (Nat Methods 2018) | transmitted light -> 3D fluorescence | structures | -- | label-noise ceiling C_max = sqrt(SNR/(1+SNR)) |
| 20 | Generalizable SSL for IFC (Microsyst. Nanoeng., Jun 2026) | custom IFC brightfield | 4-/2-/8-class leukocytes (purified) | independent datasets | frozen MoCo-v2 within ~2 % of supervised |
| 21 | Frozen hematology FMs under shift (arXiv 2608.25148, Aug 2026) | Wright-stained single cells | WBC classes | cross-dataset | in-domain macro-F1 .98-.997; cross-dataset drop 34-72 % |
| 22 | FPGA cell sorting (arXiv 2503.12622, 2025) | DIC, FACS-sorted mouse splenocytes | T4/T8/B | 80/10/10, grouping unstated | T4 vs B 98 % (likely sample-confounded) |

---------------------------------------------------------------------------------------------------------------------------------

## 3. Findings per source (URL · verbatim quote · number)

### 3.1 BSCCM itself and work that uses it

**[1] Pinkard, Liu, Nyatigo, Fletcher, Waller, "The Berkeley Single Cell Computational Microscopy (BSCCM) Dataset", arXiv
2402.06191 (v1 only, 9 Feb 2024).** https://arxiv.org/abs/2402.06191 , https://arxiv.org/html/2402.06191v1

Dataset:
- "over ~12,000,000 images of 400,000 of individual white blood cells". Table 1 lists 412,941 cells (BSCCM) and 4,304 cells
  (BSCCM-coherent).
- Optics: "The LED array and fluorescence images were collected using a 20×0.5 NA objective lens". Pixel 0.166 um, 128x128.
- "The default version of the dataset (BSCCM) has 22 images of each cell".
- Batches: "The batch index captures the two biological replicates (i.e. distinct cell isolations) on two different dates."

Label-side caveats relevant to our four drag targets:
- Shared fluorophore: "CD3, CD19 and CD56 were also combined into a single channel". All three are BV510; CD123 is BV421,
  HLA-DR BV570, CD45 BV605, CD14 BV650, CD16 BV711.
- Unmixing: "this is an underdetermined problem, with only six measurement channels"; "Cells also showed varying amounts of
  autofluorescence."
- Noisy pooled channel: "The CD123/CD14/HLA-DR channel was deemed too noisy be useful" (when building the cell-type labels).
- CD19 under-dosing in batch 1 all-antibody cells: "the amount CD19 antibody used was 35 percent of the amount used in other
  CD19 stains".
- Input drift: "From slide-to-slide and batch-to-batch, there are variations in the average intensity of images in a given
  channel."
- **No prediction baselines are reported.**

BSCCM-coherent subset:
- 566 single-LED images per cell ("566 different illumination patterns, each corresponding to illumination by a different
  single LED").
- Batch 1 only; "contains only all-antibodies and no antibody conditions".
- Its stated motivation is Fourier ptychography and diffraction tomography.

**[2] Pinkard, Kabuli, Markley, Chien, Jiao, Waller, "Information-Driven Design of Imaging Systems", NeurIPS 2025 / arXiv
2405.20559 (v5).** https://arxiv.org/html/2405.20559v5 ; blog https://bair.berkeley.edu/blog/2026/01/10/information-driven-imaging/

- Uses BSCCM with "added simulated photon shot noise to equalize photon counts".
- Predictor: "The network uses a DenseNet121 backbone pretrained on ImageNet". It has per-marker heads that output Gaussian-mixture
  parameters for CD3, CD19, CD56, CD123, HLA-DR, CD14, CD16 and CD45, and reports the "lowest negative log likelihood on test
  data".
- Result: "Our information estimates correlated with protein prediction accuracy across three illumination patterns" (brightfield,
  DPC, single LED). "Higher information estimates generally predicted better neural network classification performance, though
  single-LED showed inflated estimates due to optical imperfections."
- Blog: "In all cases, higher information meant better downstream performance."
- **No per-marker numbers appear in the text** (figure only).

**[3] Nazir & Behera, "Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning", arXiv 2605.14717 (14 May 2026;
ICPR 2026).** https://arxiv.org/html/2605.14717

Setup: BSCCMNIST 28x28, 4-channel DPC. Test split of 1,418 cells; whether it is batch-held-out is not stated.

Table 4 (Pearson r per marker group):

| Marker group | r |
|---|---|
| CD16 | .819 |
| CD45 | .799 |
| CD3/CD19/CD56 | .768 |
| CD123/HLA-DR/CD14 | .339 |

- Full model vs regression-only: r .7263 vs .7012.
- Their explanation for the weak group: "These markers reflect functional activation states rather than stable morphological
  phenotypes."
- Relevance: the only other per-marker BSCCM numbers in the last 12 months. Low resolution, unclear split, and no individual
  CD19/CD56 targets, so it is a weak baseline only.

**Also found:** arXiv 2604.05211 (Apr 2026) uses BSCCM-tiny (N = 1000) for dictionary learning; "lymphoid-vs-myeloid
separation at ARI=0.575". It has no marker regression and is not relevant to subsets.

### 3.2 Imaging flow cytometry, brightfield + darkfield

**[4] Timonen et al., "DeepIFC: virtual fluorescent labeling of blood cells in imaging flow cytometry data with deep learning",
Cytometry A 2023; bioRxiv 10.1101/2022.08.10.503433.** https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full ;
https://doi.org/10.1002/cyto.a.24770

Setup:
- "Seven PBMC samples of six blood donors", ImageStream X Mk II, "60× magnification", pixel 0.33 um. Inputs are BF (Ch01, Ch09)
  and darkfield (Ch12).
- Split: "all images from two samples into a test dataset" (200,000 cells); training 247,107 cells.

Per-marker r:
- "CD45 (r=0.90), CD14 (r=0.88) and CD3 (r=0.87)".
- "Fluorescent images for the surface markers CD56 (r=0.53, AUROC=0.805), CD19 (r=0.61, AUROC=0.800)"; also "CD8 (r=0.41,
  AUROC=0.728)".
- **How r is computed (per-cell intensity or pixel-wise) is not stated in the text.**

Balanced training:
- "Similarly, recall of NK, cytotoxic T and B cells increased from 25% to 67%, 13% to 68% and 5% to 54%, respectively."
- "For most cell types, the saturation point in performance increase was reached at 3200 cells".

Their explanations for the weak markers:
- "likely due to the relatively small numbers of cells in the complete dataset exhibiting these markers".
- "Morphological differences between T cell subtypes (CD56, CD8) were not found to be visible to the human eye" (in brightfield
  images).

**[5] Lippeveld et al., "Classification of Human White Blood Cells Using Machine Learning for Stain-Free Imaging Flow Cytometry",
Cytometry A 97:308 (2020).** https://doi.org/10.1002/cyto.a.23920 ; preprint https://www.biorxiv.org/content/10.1101/680975v1.full

Setup: 60x ImageStream, brightfield + darkfield. Four whole-blood samples, two of them for the 8-class WBC set. Stratified 5-fold
CV, not donor-split.

Results:
- Published abstract (Europe PMC): "On the white blood cell data set, the best obtained results were 0.778 and 0.703 balanced
  accuracy for classical ML and DL, respectively."
- "the DL approaches tested in this work do not outperform the approaches based on manually engineered features".
- Preprint, DL: "recalls for CD4+ and CD8+ T-cells did not exceed 0.454 and 0.299, respectively".
- "The biggest improvement over RN occurred in B-cell classification (RN: 0.391, versus DF: 0.632 recall)".

Their hypothesis on resolution: "because of the relatively low image resolution of current imaging flow cytometers, necessary
data for cell subtyping is potentially not captured in the stain-free imagery". This is a hypothesis; they did not test it.

**[6] Nassar et al., "Label-Free Identification of White Blood Cells Using Machine Learning", Cytometry A 95:836 (2019).**
https://carpenter-singh-lab.broadinstitute.org/files/anne/files/123_Nassar_CytometryA_2019.pdf

- Abstract: "It achieved an average F1-score of 97% and two subtypes of WBCs, B and T lymphocytes, were distinguished from each
  other with an average F1-score of 78%, a task previously considered impossible for unlabeled samples."
- Split: "validation was performed in 13 iterations (the training data set contains 13 blood donors). In each iteration, the data
  set from one blood donor was separated for validation".
- Features: "For lymphocyte T versus B cell classification, the overall shape of the cell was important in addition to mostly
  brightfield intensity and granularity features." Top features include brightfield edge intensity, radial distribution and
  granularity, plus darkfield edge intensity.
- Warning against cell-shuffled splits: "This method is prone to overfitting as the classifier has already seen blood cells from a
  particular subject in the training set".

**[7] Ugawa, Kawamura et al., "In silico-labeled ghost cytometry", eLife 10:e67660 (2021).**
https://elifesciences.org/articles/67660 (PDF https://cdn.elifesciences.org/articles/67660/elife-67660-v1.pdf)

- Labels came from CD45/CD123/CD14/CD16/CD34, covering the 5-part differential incl. basophils.
- Donor shift is small: "The macro- average F1- scores were 0.911 and 0.904 for (B) and (C), respectively" (intra- vs
  inter-donor).
- Rare classes: "false negatives can be seen for basophils, which can be thought to be due to their limited number in the
  population".
- Subsets: "differentiation of CD4 and CD8 T cells are one example which have been challenging to both ghost cytometry and our
  eyes."

### 3.3 Quantitative phase / holography / refractive-index tomography

**[8] Gupta, Hempler, Malcolm, Dholakia, Powis, "High throughput hemogram of T cells using digital holographic microscopy and deep
learning", bioRxiv 10.1101/2021.12.23.473983; Optics Continuum 2023 (10.1364/OPTCON.479857).**
https://www.biorxiv.org/content/10.1101/2021.12.23.473983v1.full

Setup: CD4 and CD8 cells isolated separately by negative depletion; six donors. Split "such that the training/validation set and
testing set came from different donors".

Test accuracy rises with magnification and NA:

| Optics | Test accuracy |
|---|---|
| 20x / 0.4 NA | "59.43 % ± 1.99 % as the test accuracy" |
| 60x / 0.8 NA | "69.31 % ±2.04 % test accuracy" |
| 100x / 0.9 NA | "76.2 % ± 5.27 % for the test set" |

- A cycle-GAN mapping 20x to 100x-like images, with retraining, reported "the sensitivity and specificity of 79.77 % ± 3.32 % and
  81.77 % ± 1.51 % for test set respectively". Before retraining, the transformed images scored 47.9 %.
- **This is the cleanest donor-held-out dose-response for resolution on lymphoid subsets.**
- Caveat: separately isolated populations, so isolation batch could leak.

**[9] Yoon et al., "Identification of non-activated lymphocytes using three-dimensional refractive index tomography and machine
learning", Sci Rep 7:6654 (2017).** https://doi.org/10.1038/s41598-017-06311-y (full text via Europe PMC PMC5532204)

Setup: mouse, FACS-sorted. n = 149 B, 95 CD4+ T, 112 CD8+ T.

Results:
- B vs T: "The accuracy of the optimized classifier was 93.15% and 89.81% for the training and test cases" (selected features:
  surface area, volume, sphericity, protein density, dry mass).
- CD4 vs CD8 test: 84.38 % overall.
- 3-class test: 75.93 %.
- "statistical classification performed by morphological information only (surface area, cellular volume, and sphericity) lowers
  overall accuracy". **RI-derived mass/density adds information beyond shape.**

**[10] Ryu et al., "Label-Free White Blood Cell Classification Using Refractive Index Tomography and Deep Learning", BME Frontiers
2021:9893804.** https://spj.science.org/doi/10.34133/2021/9893804 (Europe PMC PMC10521749); preprint
https://www.biorxiv.org/content/10.1101/2020.11.13.381244v1.full

Setup: bone marrow, 11 donors; 508 B, 913 T, 516 monocytes, 610 myelocytes. Optics: "theoretical resolutions of the ODT system
are 110 nm (lateral) and 330 nm (axial)". Split: "randomly shuffled the entire dataset and split the training, validation, and
test set by a 7:1:2 ratio". **Cell-level split, so donor leakage is possible.**

Representation matters:
- "We also tested another deep neural network that classifies the four types of WBCs with a test accuracy of 96.7%."
- "While the 2D MIP obtained directly from the 3D tomogram achieved an accuracy of 87.9%, which is the closest to our approach,
  the network trained with 2D phase and amplitude presented accuracies of 80.1% and 76.6%."

**[16] Dannhauser et al., "Non-invasive and label-free identification of human natural killer cell subclasses by biophysical
single-cell features in microfluidic flow", Lab Chip 2021.** https://doi.org/10.1039/d1lc00651g

- "We trained our ML algorithm with sorted NK cell subclasses (≥86% accuracy)" (CD56bright vs CD56dim, light scattering).
- Relevance: CD56 intensity level, not only NK identity, carries label-free biophysical signal.

**Holographic image representation (bioRxiv 10.64898/2026.02.26.708207, Feb 2026)**, NK vs breast-cancer cells:
"reconstructed phase images provide the highest accuracy, whereas simpler representations or accelerated reconstruction methods
significantly reduce processing time with minimal loss of accuracy." This is a cancer-vs-NK task, not subsets, so it is only
weak evidence that physics-based reconstruction helps.

### 3.4 Spectral, metabolic and temporal label-free contrast

**[11] "Nondestructive, High-Resolution T Cell Characterization and Subtyping via Deep-UV Microscopy", BME Frontiers (18 Feb
2026), doi 10.34133/bmef.0227 (Europe PMC PMC12914059).**

- Static images failed: "we initially attempted to subtype activated CD4 + and CD8 + T cells using only static images and similar
  static image features. However, this approach did not yield distinct separation between the 2, suggesting that static
  morphological features alone are insufficient for subtyping".
- Dynamics worked: "Dynamic deep-UV images revealed unique intracellular activity that enabled accurate subtyping of CD4 + and
  CD8 + T cells, with a sensitivity and specificity of ~90%".
- Resolution: "The image stacks used in this work have a diffraction-limited lateral resolution of approximately 300 nm ...
  However, blurring and downsampling these image stacks by a factor of 2 (corresponding with a lateral resolution of
  approximately 600 nm) significantly reduces subtyping accuracy".

**[12] "Autofluorescence lifetime imaging resolves cell heterogeneity within peripheral blood mononuclear cells", Biophotonics
Discovery 3:035003 (Jul 2026), doi 10.1117/1.bios.3.3.035003 (Europe PMC PMC13557457; bioRxiv 10.64898/2026.03.06.710224).**

- Abstract: "identified NK cells within quiescent and activated PBMCs with 74% recall"; "classification of NK cells was moderately
  accurate ( AUC = 0.91 , Recall = 73.6 % )".
- "Cytotoxic T cells, helper T cells, and B cells were not strongly resolved based on OMI variables".
- Setup: three donors; ground truth from one antibody per well (CD4, CD8, CD14, CD19, CD56).

**[13] Walsh et al., "Classification of T-cell activation via autofluorescence lifetime imaging", Nat Biomed Eng 5:77 (2021),
doi 10.1038/s41551-020-0592-z.** Abstract (Europe PMC):
"classified T cells according to activation state with 97-99% accuracy, and according to activation state (quiescent or
activated) and subtype (CD3+CD8+ or CD3+CD4+) with 97% accuracy."

**[14] Cell-MICS, J Biophotonics 2026 (doi 10.1002/jbio.70260); arXiv https://arxiv.org/html/2506.14449v1**

- "0.689 F1 score, 0.697 precision, 0.748 recall for multi-class classification between six isolated cell types".
- "most of the mis-classified examples were B cells that were falsely predicted as CD8+ T cells".
- "A model that was solely trained on the Dodt signal is close to a random guess (0.5 AUC-ROC)" (Dodt = transmitted-light
  gradient contrast).

### 3.5 Illumination design, data size, model scale, label ceilings, shift

**[17] Muthumbi et al., "Learned sensing: jointly optimized microscope hardware for accurate image classification", Biomed Opt
Express 10:6351 (2019).** https://doi.org/10.1364/boe.10.006351
"automatically identify malaria-infected cells with up to 5-10% greater accuracy than standard and alternative microscope lighting
designs". This is malaria, not leukocytes.

**[18] Cheng et al., "Single-cell cytometry via multiplexed fluorescence prediction by label-free reflectance microscopy", Sci Adv
7:eabe0431 (2021).** https://par.nsf.gov/servlets/purl/10210192
"the additional drDPC channels substantially improve the actin IF label predictions by ~9% because drDPC highlights cell
topography and boundaries as compared to the plain dark-field images." HeLa cells, not blood.

**[15] Chlis et al., "Predicting single-cell gene expression profiles of imaging flow cytometry data with machine learning", NAR
48:11335 (2020).** https://academic.oup.com/nar/article/48/20/11335/5943188

- Input: 32 x 32 px brightfield. Label-free: "is 0.38 ± 0.16 for CD34 and 0.5 ± 0.016 for FcgR".
- "label-free prediction is only expected to work if there is sufficient morphological information".

**[19] Ounkomol et al., "Label-free prediction of three-dimensional fluorescence images from transmitted-light microscopy", Nat
Methods 15:917 (2018) (Europe PMC PMC6212323).**

- Defines the label-noise ceiling C_max = sqrt(SNR/(1+SNR)): "a theoretical model which is able to perfectly predict the spatial
  fluctuations of the signal but is unable to predict the random fluctuations in the target image".
- "A theoretical upper bound for model performance based upon an estimate of signal-to-noise ratio (SNR) of fluorescence images
  used for training was determined".

**[20] "Generalizable self-supervised learning for imaging flow cytometry on multi-dataset leukocyte differential", Microsystems &
Nanoengineering (10 Jun 2026), doi 10.1038/s41378-026-01236-x (Europe PMC PMC13250045).**

- Pretraining: 80,347 unlabelled images from 4 donors; downstream sets are purified populations.
- Abstract: "achieves classification accuracies of 96.16%, 96.68%, and 92.24% across three independent datasets, performing
  comparably to supervised baselines (with differences of less than 2%)".
- Takeaway: SSL buys label efficiency, not extra accuracy.

**[21] "Can You Trust Frozen Hematology Foundation Models under Acquisition Shift?", arXiv 2608.25148 (25 Aug 2026).**
https://arxiv.org/abs/2608.25148
- "In-domain linear-probe macro-F1 is saturated (0.98-0.997), yet cross-dataset macro-F1 drops 34-72% and rankings re-order".
- Stained cells, not label-free. Relevant only as a warning that in-domain rankings do not predict shift robustness.

**[22] "Real-Time Cell Sorting with Scalable In Situ FPGA-Accelerated Deep Learning", arXiv 2503.12622 (Mar 2025).**
https://arxiv.org/html/2503.12622v2

- "Our teacher model attained 98% accuracy in differentiating T4 cells from B cells".
- Cells were "FACS-sorted into T4, T8, and B subsets, imaged by DIC microscopy". The split grouping is not stated.
- **Treat as likely sample/batch-confounded.** It is a cautionary example of how high per-subset numbers arise when each class
  comes from its own sorted sample.

**Polarization-multiplexed LED intensity diffraction tomography (Opt Lett 2024, doi 10.1364/ol.533430):** "near incoherent
diffraction-limited lateral resolution of 690 nm and an axial resolution of 4.68 μm". At low/moderate NA, LED-array 3D RI has
axial resolution comparable to half a lymphocyte. So the large 3D-vs-2D gain in [10] (110/330 nm ODT) **should not be assumed
to carry over to 20x/0.5 NA**.

---------------------------------------------------------------------------------------------------------------------------------

## 4. What makes the difference: mechanisms ranked by strength of evidence

1. **Optical resolution / effective NA (strong).** The controlled evidence is consistent:
   - [8] donor-held-out CD4/CD8, 59 -> 69 -> 76 % for NA 0.4 -> 0.8 -> 0.9;
   - [11] blur ablation 300 -> 600 nm "significantly reduces" subtyping;
   - [5] authors' hypothesis.

   Lymphoid subsets differ in sub-micron structure: cytoplasm amount, nuclear folds, nucleoli and NK granules ([6] cites these).
   BSCCM at 0.5 NA corresponds to the bottom of the [8] curve, where CD4/CD8 is near chance. B vs T is easier than CD4 vs CD8 in
   every study ([6] F1 .78; [9] ~90 % vs ~84 %).

2. **Physical contrast that encodes composition (strong for some pairs, not universal).**
   - 3D RI and dry mass/protein density [9][10]: 3D vs 2D phase +16.6 points, random split.
   - Metabolic or temporal signals [11][12][13]. Metabolic imaging resolves NK and monocytes but not B vs CD4 vs CD8 in PBMC
     [12]; static UV does not resolve CD4/CD8 [11].
   - None of these contrasts is available in BSCCM default data. The 3D/RI benefit at 0.5 NA is uncertain (axial ~4-5 um).

3. **Labelled examples per rare subset (strong, and directly actionable).** [4] shows 2.7x to 10x recall gains for NK/CD8/B from
   balanced training, saturating at ~3,200 cells per type. [7] attributes basophil misses to small counts. Dim/rare markers in
   every IFC study ([4][7][15]) land at r ~.4-.6 or with low recall.

4. **Illumination diversity / information content (moderate).** Information estimates track marker-prediction accuracy on BSCCM
   itself [2]. Learned illumination gives +5-10 % [17]; extra DPC-type contrast +~9 % [18]. In-project K014 already shows
   brightfield alone carries 92 % of the signal and CD123 is darkfield-borne.

5. **Model family / scale / SSL (weak).** Engineered features beat DL [5]; SSL is within 2 % of supervised [20]; a CNN+ViT
   hybrid adds +.025 r over regression-only [3]. No study shows model scale unlocking lymphoid subsets.

6. **Donor shift (small for major classes, unknown for subsets).**
   - Ghost cytometry: intra- vs inter-donor macro-F1 .911 vs .904 [7].
   - Leave-one-donor-out B vs T: .78 [6].
   - DeepIFC tested on held-out samples [4].
   - No study isolates donor shift for CD19/CD56/CD123/HLA-DR regression.

7. **Context (none found).** No label-free leukocyte study reports neighbourhood or field context helping subset prediction. In
   BSCCM, context mostly carries slide-position artefacts (K007).

**Cross-study ceiling band for morphology-poor markers.** Label-free r on dim/rare markers is .38-.61 (CD34 .38, CD8 .41, FcgR
.50, CD56 .53, CD19 .61), even at 60x with dedicated per-marker fluorophores and 6 donors.

What hitting that band would mean for us:
- Matching DeepIFC on CD19 and CD56 (about +.14/+.33 and +.12/+.14 in-distribution) would add about +.025-.03 to the 11-target
  mean.
- Getting there would need higher-NA optics and per-marker labels, which BSCCM does not provide.
- The literature therefore supports the N038 view: .67 is at or above what this optical system and labelling allow, unless the
  four drag targets are label-limited and can be cleaned or completed.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Ideas for our problem (per-cell information for lymphoid subsets / dim markers)

GPU estimates use project timings: one GOAL base model takes ~11-28 min on 1 H100 (N037). All runs are inductive (no test-draw
images), 1 seed, screen only.

### Idea 1: per-target label-count learning curve, then inductive label completion for CD19/CD56/CD123

**Mechanism:** supervision count per rare subset [4][7]. Each of these targets is labelled only on its own single-stain slides
(K002). CD19 has 12,768 / 7,736 stained cells (batch 0 / 1); CD56 14,212 / 23,141; CD123 18,224 / 30,890 [N001 audit]. If B, NK
and basophil/pDC cells are a few % or less of leukocytes, positives per draw are O(10^2-10^3). DeepIFC's B/NK recall rose 5 ->
54 % and 25 -> 67 % when more positives were seen, saturating at ~3,200 per type.

**Why it moves the bottleneck:** it acts in-distribution, where the four drag targets are already low (D1b CD19 .28/.47, CD56
.41/.39). It is the only listed lever that adds per-cell supervision without new acquisition.

**Build after a positive screen:** inductive self-training on the training draw only.
- The GOAL model pseudo-labels CD19/CD56/CD123 on all ~150k training-draw cells, i.e. all slides of that draw.
- A student trains on measured + pseudo labels, confidence-weighted and gated by the lineage heads; for example CD19
  pseudo-labels only on cells the model calls lymphoid.
- Variant: use grp:CD3/19/56 on all-antibody slides as a weak constraint on CD19 + CD56 given the predicted CD3. Exclude batch-1
  all-antibody cells for CD19 (35 % dose [1]).

**Expected gain (reasoned):** if the 50 % -> 100 % step gains Δr >= .03 on CD19/CD56/CD123, doubling effective positives
plausibly adds +.03-.05 on those three targets, i.e. +.01-.015 on the 11-mean. Pseudo-labels are noisier than real ones, so take
the lower half. If the curve is flat, the lever is dead.

**Cheapest falsifier:**
- Mask the CD19, CD56 and CD123 labels to 25 % and 50 % of their training cells. Keep the images, so the trunk sees the same
  data.
- Run b1to0 + b0to1, base model only: 2 levels x 2 directions = 4 runs, about 1.5-2 GPU-h. The 100 % point already exists
  (N031/N037).
- Also score the FOV in-distribution split on the same cells.
- **Kill:** 50 % -> 100 % Δr < .01 on all three targets.

**Build cost:** label-mask flag about 1 h; self-training stage about 1 day.

### Idea 2: retrain on low-pass-filtered inputs to test whether CD19/CD56/CD123/HLA-DR use high spatial frequencies

**Mechanism:** effective optical resolution [8][11][5]. N026 tried zoom2 (+.002 single) and no-maxpool (hurts). Upsampling
cannot add optical bandwidth, so those arms cannot tell "high-frequency information absent" apart from "present but unused".
Retraining on blurred inputs, as in the deep-UV paper [11], can.

**Why it moves the bottleneck:** it decides between image-limited and label-limited (diagnosis hypotheses a vs b) for exactly the
four targets.
- If blurring costs them a lot but costs lineage little, the subsets ride on the finest detail at 0.5 NA. The remaining in-data
  levers are then high-frequency-preserving encoders: a separate stride-1 stem for the darkfield LED group (CD123 is darkfield-borne,
  K014), a 3x-zoom ensemble member, or patch-4 ViT.
- If nothing drops, resolution work is closed for this dataset. The literature then predicts subsets at 0.5 NA are near the floor
  ([8] 20x/0.4: 59 % on CD4/CD8).

**Expected gain:** none directly; it is a decision test. If positive, a darkfield-group high-resolution stem plus a 3x member
might add +.01-.02 on CD123/CD56 (+.005 on the mean). Modest, because N026 already captured the easy part.

**Cheapest falsifier:**
- Train the GOAL base recipe on inputs low-pass filtered to half the spatial cutoff (Gaussian or OTF truncation, applied to
  train and test alike).
- Run b1to0 + b0to1 + one FOV in-distribution run: 3 runs, about 1-1.5 GPU-h. Report per-target Δr.
- **Image-limited if** Δr <= -.05 on >= 2 of the 4 drag targets and > -.02 on CD14/CD16/CD45.

**Build cost:** one transform, about 1 h.

### Idea 3: BSCCM-coherent synthetic-aperture probe (566 single LEDs vs the 22 multiplexed patterns, same cells)

**Mechanism:** angular diversity, synthetic NA and 3D/RI [2][9][10][8]. Fourier ptychography / intensity diffraction tomography
can only be reconstructed from single-LED data. BSCCM-coherent has 4,304 batch-1 cells, all-antibody or unstained only [1].

**Why it moves the bottleneck:** it is the only way, without new acquisition, to test whether the multiplexed 22-pattern design
discards subset information, i.e. whether the GOAL ceiling is set by the hardware.
- Arm (i): images synthesized by summing single-LED images into the 22 default patterns, which carries the same information as
  the GOAL input.
- Arm (ii): the full 566-image stack, channel-compressed with 1x1 convolutions, or an FPM high-resolution complex-field
  reconstruction.
- Same small encoder, FOV-grouped 5-fold CV.
- Targets: CD16 within lymphocytes (NK proxy, since CD56dim NK are CD16+), lymphoid vs myeloid, grp:CD3/19/56. Individual
  CD19/CD56 do not exist in coherent data, and pooled CD123/HLA-DR/CD14 fails QC in batch 1 (K019).

**Expected gain:** a diagnostic only; it does not move the GOAL metric directly.
- (ii) - (i) >= +.05 within-lymphocyte r: the default acquisition loses subset information, .67 is hardware-capped, and future
  acquisition should keep single-LED data.
- About 0: the angular-information route is closed.
- Prior from [8]/[10]: positive at high NA; at 0.5 NA objective + dome LEDs, uncertain.

**Cheapest falsifier:** two arms x 5 folds on 4.3k cells, small encoder: about 2-4 GPU-h. Paired bootstrap on the same cells.
With ~860 test cells per fold, the SE of a paired Δr is about .02-.03.

**Build cost:** 1-2 days: download the coherent subset, synthesize patterns, optionally implement multiplexed FPM. FPM
reconstruction is CPU-heavy; batch it on preempt_cpu.

### Idea 4: per-target label-reliability ceiling from the unstained-slide noise floor (C_max, Ounkomol [19])

**Mechanism:** label noise caps r (r_obs <= r_true x C_max).
- BSCCM's own authors flag the pooled CD123/CD14/HLA-DR channel as "too noisy", the BV510 sharing of CD3/CD19/CD56,
  underdetermined unmixing, and 35 % CD19 dose in batch-1 all-antibody cells [1].
- The diagnosis already lists a reliability falsifier. The literature supplies a concrete estimator.

**How:** push unstained-slide cells (16,338 in batch 0) through the identical unmixing + detrend + per-slide z pipeline to obtain
each target's noise-floor variance. Then:
- SNR_t = (Var_stained,t - Var_unstained,t) / Var_unstained,t
- C_max,t = sqrt(SNR_t / (1 + SNR_t))

Cross-check with a split-half estimate (disjoint emission channels or a pixel checkerboard of the same cell's fluorescence crop).
Report disattenuated r / C_max per target.

**Why it moves the bottleneck:**
- If C_max(CD19/CD56/CD123/HLA-DR) is ~.5-.7, our in-distribution r is already 50-90 % of the ceiling. The target .67 is then
  label-capped and must be re-scoped, and label work (the open reliability-weighted-loss idea) becomes the only lever.
- If C_max >= .9, the labels are reliable and Ideas 1-3 are where the gains are.

**Expected gain:** decision value only.

**Cheapest falsifier:** CPU only, 0 GPU-h, minutes on existing targets_v5 arrays.

**Build cost:** about 2-4 h.

### Idea 5 (lower priority, draw shift only): physics-calibrated photon / per-LED gain / defocus augmentation

**Mechanism:** acquisition invariance. BSCCM documents per-channel intensity drift between slides and batches [1]. Pinkard et al.
equalize photon counts by adding simulated shot noise [2]. Defocus changes LED-array images in an angle-dependent way (digital
refocusing works by shifting single-LED images; Phillips et al., PLoS One 2015, doi 10.1371/journal.pone.0124938). Augmentations:
- independent per-LED-group gain jitter;
- Poisson noise matched to the photon-count range of both draws;
- small pattern-specific defocus kernels, calibrated from per-FOV z.

**Why:** this targets only the .016 draw-shift part. Per D1b it can reach ~.658 single / ~.668 ensemble at best; it does not add
per-cell information.

**Expected gain:** +.003-.008 on the GOAL. HLA-DR, CD19 and CD56 carry most of the shift.

**Cheapest falsifier:** base + zoom2 x 2 directions, about 1.5 GPU-h. **Kill** if < +.004.

**Build cost:** about 0.5-1 day. First check which of these augmentations the SSL(aug) / fine-tune recipe already applies.

**Not recommended on literature grounds:**
- Larger backbones or other SSL objectives for subsets ([5][20]).
- Mixture-density heads as a metric lever. The GMM heads in [2] are for NLL/information estimation; the mean of a calibrated
  mixture equals the MSE optimum, so no r gain is expected.
- Spectral, autofluorescence or temporal channels ([11]-[14]). They are effective but excluded by the input rules and the
  no-acquisition ruling.

---------------------------------------------------------------------------------------------------------------------------------

## 6. Caveats and gaps

- **Few studies regress continuous dim-marker intensities from label-free images.** Most classify sorted or gated populations.
  Sorted-population designs ([9][10][22], partly [8]) risk sample confounds; only [4][6][8] hold out donors or samples.
- **DeepIFC does not define its r** (per-cell intensity vs pixel-wise). The comparison to our per-cell r is approximate.
- **[2] reports no BSCCM per-marker numbers in text.** [3] does not say whether its BSCCM split holds out a batch.
- **Not found:** any label-free study of HLA-DR or CD123 (pDC) intensity; any study of context/neighbourhood for leukocyte
  subsets; any donor-shift analysis specific to CD19/CD56 regression. The searches covered arXiv (BSCCM, label-free +
  lymphocyte/leukocyte/LED array) and Europe PMC (2025-06 to 2026-12 and all-years queries on RI/QPI + NK/B cells, HLA-DR +
  label-free).
- **The cell-type fractions used in Idea 1 are assumptions, not data.** "A few % B/NK, ≲1 % basophil/pDC" is my reasoning, not
  a cited number. Compute the actual positive counts per target and draw from targets_v5 before running the falsifier.
