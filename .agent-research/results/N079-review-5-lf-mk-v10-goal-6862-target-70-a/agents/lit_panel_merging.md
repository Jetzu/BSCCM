# Lit panel: per-cell full-profile imputation when each sample measures one exploratory marker on a shared backbone (N079)

Literature tools worked (WebSearch + WebFetch + Europe PMC full text). The quotes are verbatim: full-text greps (Europe PMC XML) or
abstracts (Europe PMC API) where marked [FT]/[ABS]; quotes marked [WF] come from WebFetch extraction (the extractor caps quotes at
about 125 characters).

## Mapping to BSCCM
Infinity Flow: backbone = about 14 shared antibodies in every well, plus 1 exploratory PE antibody per well, with one regressor per
exploratory marker. BSCCM: backbone = 24 LED-array channels (the image), plus 1 single-stain marker per slide, plus 4 pooled
channels on 'all' slides. In both, no cell has two exploratory markers measured. The structure is the same; our "backbone" is an
image representation, not antibodies.

## (1) Model classes
- **Infinity Flow (Becht 2021, Sci Adv; https://pmc.ncbi.nlm.nih.gov/articles/PMC8457665/)**
  - Per-marker regressors on the backbone: "One regression model per algorithm and Infinity antibody is fit." [WF]
  - Algorithms: XGBoost, SVM, NN (keras), and LASSO on polynomials. "XGBoost was more accurate than other nonlinear models for
    virtually any marker-algorithm pair" [FT].
  - Linear models do much worse: "linear model performed much worse (median AUC of 0.71)" vs nonlinear "median AUC between 0.88
    and 0.89" [WF].
- **pyInfinityFlow (Ferchen 2023, Bioinformatics; https://pmc.ncbi.nlm.nih.gov/articles/PMC10166583/)**: also XGBoost.
  "impute new surface markers into an existing flow cytometry backbone panel using XGBoost regression" [WF]
- **kNN / copy-from-neighbours panel merging**
  - CyTOFmerge (Abdelaal 2019; https://pmc.ncbi.nlm.nih.gov/articles/PMC6792069/): "imputed by taking the median values of those
    markers from the k-most-similar cells", with k = 50 best [WF].
  - CytoBackBone (https://pmc.ncbi.nlm.nih.gov/articles/PMC6792066/): "based on nearest-neighbor imputation, but introduces the
    notion of acceptable and non-ambiguous nearest neighbors" [ABS].
  - cyCombine (Pedersen 2022, Nat Commun; https://pmc.ncbi.nlm.nih.gov/articles/PMC8971492/): SOM clusters on the shared markers,
    then a "multi-dimensional kernel density estimate" draw [WF].
- **Joint latent (multi-output) models**
  - CytoVI (Ingelfinger et al., bioRxiv Sep 2025; https://www.biorxiv.org/content/10.1101/2025.09.07.674699.full): a totalVI-style
    mask; the encoder sees the shared markers and "the generative network reconstructs the union of all markers present in the
    antibody panels" [WF].
  - CyStainer (Ivanov et al., bioRxiv 30 Jun 2026; https://sciety.org/articles/activity/10.64898/2026.06.30.735235): a transformer
    VAE that imputes "without a set of shared backbone markers" and is benchmarked on InfinityFlow data. Only the abstract is
    available, with no numbers.

## (2) How joint / co-expression correctness is validated; accuracy; failure modes
- **Infinity Flow**
  - Validation is per marker on held-out cells of the same well: "randomly split into a training set (50% of the events)" [WF].
    The metric is AUC against manual gates, "within a range of AUC from 0.72 to 0.98" [WF].
  - Cross-well generalisation is tested by treating backbone markers as fake exploratory markers: "rotating out Backbone markers
    one at a time and using the remaining Backbone markers to predict the intensity of the left-out marker... (median Δ R 2 from
    −0.015 to −0.012)" [FT].
  - Joint co-expression was checked only with a NEW wet experiment: "Staining with these markers in an independent flow cytometry
    experiment validated the predicted coexpression patterns for these markers (fig. S10)" [FT].
  - Failure modes:
    - "The expression of an Infinity marker requires a unique signature in the Backbone data to be accurately imputed. A TCR Vβ7 +
      T cell, however, resembles any other T cell" [FT]
    - "Markers restricted to rare cell populations, in general, gave lower AUC values" [FT]
    - "predictions include a significant and underdispersed contribution from nonspecific signal" [FT]
    - "the Backbone is an information bottleneck in MPC assays" [FT]
- **Mocking et al. 2023 (Cytometry A 103:818; DOI 10.1002/cyto.a.24774)**
  - An independent benchmark of Infinicyt, CyTOFmerge, CytoBackBone and cyCombine on split panels with known truth [ABS]:
    "CyTOFmerge showed the most accurate approximation ... with a mean F-score between 0.53 and 0.87 when retrieving cell
    populations"; "Performance remained inadequate for all methods, with only limited similarity at the cell level."
  - This is the clearest statement of the "cluster-level correct, cell-level not" failure.
- **CyTOFmerge**
  - Validated at the population level: "imputed expression profiles of the 121 populations are remarkably similar (average
    correlation of 0.998)" and "adjusted Rand-index of 0.81" [WF].
  - Per-cell correlation was dropped: "this approach turned out not to be appropriate for our data" [WF].
  - Accuracy depends on how many backbone dimensions are informative: "performances improve when the number of shared markers
    increases ... saturate at m = 16" [WF].
- **cyCombine**: "imputed marker expressions should generally only be used for visualization purposes" and "we do not recommend
  basing differential expression analyses directly on imputed values" [WF].
- **CytoVI**
  - Uses semi-synthetic held-out markers: "Lineage markers such as CD4, CD14 or CD3 were imputed with high accuracy (Pearson's r >
    0.9), while Ki67 or CD103 achieved lower imputation results" [WF].
  - Its uncertainty "strongly correlated with the" imputation error [WF].
- **Over-correlation failure (joint model)**: HistoPlexer (Nat Mach Intell 2025; https://pmc.ncbi.nlm.nih.gov/articles/PMC12364712/)
  - Validates co-expression with marker-pair Spearman correlations of prediction vs truth.
  - "for some protein pairs (for example, CD3:CD8a and CD20:CD3), the SCC in MP exceeds that of the GT ... overestimated
    co-localization patterns" [FT].
- **Bottom line for (2)**
  - No paper validates joint correctness against truly co-measured exploratory markers at scale. The checks used are:
    - held-out cells of the same well (marginal only);
    - leave-one-backbone-marker-out;
    - population or gating F-score;
    - small wet-lab co-stains.
  - Reported cell-level accuracy is modest and depends on the marker.
  - The dominant failure is that accuracy is set by the backbone: imputation is cluster-level, and markers without a backbone
    signature (or on rare cells) fail.

## (3) Do joint (multi-output) models beat per-marker regressors?
- **For: HistoPlexer (H&E→IMC, all markers co-measured in training).**
  - "The MP setting consistently outperforms SP across all methods" [FT].
  - "The MP model achieves an MSE approximately an order of magnitude lower than that of the SP model" on marker-pair correlations [FT].
  - Mechanism: "learning intermarker correlations ... and leveraging auxiliary morphological information from abundant markers" [FT].
  - Caveat: it needs co-measured targets, which BSCCM does not have.
- **Against or neutral**
  - CITE-seq protein-imputation benchmark (Li et al., Genome Biol 2025-03; https://pmc.ncbi.nlm.nih.gov/articles/PMC11881419/):
    across 12 methods, "Seurat v4 (PCA) and Seurat v3 (PCA) demonstrate exceptional performance" [ABS]. These are kNN/anchor
    transfer methods, which beat the deep joint models (totalVI, sciPENN) [ABS].
  - The Mocking benchmark winner is kNN (CyTOFmerge).
  - CytoVI beats only cyCombine and kNN, not per-marker XGBoost [WF; comparison absent].
- **Transfer evidence for shared trunks**: in silico labeling (Christiansen 2018, Cell;
  https://research.google/blog/seeing-more-with-in-silico-labeling-of-microscopy-images/): "new labels can be learned with
  minimal additional training data" [WF].
- **Project check**: the project's own joint head (N062 panel-matrix latent) was flat. Pooled outputs gained .015 to .031, bought
  with -.01 to -.02 on singles. This is consistent with the literature: joint heads add nothing without co-measured targets.

## (4) Label-free image backbones
- **Reflectance microscopy (bioRxiv 2020.07.31.231613; https://www.biorxiv.org/content/10.1101/2020.07.31.231613v1.full)**
  - Per-marker networks: "multiple DL models are independently trained for individual IF labels".
  - Single-stain batches are combined: "digital multiplexing by feeding the same reflectance input to each network" [WF].
  - This is the BSCCM design. The combined predictions were not validated against co-stains beyond one Hoechst/EdU pair [WF].
- **DeepIFC (Timonen 2023, Cytometry A; DOI 10.1002/cyto.a.24770)**: BF/DF → fluorescence images; types cells by thresholding the
  predicted channels. "rarer cell types are predicted well when a balanced data set is used to train the model" [ABS].
- **Nazir & Behera, arXiv 2605.14717 (May 2026, ICPR 2026; BSCCM)**: a multi-task model (WBC class + protein regression) on DPC.
  "a 0.72 Pearson correlation for CD16 expression regression on BSCCM" [WF]. No joint/co-expression validation and no
  single-task ablation on the abstract page.
- **Not found**: no 2025-26 paper decodes a joint multi-marker profile from single-stain label-free data with a co-expression
  validation. CyStainer (2026) is the closest, but it works on cytometry and not on images.

## Ideas for BSCCM (declined.md respected: no exclusivity/lineage priors in training; validation checks only)

**A. Joint-correctness card (Mocking / HistoPlexer style).**
- Mechanism: on held-out 'all'-slide cells, compare three things between the measured pooled channels and the composed predictions:
  - the 4x4 correlation matrix (HistoPlexer pair-SCC MSE; detects over-correlation from regression to type means);
  - a within-type-gate correlation (cell-level vs cluster-level; N022 within-class r was only .15 to .22);
  - gating F-score (Mocking; F1 of populations from predicted vs measured gates).
- Add ensemble-variance vs |error| calibration (CytoVI) and exclusive-pair double-positive rates as checks only.
- Why it helps joint decoding: N079 showed per-channel composition r of .778. It did not show whether pairwise structure and
  within-type variation are right, which is the failure the literature reports.
- Gain: no GOAL change. It is the evidence base for any joint claim to the user.
- Falsifier: predicted pair-correlations within ±.05 of measured, and within-gate r > .3. If not, the predictions are only
  cluster-level.
- Cost: 0 GPU-h, CPU under 1 h, about half a day to build.

**B. Co-measured external cells as auxiliary multi-output supervision (HistoPlexer MP > SP; ISL transfer).**
- Mechanism: a shared trunk with a second stem for an external dataset where all markers are on every cell (e.g. S-BIAD452).
  Joint heads are trained only where co-measured; BSCCM single-stain heads are unchanged.
- Why it helps: the literature's joint gains come only from co-measured targets ("auxiliary morphological information from
  abundant markers"). This is the only route that adds genuine joint observations. It also adds positive events for CD19 and CD56
  (Infinity: rare-population markers have the lowest AUC).
- Expected gain: +.003 to .01 GOAL, if cross-modality transfer is about 25% efficient given K033 (+.015 to .02 per data doubling).
- Falsifier: 1 seed x 2 directions vs the existing control. Kill if under +.0054 overall and CD19/CD56 gain under .02.
  About 2 to 4 GPU-h, H100 (r50 at about .28 GPU-h per run plus the external epochs).
- Cost: 1 to 2 days for data plus the stem.

**C. Anchor/kNN imputation in the learned embedding (CyTOFmerge k = 50; Seurat-PCA won the 2025 benchmark).**
- Mechanism: on frozen trunk embeddings, predict each marker as the median over the 50 nearest training-draw cells of that
  marker's slides. Predict pooled profiles by copying from real 'all'-slide neighbours, which are jointly coherent by construction.
- Why it helps: a different inductive bias from the heads (non-parametric, no shrinkage to a regression mean). It also gives a
  real co-measured profile per cell for the pooled channels.
- Expected gain: +0 to .004 as a blend (the ensemble is saturated at +.002 per member).
- Falsifier: per-marker blend gain under .0027 (noise), reached at 0 GPU-h (CPU FAISS; about 0.1 GPU-h for embedding extraction).
- Cost: about 2 h.

**D. Infinity-style leave-one-marker-out transfer test (ISL "new label from minimal data").**
- Mechanism: drop one weak marker's slides from training, then fit a linear probe on 1 to 10% of its cells.
- Why it helps: it measures whether the trunk encodes a shared cell state that serves markers never co-measured. That is the
  premise of decoding profiles from single stains.
- Falsifier: probe r at 10% under 0.8 x the full-head r means the representation is marker-specific, so joint decoding rests on
  per-marker heads only.
- Cost: about 1 GPU-h (2 retrains), half a day.
