# N088 lit B: decoding many targets when each cell has one (or a few) observed, and learning the joint target structure

Agent B, 2026-11-24. Scope: multi-output regression with missing labels, output priors / structured prediction, single-cell protein
imputation and panel merging, aggregate (pooled / set-level) supervision, distributional heads, and group-correlated label noise.
Read first: context.md, the ideas/declined lists of N047/N059/N066/N079, N079 agents/lit_panel_merging.md, N080/N081/N087 records.

How the sources were checked (no citation is invented; each was found by WebSearch and its title, venue and year confirmed):
- [FT] = full text read via Europe PMC XML.
- [WF] = full paper (ar5iv/PMC) read through WebFetch; numbers come from that extraction.
- [ABS] = abstract or proceedings page only.
- [via N079] = quote verified by the N079 literature agent (agents/lit_panel_merging.md); not re-read here.
- [derived] = my own reasoning, not a literature claim.

Our setting in one line:
- Inputs: LF image features. Outputs: 13 targets (8 singles, 4 pooled, autofluorescence); GOAL = 11 of them.
- Single-stain cells observe 1 target. 'All'-slide cells observe 4 pooled channels, where one dye carries the sum CD3+CD19+CD56 and
  another CD123+HLA-DR+CD14.
- No cell has 2 singles (K045).
- External S-BIAD452 measures 9 markers on every cell. 5 overlap with ours: CD3, CD19, CD56, CD14, CD45.
- N080 sandbox: single .526, full .596, type oracle .814, within-type r .15-.20. N082: GOAL is label-limited.

---------------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **[derived] Two kinds of joint structure; only one can move the GOAL.** Write y = f(x) + e.
   - **(i) Function-level coupling.** The f_k share structure: low rank, shared cell states.
     - This can raise per-marker r, but only if it adds information the shared trunk does not already have.
     - Bonilla et al. 2007 prove that in a noise-free block design "a cancellation of inter-task transfer occurs".
     - A shared-trunk multi-head net already implements function-level coupling. This is consistent with the flat panel-latent head
       (N062 .6704 vs .6706) and predicts a flat N087 rank-k sweep.
   - **(ii) Residual coupling, Cov(e_j, e_k | x).** Example: CD4 vs CD8 among look-alike T cells.
     - It cannot change E[y_k|x] (tower property), so it does not move per-marker Pearson.
     - It decides co-expression and gating, i.e. the user's multi-marker priority. N080 shows single-stain training loses it
       (CD4-CD8 in T: true -.81, full -.36, single -.07).
2. **The one information channel still unused is the observed marker of each training cell.**
   - Today y_j trains only head j.
   - Through a cell-state model it also says something about the cell's other markers: an observed CD3-high lymphocyte is not a B or
     NK cell. Using this creates supervision for the label-starved heads (N082: labels, not steps, limit the GOAL).
   - Partial-label literature reports large gains from exactly this:
     - Cole 2021, single positive per image: VOC 86.5 vs 86.7 mAP with full labels, linear setting.
     - Chen 2022 (SST): +5.6 mAP at 10 % known labels.
     - Li 2022 (MTPSL), one task label per image: NYUv2 segmentation 25.8 -> 30.4 (full labels 37.0); on Cityscapes it beats full
       labels.
   - Hard caveat from SST: completions must be image-specific. Dataset-level co-occurrence statistics HURT (55.3 vs 64.1 mAP at
     10 %). For us, a global residual covariance is the wrong prior; a state-conditional (mixture) prior is the right one.
3. **External joint data is the only source of the cross-marker emission structure.** BSCCM cannot identify it (K045).
   - The marker-matrix cell-typing models (CellAssign, Astir) are the template: a probabilistic cell state plus a marker-by-state
     emission matrix.
   - Here the matrix would be LEARNED from S-BIAD452, not hand-coded. Hand-coded exclusivity, as in the exclusion loss of Shi 2021,
     stays declined.
4. **Pooled channels have an exact likelihood.**
   - Gaussian closure under sums (Zhang 2020, aggregate observations) and the "marginal loss" of Shi 2021 (merged classes =
     summed probabilities) give the right observation model for a dye that carries CD3+CD19+CD56.
   - The sum is taken in linear intensity, then mapped to asinh.
   - N062's panel head was a linear-in-asinh approximation.
5. **Domain ceilings: expect small per-marker gains, real joint-coherence gains, and a risk of over-correlation.**
   - Single-cell imputation: per-protein accuracy is set by anchor/kNN methods. Seurat v4/v3 (PCA) top a 12-method benchmark (Li 2025).
   - Joint generative models (totalVI, sciPENN, CytoVI) win on uncertainty and integration, not clearly on per-protein r.
   - Imputation is cluster-level: Mocking 2023 F 0.53-0.87, "limited similarity at the cell level".
   - A marker without a backbone signature is not imputable: Infinity Flow's TCR Vβ7.
   - Joint image models over-correlate marker pairs (HistoPlexer CD3:CD8a).
6. **Not proposed** (section 3, end): reduced-rank / task-covariance heads (N087-A covers them), output flows (N087-E covers them),
   label-proportion learning (set-level is known to fail here), and group-noise losses (N061 flat).

---------------------------------------------------------------------------------------------------------------------------------

## 1. Annotated references (35, grouped)

### A. Multi-output / multi-task regression with missing labels, low-rank output coupling, task relations

**A1. Bonilla, Chai, Williams. "Multi-task Gaussian process prediction." NIPS 2007.**
- URL: https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction [ABS]
- What it does: a shared input kernel times a free-form task covariance.
- Key claim (abstract): "under the assumption of noise-free observations and a block design, predictions for a given task only depend
  on its target values and therefore a cancellation of inter-task transfer occurs."
- Mapping: our design is the opposite extreme, heterotopic (one task per input), so transfer is possible in principle. But it flows only
  through function-level coupling, which a shared trunk already provides. This is the theory behind "joint heads add ~0" (N062).

**A2. Yu, Jain, Kar, Dhillon. "Large-scale multi-label learning with missing labels" (LEML). ICML 2014, PMLR 32:593-601.**
- URL: https://proceedings.mlr.press/v32/yu14.html [ABS]
- What it does: low-rank linear model Z = W H^T trained by ERM on observed entries only.
- Key results: bounds favour trace-norm over Frobenius regularisation; scales to over 200k labels.
- Mapping: = N087-A's reduced-rank decoder on frozen features. Nothing new to run.

**A3. Luo, Liang, Li, Wang, Zhang, Dey, Chen. "Leveraging mixed and incomplete outcomes via reduced-rank modeling" (mRRR). J Multivariate Analysis 167:378-394, 2018.**
- URL: https://ideas.repec.org/a/eee/jmvana/v167y2018icp378-394.html [ABS]
- What it does: reduced-rank regression for partially observed outcomes of mixed type (exponential family), so tasks share a low-dim
  subspace.
- Mapping: the classical statistics version of the masked multi-head loss plus a rank constraint. Same conclusion as A2.

**A4. Long, Cao, Wang, Yu. "Learning multiple tasks with multilinear relationship networks" (MRN). NeurIPS 2017.**
- URL: https://proceedings.neurips.cc/paper/2017/hash/03e0704b5690a2dee1861dc3ad3316c9-Abstract.html [ABS]
- What it does: tensor-normal priors over task-specific layers learn the task covariance; the earlier layers are shared.
- Mapping: a learned task-covariance prior on our 13 head rows. It is function-level coupling: low expected value given A1 and N062.

**A5. Tai, Lin. "Multilabel classification with principal label space transformation" (PLST). Neural Computation 24(9):2508-2542, 2012.**
- URL: https://dblp.org/rec/journals/neco/TaiL12 [ABS]
- What it does: regresses onto the top principal directions of the label matrix and decodes back.
- Mapping: exactly N087-B's "linear (PCA) prior". For us the PCA basis must come from external joint labels, because BSCCM's label
  matrix is never jointly observed.

**A6. Hsu, Kakade, Langford, Zhang. "Multi-label prediction via compressed sensing." NIPS 2009.**
- URL: https://papers.nips.cc/paper/3824-multi-label-prediction-via-compressed-sensing [ABS]
- What it does: predicts a few random linear projections of a sparse label vector, then recovers it by sparse decoding. The number of
  regressions grows only logarithmically in the number of labels.
- Mapping: our pooled dyes are fixed linear projections (sums) of the marker vector. Recovering the components from a sum needs a
  sparsity/structure prior (one lineage active per cell). Here that prior must be learned from external data, not imposed by hand.

### B. Partial / single-label supervision of many outputs (closest ML analogues of "one marker per cell")

**B1. Cole, Mac Aodha, Lorieul, Perona, Morris, Jojic. "Multi-label learning from single positive labels." CVPR 2021.**
- URL: https://openaccess.thecvf.com/content/CVPR2021/html/Cole_Multi-Label_Learning_From_Single_Positive_Labels_CVPR_2021_paper.html [WF]
- What it does: each image has one observed positive label.
  - EPR constrains the batch-mean predicted label count to the known expected count k.
  - ROLE also learns an N x L matrix of label estimates jointly with the classifier (stop-gradient both ways, 10x LR on the
    estimates).
- Key results (mAP):

  | setting | VOC12 | COCO | NUS | CUB |
  |---|---|---|---|---|
  | full labels, linear | 86.7 | 70.0 | 50.7 | 29.1 |
  | single positive, best method, linear | 86.5 (ROLE) | 66.3 | 49.5 | 18.0 |
  | assume-negative baseline | 84.2 | 62.3 | 46.2 | 17.2 |

  CUB, with about 31 positives per image, is the weak case.
- Mapping: "online label estimation" is the end-to-end form of the completion in P1. The per-image cross-label constraint (EPR) is
  what a learned state prior supplies for markers.

**B2. Chen, Pu, Wu, Xie, Lin. "Structured semantic transfer for multi-label recognition with partial labels" (SST). AAAI 2022.**
- URLs: https://ojs.aaai.org/index.php/AAAI/article/view/19910, arXiv 2112.10941 [WF]
- What it does: an image-specific C x C co-occurrence predictor infers unknown labels from known positives. Pseudo-labels are made
  once the summed co-occurrence passes θ, scheduled from .95 down to .75.
- Key results:
  - MS-COCO at 10 % / 30 % known labels: 68.1 / 75.9 mAP vs SSGRL 62.5 / 73.2.
  - Ablation: using **dataset-level (statistical) co-occurrence instead of image-specific drops to 55.3 / 65.9, below the baseline**
    (false positives where the global correlation does not hold).
- Mapping: completions of a cell's unobserved markers must be conditioned on the cell (image plus observed marker through a state
  posterior), never on a global marker correlation matrix. This is the design rule behind P1.

**B3. Li, Liu, Bilen. "Learning multiple dense prediction tasks from partially annotated data" (MTPSL). CVPR 2022, pp. 18879-18889.**
- URLs: https://openaccess.thecvf.com/content/CVPR2022/html/Li_Learning_Multiple_Dense_Prediction_Tasks_From_Partially_Annotated_Data_CVPR_2022_paper.html, arXiv 2111.14893 [WF]
- What it does: maps one task's prediction and another task's ground truth into a learned joint pairwise space (a FiLM-conditioned
  mapping per task pair) and applies a cosine consistency loss. A regulariser ties the mapping output to the encoder's image features,
  which prevents trivial cross-task maps.
- Key results, "one label per image":

  | dataset | metric | supervised-only | MTPSL | full-label bound |
  |---|---|---|---|---|
  | NYUv2 | seg mIoU | 25.75 | 30.36 | 36.95 |
  | NYUv2 | depth aErr | .651 | .609 | .551 |
  | NYUv2 | normal mErr | 33.7 | 32.1 | 29.5 |
  | Cityscapes | seg mIoU | 69.50 | **74.90** | 73.36 |

- Mapping: the same regime as BSCCM, with no external prior. Caveat: the tasks are dense maps with strong shared geometry (edges); our
  targets are scalars. This is the "no external data" control for P1/P2 (P3).

**B4. Shi, Xiao, Chen, Zhou. "Marginal loss and exclusion loss for partially supervised multi-organ segmentation." Medical Image Analysis 70:101979, 2021.**
- URL: arXiv 2007.03868 [ABS]
- What it does: trains one segmentation network on a union of datasets that each label different organs.
  - The "marginal loss" scores a merged label by the sum of its components' probabilities.
  - The "exclusion loss" uses the fact that organs do not overlap.
- Mapping:
  - The marginal loss is the right likelihood for pooled dyes: the probability or intensity of "CD3 or CD19 or CD56" = a sum over
    components.
  - The exclusion loss is a hand-coded exclusivity prior (declined by the user 11-17). Only learned emissions are allowed.

### C. Output priors and structured output prediction

**C1. Belanger, McCallum. "Structured prediction energy networks" (SPEN). ICML 2016, PMLR 48:983-992.**
- URL: https://proceedings.mlr.press/v48/belanger16.html [ABS]
- What it does: a deep energy E(x, y) over the whole label vector. Prediction = gradient descent on y. Beats the baselines on multi-label
  benchmarks.
- Mapping: a soft, learned joint prior, but it needs joint labels to train the energy. Its only use for us would be an energy trained
  on S-BIAD452 marker vectors and used as a test-time prior. That is heavier than P2's mixture for the same idea.

**C2. Oktay et al. "Anatomically constrained neural networks (ACNN)." IEEE TMI 37(2):384-395, 2018.**
- URL: arXiv 1705.08302 [WF]
- What it does: an autoencoder trained on label maps provides a 64-d code. Training adds λ1 ||f(pred) - f(gt)||², with λ1 = .01.
- Key results (vs the same net without the prior):
  - Endocardium Dice .923 -> .939, Hausdorff 10.3 -> 7.9 mm.
  - Myocardium Dice .773 -> .811.
  - The main effect is fewer artefact-driven implausible outputs.
- Mapping: the soft-prior counterpart of N087-B's hard decoder prior. The "plausibility, not accuracy" effect is what we expect on
  joint coherence. With only one revealed marker, f(gt) is undefined; the soft version would be a reconstruction penalty
  ||ŷ - D(E(ŷ))||² with a frozen external marker AE.

**C3. Lu, Huang. "Structured output learning with conditional generative flows" (c-Glow). AAAI 2020, 34(04):5005-5012.**
- URL: https://ojs.aaai.org/index.php/AAAI/article/view/5940 [ABS]
- What it does: a conditional normalising flow over the output with exact likelihood. Comparable to or better than baselines on 5
  structured tasks.
- Mapping: a flow over the 13-d output needs joint outputs to fit. N087-E already tests a conditional flow-matching decoder (full
  regime / per-target 1-D on BSCCM).

**C4. Graikos, Malkin, Jojic, Samaras. "Diffusion models as plug-and-play priors." NeurIPS 2022.**
- URL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/5e6cec2a9520708381fe520246018e8b-Abstract.html [ABS]
- What it does: a separately trained diffusion prior is combined with a differentiable constraint at inference, with no retraining.
- Mapping: the "external prior plus per-cell likelihood" pattern. The prior would be trained on S-BIAD452 vectors; the likelihood would
  be the image-predicted marginals plus the observed marker.
- For 5-13-d marker vectors a Gaussian mixture is a closed-form prior that is just as expressive in practice ([derived]). That is why
  P1/P2 use a mixture.

**C5. Yeh, Wu, Ko, Wang. "Learning deep latent spaces for multi-label classification" (C2AE). AAAI 2017.**
- URL: arXiv 1707.00418 [ABS]
- What it does: jointly embeds features and labels (deep CCA plus a label autoencoder) and decodes labels from the shared latent. The
  authors say it extends to missing labels without changing the architecture.
- Mapping: the "marker-AE prior" (N087-B) with an image-to-label-latent alignment term.

### D. Distributional / joint heads for multimodal, zero-inflated targets

**D1. Kong, Bai, Lee, Chen, Allyn, Stuart, Pinsky, Mills, Gomes. "Deep hurdle networks for zero-inflated multi-target regression." IJCAI 2020:4375-4381.**
- URL: https://www.ijcai.org/proceedings/2020/603 [ABS]
- What it does: a multivariate probit for presence plus a multivariate log-normal for positive abundance, with the two covariances tied
  by a penalty. Beats baselines on bird and fish multi-species data.
- Mapping: the joint version of our two-part head (N060: twopart .6756 single, flat in the ensemble). Species co-occurrence is
  analogous to marker co-expression. The labels there are fully observed, so it does not solve our missing-label problem by itself.

**D2. Chen, Xue, Gomes. "End-to-end learning for the deep multivariate probit model." ICML 2018, PMLR 80:932-941.**
- URL: https://proceedings.mlr.press/v80/chen18o.html [ABS]
- What it does: a GPU-sampled multivariate probit that learns label correlations end-to-end. Training is at least 10x faster than
  classical MVP, with a better joint likelihood.
- Mapping: a learned residual correlation of binary marker states. With one marker per cell that correlation is not identifiable from
  BSCCM. It would have to come from S-BIAD452 (P2).

(totalVI's protein background/foreground mixture, E1, is the domain example of a per-marker zero-inflation head. Our HL-Gauss and
two-part heads were flat, N060.)

### E. Single-cell protein imputation and joint latent models

**E1. Gayoso et al. "Joint probabilistic modeling of single-cell multi-omic data with totalVI." Nat Methods 18:272-282, 2021.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7954949 [FT]
- What it does:
  - Each protein is a mixture of background and foreground components, weighted by the cell's latent state.
  - It integrates datasets "with and without missing proteins" and imputes the unmeasured proteins.
- Key results:
  - Held-out panel imputation: totalVI's error is better than Seurat v3's "for approximately 68% of the 80 proteins".
  - On PBMC, totalVI and Seurat v3 "performed more similarly, while outperforming cTP-net".
- Mapping: masked joint-latent generative model = our masked multi-head plus a latent state. Its per-protein advantage over
  anchor/kNN is small.

**E2. Lakkis, Schroeder, Su, Lee, Bashore, Reilly, Li. "A multi-use deep learning method for CITE-seq and single-cell RNA-seq data integration with cell surface protein prediction and imputation" (sciPENN). Nat Mach Intell 4:940-952, 2022.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC9979929 [WF]
- What it does:
  - Partially overlapping panels are handled by a "censored" loss computed "only for measured proteins". This equals our masked loss.
  - A quantile head gives uncertainty.
- Key results:
  - 50 % / 80 % prediction-interval coverage 22.1 / 44.6 % vs totalVI 9.8 / 18.3 % (MALT). Both are badly under-covered.
  - Cell-type transfer 83.9 % vs Seurat 4 78.5 %.
  - Correlation/RMSE wins are reported only in figures.
- Mapping: confirms that the masked loss is the field standard. No cross-protein prior.

**E3. Zhou, Ye, Wang, Zhang. "Surface protein imputation from single cell transcriptomes by deep neural networks" (cTP-net). Nat Commun 11:651, 2020.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6994606 [ABS]
- What it does: a multiple-branch DNN (shared trunk plus per-protein branches) trained on paired CITE-seq data, then transferred.
- Key results: beats Seurat v3 for cell types absent from training; worse than a model trained on the same population.
- Mapping: the same architecture as ours (trunk plus per-marker heads). Cross-population transfer loss is the analogue of our
  draw shift.

**E4. Hao et al. "Integrated analysis of multimodal single-cell data" (Seurat v4, WNN). Cell 184(13):3573-3587, 2021.**
- URL: https://pubmed.ncbi.nlm.nih.gov/34062119/ [ABS]
- What it does: weighted nearest neighbours across modalities. Proteins are imputed by averaging neighbours (211k PBMCs, up to 228
  antibodies).
- Mapping: kNN/anchor imputation. N079 idea C and N087-A already run kNN decoders.

**E5. Hao et al. "Dictionary learning for integrative, multimodal and scalable single-cell analysis" (bridge integration). Nat Biotechnol 42:293-304, 2024.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517 [ABS]
- What it does: a multi-omic "bridge" dataset whose cells act as dictionary atoms to map a unimodal query into a reference space.
  Scales to 8.6 M cells, including mass cytometry.
- Mapping: S-BIAD452 as a bridge in MARKER space: predicted overlap markers -> dictionary of jointly measured cells -> unmeasured
  markers (P4).
- The image-space bridge failed: the N081 shared trunk gave -.0057, and LF↔Giemsa retrieval collapses across slides.

**E6. Li et al. "Benchmarking single-cell cross-omics imputation methods for surface protein expression." Genome Biol 26:46, 2025.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11881419 [FT]
- What it does: benchmarks 12 methods on 11 datasets in 6 scenarios.
- Key results:
  - Conclusion: "Seurat v4 (PCA) and Seurat v3 (PCA) demonstrate exceptional performance".
  - Seurat methods lead protein-level metrics except in 2 cross-tissue cases.
  - sciPENN is best for cell-level PCC in most experiments.
- Mapping: on per-marker accuracy, the simplest neighbour-based transfer is as good as deep joint models. Expect GOAL gains from joint
  modelling to be small (consistent with N062 and N080 kNN .507 vs model .526).

**E7. Ingelfinger et al. "CytoVI: Deep generative modeling of antibody-based single cell technologies." bioRxiv 2025.09.07.674699.**
- URLs: https://www.biorxiv.org/content/10.1101/2025.09.07.674699.full.pdf, scvi-tools CYTOVI [ABS + via N079]
- What it does: a totalVI-style masked VAE for flow, mass cytometry and CITE-seq. It "impute[s] missing markers when overlapping
  antibody panels are applied".
- Key results: lineage markers CD4/CD14/CD3 imputed at r > .9; Ki67/CD103 lower; uncertainty correlates with error [via N079].
- Mapping: the closest cytometry joint latent. It needs partial panel overlap, which BSCCM singles do not have.

### F. Cytometry panel merging / backbone imputation, marker-matrix cell typing, multiplexed-image imputation

**F1. Becht et al. "High-throughput single-cell quantification of hundreds of proteins using conventional flow cytometry and machine learning" (Infinity Flow). Science Advances 2021.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8457665 [FT]
- What it does: every well has the same backbone plus ONE exploratory marker, and "one regression model per algorithm and Infinity
  antibody is fit". This is our design, with an image in place of the backbone.
- Key results:
  - Nonlinear models: "median AUC between 0.88 and 0.89"; linear: "0.71".
  - Cross-well generalisation: ΔR² -.015 to -.012.
  - Failure: "The expression of an Infinity marker requires a unique signature in the Backbone data ... A TCR Vβ7+ T cell, however,
    resembles any other T cell".
  - "the Backbone is an information bottleneck".
- Mapping:
  - Per-marker regressors are the field standard and there is no cross-marker model.
  - Our CD19/CD56 among look-alike lymphocytes is the TCR Vβ7 case: if the image lacks the signature, no decoder recovers it. A
    joint prior can only make the profile coherent.
  - Follow-up pyInfinityFlow (Ferchen, Salomonis, Grimes; Bioinformatics 2023, DOI 10.1093/bioinformatics/btad287) keeps XGBoost.

**F2. Abdelaal et al. "CyTOFmerge: integrating mass cytometry data across multiple panels." Bioinformatics 2019, DOI 10.1093/bioinformatics/btz180.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6792069 [via N079]
- What it does: imputes the non-shared markers by the median of the k = 50 nearest cells on the shared markers.
- Key results: population-level profiles correlate .998 and ARI is .81, but "per-cell correlation ... turned out not to be
  appropriate". Accuracy saturates at about 16 shared markers.
- Related: CytoBackBone (Leite Pereira et al., Bioinformatics 2019, DOI 10.1093/bioinformatics/btz212) is NN imputation with
  "acceptable and non-ambiguous" neighbours.
- Mapping: the kNN-in-shared-space baseline (P4 uses it in marker space).

**F3. Pedersen et al. "cyCombine allows for robust integration of single-cell cytometry datasets within and across technologies." Nat Commun 2022, DOI 10.1038/s41467-022-29383-5.**
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8971492 [via N079]
- What it does: SOM clusters on shared markers, then a kernel density draw for the missing markers.
- Key statement: imputed values "should generally only be used for visualization purposes".
- Mapping: a warning label for any joint-profile claim on BSCCM.

**F4. Mocking et al. "Merging and imputation of flow cytometry data: a critical assessment." Cytometry A 103(10):818-829, 2023, DOI 10.1002/cyto.a.24774.**
- [ABS + via N079]
- What it does: benchmarks Infinicyt, CyTOFmerge, CytoBackBone and cyCombine against known truth.
- Key results: CyTOFmerge is best, F-score .53-.87 for populations; "Performance remained inadequate for all methods, with only limited
  similarity at the cell level."
- Mapping: the realistic expectation for within-type joint decoding (N080 within-type r .15-.20).

**F5. Zhang, O'Flanagan, Chavez, Lim et al. "Probabilistic cell-type assignment of single-cell RNA-seq for tumor microenvironment profiling" (CellAssign). Nat Methods 16:1007-1015, 2019.**
- URL: https://www.nature.com/articles/s41592-019-0529-1 [ABS]
- What it does: a cell-state mixture with a marker-by-type matrix and EM fitting; cells that match no type stay "unassigned".
- Mapping: the generative template for P1/P2. Our "marker matrix" is a continuous emission matrix LEARNED from S-BIAD452, never a
  binary hand-coded one.

**F6. Geuenich, Hou, Lee, Ayub, Jackson, Campbell. "Automated assignment of cell identity from single-cell multiplexed imaging and proteomic data" (Astir). Cell Systems 12(12):1173-1186.e5, 2021.**
- URL: https://pubmed.ncbi.nlm.nih.gov/34536381/ [ABS]
- What it does: the same idea for cytometry/IMC, with an amortised recognition network that maps cells to type posteriors. Tested on
  2.1 M cells; more robust to imbalanced proportions.
- Mapping: recognition network = our image encoder; emissions = the state profiles. This is the end-to-end form used in P2.

**F7. Wu, Trevino et al. "7-UP: Generating in silico CODEX from a small set of immunofluorescence markers." PNAS Nexus 2(6):pgad171, 2023.**
- URL: https://pubmed.ncbi.nlm.nih.gov/37275261/ [ABS]
- What it does: 7 markers plus morphology predict a 40-plex panel. A concrete autoencoder picks the 7.
- Key results: mean patch-wise Pearson .534; cell-type F1 .727; generalises across sites.
- Mapping: a "backbone + morphology -> many markers" ceiling of about .5 r. It is trained with all markers co-measured, which we do
  not have.

**F8. Wenckstern et al. "AI-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery" (VirTues). arXiv 2501.06039, 2025 (an ICLR 2025 listing exists; the code repository cites a later journal version, not verified).**
- URL: https://arxiv.org/abs/2501.06039 [ABS]
- What it does: masked autoencoding over marker tokens, so the model handles arbitrary panels. Trained on 15 IMC datasets with 147
  markers.
- Mapping: the panel-agnostic marker-token route for pooling heterogeneous panels. Relevant if more external all-marker sets join
  S-BIAD452. Not needed for 5 shared markers.

**F9. HistoPlexer (Nat Mach Intell 2025; https://pmc.ncbi.nlm.nih.gov/articles/PMC12364712/).**
- [via N079]
- What it does: H&E -> multiplexed IMC.
- Key results: the multi-plex output model beats single-plex in all settings and cuts pair-correlation MSE by about 10x, but it
  "overestimated co-localization patterns" for CD3:CD8a and CD20:CD3.
- Mapping: joint training helps only with co-measured targets. Joint priors risk over-separation, which N080 already sees: CD3-CD19 in
  lymphocytes, all regimes -.83..-.89 vs true -.73.

### G. Aggregate / set-level / proportion supervision

**G1. Quadrianto, Smola, Caetano, Le. "Estimating labels from label proportions." JMLR 10:2349-2374, 2009.**
- URL: https://jmlr.org/papers/v10/quadrianto09a.html [ABS]
- What it does: consistent estimators of instance labels from bag-level proportions.
- Mapping: classical LLP. Context.md: set-level supervision works here only with small sets of the SAME cells, so this is not a GOAL
  route. Its clinical use (Giemsa cohorts) is already in IDEAS.md.

**G2. Zhang, Charoenphakdee, Wu, Sugiyama. "Learning from aggregate observations." NeurIPS 2020.**
- URL: https://papers.nips.cc/paper/2020/hash/5b0fa0e4c041548bb6289e15d865a696-Abstract.html [WF]
- What it does: maximum likelihood with instance-level models when only aggregates are observed. For regression from means, the Gaussian
  "is closed under linear combination given independence", so the loss is MSE between the observed mean and the mean of the
  predictions.
- Key result: comparable to direct supervision on UCI with groups of K = 4 (the table did not extract cleanly; no numbers quoted).
- Mapping: a pooled dye is an aggregate over TARGETS within one cell, not over instances. The same closure gives the exact likelihood
  of a pooled channel as a sum of per-marker intensities in linear space (P2). It needs a model of the raw-intensity scale (sinh of
  the asinh outputs times a learnable dye brightness per marker).

### H. Group-correlated label noise (FOV-level errors shared within an image)

**H1. Wang, Yao, Gong, Liu, Gong, Yang, Han. "Learning with group noise." AAAI 2021, 35(11):10192-10200.**
- URL: https://ojs.aaai.org/index.php/AAAI/article/view/17222 [ABS]
- What it does: the group is correctly linked to a target but individual members may not be ("Max-Matching": train on the most
  confident member).
- Mapping: our structure is different (an additive FOV offset on all members). The mixed-model / ARMED / LMMNN / rmcorr literature was
  covered in N047 (lit_group_noise.md) and N066 (lit_labelnoise.md). Within-FOV and random-effect losses were flat (N061 .6742 /
  .6740).
- New relevance for this topic only: **FOV-level label error is CORRELATED ACROSS TARGETS that share a dye** (CD3, CD19 and CD56 share
  the dim channel; K028 FOV-level variance .34-.49). Any cross-marker conditioning (P1) must therefore not pass one marker's FOV offset
  into another marker's pseudo-label. In P1 this is handled by inflating the observation variance by the FOV-level variance τ_j².

---------------------------------------------------------------------------------------------------------------------------------

## 2. Synthesis for BSCCM

1. **Why heads were flat (N060-N063) and why rank-k should be flat (N087-A).** [derived from A1 + A2/A3 + N062]
   - Masked multi-head training with a shared trunk already does function-level coupling.
   - Low-rank or covariance-tied heads re-parameterise the same coupling and add no information.
2. **Where information can still enter.** Three places. All need a state model whose emissions are learned from all-marker data.
   - (a) The observed marker of every training cell, used for the OTHER markers through the cell's state posterior (B1-B3, F5-F6).
   - (b) External emission structure (S-BIAD452), which BSCCM cannot identify.
   - (c) The pooled sums, through their exact likelihood (B4, G2).
3. **What it can buy.**
   - GOAL: label multiplication for label-starved lymphoid heads (N082: D_data .0155 > D_steps .0115; K033: weak targets +.035 from
     50 -> 100 % labels).
   - Joint decoding: coherent profiles, i.e. restored CD4-CD8 exclusivity.
   - Domain ceilings (E6, F1, F4) say per-marker gains will be modest and cluster-level.
   - SST (B2) says global co-occurrence statistics hurt, so the prior must be state-conditional.
   - HistoPlexer (F9) says watch for over-separation.
4. **One unresolved question decides how much to invest.** Is N080's full-vs-single gap (.596 vs .526) about co-measurement or just
   label count?
   - The full regime has 9x the label entries.
   - A step-matched, label-matched full run (1/9 of the cells, all 9 labels) separates the two (P2a). It costs about 0.5 H100-h with
     the existing `--max-train` flag.
   - Pitfall: `steps = epochs x len(train)` in train_ifc_sandbox.py, so the epochs must be scaled x9. This is the K033 confound again.

---------------------------------------------------------------------------------------------------------------------------------

## 3. Experiment proposals, ranked by expected value for the GOAL and multi-marker decoding

All proposals are inductive:
- Nothing from the held-out draw is used.
- In the sandbox, priors come from the VALIDATION donor only, never the test donor.
- S-BIAD452 is external, so all its donors may be used for BSCCM.
- Emissions are learned from jointly measured data; there are no hand-coded exclusivity rules (user 11-17 ruling; marker-AE/learned
  prior OK'd 11-24).
- Total if every gate passes: about 12 H100-h, under the 25 GPU-h approval line.

### P1 (rank 1). Posterior-state label completion: an EM step that turns each cell's observed marker into supervision for its other markers

**Hypothesis.**
- In the one-marker-per-cell regime, a cell's observed marker carries state information the image teacher does not have (for example
  T vs B vs NK among similar lymphocytes; N080 within-type r .15-.20).
- Pushing it through a state model anchored on S-BIAD452 into image- and state-conditional completed labels for the cell's
  UNOBSERVED markers adds informative supervision where the GOAL is label-limited.
- This uses the ROLE/SST mechanism (B1, B2) and respects SST's rule that there are no dataset-level statistics.
- It differs from the N040 noisy student (+.0103 single / +.0037 ensemble on v5; round 2 worse = confirmation bias). The new signal is
  real data (the observed y_j), not the teacher's own output.

**Recipe.**

0. **State prior (CPU).**
   - Fit a K-component full-covariance Gaussian mixture (K by BIC, about 8-12: neutrophil, eosinophil, monocyte, T4, T8, B, NK, rest).
   - Data: S-BIAD452 standardised all-9 labels. Use all donors for BSCCM; use the validation donor only in the sandbox.
   - Keep per-state mean and variance on the 5 overlap markers.
1. **Emission model in BSCCM target space (13 targets).**
   - Overlap markers: m_sk = a_k μ_sk^ext + b_k, with a per-marker affine fitted on BSCCM (cross-instrument scale).
   - CD16, CD123, HLA-DR, autofluorescence and the 4 pooled channels: free per-state means and variances.
   - Add 2 extra free states (CD123+/basophil-pDC-like, other).
2. **Image prior π_s(x).**
   - A multinomial logistic on frozen features (N087 dumps, jobs 1442342/3: n051_v10_base seed 0, PCA-64) plus the teacher's 13-d
     prediction.
   - Fit by EM on the training draw.
   - E-step: r_is ∝ π_s(x_i) · Π_{j∈O_i} N(y_ij; m_sj, σ_sj² + τ_j²).
     - τ_j² = target j's FOV-level label variance (K028/K041). This makes the update ignore FOV offsets that are shared across the
       CD3/CD19/CD56 dye.
   - M-step: weighted cross-entropy for π; update the emissions.
   - For the student, use CROSS-FITTED features: teacher trained on FOV-half A, features on half B, and vice versa. Otherwise π(x)
     already contains y_j, which double counts.
3. **Completion for unobserved k.**
   - ỹ_ik = ŷ_ik + Σ_s (r_is − π_s(x_i)) m_sk: the teacher's prediction plus the observation-driven state shift. This keeps the
     teacher's within-state signal.
   - Per-entry weight w_ik = 1 − posterior variance / prior variance of y_k.
4. **Student.**
   - The GOAL recipe (train_lf.py, SSL r50, v10, 10 epochs).
   - Loss: masked MSE on real labels + 0.3 · w_ik · (ŷ − ỹ)² on completed entries. 0.3 = N040's pseudo weight.
   - Arms: (i) control (real only); (ii) plain teacher pseudo-labels at the same weight (N040-style; isolates the observation-driven
     part); (iii) completion.

**Gates and kill criteria.**
- **G0 (CPU, sandbox, primary).**
  - Fit the prior on the validation donor.
  - On each test donor, reveal one random marker per cell and complete the other 8 from the N080 single-model prediction plus the
    state posterior.
  - The test truth is used for scoring the completion only; nothing is trained on it.
  - Kill if Δr(completed vs truth) − r(model vs truth) < +.02 averaged over hidden entries, OR < +.03 on the lymphoid hidden markers
    (CD3/CD4/CD8/CD19/CD56).
  - Also run an "overlap5-only emissions" variant (prior knows only CD3/CD19/CD56/CD14/CD45, the BSCCM-realistic case). It must keep
    at least half of the gain.
- **G0b (CPU, BSCCM sanity).**
  - On 'all'-slide cells (4 pooled channels jointly measured), reveal 1 channel, complete the other 3, and compare with the teacher.
  - Cross-fit by FOV.
  - Expected small, because pooled channels mostly encode type (K045). It is a plumbing check, not a kill.
  - Also check that the mixture's marginal for each overlap marker matches the observed BSCCM single-slide histogram (W1). A mismatch
    means the cross-instrument affine failed.
- **G1 (sandbox GPU).**
  - Student trained on completed single-regime labels vs N080 single (.526 mean of 3 donors).
  - Kill if macro r < single + .010, or if the plain-pseudo arm explains ≥ 80 % of the gain.
- **G2 (BSCCM, pre-registered).**
  - Single-model mean of 2 seeds × 2 directions ≥ control + .0054 AND ≥ plain-pseudo arm + .0027.
  - Then as an ensemble member, against the .6916 bar.

**Expected effect (reasoned, not measured).**

| stage | expected |
|---|---|
| G0, lymphoid hidden markers | Δr +.03-.08 (a revealed lymphoid marker resolves the T/B/NK states the image does not) |
| G0, all hidden markers | Δr +.01-.04 |
| Sandbox student | +.01-.03 macro r (15-40 % of the .07 single->full gap) |
| BSCCM GOAL, single model | +.003-.010 |
| BSCCM GOAL, ensemble | +.002-.005 |

**Compute.**

| step | H100-h |
|---|---|
| G0 | CPU < 2 h, plus ≤ 0.3 if training-donor feature dumps are needed |
| G1: cross-fit teachers (3 folds × 2 halves × ~0.1) | ~0.6 |
| G1: students (3 folds × 2 arms × ~0.15) | ~0.9 |
| G2: cross-fit teachers (2 halves × 2 directions × 0.3) | 1.2 |
| G2: students (2 arms × 2 seeds × 2 directions × 0.3) | 2.4 |
| **Total** | **~5-6** (preempt, 1 × H100 each) |

**Overlap with N087.**
- None in mechanism. N087 changes the prediction path (decoders, priors); P1 changes the TRAINING TARGETS.
- It reuses N087's frozen-feature dumps and can share N087-B's validation-donor prior-fitting code.
- If N087-B shows that an AE prior restores joint structure, P1 can use that AE's posterior instead of the GMM (same interface).

**Risks.**
- Cross-instrument emission mismatch: checked by G0b's marginal check.
- FOV-offset leakage: handled by τ_j².
- Confirmation bias: avoided by the cross-fitted teacher and a single EM step. A second round is not planned (the N046 lesson).

### P2 (rank 2). External-prior mixture head trained on whichever channels are observed (singles and pooled sums), plus the label-matched control

**Hypothesis.**
- (a) Part of N080's .07 full-vs-single gap is label count, not co-measurement. A step-matched, label-matched full run measures how
  much is structure.
- (b) Tying the outputs through externally anchored state emissions gives coherent joint profiles (CD4-CD8 exclusivity) and pools all
  markers' labels into one state classifier.
  - p(y|x) = Σ_s π_s(x) N(y; m_s, Σ_s), the CellAssign/Astir template (F5/F6) with learned emissions.
  - Trained by the marginal likelihood of the observed channels; pooled dyes use the exact sum likelihood (B4/G2).

**Recipe.**
- **(a) Label-matched control** (zero code):
  - `train_ifc_sandbox.py --regime full --max-train ⌊n_train/9⌋ --epochs 135`. These are the same optimiser steps as single at 15
    epochs. Also report `--epochs 15`.
  - 3 folds.
  - Decision rule:
    - If label-matched full ≥ single + .02 macro r, co-measurement carries information per label, and P1/P2/P3 have headroom on
      marginals.
    - If ≤ single + .005, joint structure matters only for joint metrics: pursue P2 for the user's multi-marker priority, not for the
      GOAL.
- **(b1) Mixture decoder on frozen features (CPU; one more decoder family in N087-A's n087_decoders.py).**
  - π = softmax(MLP 2048->256->K); emissions as in P1.
  - Prediction E[y_k|x] = Σ_s π_s m_sk.
  - Loss: masked marginal negative log-likelihood.
  - Score on the GOAL with the N087-A harness, which must reproduce the .6730 own-head reference.
- **(b2) End-to-end in the sandbox** (add `--marker-prior mixture` to N087-B's patched train_ifc_sandbox.py).
  - Head outputs K logits, optionally plus a k-dim within-state factor with frozen loadings (MFA).
  - Loss: −log Σ_s π_s N(y_O; m_sO, Σ_sOO) over the revealed entries.
  - Pooled cells: the likelihood of the dye sum in linear intensity (sinh back-transform, per-marker brightness), marginalised over
    states.
  - Arms: emissions for all 9 markers, vs overlap5 only (the BSCCM-realistic case: CD4/CD8/CD15/Siglec8 emissions free), vs N080
    single and full.
  - Metrics: macro r, within-type r, CD4-CD8 in T, CD3-CD19 in lymphocytes, and the registered 36-pair error.
- **(b3) BSCCM** (only if b2 passes): train_lf.py with a mixture head (13 targets, overlap5 anchored, pooled-sum likelihood); 2 seeds ×
  2 directions.

**Expected (reasoned).**
- (a) Label-matched full ≈ single + .01-.04 (HistoPlexer MP > SP points to a positive value).
- (b1) GOAL ±.003 vs the linear head.
- (b2) Macro r +0-.02. CD4-CD8 in T moves from −.07 to ≤ −.4. Risk of over-separation as with CD3-CD19 (−.83..−.89 vs −.73).
- (b3) GOAL 0-+.004.

**Kill.**
- (b2): macro r < single + .010 AND CD4-CD8 in T > −.30.
- Or: the CD3-CD19 lymphocyte over-separation is worse than N080 single's.
- Or: a 36-pair error worse than single's (N087-B's registered criterion is ≤ .21 vs .229).
- (b3): the GOAL bar of P1-G2.

**Compute.**

| step | H100-h |
|---|---|
| (a) step-matched, 3 × ~0.15 | ~0.5 |
| (b1) | CPU (≤ 0.2 GPU-h) |
| (b2) 3 folds × 2 arms × 0.15 | ~1 |
| (b3) | 1.2 |
| **Total** | **~3** |

**Overlap with N087.**
- (b2) extends N087-B. N087-B uses a continuous PCA/AE decoder trained with MSE on all-9 validation-donor emissions. P2 adds discrete
  states, likelihood training, overlap5 realism and the pooled-sum likelihood.
- If N087-B already meets its criterion, run only the overlap5 and pooled arms of P2.
- (b1) is one row in N087-A's decoder table.
- (a) should run alongside N087-B, since both use the same harness and the result is needed to interpret both.

### P3 (rank 3). Learned cross-marker consistency without an external prior (MTPSL-style joint pairwise space)

**Hypothesis.**
- The cross-marker relation can be learned from BSCCM itself.
- A pair-conditioned mapping network projects marker k's prediction and marker j's revealed label into a shared space. A cosine
  consistency loss, regularised to keep image information, gives head k gradients on cells labelled only for j.
- MTPSL (B3) closed about 41 % of the NYUv2 one-label segmentation gap and beat full labels on Cityscapes.
- This is the no-external-data control that tells whether P1/P2 gains need S-BIAD452.

**Recipe.**
- Sandbox single regime, shared r50 trunk, 9 heads.
- Mapping M(v, h | j, k): input = scalar v (a prediction or revealed label) concatenated with stop-grad trunk features h (PCA-64);
  FiLM-conditioned on a learned (j,k) pair embedding; 2-layer MLP -> 32-d.
- Loss: masked MSE + λ Σ_{j revealed, k≠j} [1 − cos(M(ŷ_k, h | j,k), M(y_j, h | j,k))] + μ [1 − cos(M(·), P h)].
- Sweep λ ∈ {.1, .3} on the validation donor; μ = .1.
- BSCCM port only if the sandbox passes: 13 targets; revealed pairs from single slides plus the 4 jointly observed pooled channels.

**Caveat.** MTPSL's tasks are dense maps with shared geometry. For scalar targets the joint space can only carry state information. It
should therefore not beat P1/P2 when the external prior is good; if it does, the prior is miscalibrated.

**Expected.** Sandbox +0-.015 macro r; GOAL 0-+.003.

**Kill.** Sandbox macro r < single + .010 (mean of 3 donors).

**Compute.** Sandbox 3 folds × 2 λ × ~0.2 = 1.2; BSCCM 1.2; total **~2.4 H100-h**. Build about 1 day (Codex writer, Claude red-team).

**Overlap.** None with N087. Report it in the same table as N087-B/P2.

### P4 (rank 4). Marker-space bridge to S-BIAD452 for markers BSCCM never measured (CD4, CD8, CD15, Siglec8); joint-profile deliverable, GOAL-neutral

**Hypothesis.**
- Bridge integration (E5) and CyTOFmerge (F2) logic: a jointly measured reference indexed by shared coordinates extends a panel.
- The shared coordinates here are the predicted overlap markers (CD3/CD19/CD56/CD14/CD45) plus the P2 state posterior. The image-space
  bridge already failed (N081).
- This gives BSCCM cells coherent profiles over 4 extra markers, for the user's multi-marker priority, with a sandbox-measured
  accuracy.

**Recipe.**
- Sandbox validation:
  - A model trained on the training donor with overlap5 single labels only predicts overlap5 for the test donor.
  - Bridge = validation-donor cells with the SAME model's overlap5 predictions (matched shrinkage) and their TRUE 9 labels.
  - Impute CD4/CD8/CD15/Siglec8 for each test cell as the median of the k = 50 nearest bridge cells (CyTOFmerge), or with the P2
    mixture's conditional mean.
  - Compare with the full-regime model's direct r on these markers (upper bound).
- BSCCM: bridge predictions come from the sandbox model, and BSCCM cells from the GOAL model. Per-marker quantile alignment is done
  within matched states. This is flagged as unvalidated on BSCCM; only the sandbox numbers count.

**Expected.**
- Imputed CD15/Siglec8 r about .5-.7 (granulocyte types).
- CD4/CD8 about .2-.4 (cluster level, F4).
- GOAL unchanged.

**Kill.** Imputed r < .7 × the full-regime direct r on the same markers in the sandbox.

**Compute.** CPU plus ≤ 0.5 H100-h.

**Overlap.**
- N079 idea C / N087-A kNN decoders work in IMAGE-embedding space on BSCCM's own markers.
- P4 works in predicted-MARKER space toward new markers.

### Considered and not proposed (reason)

- **Reduced-rank / task-covariance / multi-task-GP heads (A2-A4).** N087-A's rank-k sweep covers them. Theory (A1) and N062 predict
  flat results.
- **Output normalising flows, diffusion priors, SPEN (C1, C3, C4).** N087-E (conditional flow-matching decoder) covers distributional
  decoding. A test-time energy or diffusion prior is a heavier P2 with the same information.
- **Joint hurdle / probit heads (D1, D2).** HL-Gauss and two-part were flat (N060). Their joint form needs residual correlations that
  only P2's external prior can provide.
- **LLP / set-level (G1).** Set-level supervision works only with small sets of the same cells (context.md). The aggregate likelihood
  is used inside P2 for pooled sums only.
- **Group-noise robust losses (H1).** Within-FOV and random-effect losses were flat (N061). The only new item is the cross-target FOV
  correlation, which is folded into P1 (τ_j²).
- **Exclusion loss (B4).** A hand-coded lineage prior, declined (user 11-17). Learned emissions only.

## 4. Gaps / unverified
- SST's ablation numbers, the MTPSL table and ROLE's numbers come from WebFetch extraction of ar5iv full texts; the tables were not
  re-checked against the PDFs.
- Learning-from-aggregate-observations numbers did not extract cleanly; none are quoted.
- sciPENN/totalVI correlation values are figure-only.
- VirTues journal version: the repository cites a later journal paper; only the arXiv version is verified.
- No paper was found that decodes joint multi-marker profiles from single-stain LABEL-FREE images using an external joint prior. The
  closest are Infinity Flow (per-marker, cytometry backbone) and CytoVI/totalVI (joint latent, overlapping panels).
- All expected effect sizes in section 3 are [derived] estimates from N080/N082/K033 and the cited ratios, not measurements.
