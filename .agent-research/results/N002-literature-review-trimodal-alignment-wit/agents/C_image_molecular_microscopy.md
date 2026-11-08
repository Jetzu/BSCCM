# Strand C: linking images to molecular readouts, image-to-image translation, channel-adaptive encoders, and illumination information

Written 2026-11-08 (strand C of N002). I read each source through its arXiv, PMC or bioRxiv text, or through the publisher's abstract page where that was all I could get. Claims I did not check against a primary source are marked **[unverified]**. Numbers are the authors' own unless stated otherwise.

Project mapping used throughout: **L** = the 23 LED-array label-free patterns, imaged for all cells. **M** = per-cell surface-marker abundance, mostly one antibody per cell. **G** = Wright/Giemsa RGB images of 15.7k unstained cells. The data contain G–L and L–M pairs but never G–M pairs.

---

## 0. Main findings

1. **With paired data, a regression head on strong image features beats GAN or unpaired translation for quantitative molecular readout.**
   - HEX (H&E to 40-plex CODEX) puts a regression head on a pathology foundation model. It reached cross-validated mean Pearson r = 0.790, against a value 46 % lower for the second-best model, a conditional GAN. The baselines were the CGAN and VirtualMultiplexer, which is CUT-based ([HEX, PMC12823406](https://pmc.ncbi.nlm.nih.gov/articles/PMC12823406/)).
   - MIPHEI-ViT beat ROSIE, HEMIT and Pix2Pix on cell-level AUPRC: 0.517 vs 0.433, 0.390 and 0.287 ([arXiv 2505.10294](https://arxiv.org/html/2505.10294v2)).
2. **Whether a foundation encoder helps is task-dependent.** Four results point in different directions:
   - ROSIE: a ConvNeXt pretrained on ImageNet beat a histopathology ViT-L/16 with more than 300M parameters ([ROSIE](https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full)).
   - HistoPlexer: end-to-end training without UNI features gave lower MSE than training with them ([PMC12364712](https://pmc.ncbi.nlm.nih.gov/articles/PMC12364712/)).
   - MIPHEI: H-optimus-0 adapted with LoRA scored 0.564 AUPRC, against 0.508 for ConvNeXt-v2 ([arXiv 2505.10294](https://arxiv.org/html/2505.10294v2)).
   - HEST: an ImageNet ResNet50 was the worst of 11 encoders ([HEST](https://arxiv.org/html/2406.16192v2)).

   No foundation model exists for 23-pattern LED data. The decision has to be made empirically, against a strong from-scratch baseline.
3. **Pixel metrics are weak proxies for biological fidelity.**
   - ROSIE's pixel Pearson is only 0.285, yet cell typing and structure discovery from its predictions are useful ([ROSIE](https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full)).
   - Virtual staining cannot add information (data processing inequality). It helps only low-capacity or data-starved downstream networks and can hurt high-capacity ones ([Sengupta et al. 2025](https://www.biorxiv.org/content/10.1101/2025.08.04.668552v1.full)).
   - Evaluate at cell, population and human levels, and always against a noise ceiling and trivial baselines.
4. **Label-free images predict lineage markers well and subtype or activation markers poorly.** The results are directly relevant to BSCCM:
   - DeepIFC (brightfield and darkfield imaging flow cytometry): CD45 r = 0.90, CD14 0.88, CD3 0.87, CD19 0.61, CD56 0.53, CD8 0.41 ([DeepIFC](https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full)).
   - Stain-free classification could not separate CD4+ from CD8+ T cells ([Lippeveld 2020 preprint](https://www.biorxiv.org/content/10.1101/680975v1.full)).
   - A 2026 study on BSCCM with 4-channel DPC input reports CD16 r = 0.819, CD45 0.799, the CD3/CD19/CD56 composite 0.768, and the CD123/HLA-DR/CD14 composite 0.339 ([Nazir & Behera 2026](https://arxiv.org/html/2605.14717)).
5. **BSCCM has already been used to estimate information content per illumination pattern.** Pinkard et al.'s IDEAL (NeurIPS 2025) did the following:
   - estimated mutual information with a PixelCNN upper bound on 40×40 patches;
   - predicted 8 proteins with a DenseNet121 Gaussian-mixture head, scored by negative log-likelihood (NLL);
   - found that higher information generally predicted better protein prediction, except that single-LED estimates were inflated by "system" information measured on empty patches ([IDEAL](https://arxiv.org/html/2405.20559v5)).

   The ranking is therefore not starting from zero, and that confound must be controlled.
6. **For learned illumination, joint optimisation of a "physical layer" with the network beats standard brightfield, darkfield, DPC and random patterns.**
   - Malaria classification: 0.932 accuracy for the optimised pattern vs 0.896 for brightfield and 0.859 for DPC ([Horstmeyer 2017](https://ar5iv.arxiv.org/html/1709.07223)).
   - Virtual fluorescence: the gain grows with the target's information content ([Cooke 2021](https://ar5iv.labs.arxiv.org/html/2004.04306)).
   - Because BSCCM patterns are incoherent intensity sums, learned linear combinations of the 23 measured channels approximate this, with the caveats in §8.

---

## 1. H&E to protein / multiplex immunofluorescence (mIF) prediction

| Method | Input → target | Pairing | Model | Evaluation | Key result |
|---|---|---|---|---|---|
| **ROSIE** (Nat Commun 2025) | H&E → 50 CODEX markers | same section, pixel-registered | ConvNeXt-S (ImageNet init), 128 px patch → mean of central 8×8 px, stride 8; masked MSE across panels | patch Pearson/Spearman; sample C-index; cell-type F1 (7 types); tissue-structure ARI; TIL/LNE proportions | Pearson 0.285, Spearman 0.352, C-index 0.706 overall; cell-type F1 0.507 (Stanford-PGC). "H&E intensity" and "cell morphology MLP" baselines near random. ConvNeXt beat a 300M+ histopathology ViT-L ([bioRxiv v2](https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full); [HF](https://huggingface.co/ericwu09/ROSIE/blob/main/README.md)) |
| **MIPHEI-ViT** (2025/26) | H&E → ~16 mIF channels | same section (Orion CRC, 41 slides) | U-Net with H-optimus-0 ViT encoder + LoRA, ViTMatte-style detail branch; per-marker inverse-std weighted MSE | PSNR/SSIM/Pearson; cell-level logistic regression vs mIF pseudo-labels (AUPRC, F1) | Cell AUPRC 0.517 vs ROSIE 0.433, HEMIT 0.390, Pix2Pix 0.287, nuclear morphometry 0.247. Encoder: H-optimus-0 0.564 > UNIv2 0.554 > ConvNeXt-v2 0.508 > CTransPath 0.472. LoRA > frozen (0.564 vs 0.541). A GAN discriminator slightly hurt. CD8a, FOXP3 and PD-L1 are hard ([arXiv 2505.10294](https://arxiv.org/html/2505.10294v2)) |
| **HEX** (Nat Med 2026) | H&E → 40 CODEX markers (tile mean) | co-registered WSIs | MUSK foundation encoder + MLP head; feature-distribution smoothing + adaptive loss | per-marker Pearson; external cohorts; patient-level prognosis / immunotherapy response | CV r 0.790; Stanford-TMA 0.738 vs CGAN 0.370; external Bern 0.658 vs 0.210. Virtual vs true CODEX gave no significant difference in prognostic C-index ([PMC12823406](https://pmc.ncbi.nlm.nih.gov/articles/PMC12823406/)) |
| **GigaTIME** (Cell 2025) | H&E → 21 mIF channels | paired, 40M cells | NestedUNet | pixel, cell and slide-level metrics vs CycleGAN; population-scale association study | Reported to beat CycleGAN on all three levels; generated about 300k virtual mIF slides ([MSR blog](https://www.microsoft.com/en-us/research/blog/gigatime-scaling-tumor-microenvironment-modeling-using-virtual-population-generated-by-multimodal-ai/)). Numbers not checked against the Cell paper **[unverified]** |
| **HistoPlexer** (Nat Mach Intell 2025) | H&E → 11 IMC markers | serial sections (misaligned) | cGAN with Gaussian-pyramid L1 (tolerates misalignment) + patch InfoNCE | MS-SSIM/PSNR/RMSE-SW; marker co-localisation Spearman; expert HYPE realism test; cell-type F1; TCGA survival / immune subtype | **Multiplex (one joint model) beat singleplex (one model per marker)** on all metrics, with roughly 10× lower co-localisation error. UNI features did not help. Generated features raised survival C-index by 3.18 % ([PMC12364712](https://pmc.ncbi.nlm.nih.gov/articles/PMC12364712/)) |
| **VirtualMultiplexer** (Nat Mach Intell 2024) | H&E → 6 IHC markers | **unpaired** | CUT-style patch-contrastive GAN | downstream graph-transformer clinical prediction | No serial sections or registration needed; the authors report transfer across cohorts ([PubMed 39309216](https://pubmed.ncbi.nlm.nih.gov/39309216/); [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.29.568996.full.pdf)) |
| **HEMIT** (MICCAI-W 2024) | H&E → mIHC (DAPI, CD3, panCK) | same section (mIHC → bleach → H&E) | dual-branch CNN + Swin pix2pix | SSIM, Pearson, PSNR | 5,292 aligned pairs, public ([arXiv 2403.18501](https://arxiv.org/pdf/2403.18501)) |
| **7-UP** (PNAS Nexus 2023) | 7 measured mIF markers → ~40 markers (not from H&E) | same cells | **concrete autoencoder** marker selection; ResNet-50 morphology features + XGBoost per-cell regression | patch Pearson, cell-type weighted F1, patient-outcome AUC via GNN | Mean imputation PCC 0.474 from 7 markers, 0.534 with morphology; cell-type F1 0.667 → 0.727 ([PMC10236358](https://pmc.ncbi.nlm.nih.gov/articles/PMC10236358)) |
| **MIM-CyCIF** (Commun Biol 2024) | subset of CyCIF channels → rest | same cells | MAE with **channel-wise masking**; iterative greedy marker selection maximising held-out Spearman | per-marker Spearman | 9 markers recover the information of 25; +0.22 correlation over the previous approach ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.05.10.540265.full.pdf); [Commun Biol](https://www.nature.com/articles/s42003-024-06110-y)) |

**Lessons for BSCCM**
- **Masked loss over heterogeneous labels.** ROSIE trains on studies with different marker panels using a masked MSE ([ROSIE](https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full)). This is exactly the BSCCM single-antibody situation: one shared multi-output head, with loss only on the measured marker.
- **One joint model beats one model per marker** (HistoPlexer, above). DeepIFC by contrast trained 7 separate models ([DeepIFC](https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full)). Prefer a single multi-head model.
- **Evaluate in tiers**: pixel or patch, then cell-level phenotype, then sample or population. Always include trivial baselines: an intensity proxy, a morphology MLP, and "bulk phenotyping" (assign by population proportions), as ROSIE does ([ROSIE](https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full)).
- **Rare or functional markers are consistently the hardest** (MIPHEI, ROSIE). Expect the same for CD123, HLA-DR and CD56.

---

## 2. H&E and spatial transcriptomics: contrastive and regression models

- **BLEEP** (NeurIPS 2023).
  - Method: a CLIP-like joint embedding of a ResNet50 image patch and an expression MLP. The contrastive targets are softened by within-modality similarity. Expression is imputed as the average over the top-k (k = 50) reference spots.
  - Results on held-out liver Visium (Pearson): marker genes 0.217, highly expressed genes 0.175, highly variable genes 0.173, vs ST-Net 0.099/0.126/0.091 and HisToGene 0.097/0.072/0.071.
  - Limitations noted by the authors: retrieval averaging underestimates variance, and ResNet50 beat ViTs ([arXiv 2306.01859v2](https://arxiv.org/html/2306.01859v2)).
- **OmiCLIP / Loki** (Nat Methods 2025).
  - Method: a CoCa-style ViT paired with a text encoder over "gene sentences" (each spot's top-50 genes), trained with a symmetric contrastive loss on 2.2M Visium patch–transcriptome pairs (ST-bank). Expression prediction ("PredEx") is a similarity-weighted sum over reference spots.
  - Results: best MSE in 28 of 39 heart samples and best PCC in 16 of 39, against Hist2ST, HisToGene, BLEEP and mclSTExp.
  - Limitations noted by the authors: the model is contrastive and cannot generate novel profiles; averaging image and transcriptome embeddings did not beat single-modality embeddings ([PMC12240810](https://pmc.ncbi.nlm.nih.gov/articles/PMC12240810/)).
- **TANGLE** (CVPR 2024): slide-level contrastive alignment of whole-slide-image and bulk-expression encoders. It improves few-shot slide classification and retrieval over supervised and self-supervised baselines ([CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Transcriptomics-guided_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html)).
- **mSTAR** (Nat Commun 2025): slide-level contrastive learning over WSI, pathology reports and RNA-seq (26k slides). The slide-level knowledge is then "self-taught" back into a patch extractor initialised from UNI ([arXiv 2407.15362](https://arxiv.org/html/2407.15362v3)). The "teach the patch encoder from a slide-level aligned teacher" step is the transferable idea.
- **ST-Align** (ICLR 2026 submission): spot- and niche-level image–gene contrastive objectives on 1.3M pairs. Reported results: domain identification ARI 0.340 vs 0.256 for the best multimodal baseline, and gene MSE 0.168 vs 0.184 ([arXiv 2411.16793](https://arxiv.org/abs/2411.16793); [ICLR](https://iclr.cc/virtual/2026/10014521)).
- **Hist2ST** (Brief Bioinform 2022): ConvMixer, then a Transformer, then a GNN, with a zero-inflated negative binomial output and self-distillation ([OUP](https://academic.oup.com/bib/article/23/5/bbac297/6645485)).
- **iStar** (Nat Biotechnol 2024): hierarchical histology features used to super-resolve spot expression to near single-cell resolution ([s41587-023-02019-9](https://www.nature.com/articles/s41587-023-02019-9.pdf)).
- **HEST-1k benchmark**: the standard encoder-comparison protocol.
  - Protocol: PCA to 256 dimensions, then ridge regression, scored by Pearson on 50 highly variable genes, with patient-stratified k-fold splits.
  - Results: H-Optimus-0 best (0.4146) and ImageNet ResNet50 worst (0.326). Performance scales log-linearly with encoder parameter count (R = 0.81) but not significantly with the number of pretraining patches ([arXiv 2406.16192v2](https://arxiv.org/html/2406.16192v2)).

**Lessons for BSCCM**
- **(a)** A frozen-feature linear or ridge probe with fixed PCA dimensionality, as in HEST, is a fair and cheap way to compare candidate L encoders on M regression. Use donor/slide-stratified splits.
- **(b)** Retrieval-based prediction in the style of BLEEP or Loki shrinks variance. Use it as a baseline, not as the final M head.
- **(c)** Soft contrastive targets (BLEEP's within-modality similarity) anticipate the S2L/CWCL losses in §3. BSCCM has many near-identical marker profiles, so hard one-hot InfoNCE produces false negatives.

---

## 3. Cell Painting and molecules/omics

- **CLOOME** (Nat Commun 2023): contrastive learning over Cell Painting images and molecular structures using InfoLOOB with Hopfield retrieval. Retrieving the correct structure among about 2,000 candidates is more than 70× better than random. Its image embeddings beat CellProfiler features in transfer ([Nat Commun](https://www.nature.com/articles/s41467-023-42328-w); [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.17.516915v2.full.pdf)).
- **MolPhenix** (NeurIPS 2024). This is the most relevant to the SigLIP plan.
  - **Pretrained phenomics embeddings beat raw images.** Contrastive training on embeddings from a pretrained phenomics MAE (Phenom1) instead of raw images raised CLOOME's active-unseen retrieval from 0.0756 to 0.4659.
  - **Soft targets beat SigLIP.** The new S2L loss is SigLIP with soft targets taken from phenomic-space distance. Top-1 % retrieval: CLIP 0.337, CLOOME 0.357, CWCL 0.709, SigLIP 0.776, **S2L 0.910** (cumulative concentrations, active molecules, unseen images).
  - **Inactive molecules are handled** by undersampling and partial-match weighting ([arXiv 2409.08302](https://arxiv.org/html/2409.08302v1)).
- **CellCLIP** (NeurIPS 2025): perturbations are encoded as text; images go through a channel-aware "CrossChannelFormer" on top of pretrained vision models. A continuously weighted contrastive loss handles the many-images-to-one-label problem. It beats CLOOME and MolPhenix on retrieval ([arXiv 2506.06290](https://arxiv.org/html/2506.06290v3)).
- **Haghighi et al. 2022** (Nat Methods): four datasets profiled with both L1000 and Cell Painting, with cross-modal prediction in both directions. A companion paper shows the two modalities are complementary rather than redundant ([PubMed 36395727](https://pubmed.ncbi.nlm.nih.gov/36395727/); [Carpenter–Singh lab PDF](https://carpenter-singh-lab.broadinstitute.org/files/anne/files/168_haghighi_natmethods_2022.pdf)). Quantitative accuracies not read **[unverified]**.
- **MorphDiff** (Nat Commun 2025): latent diffusion that generates Cell Painting images conditioned on L1000 profiles ([PMC12405515](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12405515/)).
- **CellFlux** (ICML 2025): flow matching from control-cell to perturbed-cell image distributions. The authors report a 35 % better FID and 12 % higher mode-of-action accuracy than prior methods ([arXiv 2502.09775](https://arxiv.org/abs/2502.09775v3)). This is a working precedent for distribution-to-distribution flow matching on multi-channel cell images.

**Lessons for BSCCM**
- **(a)** Pretrain the L encoder first (self-supervised), then align with SigLIP on frozen or lightly tuned features: the "locked image" pattern, which MolPhenix shows is far better than learning images inside the contrastive loss.
- **(b)** Use soft targets in the S2L/CWCL style for L–M. Positives between cells with similar marker values should not count as negatives. M is a scalar for one antibody per cell, so the soft weight can be a kernel on the measured marker value, applied within each antibody's subset.
- **(c)** Report retrieval in a size-invariant way, as MolPhenix does with top-1 % recall, and separately for informative (antibody-positive) and uninformative cells. This mirrors MolPhenix's active/inactive split.

---

## 4. Label-free images to molecular readouts at the single-cell level (closest to L→M)

- **In silico labeling** (Christiansen et al., Cell 2018).
  - Verified: the model predicts nuclei, cell type (neuron) and state (death) from transmitted-light z-stacks across brightfield, phase and DIC. A multi-scale, Inception-inspired architecture was designed with automated search, and transfer learning lets new labels be learned from very little data ([Google Research blog](https://www.research.google/blog/seeing-more-with-in-silico-labeling-of-microscopy-images/); [Cell record](https://research.google/pubs/in-silico-labeling-predicting-fluorescent-labels-in-unlabeled-images/)).
  - Not verified: the per-pixel discrete intensity distribution output and the 98 % live/dead figure, which comes from a press release ([EurekAlert](https://www.eurekalert.org/news-releases/692666)) **[unverified]**.
- **fnet** (Ounkomol et al., Nat Methods 2018). This introduces the key evaluation idea: a **noise-ceiling C_max**.
  - The target is modelled as T = S + N. The noise variance is estimated from unlabeled images taken with identical settings, giving the best achievable Pearson.
  - Per-structure performance is well bounded by C_max. DNA, nuclear envelope and nucleoli are good; desmosomes and actomyosin are poor.
  - Each structure was trained on only 30 image pairs, using a U-Net with MSE ([bioRxiv 289504v2](https://www.biorxiv.org/content/10.1101/289504v2.full)).
- **DeepIFC** (bioRxiv 2022): brightfield + darkfield (imaging flow cytometry, 60×) → 7 virtual fluorescence markers in PBMCs, 527k cells, one Inception-U-Net per marker.
  - Per-marker r: CD45 0.90, CD14 0.88, CD3 0.87, 7-AAD 0.79, CD19 0.61, CD56 0.53, CD8 0.41. AUROC 0.73–0.98.
  - Strict cell-typing recall from virtual stains: B cells 5 %, cytotoxic T 13 %. **Class-balanced training raised these to 54 % and 68 %.**
  - Held-out donors were used, and unseen red blood cells formed their own cluster ([DeepIFC](https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full)).
- **Lippeveld et al.** (Cytometry A 2020): stain-free brightfield/darkfield classification of 8 WBC types. Best balanced accuracy was 0.778 with classical ML and 0.703 with deep learning. "No distinction can be made between CD4+ and CD8+ T-cells". Granulocytes and monocytes were easy ([bioRxiv](https://www.biorxiv.org/content/10.1101/680975v1.full)).
- **Nazir & Behera** (ICPR 2026, BSCCM): 4-channel DPC (left/right/top/bottom) into a CNN+ViT hybrid with multi-task classification and regression. Accuracy 91.3 % (3 classes). Per-marker r: CD16 0.819, CD45 0.799, CD3/CD19/CD56 0.768, CD123/HLA-DR/CD14 0.339.
  - The test set was only 1,418 cells.
  - Composite markers imply the all-antibody condition was used.
  - No comparison across illumination patterns ([arXiv 2605.14717](https://arxiv.org/html/2605.14717)).
  - **These r values are probably dominated by between-class variance** (lymphocyte/monocyte/granulocyte) **[my inference]**. Within-class correlation is the stronger test.
- **Pinkard et al. IDEAL** (NeurIPS 2025, uses BSCCM): DenseNet121 (ImageNet init, single-channel input) with 8 parallel Gaussian-mixture heads (CD3, CD19, CD56, CD123, HLA-DR, CD14, CD16, CD45), scored by test NLL. Information estimates by modality (brightfield, DPC, single LED) generally tracked protein-prediction NLL, with the single-LED exception in §8 ([arXiv 2405.20559v5](https://arxiv.org/html/2405.20559v5)).
- **Raman2RNA** (Nat Biotechnol 2024). This is the best analogue for the **never-paired G–M** problem: Raman and scRNA-seq are never measured on the same cell.
  - **Anchor-based route**: Raman → smFISH on the same cells (9 anchor genes) → Tangram mapping to full scRNA-seq.
  - **Anchor-free route**: adversarial autoencoders aligning unpaired latent spaces. This failed without cell-type label guidance and was "relatively unstable".
  - Results: single-cell anchor cosine about 0.8. Brightfield models reached cosine < 0.15 on the same task, so label-free modality choice matters ([PMC11233426](https://pmc.ncbi.nlm.nih.gov/articles/PMC11233426)).
  - For BSCCM, **L plays the smFISH-anchor role** between G and M.
- **Cross-Zamirski et al.** (Sci Rep 2022): brightfield z-stack → 5 Cell Painting channels with U-Net and cWGAN-GP.
  - Evaluated with **CellProfiler features** computed on predicted images: 611 features, mean correlation 0.45, 30 features with r > 0.8. Pixel Pearson was 0.84.
  - The gap between pixel and feature fidelity is the lesson ([search summary of Sci Rep / bioRxiv](https://pmc.ncbi.nlm.nih.gov/articles/PMC9200748)). I could not open the full text, so the numbers come from the abstract and preprint snippets **[partly unverified]**.
  - Follow-up: class-guided image-to-image diffusion based on Palette improves downstream content ([ICCVW 2023](https://openaccess.thecvf.com/content/ICCV2023W/BIC/html/Cross-Zamirski_Class-Guided_Image-to-Image_Diffusion_Cell_Painting_from_Brightfield_Images_with_Class_ICCVW_2023_paper.html)).

---

## 5. Virtual staining of label-free images

**Paired GAN era**
- **Rivenson et al. 2019** (Nat Biomed Eng): a GAN maps autofluorescence of unstained sections to H&E, Jones or Masson's trichrome. Board-certified pathologists, blinded to whether images were virtual or chemical, found "no clinically significant difference" ([Springer Nature community](https://communities.springernature.com/posts/deep-learning-based-virtual-histology-staining-of-unlabelled-tissue); [record](https://cris.bgu.ac.il/en/publications/virtual-histological-staining-of-unlabelled-tissue-autofluorescen-2/)).
- **Bai et al. 2023 review** (Light Sci Appl):
  - Registration recipe: coarse cross-correlation, then an affine fit, then a briefly trained "pseudo-model" that enables elastic pyramidal registration.
  - A GAN trained only with an adversarial loss can produce "severe hallucinations at the micro-scale", so pixel losses are added.
  - Unsupervised methods such as CycleGAN are generally inferior to supervised ones.
  - Algorithmic scores "cannot always accurately reflect" diagnostic value, so pathologist studies and downstream tasks are needed ([PMC9981740](https://pmc.ncbi.nlm.nih.gov/articles/PMC9981740/); [arXiv 2211.06822](https://arxiv.org/abs/2211.06822)).

**Blood-specific precedents**
- **Ojaghi et al.** (PNAS 2020): deep-UV label-free hematology with a pseudo-colourisation that resembles Giemsa. Hematologists evaluated it blind; the method gives a 5-part differential ([abstract record](https://sigmaaldrich.com/NL/en/tech-docs/paper/1388051)).
- **Kaza et al. 2022**: a cGAN (U-Net generator with total-variation regularisation) predicts the a/b channels in Lab colour space from a single 260 nm UV image. The training target is the multispectral pseudo-colour, not real Giemsa. Real Giemsa was used only for visual comparison.
  - MS-SSIM 0.913 and SSIM 0.781.
  - **No hematologist scoring of the virtual stains, and no classifier run on them.**
  - Eosinophil cytoplasm was rendered less orange than in the target ([PMC10521747](https://pmc.ncbi.nlm.nih.gov/articles/PMC10521747/)).
  - **BSCCM's G–L pairs are real Wright-stained images of the same cells**, which is a stronger target than Kaza's pseudo-colour.

**Diffusion, bridge and flow models**
- **Ozcan lab, Brownian-bridge diffusion** (Nat Commun 2025): super-resolved virtual H&E from low-resolution autofluorescence of lung tissue, with a 4–5× super-resolution factor. The authors add sampling techniques at inference to reduce output variance ([arXiv 2410.20073](https://arxiv.org/abs/2410.20073)). I did not read the details of those techniques **[unverified]**. This is relevant because G is 63×/1.4 NA and L is 20×/0.5 NA, so virtual G is partly super-resolution.
- **I²SB** (ICML 2023): a tractable Schrödinger bridge from the degraded to the clean image. It needs paired data and degrades little at low step counts ([arXiv 2302.05872](https://arxiv.org/abs/2302.05872v3)).
- **Palette-style conditional diffusion** for brightfield → fluorescence: see the Cross-Zamirski ICCVW paper above.
- **Flow matching in microscopy**: CellFlux (distribution-to-distribution), HazeMatching (conditional flow matching for dehazing), ResMatching (super-resolution) ([HazeMatching arXiv 2506.22397](https://arxiv.org/abs/2506.22397v5); [ResMatching arXiv 2510.26601](https://arxiv.org/abs/2510.26601v1)). I did not find a peer-reviewed flow-matching virtual-staining paper for label-free → stain as of this search. One preprint (STREAM, Riemannian flow matching in pathology) appeared but I could not date it **[unverified]**.
- **AQuA** (Nat Biomed Eng 2025): detects hallucinations without ground truth. An ensemble scores image quality using cycles of virtual staining and a learned "unstaining" model. It reports 99.8 % accuracy at separating acceptable from unacceptable stains and 98.5 % agreement with pathologists ([arXiv 2404.18458](https://arxiv.org/abs/2404.18458v2); [UCLA](https://www.ee.ucla.edu/autonomous-detection-of-ai-hallucinations-in-digital-pathology/)).

**Unpaired and weakly paired translation**
- **CycleGAN, CUT and UNSB**:
  - CUT (ECCV 2020) uses a PatchNCE loss with negatives drawn from the same image ([arXiv 2007.15651](https://arxiv.org/abs/2007.15651)).
  - UNSB (ICLR 2024) builds a Schrödinger bridge from adversarial sub-problems ([arXiv 2305.15086](https://arxiv.org/abs/2305.15086v2)).
  - **Cohen et al. 2018**: distribution-matching losses can hallucinate features, and translated images should not be used for direct interpretation ([arXiv 1805.08841](https://arxiv.org/abs/1805.08841)).
  - In a stain-translation study, unpaired CycleGAN added structures that were absent from the reference; loosely paired training removed them ([arXiv 2306.08583](https://arxiv.org/pdf/2306.08583)).
- **Misalignment-tolerant paired losses**:
  - RegGAN treats a misaligned target as a noisy label and trains a registration network jointly. It helped every method tested, and the version without cycle loss was best ([arXiv 2110.06465](https://arxiv.org/abs/2110.06465)).
  - ASP (MICCAI 2023) is a supervised PatchNCE that down-weights inconsistent regions in weakly paired H&E→IHC ([arXiv 2303.06193](https://arxiv.org/pdf/2303.06193)).
  - HistoPlexer uses a Gaussian-pyramid L1 (§1).
  - One method avoids co-registration altogether by training only on stain-free refractive-index tomograms (holotomographic flow cytometry) ([bioRxiv 2025.07.22.666145](https://www.biorxiv.org/content/10.1101/2025.07.22.666145.full.pdf)).

**Utility and evaluation critiques**
- **Sengupta et al. 2025** (QPI → fluorescence, SW480 cells): virtual stains improved segmentation and classification only for low-capacity networks or small training sets. **High-capacity classifiers did better on the label-free input directly.** The authors explain this with the data processing inequality ([bioRxiv 2025.08.04.668552](https://www.biorxiv.org/content/10.1101/2025.08.04.668552v1.full)).
- **Spotlight** (Kalinin et al. 2025): pixel losses reproduce background noise. A foreground-masked MSE plus a Dice term worsens whole-image metrics but improves segmentation and morphology fidelity ([arXiv 2507.05383](https://arxiv.org/html/2507.05383v1)).
- **Metric critiques**: SSIM and PSNR are sensitive to small shifts, and FID uses ImageNet features. Proposed replacements are classifier-based "semantic fidelity" scores and pathology-aware metrics (PaPIS) ([arXiv 2511.18946](https://arxiv.org/pdf/2511.18946); [arXiv 2507.12624](https://arxiv.org/html/2507.12624v1)). These are recent preprints that I did not read in full **[partly unverified]**.

---

## 6. How these fields evaluate: synthesis

| Level | Typical metrics | Examples | Pitfall |
|---|---|---|---|
| Pixel | SSIM, MS-SSIM, PSNR, pixel Pearson, LPIPS/FID | HEMIT, Kaza, Rivenson | Shift-sensitive. Rewards blur and background reproduction (Spotlight). Weakly related to biology (ROSIE pixel r 0.285 is still useful downstream) |
| Feature / cell | per-cell marker Pearson/Spearman; marker-positive AUROC; CellProfiler-feature correlation; cell-type F1/AUPRC from predicted markers | DeepIFC, MIPHEI, ROSIE, Cross-Zamirski, 7-UP | Between-class variance inflates r. Thresholds for positivity are arbitrary. Rare types collapse unless training is class-balanced (DeepIFC) |
| Population / sample | cell-type proportions, C-index on sample ranking, co-localisation | ROSIE, HistoPlexer, DeepIFC (fractions vs IDEAS gating) | Few independent units. Split by donor, slide or batch (HEST patient-stratified; ROSIE held-out studies) |
| Downstream clinical | survival, response, retrieval | HEX, HistoPlexer, VirtualMultiplexer | No patients in BSCCM; the analogue is held-out donor/batch generalisation |
| Human | blinded pathologist diagnosis concordance; HYPE/Turing realism | Rivenson, HistoPlexer, Ojaghi | Passing a Turing test does not establish diagnostic equivalence ([search summary of HER2 virtual-staining paper](https://arxiv.org/pdf/2604.08305)) **[unverified for that paper]** |
| Ceiling and controls | noise ceiling C_max (fnet); trivial baselines (intensity, morphology, bulk); reference-free hallucination audit (AQuA) | fnet, ROSIE, AQuA | Usually omitted. Without them a reported r cannot be interpreted |

---

## 7. Channel-adaptive vision transformers for multi-channel microscopy

- **ChannelViT** (ICLR 2024): one patch token per (channel, patch), plus a learned channel embedding. **Hierarchical Channel Sampling (HCS)** trains on random channel subsets, so the model stays accurate when only some channels are present at test time. Tested on JUMP-CP, So2Sat and ImageNet ([arXiv 2309.16108](https://arxiv.org/abs/2309.16108)). The cost is a token sequence C× longer.
- **ChAda-ViT** (CVPR 2024): intra- and inter-channel attention over any number, order or type of channel. Self-supervised on IDRCell100k (79 experiments, 7 modalities, 1–10 channels) ([arXiv 2311.15264](https://arxiv.org/abs/2311.15264v2); [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Bourriez_ChAda-ViT__Channel_Adaptive_Attention_for_Joint_Representation_Learning_of_CVPR_2024_paper.pdf)).
- **DiChaViT** (NeurIPS 2024): adds a channel-diversification loss, a token-diversification loss and diverse channel sampling.
  - CHAMMI average 69.7 vs ChannelViT 64.9. JUMP-CP full channels 69.19 vs 67.51.
  - ChannelViT's channel tokens are redundant (high mutual information), and its attention concentrates on a few channels ([arXiv 2405.16419v2](https://arxiv.org/html/2405.16419v2)).
  - This matters for BSCCM because many brightfield/DPC patterns are highly correlated.
- **CHAMMI** (NeurIPS 2023 Datasets & Benchmarks): WTC-11 (3 channels), HPA (4) and Cell Painting (5). Evaluation uses 1-NN on frozen features, scored by macro-F1, with weight on out-of-distribution tasks.
  - From scratch: HyperNet 0.537 > Depthwise 0.517 > FixedChannels 0.500.
  - **Fine-tuning from an ImageNet-22k ConvNeXt raised FixedChannels from 0.500 to 0.614.**
  - SimCLR alone was weak (0.360 vs 0.623 supervised).
  - ChannelReplication (one shared model applied to each channel, features concatenated) scored 0.344 at about 4× inference cost ([ar5iv 2310.19224](https://ar5iv.labs.arxiv.org/html/2310.19224)).
- **IC-ViT** (BMVC 2025): single-channel pretraining, sampling one random channel per image, then multi-channel fine-tuning. Reported +4–14 points on JUMP-CP, CHAMMI and So2Sat ([arXiv 2503.09826](https://arxiv.org/abs/2503.09826v1)).
- **ChA-MAEViT** (NeurIPS 2025): MAE with dynamic channel-patch masking, memory tokens and a channel-aware decoder. Reported +3–21.5 % over multi-channel ViTs ([arXiv 2503.19331](https://arxiv.org/abs/2503.19331v3)).
- **CA-MAE / OpenPhenom** (CVPR 2024): a channel-agnostic MAE that accepts any number and order of channels and transfers to JUMP-CP ([arXiv 2404.10242](https://arxiv.org/abs/2404.10242); [HF model card](https://huggingface.co/recursionpharma/OpenPhenom/blob/ebfee133915e3fee1c5b736343b3b4e558509bdc/MODELCARD.md?code=true)).
- **Physically conditioned channel embeddings (remote-sensing analogue)**: DOFA uses a hypernetwork that generates patch-embedding weights from each band's central wavelength. This lets one backbone accept arbitrary band sets, including unseen sensors ([arXiv 2403.15356](https://arxiv.org/html/2403.15356v3)).
  - **[proposal, not literature]** For BSCCM, the analogue is conditioning each channel's embedding on illumination geometry: the LED set's NA range, azimuth, and brightfield/darkfield/DPC type. This would let the model interpolate to unseen patterns, such as BSCCM-coherent single LEDs.
- **DinoBloom** (MICCAI 2024): DINOv2 trained on 13 public blood/bone-marrow smear datasets (more than 380k WBCs), without the local-crop loss. External Acevedo 1-NN weighted F1: DinoBloom-S 86.4 vs DINOv2-S 71.9 ([arXiv 2404.05022](https://arxiv.org/html/2404.05022v1)). It is an RGB-smear encoder for G, not for L.
  - Note: BSCCM's histology RGB is formed by **sequential red/green/blue LED illumination**, not a colour camera ([BSCCM paper](https://arxiv.org/html/2402.06191v1)). Expect a colour domain shift relative to DinoBloom's training smears, and check stain-colour normalisation.

---

## 8. Quantifying information content per channel or illumination pattern

1. **Joint "physical layer" optimisation** (learned illumination):
   - **Horstmeyer et al. 2017 / Muthumbi et al. 2019** (Biomed Opt Express): 29 LEDs, 20×/0.5 NA (the same objective class as BSCCM), malaria-infected red blood cells.
     - Accuracy by illumination: optimised 0.932 (majority vote 0.957); centre-LED brightfield 0.896; DPC 0.859; off-axis 0.832; random 0.813; all LEDs 0.746; full Fourier ptychography with 29 images 0.951.
     - The outer ring held about 75 % of the weight energy, so high spatial frequencies matter. 14 runs converged to similar patterns ([ar5iv 1709.07223](https://ar5iv.arxiv.org/html/1709.07223); [Muthumbi 2019 record](https://pmc.ncbi.nlm.nih.gov/articles/PMC6913384)).
   - **Kellman et al. 2019** (IEEE TCI): learned coded illumination for quantitative phase. 2 learned measurements matched the accuracy of 69-image Fourier ptychography ([arXiv 1808.03571](https://arxiv.org/abs/1808.03571v3)).
   - **Cooke et al. 2021** (ICCV): a learned LED pattern (15×15 LEDs × 3 colours = 675 weights) combined with a U-Net for virtual DAPI/membrane fluorescence.
     - It beat centre, all-LED, differential, off-axis and random patterns on MSE and SSIM.
     - **The margin grew with the target's bit depth (information content).**
     - An L1 penalty on LED weights gave smoother patterns at equal MSE. Learned patterns mixed differential brightfield and darkfield LEDs with colour contrast ([ar5iv 2004.04306](https://ar5iv.labs.arxiv.org/html/2004.04306)).
   - Implication: the advantage of a pattern depends on the task. Rank LED patterns **per target** (each M marker, cell type, G reconstruction), not globally.
2. **Task-agnostic mutual-information estimation (IDEAL; Pinkard et al. NeurIPS 2025, on BSCCM)**:
   - Method: an upper bound on I(object; measurement) = H(Y) − H(Y|X). H(Y) comes from the held-out cross-entropy of a fitted PixelCNN or Gaussian-process model on patches; H(Y|X) comes from a Poisson/Gaussian noise model.
   - Photon levels were simulated by adding Poisson noise. The minimum estimate over 15 splits was reported ([arXiv 2405.20559v5](https://arxiv.org/html/2405.20559v5); [BAIR blog](https://bair.berkeley.edu/blog/2026/01/10/information-driven-imaging/)).
   - **Pitfall:** single-LED images carried more information on *empty* patches, which the authors call system variation, so their estimate was inflated relative to task performance. Subtract or condition on empty-patch information.
   - The authors also note that total information need not track task-specific performance.
3. **Channel / marker selection from imputation literature**:
   - Concrete autoencoder (7-UP), which a naive greedy search matched.
   - Iterative greedy addition that maximises held-out reconstruction Spearman, with a channel-masked MAE (MIM-CyCIF) (§1).
4. **Missing-channel robustness**:
   - ChannelViT's HCS and DiChaViT's diverse sampling train **one** model that can be evaluated on any subset of channels (§7). This makes leave-one-out, single-channel, greedy forward selection and Monte-Carlo Shapley ablation over 23 channels cheap: inference only, no retraining.
   - Caveat **[my inference]**: subset scores from an HCS model are biased if the model under-fits small subsets. Spot-check the top subsets with dedicated retrained models.
5. **Optics prior.**
   - Fourier ptychography forms a larger synthetic NA by stitching angle-shifted pupils. Oblique and darkfield LEDs therefore shift high spatial frequencies into the objective's passband ([Ou et al. 2013, via Caltech PDF](https://biophot.caltech.edu/documents/21042/64-XO-OptLett-15November2013.pdf); [Zheng et al. 2013](https://biophot.caltech.edu/documents/21043/63-GZ-NatPhot-September2013.pdf)).
   - The DPC/brightfield/darkfield-by-LED-pattern design is described in [Liu et al. 2014 J Biomed Opt](https://spiedigitallibrary.org/journals/Journal-of-Biomedical-Optics/volume-19/issue-10/106002/Real-time-brightfield-darkfield-and-phase-contrast-imaging-in-a/10.1117/1.JBO.19.10.106002.full) and [Tian et al. 2014 BOE](https://pmc.ncbi.nlm.nih.gov/articles/PMC4102371).
   - **Implication [inference]**: darkfield channels are the main carriers of sub-0.5-NA detail (granularity, nuclear lobes) and should matter most for virtual G (a 1.4 NA target) and for granulocyte subtyping.
   - Darkfield images have fewer photons and are noisier (Horstmeyer noted lower high-angle transmission). Compare patterns at **matched photon budgets**.

---

## 9. Recommendations for BSCCM

### 9.1 Encoder for the 23-channel LED data (L)

- **Leakage guard (critical).** M is derived from the 6-channel fluorescence images ([BSCCM paper](https://arxiv.org/html/2402.06191v1)). The L encoder must see **only the 23 LED patterns**, never the fluorescence images, in any L→M or L–M contrastive task. Autofluorescence in no-antibody cells is a legitimate label-free signal, but keep it out of the main L definition. If wanted, add it as a separately reported arm.
- **Primary candidate: a channel-adaptive ViT** (ChannelViT tokenisation + HCS + DiChaViT diversity losses, or ChA-MAEViT/IC-ViT-style pretraining), **self-supervised on all ~413k cells** (MAE or DINO). Reasons:
  - DinoBloom shows that about 380k in-domain cells suffice for a useful DINOv2 pretraining.
  - The CHAMMI, HEST and MolPhenix results show that pretraining-then-probing beats training inside the downstream loss.
  - The HCS property gives illumination ranking for free (§9.4).
- **Token budget [inference]**: a 128×128 crop at patch size 16 gives 64 patches per channel. With 23 channels that is 1,472 tokens, which is fine for ViT-S/B. Patch size 8 gives about 5.9k tokens, which is costly. Alternatives:
  - group channels physically (brightfield; darkfield rings; DPC pairs) as token groups;
  - use the IC-ViT approach of single-channel pretraining.
- **Mandatory baselines**, all on donor/slide/batch-stratified splits:
  - (a) a from-scratch CNN (ConvNeXt or DenseNet) with a 23-channel stem. IDEAL's DenseNet121 + Gaussian-mixture head is the existing BSCCM reference ([IDEAL](https://arxiv.org/html/2405.20559v5)).
  - (b) ImageNet-initialised ConvNeXt with an inflated stem. CHAMMI shows ImageNet fine-tuning gave +22 %.
  - (c) DINOv2 or DinoBloom used as ChannelReplication (per-channel frozen features, concatenated). This is the cheap foundation-model baseline; CHAMMI found it weak, but it is cheap to try.
- **Compare encoders with a fixed-capacity probe** in the HEST style (PCA to a fixed dimension, then ridge or a small MLP) on M regression and 3/10-class cell type. Then fine-tune the best one.
- For the SigLIP stage, **lock or LoRA the pretrained L encoder**, following the MolPhenix frozen-embedding lesson. Use **soft targets** (S2L/CWCL) built from marker similarity within each antibody subset, and from DinoBloom-feature similarity on the G side.

### 9.2 Marker regression heads (L→M)

- Use one shared encoder with a multi-output head trained with a **masked loss** over the measured antibody (ROSIE; HistoPlexer's multiplex > singleplex). Use a heteroscedastic or Gaussian-mixture output as IDEAL does, since single-cell marker noise is large.
- Train with class balancing or reweighting for rare subsets: DeepIFC's B-cell recall rose from 5 % to 54 %.
- Expect, based on DeepIFC, Lippeveld and Nazir & Behera, strong CD45/CD16/CD14, moderate CD3/CD19, and weak CD56, CD123 and HLA-DR. Treat CD4 vs CD8-like subtype distinctions as likely infeasible from morphology.

### 9.3 Virtual Giemsa (L→G) model family

- G–L is **paired per cell but not pixel-registered**: the objectives differ (20×/0.5 NA vs 63×/1.4 NA), and fixation and staining happen after LED imaging.
- **Stage 1 (baseline)**: a U-Net/pix2pix regressor with a misalignment-tolerant loss (Gaussian-pyramid L1 as in HistoPlexer, or a jointly trained registration network as in RegGAN). Add a perceptual term in DinoBloom feature space. Use an L1/Spotlight-style foreground mask to avoid background fitting. Rigid or affine pre-registration of each LED crop to its G crop at matched scale should come first, following the Bai et al. coarse-to-fine recipe.
- **Stage 2 (main)**: a paired conditional diffusion or bridge model (I²SB / Brownian-bridge / Palette-style), or a conditional flow-matching model. It should start from the upsampled stage-1 prediction or from L, and be conditioned on all 23 channels through the L encoder.
  - G is a 1.4 NA target and L is a 0.5 NA source, so a stochastic model is appropriate for the unrecoverable detail.
  - **Report the variance across samples**, and average several samples for any quantitative use, following the Ozcan lab's variance-reduction concern.
- **Do not use unpaired CycleGAN/CUT/UNSB as the main model.** Paired data exist, and distribution-matching losses hallucinate (Cohen 2018). Keep CUT as a baseline only.
- With 15.7k pairs, consider class guidance from M- or L-predicted cell type, following Cross-Zamirski's class-guided diffusion.

### 9.4 Evaluation protocol for L→M and L→G

**Splits.** Hold out whole slides or batches (and donors, if multiple). Report in-distribution and held-out batch results separately.

**L→M**
- Per-marker Pearson and Spearman, **within each of the 3 main classes** as well as overall.
- AUROC for marker-positive vs marker-negative.
- Cell-type F1 derived from predicted markers.
- Population fractions per slide.
- Comparison to a **noise ceiling**: estimate C_max from the no-antibody autofluorescence distribution and the single-cell measurement noise, following fnet.
- Baselines: predict the class mean ("bulk phenotyping"); handcrafted size and darkfield-intensity features; and the IDEAL DenseNet reference.

**L→G**
- **Semantic fidelity**: DinoBloom 5-part differential on virtual vs real G for the same cell, scored by Cohen's κ, plus kNN agreement.
- Fréchet distance in **DinoBloom** feature space rather than an ImageNet FID.
- Pixel SSIM/PSNR/LPIPS only after registration, as secondary metrics.
- **Bridging test, enabled only by virtual G [proposal]**: apply virtual G to L–M cells that have no Giemsa image, classify with DinoBloom, and score against marker-derived cell types. This is the only direct check of G–M consistency, analogous to Raman2RNA's anchor chain.
- **Information check**: virtual G must not beat a classifier trained directly on L (data processing inequality; Sengupta 2025). If it does, suspect leakage.
- Hallucination audit: L→G→L cycle consistency (AQuA-style) and per-pixel sample variance.
- A small **blinded hematologist** test of real vs virtual images, with diagnosis concordance rather than realism alone.

### 9.5 Ranking LED patterns by information content

Run three complementary estimators and report them **per target**: each marker, 3/10-class type, G reconstruction, and task-agnostic information.

1. **Model-based ablation with one HCS-trained channel-adaptive model**:
   - single-channel score, leave-one-out drop, greedy forward selection (MIM-CyCIF style), and Monte-Carlo Shapley values over the 23 channels;
   - confirm the top-k subsets with retrained dedicated models.
   - Report marginal importance (alone) and conditional importance (given the others) separately, because brightfield and DPC channels are redundant.
2. **Learned illumination surrogate**: a learned linear "physical layer" over the 23 measured channels, with L1/group sparsity (Cooke; Horstmeyer), trained jointly with the probe.
   - **Caveat [inference]**: digitally summing measured images adds per-capture read and shot noise, unlike a single physical exposure. Simulate a matched photon budget by adding Poisson noise, as IDEAL does.
   - BSCCM's 22 multi-LED patterns span only a subspace of possible LED patterns. The BSCCM-coherent subset (566 single LEDs, about 4.3k cells) allows a true per-LED learned design ([BSCCM](https://arxiv.org/html/2402.06191v1)).
3. **Task-agnostic mutual information (IDEAL)**: PixelCNN upper bounds per pattern at equalised photon counts, **minus the information measured on empty background patches**, to remove the single-LED-style inflation.

**Pre-registered expectations to test.** Darkfield and oblique patterns should rank higher for granularity-dependent targets (granulocyte subtypes, CD16) and for G reconstruction. DPC and brightfield should rank higher for size or shape-dependent lineage markers. Disagreement between the information estimate and model ablation is itself a finding (IDEAL's caveat).

---

## 10. Gaps, caveats, and unverified items

- Not opened in full text: Cell 2018 (Christiansen), Nat Commun CLOOME, Cross-Zamirski Sci Rep, GigaTIME Cell paper, Rivenson 2019 full paper, Haghighi 2022 numbers. Statements about them rest on abstracts, author pages or preprints, as cited.
- The Nazir & Behera BSCCM preprint has internal inconsistencies: abstract CD16 r 0.72 vs table 0.819, and a mismatch between the marker list and the table ([arXiv 2605.14717](https://arxiv.org/html/2605.14717)).
- ROSIE's sample counts are inconsistent across abstract and results (over 1,000 vs over 2,000 samples), and HEX's tile counts are also inconsistent (see the respective links).
- BSCCM's histology is described in the dataset paper as **Wright's stain**, not "Wright–Giemsa" ([BSCCM](https://arxiv.org/html/2402.06191v1)). The project context's "Giemsa" label should be checked against the data README.
- No published work was found that ranks all 23 BSCCM LED patterns for marker prediction. IDEAL compared only the brightfield, DPC and single-LED modalities. Nazir & Behera used DPC only.
- No peer-reviewed flow-matching virtual-staining paper for label-free → stain was found as of this search **[unverified absence]**.

## Sources (primary links used above)
ROSIE https://www.biorxiv.org/content/10.1101/2024.11.10.622859v2.full · MIPHEI-ViT https://arxiv.org/html/2505.10294v2 · HEX https://pmc.ncbi.nlm.nih.gov/articles/PMC12823406/ · GigaTIME https://www.microsoft.com/en-us/research/blog/gigatime-scaling-tumor-microenvironment-modeling-using-virtual-population-generated-by-multimodal-ai/ · HistoPlexer https://pmc.ncbi.nlm.nih.gov/articles/PMC12364712/ · VirtualMultiplexer https://pubmed.ncbi.nlm.nih.gov/39309216/ · HEMIT https://arxiv.org/pdf/2403.18501 · 7-UP https://pmc.ncbi.nlm.nih.gov/articles/PMC10236358 · MIM-CyCIF https://www.nature.com/articles/s42003-024-06110-y · BLEEP https://arxiv.org/html/2306.01859v2 · OmiCLIP/Loki https://pmc.ncbi.nlm.nih.gov/articles/PMC12240810/ · TANGLE https://openaccess.thecvf.com/content/CVPR2024/html/Jaume_Transcriptomics-guided_Slide_Representation_Learning_in_Computational_Pathology_CVPR_2024_paper.html · mSTAR https://arxiv.org/html/2407.15362v3 · ST-Align https://arxiv.org/abs/2411.16793 · Hist2ST https://academic.oup.com/bib/article/23/5/bbac297/6645485 · iStar https://www.nature.com/articles/s41587-023-02019-9.pdf · HEST https://arxiv.org/html/2406.16192v2 · CLOOME https://www.nature.com/articles/s41467-023-42328-w · MolPhenix https://arxiv.org/html/2409.08302v1 · CellCLIP https://arxiv.org/html/2506.06290v3 · MorphDiff https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12405515/ · CellFlux https://arxiv.org/abs/2502.09775v3 · fnet https://www.biorxiv.org/content/10.1101/289504v2.full · In silico labeling https://www.research.google/blog/seeing-more-with-in-silico-labeling-of-microscopy-images/ · DeepIFC https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full · Lippeveld https://www.biorxiv.org/content/10.1101/680975v1.full · Nazir & Behera https://arxiv.org/html/2605.14717 · Raman2RNA https://pmc.ncbi.nlm.nih.gov/articles/PMC11233426 · Kaza https://pmc.ncbi.nlm.nih.gov/articles/PMC10521747/ · Bai review https://pmc.ncbi.nlm.nih.gov/articles/PMC9981740/ · Rivenson https://communities.springernature.com/posts/deep-learning-based-virtual-histology-staining-of-unlabelled-tissue · AQuA https://arxiv.org/abs/2404.18458v2 · Brownian bridge https://arxiv.org/abs/2410.20073 · I2SB https://arxiv.org/abs/2302.05872v3 · CUT https://arxiv.org/abs/2007.15651 · UNSB https://arxiv.org/abs/2305.15086v2 · Cohen 2018 https://arxiv.org/abs/1805.08841 · RegGAN https://arxiv.org/abs/2110.06465 · ASP https://arxiv.org/pdf/2303.06193 · Sengupta 2025 https://www.biorxiv.org/content/10.1101/2025.08.04.668552v1.full · Spotlight https://arxiv.org/html/2507.05383v1 · ChannelViT https://arxiv.org/abs/2309.16108 · ChAda-ViT https://arxiv.org/abs/2311.15264v2 · DiChaViT https://arxiv.org/html/2405.16419v2 · CHAMMI https://ar5iv.labs.arxiv.org/html/2310.19224 · IC-ViT https://arxiv.org/abs/2503.09826v1 · ChA-MAEViT https://arxiv.org/abs/2503.19331v3 · CA-MAE https://arxiv.org/abs/2404.10242 · DOFA https://arxiv.org/html/2403.15356v3 · DinoBloom https://arxiv.org/html/2404.05022v1 · BSCCM https://arxiv.org/html/2402.06191v1 · IDEAL https://arxiv.org/html/2405.20559v5 · Horstmeyer 2017 https://ar5iv.arxiv.org/html/1709.07223 · Muthumbi 2019 https://pmc.ncbi.nlm.nih.gov/articles/PMC6913384 · Kellman 2019 https://arxiv.org/abs/1808.03571v3 · Cooke 2021 https://ar5iv.labs.arxiv.org/html/2004.04306 · Tian 2014 https://pmc.ncbi.nlm.nih.gov/articles/PMC4102371
