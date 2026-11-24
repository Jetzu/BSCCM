# N088 lit A: label-free -> marker prediction, and representation learning (autoencoders, low-dimensional bottlenecks, projectors)

Agent: read-only literature agent (Claude), 2026-11-24 18:27. Scope: topic A of the N088 brief.
1. Predicting markers from label-free single-cell images, with per-marker accuracies.
2. Autoencoders vs SSL latents for clustering, and the projector / bottleneck question.

Constraints read first:
- context.md;
- ideas.md + declined.md of N047, N059, N066, N079;
- N066 agents/lit_labelfree.md (not repeated here);
- N087, N078, N083 item records;
- KNOWLEDGE K017, K036, K042, K046.

Access:
- WebSearch and WebFetch for arXiv, ICLR/ICML/NeurIPS/CVPR pages and PLoS.
- Europe PMC REST for PubMed abstracts and full-text XML of PMC9349041 (cytoself) and PMC11811211 (Kim 2025).
- PubMed and PMC HTML were blocked (cookie wall), so Europe PMC was used instead.

Verification tags:
- [full text]: numbers read in the paper body or tables.
- [abstract]: abstract or proceedings page only.
- [via N066/N038]: taken from earlier verified project reviews, not re-fetched.
- [derived]: my inference.

No personal or project identifiers went into any query.

---------------------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Autoencoder latents cluster cell identity worse than SSL or pretext-task features.** Every direct microscopy comparison found points the same way, and none goes the other way.
   - Lu 2019, yeast localisation kNN: AE **42.5 %** vs paired-cell-inpainting SSL **88.0 %** vs supervised 92.4 %.
   - cytoself ablation: a pure VQ-VAE (no identity pretext) clusters 3-5x worse (score 1.13 / 1.26 vs 3.41 / 5.96). Removing the decoder costs only ~13-25 %.
   - Ternes 2022: VAEs "often fail to identify biologically informative features due to uninformative technical variation".
   - Theory (Balestriero & LeCun, ICML 2024): reconstruction spends capacity on the top-variance pixel subspace, which is the least informative for perception.
   - A SMALLER AE bottleneck keeps only the highest-variance directions. In LED-array data those are illumination, background, size and focus. [derived] So k = 16 should be the worst of N087 C's arms.

2. **Post-projector features are worse than backbone features.**
   - SimCLR: > 10 points under linear evaluation.
   - Guillotine regularisation: up to 30 points.
   - Fine-grained / subclass information is lost after the projector: CIFAR-100 fine labels 36.0 vs 25.7 (SupCon), 33.7 vs 29.7 (supervised).
   - SIGReg pushes the projector output to an isotropic Gaussian. [derived] That space therefore has no cluster structure by construction; clusters must be read from the backbone.
   - A small projector is harmless. LeJEPA linear probe: proj. dim 64 gives 75.65 and 1024 gives 74.79.
   - A narrow backbone is not harmless: effective rank tracks downstream quality (RankMe, ICML 2023).
   - "Small from the start" in the literature means wide backbone + nested / small projection (Matryoshka: 14x smaller embedding at equal ImageNet accuracy), not a narrow encoder.

3. **Label-free marker accuracies in the literature sit at or below our numbers.** Lineage markers reach r .8-.9 with any optics. Dim lymphoid markers are .4-.6 even at 60x IFC (DeepIFC CD19 .61, CD56 .53, CD8 .41). Nothing reports cross-batch (different-draw) marker regression. No new BSCCM user was found since N066.

4. **The one untested lever the literature strongly supports for our two-draw shift is feature-preserving fine-tuning.** That means LP-FT, a lower encoder LR, and WiSE weight interpolation.
   - Literature: LP-FT ~+10 points OOD over full fine-tuning on 10 shift benchmarks.
   - The project already has a strong in-house signal. The external Cytoland encoder fine-tuned at encoder lr x1.0 scored .6464 vs .6671 at x0.1 (+.021, N053).
   - The in-domain SIGReg r50 / ConvNeXt-S encoders have only ever been fine-tuned at full LR (N078, N083: lr 5e-4, one group).

---------------------------------------------------------------------------------------------------------------------------------

## (i) Annotated reference list (35 entries, grouped)

### A. Label-free -> markers / immunophenotype in blood cells

**A1. Pinkard, Liu, Nyatigo, Fletcher, Waller. "The Berkeley Single Cell Computational Microscopy (BSCCM) Dataset." arXiv 2402.06191 (2024).** https://arxiv.org/abs/2402.06191 [via N066, full text]
- Content: ~413k WBCs; LED-array patterns + DPC; 6 fluorescence channels unmixed into 8 markers; LOWESS slide background.
- Authors call the CD123/CD14/HLA-DR channel "too noisy to be useful".
- No marker-prediction benchmark is given.
- Applies: our data and the label pipeline (source of FOV-level label error, K040/K041).

**A2. Pinkard, Kabuli, Markley, Chien, Jiao, Waller. "Information-Driven Design of Imaging Systems." NeurIPS 2025, arXiv 2405.20559.** [via N066]
- Method: DenseNet121 on ONE LED pattern; 8 Gaussian-mixture heads, NLL loss.
- Finding: a PixelCNN estimate of image information tracks protein-prediction NLL.
- Single-LED images carry the most "system-derived randomness".
- No per-marker numbers in the text.
- Applies: the only BSCCM information argument. Single LEDs are the likeliest carrier of draw-specific nuisance. [derived] A layer-wise / per-LED audit fits P4 below.

**A3. Nazir & Behera. "Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning." ICPR 2026, arXiv 2605.14717.** https://arxiv.org/abs/2605.14717 [via N066, full text; venue per the model card]
- Setup: BSCCMNIST 28x28 4-channel DPC; CNN-ViT multi-task model (WBC class + marker regression, SmoothL1 + Pearson term).
- Results: CD16 r .819, CD45 .799, CD3/19/56 group .768, CD123/HLA-DR/CD14 group .339.
- Split unstated; no batch handling.
- Applies: the only other BSCCM marker numbers. Their tricks (multi-task, Pearson loss) were already flat in-project (K042).

**A4. Timonen et al. "DeepIFC: Virtual fluorescent labeling of blood cells in imaging flow cytometry data with deep learning." Cytometry A 2023, doi 10.1002/cyto.a.24770.** [abstract; numbers via N066/N038]
- Method: Inception U-Net per marker, BF + DF -> fluorescence image, pixel BCE; ImageStream 60x, 247k cells.
- Results r (AUROC): CD45 .90, CD14 .88, CD3 .87, CD19 .61 (.800), CD56 .53 (.805), CD8 .41 (.728).
- Held-out samples share one panel and compensation.
- Applies: the best label-free per-marker reference. Our dim-marker band (CD19/CD56) matches theirs. Dense supervision already tried (N074).

**A5. Nassar, Doan, ..., Carpenter, Summers, Rees, Hennig. "Label-Free Identification of White Blood Cells Using Machine Learning." Cytometry A 2019;95:836-842.** [abstract, Europe PMC]
- Unstained IFC BF + DF, 85 donors.
- "average F1-score of 97%"; B vs T lymphocytes "average F1-score of 78%, a task previously considered impossible".
- Applies: B vs T (CD19 vs CD3) is weakly but really encoded in label-free morphology, consistent with our CD19 r ~.3-.6.

**A6. Lippeveld et al. "Classification of human white blood cells using machine learning for stain-free imaging flow cytometry." Cytometry A 2020;97:308-319.** [abstract / repository]
- Task: 8 WBC classes from BF + DF.
- Balanced accuracy .778 for hand-engineered features + classical ML vs .703 for deep learning. "DL did not outperform engineered features."
- Applies: at small n, DL is not automatically better. At our 400k cells this is reversed (SSL r50 >> probes).
- [derived] A hand-crafted morphology baseline was parked in N079 (#11); this paper is weak support for it.

**A7. Yoon, ..., Park. "Identification of non-activated lymphocytes using three-dimensional refractive index tomography and machine learning." Sci Rep 2017;7:6654.** [abstract; numbers from the JoVE companion via search]
- 3D RI tomograms, handcrafted features + kNN.
- Test accuracy: B vs T 89.8 %; CD4 vs CD8 84.4 %. Small n (149 / 95 / 112 cells).
- See also Shu et al., "AIRFIHA", Adv. Intell. Syst. 2021, doi 10.1002/aisy.202000277: QPI + cascaded ResNet; B/T and CD4/CD8 separation claimed. Numbers not retrieved.
- Applies: phase / RI carries lymphocyte-subset signal. Our computed phase channel is 2D and lower resolution.

**A8. Islam, ..., Liu. "Real-Time Cell Sorting with Scalable In Situ FPGA-Accelerated Deep Learning." arXiv 2503.12622 (2025; journal DOI 10.1039/d5dd00345h).** [abstract]
- Brightfield lymphocytes, 80k images (LymphoMNIST).
- T4 vs B "98% accuracy"; T8 vs B "93% accuracy in zero-shot classification".
- Applies: purified-population classification (balanced, clean labels) gives far higher numbers than per-cell marker regression on mixed blood. These are not comparable to our r.

**A9. Muthumbi, ..., Horstmeyer. "Learned sensing: jointly optimized microscope hardware for accurate image classification." Biomed. Opt. Express 10:6351 (2019).** [abstract / press]
- Setup: LED-array illumination jointly optimised with a CNN for malaria-infected RBCs.
- Result: "up to 5-10% greater accuracy" than standard illumination. The learned pattern was a high-angle coloured ring.
- Applies: on fixed-pattern data like ours, the analogue is a learned per-LED mixing layer, which the project has had since N053. Nothing new to propose.
- [derived] Supports darkfield-ring channels as the informative ones.

### B. Virtual staining / in silico labelling (generic)

**B1. Christiansen et al. "In Silico Labeling: Predicting Fluorescent Labels in Unlabeled Images." Cell 2018;173:792.** [abstract]
- Multi-label transmitted-light -> fluorescence network predicts nuclei, cell type and cell state.
- See also Ounkomol et al. Nat Methods 2018 (3D label-free -> fluorescence).
- Applies: the origin of multi-task dense supervision (tried in N074).

**B2. Cheng, ..., Tian. "Single-cell cytometry via multiplexed fluorescence prediction by label-free reflectance microscopy." Sci Adv 2021;7:eabe0431.** [abstract; details via N066]
- "Up to three times improvement in prediction accuracy"; multiplexed readouts "enable accurate multiparametric single-cell profiling".
- See also Cross-Zamirski et al. Sci Rep 2022: brightfield -> Cell Painting. Image-level Pearson .84 but only .45 on extracted features. Pixel metrics overstate per-cell readout quality.
- Applies: supports scalar readout over image generation for markers (K027).

**B3. Liu, Hirata-Miyasaki, ..., Mehta. "Robust virtual staining of landmark organelles" (Cytoland). bioRxiv 10.1101/2024.05.31.596901; Nat Mach Intell 2025 per the package citation.** [abstract]
- UNeXt2 (ConvNeXtV2) with FCMAE self-supervised pre-training + virtual staining.
- See also DynaCLR (Hirata-Miyasaki et al., arXiv 2410.11281): contrastive, time-aware cell embeddings, > 95 % infection-state accuracy. The project has a converted DynaCLR encoder, never tested.
- Applies: our Cytoland member (.6671 single; +.006 as ensemble member).
- Its lower-LR fine-tuning result (.6671 at x0.1 vs .6464 at x1.0) is the in-project evidence behind proposal P1.

### C. Representation learning for single-cell images: autoencoders vs SSL

**C1. Kobayashi, Cheveralls, Leonetti, Royer. "Self-supervised deep learning encodes high-resolution features of protein subcellular localization" (cytoself). Nat Methods 2022;19:995.** [full text, Table 1]
- Model: VQ-VAE-2 + protein-identity pretext.
- Clustering score (organelle / complex level):

| Variant | Organelle | Complex |
|---|---|---|
| full | 3.41 | 5.96 |
| **without identity pretext (= plain VQ-VAE)** | **1.13** | **1.26** |
| without decoder | 2.98 | 4.48 |
| without VQ | 2.98 | 4.46 |
| Lu et al. inpainting conv5 | 2.91 | 3.06 |
| CellProfiler | .13 | .12 |

- Authors: "the single most important component ... is the protein identification pretext task".
- Applies: the AE part alone clusters badly. The weak-label pretext makes the clusters.
- [derived] Our "pretext" equivalent is the marker supervision itself. A pure image AE (N087 C) lacks it.

**C2. Lu, Kraus, Cooper, Moses. "Learning unsupervised feature representations for single cell microscopy images with paired cell inpainting." PLoS Comput Biol 2019;15:e1007348.** [full text]
- Yeast localisation, kNN (k = 11), balanced accuracy: paired cell inpainting 87.98 % (best layer conv4); **autoencoder 42.50 %**; VGG16 transfer 69.33 %; supervised 92.39 %.
- Human cells, pairwise-distance score (more negative = better): inpainting conv3 -.574, transfer -.325, AE -.281.
- Applies: the cleanest head-to-head AE vs SSL on single cells. A large gap, and the best SSL layer is intermediate (conv3/4).

**C3. Ternes, ..., Chang. "A multi-encoder variational autoencoder controls multiple transformational features in single-cell image analysis." Commun Biol 2022;5:255.** [abstract]
- Quote: "single cell image analyses based on VAEs often fail to identify biologically informative features due to uninformative technical variation". The fix uses transformed images as an invariance (SSL-like) signal.
- See also Locatello et al. ICML 2019 (arXiv 1811.12359): unsupervised disentanglement (beta-VAE family) is "fundamentally impossible without inductive biases". Increased disentanglement "does not seem to lead to a decreased sample complexity" downstream (12,000 models).
- Applies: argues against beta-VAE / disentangled-latent proposals.

**C4. Balestriero & LeCun. "How Learning by Reconstruction Produces Uninformative Features for Perception." ICML 2024, arXiv 2402.11337.** [abstract]
- TinyImageNet: the top subspace explaining 90 % of pixel variance gives 45 % test accuracy; "the bottom subspace ..., accounting for only 20% of the pixel variance, reaches 55%".
- Masking can help; additive noise does not.
- Counterpoint in another modality: Richter et al., Nat Mach Intell 2025 (doi 10.1038/s42256-024-00934-3) find masked AEs better than contrastive methods in scRNA-seq, where natural augmentations are absent.
- Applies: explains why low-k pixel AEs on LED images should encode illumination / background / size. Our images have natural invariances (rotations, LED symmetry), so the vision regime applies, not the scRNA one.

**C5. He et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022, arXiv 2111.06377.** [full text via ar5iv]
- ViT-L: MAE linear probe 73.5 % (75.8 % in Appendix B) vs MoCo v3 77.6 %.
- Fine-tuning ONE block raises MAE from 73.5 to 81.0; full fine-tuning 84.9-85.9.
- "linear separability is not the sole metric"; MAE features are "stronger non-linear features".
- Applies: reconstruction features need a nonlinear or partially fine-tuned readout, which is the decoder-capacity question of N087 A. For fine-tuned encoders the point is moot.

**C6. Kraus, ..., Earnshaw. "Masked Autoencoders for Microscopy are Scalable Learners of Cellular Biology." CVPR 2024, arXiv 2404.10242.** [abstract]
- ViT MAEs trained on up to 93M Cell Painting crops beat weakly supervised classifiers ("up to 11.5% relative improvement" in known-relationship recall).
- Channel-agnostic MAE.
- Applies: the counter-example. Reconstruction wins at 10^7-10^8 images with ViT-L. We have 4x10^5 cells, where invariance SSL has won in-project and in C1, C2, C8.

**C7. Kenyon-Dean et al. "ViTally Consistent: Scaling Biological Representation Learning for Cell Microscopy." ICML 2025 (PMLR 267), arXiv 2411.02572.** [full text]
MAE-G/8 (48 blocks):
- Best block 38. RxRx1 linear probe 0.51 balanced accuracy, +8.5 % over the final block. Anax +5 %.
- Whole-genome KS .60 -> .63 and CM 16.4 -> 18.2 when trimmed; recall unchanged.

Natural-image models also gain from intermediate blocks:
- DINOv2 ViT-G/14 recall 32.7 (block 40) -> 37.5 (block 16).
- ViT-L ImageNet MAE: best RxRx1 block +27 % over the final block.
- Linear-probe proxy correlates with genome-wide recall at rho .97.

Applies: the "which layer" question. Frozen-feature readouts should scan layers (P4).

**C8. Kim, Adaloglou, ..., Marin Zapata. "Self-supervision advances morphological profiling by unlocking powerful image representations." Sci Rep 2025, doi 10.1038/s41598-025-88825-4.** [full text]
- DINO, MAE and SimCLR trained on JUMP Cell Painting.
- "DINO achieved the best results among the SSL methods". DINO "surpassed CellProfiler by a margin of 29% in perturbation mAP".
- MAE used 50 % masking and flip-only augmentation.
- Applies: in a direct same-data comparison, the prototype / self-distillation objective beats reconstruction and contrastive. Motivates P3.

**C9. Doron, ..., Caicedo. "Unbiased single-cell morphology with self-supervised vision transformers" (DINO4Cells). bioRxiv 10.1101/2023.06.16.545359 (2023).** [full text]
- In-domain DINO per dataset.
- HPA cell-line classification .93 vs supervised CNN .69. Single-cell heterogeneity peak accuracy 64 % vs supervised 48 % vs ImageNet-DINO 32 %.
- Channel dropout and warping are useful augmentations; blur, solarisation and greyscale are not.
- DINO head removed after training; features from the teacher backbone.
- See also Pfaendler, Hanimann, Lee, Snijder, scDINO (bioRxiv 10.1101/2023.01.16.524226): DINO ViTs on 5-channel PBMC immunofluorescence resolve "multilineage identities". Fluorescence input, not label-free.
- Applies: in-domain DINO clusters cell states well, including on PBMCs.

**C10. Gupta, Wefers, Kahnert, Hansen, ... "SubCell: Proteome-aware vision foundation models for microscopy capture single-cell biology." bioRxiv 10.1101/2024.12.06.627299.** [abstract]
- Multitask (weakly supervised + SSL) on HPA, 1.13M cells.
- Outperforms the self-supervised DINO4Cells-HPA and a weakly supervised ViT on localisation; transfers across datasets.
- Applies: like cytoself, weak supervision + SSL > SSL alone. Fluorescence-protein specific, not usable as an LF encoder.

**C11. Koch, ..., Marr. "DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology." MICCAI 2024, arXiv 2404.05022.** [abstract]
- DINOv2 on 380k WBC smear images, 13 datasets.
- The global-local crop loss "hampers performance" on single-cell images and was dropped.
- Applies: we tested DinoBloom-B (GOAL .6552, worse than SSL r50; K036). The no-local-crop recipe carries over to P3.

**C12. Yamachui Sitcheu, ..., Mikut. "IFC-SSL-Bench: Benchmarking Self-Supervised Learning for Annotation-Efficient Multispectral Imaging Flow Cytometry." arXiv 2610.31848 (Oct 2026).** [full text, first half]
- Methods: SimCLR, MoCo-v2/v3, DINO, Barlow Twins, BYOL. "masked autoencoders ... were not evaluated."
- Backbones: ResNet50, ConvNeXtV1/V2-T, ViT.
- Immune synapse IFC (BF + F-actin, MHCII, CD3, pCD3zeta; B/T cells), 1 % labels: DINO + ConvNeXtV2-T linear probe 47.4 vs best supervised 37.1 macro-F1.
- 100 %: linear probe .879 (MoCo-v2 ConvNeXtV1-T) vs fine-tune 93.0.
- MoCo-v2 + ConvNeXt best overall; BYOL unstable.
- Applies: the newest IFC benchmark. ConvNeXt + invariance SSL is the right family. Fine-tuning > frozen probe by ~5 points even with SSL.

### D. Projector, effective rank, bottlenecks

**D1. Chen, Kornblith, Norouzi, Hinton. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR). ICML 2020, arXiv 2002.05709.** [blog / slides]
- The layer before the projection head is "much better (>10%)" under linear evaluation than the projected output.
- Applies: predicts N087 D1. Projector output < backbone.

**D2. Bordes, Balestriero, Garrido, Bardes, Vincent. "Guillotine Regularization: Why removing layers is needed to improve generalization in self-supervised learning." TMLR 2023, arXiv 2206.13378.** [abstract]
- Removing the projector is "critical ... more than 30 percentage points can be gained" on ImageNet.
- Causes: augmentations misaligned with the downstream task and suboptimal SSL hyperparameters. "When these views are aligned with the downstream-task, the optimal layer ... become[s] closer to the last layer."
- Applies: our SSL augmentations (geometric / intensity) are only partly aligned with dim-marker readout. [derived] The best frozen layer may be stage 3, not the pooled stage-4 output (P4). After supervised fine-tuning the last layer is task-aligned.

**D3. Xue, Gan, Ni, Joshi, Mirzasoleiman. "Investigating the Benefits of Projection Head for Representation Learning." ICLR 2024, arXiv 2403.11391.** [full text]
- CIFAR-100 trained on coarse labels, evaluated on fine labels (pre vs post projection):

| Training | Fine pre | Fine post | Coarse pre | Coarse post |
|---|---|---|---|---|
| SupCon | 36.0 | 25.7 | 55.2 | 53.6 |
| supervised | 33.7 | 29.7 | 53.1 | 53.2 |

- "the subclass feature is assigned a weight of zero post-projection but has a non-zero weight pre-projection."
- UrbanCars few-shot adaptation under shift: 82.2 vanilla -> 86.1 with a projection head.
- Applies: within-cell-type marker variation (S-BIAD452 within-type r .15-.20, lost CD4/CD8 exclusivity, K046) is subclass structure. Training through a projection head and reading pre-projection features is the literature's way to keep it (P2).

**D4. Jing, Vincent, LeCun, Tian. "Understanding Dimensional Collapse in Contrastive Self-supervised Learning." ICLR 2022, arXiv 2110.09348.** [abstract]
- Embeddings collapse to a low-dimensional subspace (strong augmentation + implicit low-rank bias).
- DirectCLR (no trainable projector) beats SimCLR with a linear projector.
- Applies: the projector absorbs collapse; the backbone stays higher-rank. Measure rank where you read features.

**D5. Garrido, Balestriero, Najman, LeCun. "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank." ICML 2023, arXiv 2210.02885.** [abstract]
- Effective rank of the embeddings, no labels, no hyperparameters.
- Correlates with in- and out-of-distribution linear-probe accuracy over 110 models / 11 datasets. Hyperparameter selection with "nearly no loss" vs label-based selection.
- See also LiDAR (Thilak et al., ICLR 2024, arXiv 2312.04000): higher Spearman / Kendall correlation than RankMe for JEPAs. And alpha-ReQ (Agrawal et al., NeurIPS 2022): power-law eigenspectrum decay; best alpha "neither too large nor too small", near 1.
- Applies: cheap label-free screen of SSL encoders / settings (P4). Caveat: validated against linear probes; our GOAL uses fine-tuned encoders (K017).

**D6. Balestriero & LeCun. "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics." arXiv 2511.08544 (Nov 2025).** [full text, first half]
- SIGReg drives embeddings to an isotropic Gaussian (our SSL objective).
- In-domain pretraining (Galaxy10) beats DINOv2/v3 transfer, mirroring our N053.
- Projector-dim ablation, linear probe: 64-d 75.65 vs 1024-d 74.79. "none of the choices lead to a catastrophic collapse".
- Training loss vs linear probe Spearman ~.85, ~.99 with scaling.
- Applies: (a) a small SSL projector is free; (b) [derived] SIGReg output space is isotropic by design, so it is the WRONG place for k-means; (c) the SIGReg loss itself can rank SSL runs.

**D7. Kusupati et al. "Matryoshka Representation Learning." NeurIPS 2022, arXiv 2205.13147.** [abstract]
- Nested-prefix losses make the first m dimensions usable at every m.
- "up to 14x smaller embedding size ... at the same level of accuracy" (ImageNet-1K). Prefixes are "at least as accurate" as independently trained low-dim models; +2 % long-tail few-shot.
- Applies: the literature-backed way to "make it small from the start" without losing the wide representation (P2). Replaces N087 D3.

**D8. Alemi, Fischer, Dillon, Murphy. "Deep Variational Information Bottleneck." ICLR 2017, arXiv 1612.00410.** [abstract]
- A stochastic bottleneck with a KL penalty improves generalisation and adversarial robustness over other regularisers (abstract claim, numbers not retrieved).
- Applies: an optional VIB arm on the P2 head. [derived] Prior is weak for batch shift, see D9.

**D9. Masarczyk, ..., Trzcinski. "The Tunnel Effect: Building Data Representations in Deep Neural Networks." NeurIPS 2023, arXiv 2305.19753.** [abstract]
- Later layers of supervised nets compress representations (the "tunnel"), "hindering the model's out-of-distribution generalization".
- Applies: a learned low-dim bottleneck is not expected to fix the cross-draw shift. If anything it removes features that transfer. Predicts N087 D2 flat or slightly worse on b01.

### E. Fine-tuning under distribution shift

**E1. Kumar, Raghunathan, Jones, Ma, Liang. "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution." ICLR 2022 (oral), arXiv 2202.10054.** [abstract / proceedings]
- Over 10 shift datasets: full fine-tuning +~2 % ID but ~-7 % OOD vs linear probing.
- LP-FT (linear probe first, then fine-tune) +~1 % ID and +~10 % OOD over fine-tuning. It changes features "10-100x less".
- FT wins only for tiny shifts (CIFAR-10 -> 10.1).
- Applies: our cross-draw shift. In-project analogue: Cytoland enc-lr x0.1 vs x1.0 = .6671 vs .6464 (N053). Never tested for the in-domain encoders (P1).

**E2. Wortsman et al. "Robust fine-tuning of zero-shot models" (WiSE-FT). CVPR 2022, arXiv 2109.01903.** [abstract]
- Linear interpolation between pre-fine-tuning and fine-tuned weights.
- +4-6 points under shift over prior robust methods with +1.6 ID (ImageNet + 5 shifts); 2-10 points OOD over standard FT. Zero extra training cost.
- Applies: post-hoc arm of P1. Needs a meaningful pre-FT head, which LP-FT provides. BN statistics must be recomputed for ResNets.

---------------------------------------------------------------------------------------------------------------------------------

## Per-marker accuracy landscape (topic 1)

| System | Optics, scale | Per-marker / per-class numbers | Shift tested? |
|---|---|---|---|
| DeepIFC (A4) | IFC 60x BF+DF, 247k | CD45 .90, CD14 .88, CD3 .87, CD19 .61, CD56 .53, CD8 .41 (r) | held-out samples, same panel |
| Nazir & Behera (A3) | BSCCM 28x28 DPC | CD16 .819, CD45 .799, CD3/19/56 .768, CD123/HLA-DR/CD14 .339 | split unstated |
| Chlis 2020 (via N038) | IFC BF, mouse | CD34 .38, FcgR .50 | - |
| Nassar 2019 (A5) | IFC BF+DF, 85 donors | 5-class F1 .97; B vs T F1 .78 | donors |
| Lippeveld 2020 (A6) | IFC BF+DF | 8-class BA .778 handcrafted vs .703 DL | CV |
| Yoon 2017 (A7) | 3D RI tomography | B vs T 89.8 %, CD4 vs CD8 84.4 % (test) | small n |
| Islam 2025 (A8) | brightfield, purified | T4 vs B 98 %, T8 vs B 93 % | purified pops |
| project S-BIAD452 (N080) | IFC BF/SSC, 274k | 9 markers macro r .596 (all labels) / .526 (one per cell); within-type .15-.20 | held-out donor |
| project GOAL | LED array 24 ch, 400k | 11-target r .6766 single / .6862 ensemble | **held-out draw** (only one) |

[derived] No published system reports marker regression across separate blood draws processed as separate batches. Our GOAL is already at or above the published bands for every comparable marker. The literature offers no architecture that closes the dim-marker gap. This repeats the N066 conclusion with no new counter-evidence in the last 5 days.

---------------------------------------------------------------------------------------------------------------------------------

## (ii) What the literature says

### (a) Do autoencoder latents cluster cell types better or worse than SSL / contrastive features?

**Worse, by a wide margin, in every image-domain head-to-head found.**

| Evidence | AE / reconstruction | SSL / pretext | Source |
|---|---|---|---|
| yeast localisation kNN | AE 42.5 % | inpainting 88.0 % (supervised 92.4 %) | C2 |
| protein-localisation clustering | VQ-VAE w/o identity pretext 1.13 / 1.26 | full cytoself 3.41 / 5.96 | C1 |
| Cell Painting profiling | MAE < DINO | DINO +29 % mAP over CellProfiler | C8 |
| ImageNet linear probe | MAE ViT-L 75.8 | MoCo v3 77.6 | C5 |
| VAE on single cells | "fail ... due to uninformative technical variation" | invariance-trained ME-VAE better | C3 |

Conditions under which reconstruction catches up:
- 10^7-10^8 images with ViT-L/G (C6).
- Readout from intermediate blocks rather than the last (C7).
- A nonlinear or partially fine-tuned readout (C5: 73.5 -> 81.0 with one block).
- Data without natural augmentations (scRNA, C4 note).

None of these describes a 16-256-d bottleneck of a pixel AE trained on 4x10^5 LED crops.

**On lower dimensionality specifically.**
- Lowering k does not make AE latents cluster better.
- The bottleneck keeps the highest-variance pixel directions first (C4: the top-90 %-variance subspace is the less informative one).
- In LED-array crops those are illumination amplitude, background, cell size/position and focus. [derived]
- beta-VAE / disentanglement does not rescue this without supervision (C3 note, Locatello).
- Expected N087 C outcome [derived]:
  - kNN balanced accuracy and k-means NMI vs gated classes: AE k = 16 < k = 64 < k = 256 < SIGReg backbone (2048-d).
  - GOAL from frozen AE latents below the frozen SSL ridge, which itself is weak (K017: .52 vs .71 FOV).
- If the AE does cluster lineage (lymph / mono / gran) acceptably, that is because lineage is a high-variance morphological factor (size, granularity / darkfield). It does not mean the AE carries within-type marker information.

**Where an AE is useful.**
- As an auxiliary objective on a supervised or SSL encoder. cytoself: decoder removal costs ~13-25 % of clustering score; supervised AEs "never harm" (N066).
- As masked pretraining for a diverse ensemble member (Cytoland FCMAE).
- Not as the clustering space itself.

### (b) Features before vs after the projector; bottlenecks; "small from the start"

1. **Before > after, consistently.**
   - SimCLR > 10 points (D1); Guillotine up to 30 points (D2).
   - The gap is largest for fine-grained / subclass information (D3: 36.0 vs 25.7).
   - The same holds for SUPERVISED training with a projection head (D3: 33.7 vs 29.7).
   - Prediction for N087 D1: SSL projector 256-d output < its intermediate 2048-d layer < backbone 2048-d, for both clustering and marker ridge. [derived]
2. **SIGReg makes the projector output isotropic Gaussian (D6).** [derived] A k-means clustering of that space should be near-uninformative by design. Clusters live in the backbone, so cluster the backbone features.
3. **The layer matters for frozen readouts.** Intermediate blocks beat the last by 5-27 % in microscopy probes (C7), and the best SSL layer in C2 is conv3/4. For fine-tuned encoders the last layer is task-aligned (D2), so no gain is expected there.
4. **Small projector: free. Narrow backbone: costly.**
   - LeJEPA projector 64 vs 1024: 75.65 vs 74.79 (D6).
   - Effective rank of the backbone predicts downstream accuracy (D5); collapse sits in the projector (D4).
   - Matryoshka gives nested small prefixes with full-width accuracy preserved (D7).
   - Prediction for N087 D3 ("narrow SSL from the start"): worse, or at best equal. Replace with a nested (Matryoshka) projection (P2).
5. **Bottlenecks and the cross-draw shift.** VIB claims robustness (D8); the tunnel effect says compression hurts OOD (D9); Matryoshka is neutral in distribution (D7).
   - Prediction for N087 D2 (32-d bottleneck, fresh or SSL-projector init): GOAL flat (+-.003), b01 not better.
   - The SSL-projector init is pretext-specific (D2) and should not beat a fresh init. [derived]
6. **Decoder capacity (user's "how did you decode").**
   - It matters for frozen reconstruction features (C5: one block of nonlinearity +7.5 points).
   - For fine-tuned features it should not matter. The project already found heads flat (K042). N087 A tests it.

---------------------------------------------------------------------------------------------------------------------------------

## (iii) Experiment proposals, ranked by expected value for the GOAL / multi-marker decoding

Shared conventions (project standard):
- Bars: GOAL single model > control + .0054 (2 x noise) over 2 seeds; MM3 > control + .007.
- Inductive: training draw only.
- Fine-tune cost from N078: ~.45 H100-h per 10-epoch GOAL fine-tune per direction.
- SIGReg SSL cost: r50 40 ep 1.9 h (275k-cell draw) / .6 h (138k); ConvNeXt-S 60 ep ~2.4 / 1.2 h.

### P1 (rank 1). Feature-preserving fine-tuning of the in-domain SSL encoders: LP-FT, encoder-LR multiplier, WiSE interpolation

**Hypothesis.**
- Full-LR fine-tuning distorts the in-domain SSL features in a draw-specific way (E1).
- The held-out draw then sees features moved away from the shared SSL geometry.
- In-house evidence: Cytoland lost .021 at encoder lr x1.0 vs x0.1 (N053). The in-domain r50 / ConvNeXt-S have never been fine-tuned with a lower encoder LR or a warm-started head.
- Preserving features should help b01 most, where ConvNeXt-S fell -.015 in N078.
- It should also keep more SSL co-expression structure, measured by MM3 and the sandbox joint metrics.

**Recipe (one item, bundled).**
- Encoders: N031-config SIGReg r50 per draw (existing) and the N085 ConvNeXt-S 120-ep encoders.
- Arms, seed 0, both directions:

| Arm | Change |
|---|---|
| (a) control | n051_v10_base recipe (10 ep, lr 5e-4, bs 256, w-cls .5) |
| (b) | `--enc-lr-mult .1` (flag exists, N053) |
| (c) LP-FT | stage 1: 2 epochs, encoder frozen, BN in eval mode, heads only at lr 5e-4. Stage 2: the standard 10-ep recipe started from that head |
| (d) | LP-FT + enc-lr-mult .3 |

- Post hoc, zero training (WiSE):
  - theta(alpha) = (1 - alpha) theta_LP + alpha theta_FT for alpha in {.25, .5, .75, 1}.
  - Recompute BN running stats at each alpha on 20k TRAINING-draw cells. This is source-only, not AdaBN on the target.
  - Pre-register alpha = .5 (WiSE default). Selecting alpha on the training draw would bias towards alpha = 1.
- Sandbox: arms (a) vs (c) in the N080 single-marker regime on S-BIAD452, three held-out donors. Report macro r, 36-pair error, within-type r, CD4-CD8 correlation in T cells.
- Score: GOAL, MM3, per-direction values.
- Second seed only for an arm passing on seed 0.

**Expected effect.**
- Literature: LP-FT ~+10 points OOD on large shifts (E1); WiSE +2-10 OOD (E2). Both assume strong pretrained features and large shifts.
- Our SSL linear probe is weak (K017). The matched-model cross-draw shift is only -.026 (N059), mostly label-side.
- [derived] Realistic: GOAL single +.000 to +.008 (b01 > b10); MM3 +.000 to +.010; ensemble member +.002-.005 if errors decorrelate from full-LR members.

**Compute.**
- 4 arms x 2 directions x ~.5 h = 4 H100-h (two encoder families: x2 for ConvNeXt-S).
- WiSE evaluations ~.5 h; sandbox ~1 h.
- Total ~6-9 H100-h on 1 x H100 preempt. Fine-tunes are short and data-parallel overhead is not worth it.

**Kill criterion.**
- On seed 0, kill if the best of (b, c, d) and WiSE alpha = .5 is < control + .0054 GOAL AND < control + .007 MM3 for BOTH encoder families.
- If only b01 improves (> +.01) while b10 drops: report, don't adopt.

**N087 overlap.** None. N087 D2 changes the head shape, not the fine-tuning dynamics. (c) can be bundled with D2's `--bottleneck 32` as one extra arm.

### P2 (rank 2). Supervised projection head with Matryoshka nesting + Guillotine readouts (replaces N087 D3; extends D1/D2)

**Hypothesis.**
- (1) A nonlinear projection between pooled features and the marker heads lets the backbone keep within-type (subclass) structure that a direct linear head collapses (D3: +4-10 points fine-grained).
- This matters for JOINT decoding (co-expression, K046) more than for single-marker r.
- (2) Nested-prefix losses answer the user's "how small can it be" in one run, without hurting the full-width GOAL (D7).

**Recipe.**
- Head: pooled h (2048-d r50) -> Linear(2048, 256) + LayerNorm + GELU MLP(256, 256) = z.
- Separate linear marker heads on z[:m] for m in {4, 8, 16, 32, 64, 256}. Matryoshka loss = mean over m of the usual marker loss.
- Keep the standard full-width head on h as well, so the GOAL readout is unchanged and comparable.

Stage 1, sandbox (S-BIAD452, N080 harness, single-marker regime, 3 held-out donors). Report:
- (i) Macro r per prefix m.
- (ii) Guillotine readouts: freeze the encoder, fit ridge and kNN joint decoders on h (pre-projection), z (post) and z[:8].
- (iii) Metrics: macro r, 36-pair error, within-type r, CD4-CD8 correlation, k-means NMI/ARI vs fine types.
- Compare with N080 single (.526, pair error .229) and full (.596).
- Optional VIB arm: z stochastic with KL beta = 1e-3 (D8). Low prior.

Stage 2, BSCCM: the same head on the n051_v10_base recipe, seed 0, both directions. Report GOAL / MM3 from the full head and per prefix m, plus k-means of z[:8] vs gated classes. This gives a compact SUPERVISED cell-state space for the user's clustering question.

**Expected effect.**
- MRL: full width preserved (~0), small prefixes near full at classification.
- [derived] GOAL +-.003 at full width.
- GOAL at m = 16-32 within .005 of full if N087 A finds effective rank <= ~16.
- Sandbox: within-type r +.01-.04 and pair error -.005 to -.02 from pre-projection ridge vs the direct-head baseline. Speculative; the transfer from fine-grained classification to co-expression is untested.

**Compute.** Sandbox 3 donors x ~.3 h + BSCCM 2 x .5 h, ~2-3 H100-h. Readouts on CPU.

**Kill criterion.**
- Sandbox: if pre-projection within-type r is not > baseline + .01 AND pair error is not < .224, do not port the projector to BSCCM.
- BSCCM: if the full-width GOAL is < control - .0054, drop the nesting.

**N087 overlap.**
- D1 (frozen SSL projector) and D2 (single 32-d bottleneck, SSL-projector init) test pieces of this.
- D3 ("narrow SSL from the start", ~6 H100-h per direction) is predicted harmful (D4, D5, D6) and should be replaced by this.

### P3 (rank 3). Clustering-objective in-domain SSL (DINO-style self-distillation with prototypes, no local crops) vs SIGReg

**Hypothesis.**
- SIGReg makes the embedding isotropic (no clusters by design).
- A prototype / self-distillation objective forms soft clusters and is the best SSL objective in direct cell-profiling comparisons: C8 (DINO > MAE, SimCLR), C9 / scDINO (PBMC), C11 (WBC), C12 (DINO best at 1 % labels on immune IFC).
- Expected outcomes:
  - better k-means / kNN cell-type structure (the user's clustering question);
  - errors decorrelated from SIGReg members (ensemble diversity);
  - possibly MM3, since an SSL encoder change already moved MM3 +.0135 (N083).

**Recipe.**
- Backbone: ConvNeXt-S (N078 config, 48-ch stem, 24-ch input), training draw only.
- DINO head:
  - 3-layer MLP 2048-2048-256 + weight-normed prototype layer, K = 4096 (fewer than 65,536 for 1.4-2.8 x 10^5 cells).
  - Teacher temperature .04, student .1, centering momentum .9, EMA .996 -> 1.
  - TWO global views from the SAME augmentation pipeline as the SIGReg arm (no local crops, per C11), 60 ep, bs 512.
- Evaluate the frozen BACKBONE (pre-projector, per D1-D3):
  - kNN (k = 20) balanced accuracy on the 10 gated classes of the held-out draw;
  - k-means K = 10/20 NMI/ARI;
  - ridge GOAL; RankMe;
  - vs SIGReg ConvNeXt-S, SIGReg r50 and the N087 C AE.
- Then GOAL fine-tune (seed 0, both directions; P1's best fine-tuning mode if P1 passes) and the pre-registered union ensemble.

**Expected effect.**
- GOAL single +-.005. All in-domain encoder swaps so far landed within +-.005; ViTs were -.015 to -.021.
- MM3 +-.010.
- Ensemble +.002-.004 (Cytoland as a diverse member gave +.006; saturation ~+.002 per member, K039).
- Clustering NMI / kNN: DINO > SIGReg by a few points (the literature is qualitative, no LF-blood numbers).

**Compute.** DINO ~1.3x SIGReg cost (EMA teacher forward): ~1.5 h (138k draw) + ~3.1 h (275k draw). Fine-tunes ~1 h, evaluations ~.5 h. Total ~6 H100-h; +6 for an r50 replicate.

**Kill criterion.** Stop before fine-tuning (saves ~1 h) if frozen kNN balanced accuracy, k-means NMI and ridge GOAL are ALL <= SIGReg ConvNeXt-S on the same draw. Close if fine-tuned GOAL < control - .0054 and ensemble gain < +.002.

**N087 overlap.** None. C tests a pixel AE vs SIGReg; nothing tests a clustering SSL objective.

### P4 (rank 4). Diagnostics bundle on existing encoders: multi-layer readout + effective-rank metrics (add to N087 A/C/D analysis)

**Hypothesis.**
- (1) Stage-3 or stage-3+4 pooled features of the FROZEN SSL encoders carry more dim-marker / within-type information than stage 4 (C7: +5-27 %; D2).
- (2) RankMe / LiDAR / alpha-ReQ (D5) or the SIGReg loss (D6) of a frozen backbone rank encoders by fine-tuned GOAL / MM3. If so, SSL settings (lambda, projector dim, epochs, objective in P3) can be screened without fine-tunes.

**Recipe.**
- Dump pooled layer2/3/4 (r50) and stage 2/3/4 (ConvNeXt-S) features.
- Encoders: SIGReg r50 40/160 ep, ConvNeXt-S 60/120 ep, Cytoland, DINOv3-S, DinoBloom-B, the N087 C AE encoder (intermediate layers too), and fine-tuned n051_v10_base.
- Cells: 20k training-draw cells + held-out-draw split.
- Fit the N087 A ridge / kNN decoders per layer and per concatenation.
- Compute RankMe (entropy effective rank on 25k cells), the alpha-ReQ slope and k-means NMI vs gated classes.
- Spearman of each metric with fine-tuned GOAL and MM3 across the ~8 encoders.

**Expected effect.**
- [derived] Frozen ridge GOAL +.01-.03 for stage-3 / concat vs stage 4 on SSL encoders; ~0 on fine-tuned features.
- No direct GOAL change, unless a multi-layer head is adopted afterwards (+.000-.004, to be shown by a fine-tune).

**Compute.** ~.5 H100-h inference + CPU.

**Kill criterion.** Diagnostic, no kill needed. Adopt a multi-layer head only if the frozen gain survives a fine-tuned run > +.0054.

**N087 overlap.** Partial. A sweeps decoders on stage-4 / projector features; add layers and rank metrics to the same script.

### P5 (rank 5, park unless P3 shows the SSL objective moves MM3 / ensemble). In-domain hybrid masked reconstruction + SIGReg

**Recipe.** ConvNeXt-S with FCMAE-style 50-60 % patch masking and a light decoder, plus SIGReg on the pooled embedding of two views. Training draw only, 60 ep.

**Rationale.** Masking is the regime where reconstruction helps (C4). It scales (C6). The Cytoland FCMAE member was diverse (+.006). An in-domain version avoids the cross-domain gap.

**Expected.** Ensemble +.002-.004, single +-.005.

**Compute.** ~7 H100-h.

**Kill.** Same frozen screen as P3.

**N087 overlap.** C's AE is unmasked and low-dimensional, which is a different objective.

### Not proposed (literature says low value; listed so they are not re-proposed)

- **beta-VAE / disentangled latents for clustering or decoding:** C3, Locatello.
- **Smaller pixel-AE bottlenecks:** C1, C2, C4.
- **A plain VQ-VAE without a supervised / identity pretext:** C1.
- **k-means on the SIGReg projector output:** D6, D1-D3.
- **A narrow SSL backbone:** D4, D5.
- **VIB as a cross-draw fix:** D9; at most an arm inside P2.
- **Virtual staining as a preprocessing step:** N066 / HEmIF, B2.
- **LED-channel dropout augmentation (C9 used channel dropout):** close to the cross-illumination SSL views (K017, worse) and instrument augmentation (N041, flat).

---------------------------------------------------------------------------------------------------------------------------------

## Predictions for the running N087 arms (to check against when they land)

| N087 arm | Literature prediction | Basis |
|---|---|---|
| A rank-k / MLP / kNN on fine-tuned features | rank <= ~13 recovers the head; MLP / kNN within +.003 | C5 (nonlinearity matters for frozen reconstruction features, not task-aligned ones), K042 |
| A on frozen SSL features | MLP > ridge by a clear margin; stage 3 >= stage 4 (P4) | C5, C7, D2 |
| C image AE k = 16 / 64 / 256 | clusters worse than the SIGReg backbone; worst at k = 16; GOAL from latents far below SSL ridge | C1, C2, C3, C4 |
| D1 SSL projector outputs | 256-d output < 2048-d projector hidden layer < backbone, on clustering and markers | D1, D2, D3, D6 |
| D2 32-d bottleneck fine-tune | GOAL flat (+-.003); SSL-projector init <= fresh init; b01 not better | D7, D9, D2 |
| D3 narrow SSL from the start | worse or equal; replace by P2 (nested projection) | D4, D5, D6, D7 |

## Sources (verified URLs)

**Label-free blood / virtual staining**
- https://arxiv.org/abs/2402.06191
- https://arxiv.org/abs/2405.20559
- https://arxiv.org/abs/2605.14717
- https://doi.org/10.1002/cyto.a.24770
- PMID 31081599 (Nassar 2019)
- https://hdl.handle.net/10481/101351 (Lippeveld)
- https://doi.org/10.1038/s41598-017-06311-y
- https://doi.org/10.1002/aisy.202000277
- https://arxiv.org/abs/2503.12622
- https://doi.org/10.1364/BOE.10.006351
- https://doi.org/10.1016/j.cell.2018.03.040
- https://doi.org/10.1038/s41592-018-0111-2
- https://doi.org/10.1126/sciadv.abe0431
- https://doi.org/10.1038/s41598-022-12914-x
- https://doi.org/10.1101/2024.05.31.596901
- https://arxiv.org/abs/2410.11281

**Single-cell representation learning**
- https://doi.org/10.1038/s41592-022-01541-z
- https://doi.org/10.1371/journal.pcbi.1007348
- https://doi.org/10.1038/s42003-022-03218-x
- https://arxiv.org/abs/1811.12359
- https://arxiv.org/abs/2402.11337
- https://doi.org/10.1038/s42256-024-00934-3
- https://arxiv.org/abs/2111.06377
- https://arxiv.org/abs/2404.10242
- https://arxiv.org/abs/2411.02572
- https://doi.org/10.1038/s41598-025-88825-4
- https://doi.org/10.1101/2023.06.16.545359
- https://doi.org/10.1101/2023.01.16.524226
- https://doi.org/10.1101/2024.12.06.627299
- https://arxiv.org/abs/2404.05022
- https://arxiv.org/abs/2610.31848

**Projector, rank, bottlenecks**
- https://arxiv.org/abs/2002.05709
- https://arxiv.org/abs/2206.13378
- https://arxiv.org/abs/2403.11391
- https://arxiv.org/abs/2110.09348
- https://arxiv.org/abs/2210.02885
- https://arxiv.org/abs/2312.04000
- NeurIPS 2022 alpha-ReQ (proceedings hash 70596d70542c51c8d9b4e423f4bf2736)
- https://arxiv.org/abs/2511.08544
- https://arxiv.org/abs/2205.13147
- https://arxiv.org/abs/1612.00410
- https://arxiv.org/abs/2305.19753

**Fine-tuning under shift**
- https://arxiv.org/abs/2202.10054
- https://arxiv.org/abs/2109.01903
