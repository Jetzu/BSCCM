# Strand A: ML methods for pivot ("binding") alignment and missing modalities, applied to G-L-M on BSCCM

Agent A, literature strand. Written 2026-11-08. Every factual claim about a paper carries a URL. If I read only the abstract, the entry says so. Points I could not confirm from a primary source are marked **[unverified]**. Passages marked **[derivation]** are my own reasoning and are not claimed by any paper.

Problem recap (from the brief): L (label-free LED-array stack) exists for all ~413k cells. M (partially observed surface markers, usually one antibody per cell, all 8 for 37k) exists for ~350k. G (Wright/Giemsa RGB) exists for 15.7k cells, all of them unstained. So G-L (15.7k pairs) and L-M (~350k pairs) are observed, but **G-M never is**. L is therefore the only possible pivot.

---

## 0. Key conclusions (TL;DR)

1. **The transitive link only works under a Markov assumption, and it can only be partly tested.** Every method below (ImageBind, C-MCR, pivot MT, teacher-student distillation, bridge integration) implicitly assumes that the pivot "explains" what the two outer modalities share. For us that means M ⟂ G | L: once L is known, G adds nothing about markers. Even then, the best a G-model trained through L can learn is E[E[M|L] | G] **[derivation]**. Any marker information visible in G but not in L is unrecoverable. Any information visible in L but destroyed by fixation and staining caps accuracy. The Platonic-representation authors make the same point: modalities with different information can only *partially* align, up to their shared information ([Huh et al. 2024](https://arxiv.org/html/2405.07987)). Tjandrasuwita et al. show that alignment stops predicting downstream performance when modality-unique information dominates ([arXiv 2502.16282](https://arxiv.org/html/2502.16282)).
2. **Emergent (pivot-only) links are consistently weaker than directly supervised ones.** For example, LanguageBind's emergent RGB→audio R@1 is 10.6, against ImageBind's 36.9 with direct image-audio training ([LanguageBind](https://arxiv.org/html/2310.01852)). Zero-shot (implicit-pivot) NMT trails explicit pivoting by 6-9 BLEU, and by about 50% for distant languages ([Johnson et al. 2017](https://ar5iv.labs.arxiv.org/html/1611.04558)). Expect G→M to be clearly worse than L→M, and quantify how much worse.
3. **Distilling through the pivot beats both explicit two-step pivoting and implicit zero-shot.** In NMT, a teacher-student scheme trains the source→target student on source-pivot pairs using a pivot→target teacher. It beats the pivot pipeline by +3.0 to +5.6 BLEU ([Chen et al. 2017](https://ar5iv.labs.arxiv.org/html/1705.00753)). Back-translation through the pivot even beats explicit pivoting, 33.1 vs 31.5 BLEU ([Gu et al. 2019](https://ar5iv.labs.arxiv.org/html/1906.01181)). Cross-modal distillation transfers supervision RGB→depth on paired unlabeled images in the same way ([Gupta et al. 2016](https://ar5iv.labs.arxiv.org/html/1507.00448)). For us: train an L→M teacher on ~350k cells, then distil its *predictive distribution* into a G student on the 15.7k G-L pairs.
4. **Freeze, or otherwise protect, the pivot space.** ImageBind freezes the image/text encoders and trains only the new modality encoders ([ImageBind](https://arxiv.org/html/2305.05665v2)). Ex-MCR keeps the base space intact on purpose ([Ex-MCR](https://arxiv.org/html/2310.08884)). LanguageBind's per-branch fine-tuned text towers drifted apart (cosine 0.553 between branch embeddings of identical captions), which broke the pivot ([Fusion Embedding, arXiv 2607.18666](https://huggingface.co/papers/2607.18666.md)). EmergentBridge restricts extra alignment to the subspace orthogonal to the anchor direction so it does not damage the anchor ([arXiv 2604.11043](https://arxiv.org/html/2604.11043)).
5. **A stronger pivot encoder gives stronger emergent links.** ImageBind's emergent zero-shot scores improve with image-encoder size (ViT-B→H: about +7% depth, +4% audio). Self-supervised DINO beat DeiT for emergent audio/depth (ESC 44.7 vs 25.0) even though it does not track ImageNet accuracy ([ImageBind](https://arxiv.org/html/2305.05665v2)). Pretrain the L encoder self-supervised on **all 413k** cells.
6. **Loss: use a SigLIP-style pairwise sigmoid with soft targets, not plain InfoNCE.** Sigmoid "processes every image-text pair independently". It beats softmax "by a large margin" below 16k batch (51.5 vs 47.7 at batch 512) and is more robust to label noise ([SigLIP](https://huggingface.co/papers/2303.15343.md)). Because pairs are independent binary problems, soft or partial targets and masked pairs are trivial to add **[derivation]**. For the continuous, masked marker side, use label-distance-aware objectives: y-aware InfoNCE with an RBF kernel on metadata ([Dufumier et al. 2021](https://ar5iv.labs.arxiv.org/html/2106.08808)), Rank-N-Contrast ([Zha et al.](https://arxiv.org/html/2210.01189v2)), or X-CLR soft similarity-graph targets ([Sobal et al.](https://arxiv.org/html/2407.18134v1)). Compute label distance **only on co-observed markers**, and exclude pairs with no co-observed marker from both the positive and negative sets.
7. **Multimodal VAEs are not the main tool here.** Mixture-based multimodal VAEs (MMVAE, MoPoE) have a provable gap whenever modalities carry modality-specific information ([Daunhawer et al. 2022](https://ar5iv.labs.arxiv.org/html/2110.04121)). G (stain colour) and L (illumination physics) carry a great deal of it. But the **product-of-experts posterior over whatever subset is observed** ([MVAE](https://ar5iv.labs.arxiv.org/html/1802.05335)) and the **set encoder over (feature-id, value) tokens** ([Partial VAE/EDDI](https://ar5iv.labs.arxiv.org/html/1809.11142)) are exactly the right building blocks for encoding partially observed marker vectors.
8. **Honest G-M evaluation is possible without same-cell G-M data, using bounds.** If T(L) is the L→M teacher and g(G) the G student, then RMSE(g,M) ≤ RMSE(g,T) + RMSE(T,M) by Minkowski's inequality, and RMSE(g,M) ≥ |RMSE(T,M) − RMSE(g,T)| **[derivation]**. The first term is measurable on held-out G-L pairs, the second on held-out L-M cells. The only untestable step is that the teacher's error transports from stained to unstained cells, and that is partly checkable via L-distribution shift. Under M ⟂ G | L with a calibrated teacher, the MSEs add exactly **[derivation]**. Add the evaluation templates the literature used for unpaired links: (a) simulated never-co-observed pairs with ground truth (multiome split, [Hao et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517); StabMap multi-hop simulations, [Ghazanfar et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10869270/)), (b) label-mediated zero-shot classification (ImageBind's text-prompt protocol), and (c) confidence collapse on populations missing from the bridge (Hao et al.: pDC prediction score 0.907 → 0.514).

---

## 1. The structure of our problem in pivot terms

| Literature setting | Pivot | Observed pairs | Never co-observed | Our analogue |
|---|---|---|---|---|
| ImageBind ([paper](https://arxiv.org/html/2305.05665v2)) | image | image-text, image-audio, image-depth, ... | audio-text etc. | L = image, G = text side, M = audio side |
| Pivot / zero-resource NMT ([Chen 2017](https://ar5iv.labs.arxiv.org/html/1705.00753)) | English | src-En, En-tgt | src-tgt | G = src, L = En, M = tgt |
| Bridge CorrNet ([Rajendran 2016](https://arxiv.org/abs/1510.03519)) | pivot view V3 | V1-V3, V2-V3 | V1-V2 | same |
| BrokenBind ([arXiv 2602.06451](https://arxiv.org/html/2602.06451)) | shared modality b | dataset1 (a,b), dataset2 (b,c), no cross-dataset sample correspondence | a-c | dataset1 = G-L cells, dataset2 = L-M cells (different cells) |
| Mosaic single-cell integration ([StabMap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10869270/), [bridge integration](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517)) | shared features / bridge dataset | RNA-only, ATAC-only, multiome | e.g. ATAC↔protein | very close: "multi-hop" mosaic |

Two features of our case are not in ImageBind:

- **G-L cells and L-M cells are disjoint populations.** Unstained and stained cells may come from different donors or batches. BrokenBind targets exactly this cross-dataset situation, with an explicit "cross-data" transition ([arXiv 2602.06451](https://arxiv.org/html/2602.06451)). SOTAlign finds that the distribution shift between unpaired and paired sets, measured by spherical sliced Wasserstein distance, correlates with downstream gains ([SOTAlign](https://arxiv.org/html/2602.23353v1)). Check whether L-embeddings of G-cells and M-cells overlap, using a domain classifier or MMD, before trusting any transitive result **[derivation]**.
- **M is not an image-like modality.** For most cells it is one scalar, or a handful of fluorescence values, with a missingness mask. Cell-level "M retrieval" is ill-posed because thousands of cells share nearly the same CD-x value. M behaves more like a **continuous, partially observed label** than a view. That is why the label-distance-aware losses in §5 and the regression-style teacher in §8 matter **[derivation]**.

---

## 2. Pivot-binding models: ImageBind and successors

### 2.1 ImageBind (Girdhar et al., CVPR 2023)

- **What it does.** One embedding space for six modalities. "Only image-paired data is sufficient to bind the modalities together" ([abs](https://arxiv.org/abs/2305.05665)). Image and text encoders come from OpenCLIP and are **frozen**. Audio, depth, thermal and IMU encoders are trained with InfoNCE on naturally paired data: AudioSet video-audio, SUN RGB-D, LLVIP, Ego4D ([HTML](https://arxiv.org/html/2305.05665v2)).
- **Why the transitive link works.** Text and audio are each aligned to the *same frozen* image space, so audio-text similarity emerges if both are predictable from image content. The assumption is that the image carries the information audio and text share.
- **How the unpaired link was evaluated.** On benchmarks that do have audio-text (etc.) ground truth never used in training: zero-shot classification with CLIP text prompts (ESC-50 66.9, LLVIP 63.4, NYU-D 54.0, SUN-D 35.1, AudioSet mAP 17.6) and audio-text retrieval (Clotho R@1/R@10 6.0/28.4, AudioCaps 9.3/42.3) ([HTML](https://arxiv.org/html/2305.05665v2)).
- **How it fails.** It is well below supervised specialists: Clotho 6.0 vs ARNLQ 12.6 R@1, AudioCaps 9.3 vs 24.3, ESC 66.9 vs a 97.0 SOTA ([HTML](https://arxiv.org/html/2305.05665v2)). Coverage is limited to what the pairing data saw (thermal only outdoor, depth only indoor).
- **Ablations relevant to us** ([HTML](https://arxiv.org/html/2305.05665v2)):
  - A larger image encoder improves emergent performance.
  - A fixed temperature beats a learnable one, and the best value differs by modality.
  - A linear projection head beats an MLP (ESC 56.7 vs 51.0).
  - Longer training helps (16→64 epochs: 56.7→62.9).
  - Strong augmentation badly hurts audio (22.6 vs 56.7).
  - Spatially aligned crops matter for depth (26.7 vs 16.0).
  - Best batch size was 512 for depth and 2k for audio.
  - DINO beats DeiT as the vision backbone for emergent tasks.
- **Lesson for us:** registered G-L crops of the same cell matter (cf. spatial alignment). Avoid augmentations that destroy what links the modalities: colour jitter on G alters the stain information, illumination-channel shuffling on L alters the physics.

### 2.2 LanguageBind (Zhu et al., ICLR 2024)

- **Method.** Language is the pivot. The language encoder is frozen and other modality encoders are trained contrastively against it on VIDAL-10M ([abs](https://arxiv.org/abs/2310.01852)).
- **Unpaired-link evaluation.** "Emergent" RGB→X retrieval (Table 6): AVE RGB→audio R@1 10.6 vs ImageBind 36.9, VGGS 10.0 vs 28.7, NYU RGB→depth 17.9, LLVIP RGB→IR 7.5. The authors note "there are no 'fair' baseline models" ([HTML](https://arxiv.org/html/2310.01852)). The emergent link is far weaker than direct pairing on audio and close to text-mediated baselines on depth.
- **Failure.** There is no controlled pivot-swap ablation ([HTML](https://arxiv.org/html/2310.01852)). A 2026 analysis found that LanguageBind's per-branch fine-tuned text towers diverged (mean cosine 0.553 between audio-branch and image-branch embeddings of identical captions). The language pivot then no longer provides one shared space ([arXiv 2607.18666](https://huggingface.co/papers/2607.18666.md)). Lesson: **never maintain separate copies of the pivot encoder per branch.**

### 2.3 Point-Bind (Guo et al. 2023)

- **Method.** Aligns a 3D encoder (I2P-MAE) to frozen ImageBind image, text and audio encoders. It uses ULIP/ShapeNet 3D-image-text triplets plus ESC-50 sounds for 9 ShapeNet categories, and skips the audio loss for silent categories ([HTML](https://arxiv.org/html/2309.00615)).
- **Evaluation.** 3D-audio "emergent" ability is shown only qualitatively, via embedding arithmetic and audio-to-mesh figures, with no quantitative audio-3D numbers ([HTML](https://arxiv.org/html/2309.00615)). This is a cautionary example of an unpaired link claimed without a quantitative test.

### 2.4 UniBind (Lyu et al., CVPR 2024)

- **Problem and method.** Image-centric binding gives an "unbalanced" space. UniBind replaces the image anchor with LLM-built, modality-agnostic class-wise embedding centres and aligns all modalities to them contrastively. It reports +6.36% average zero-shot gain ([abs only](https://arxiv.org/abs/2403.12532)).
- **Relevance.** The analogue would be anchoring all three modalities to cell-type prototypes. But we only have 10k labelled cells and the labels are marker-gated (M-derived), so this would leak M into the "shared" centre. Use only as an auxiliary loss **[derivation]**.

### 2.5 EmergentBridge (Xie et al. 2026)

- **Diagnosis.** Two modalities can each align well with the anchor yet stay weakly connected to each other, and naive proxy alignment causes gradient interference that degrades the anchor ([HTML](https://arxiv.org/html/2604.11043)).
- **Method and gains.** A diffusion-predicted proxy "bridge anchor", with alignment restricted to the subspace orthogonal to the anchor direction (stop-gradient). Gains over ImageBind: +10.3 (LLVIP), +16.1 (NYU-D), +8.5 (VGG-S) zero-shot top-1, and +5.0/+6.9 R@10 on Clotho/AudioCaps ([HTML](https://arxiv.org/html/2604.11043)).
- **Evaluation.** Same held-out-benchmark protocol as ImageBind.
- **Limitations.** Gains are not uniform, and λ needs tuning.

---

## 3. Connecting pretrained spaces without paired data (C-MCR family, FreeBind, OmniBind, BrokenBind)

### 3.1 C-MCR (Wang et al., NeurIPS 2023)

- **Method.** Connects two contrastive spaces, CLAP (audio-text) and CLIP (image-text), through their overlapping modality (text). Only small projectors are trained ([abs](https://arxiv.org/abs/2305.14381), [HTML](https://arxiv.org/html/2305.14381v2)). Components:
  - **Semantic enhancement:** memory-based softmax aggregation of unpaired images and audio around each text embedding, plus Gaussian noise injection followed by re-normalisation.
  - **Inter-MCR alignment:** text-text contrastive plus pseudo audio-visual pairs.
  - **Intra-MCR alignment:** a pull-only L2 term that closes the modality gap. It is pull-only because the repulsive contrastive term maintains the gap.
- **Assumption.** Overlapping-modality embeddings in both spaces are views of the same meaning, and each source space is internally aligned.
- **Unpaired-link evaluation.** Zero-shot audio-image retrieval (AVE mAP 4.11/4.13, Flickr-SoundNet 4.57/4.92), source localisation, counterfactual recognition, and 3D ModelNet40 (64.9 vs ULIP-2 74.0) ([HTML](https://arxiv.org/html/2305.14381v2)).
- **Failure modes.** Without the text-text connection, retrieval falls to about random (~0.2 mAP). Removing noise injection causes a large drop. Removing intra-alignment pushes performance back toward the no-text-pair level ([HTML](https://arxiv.org/html/2305.14381v2)). Absolute numbers are very low.
- **Lesson.** The modality gap ([Liang et al. 2022](https://arxiv.org/abs/2203.02053)) must be closed explicitly if a head trained on one modality's embeddings is applied to another's.

### 3.2 Ex-MCR (Wang et al. 2023)

- **Method.** Freezes a base space (CLIP) and projects a leaf space (CLAP or ULIP) into it, so the base alignment is preserved. Uses modality-centric pseudo-pairs, a decoupled projector (linear intra + MLP inter) and dense InfoNCE ([HTML](https://arxiv.org/html/2310.08884)).
- **Evaluation.**
  - Audio-image mAP 4.94/4.46/6.39 vs C-MCR 4.74/4.21/5.95.
  - AudioCaps audio-text 11.19, vs CLAP's own 21.98: the inherited link loses half its quality.
  - ModelNet40 66.53 vs ULIP-2 73.06 ([HTML](https://arxiv.org/html/2310.08884)).
- **Lesson.** **Transitively inherited links lose substantial quality even when the base link is strong.**

### 3.3 FreeBind (ICML 2024) and OmniBind (2024)

- **FreeBind.** Upgrades a unified space (ImageBind) with expert spaces through "space bonds", reporting gains on 5 tasks / 9 datasets ([abs only](https://arxiv.org/abs/2405.04883)). Pseudo-pair construction not verified **[unverified]**.
- **OmniBind.** Binds 14 pretrained spaces to EVA-CLIP-18B with small MLP projectors.
  - Pseudo-pairs come from top-1 item-level retrieval over unpaired data.
  - Learned routers weight the source spaces per input.
  - A "language representation decoupling" BCE fights the knowledge interference that fixed weights cause.
  - Evaluation: held-out retrieval and classification benchmarks ([HTML](https://arxiv.org/html/2407.11895)).
- **Relevance.** An existing blood-smear or pathology foundation-model space for G can be projected into our L hub (the Ex-MCR "leaf into base" pattern). Retrieval-based pseudo-pairs (nearest L-M cells to the L image paired with each G cell) are a non-parametric teacher **[derivation]**.

### 3.4 BrokenBind (Huang et al. 2026): the closest structural match

- **Setting.** Dataset 1 has (a,b), dataset 2 has (b,c), and there is no instance correspondence across datasets. The method estimates linear "transition matrices" via Moore-Penrose pseudo-inverses of pivot embeddings, with a cross-modal transition (b→c) and a cross-data transition (dataset 2→1). It synthesises pseudo-embeddings for the missing modality and adds a Frobenius consistency term between the two estimates plus CyCLIP-style and symmetry losses ([HTML](https://arxiv.org/html/2602.06451)).
- **Evaluation.** Retrieval across held-out datasets, e.g. au→vi→te on AVE→MSRVTT: ImageBind 11.4 → 35.3 mAP ([HTML](https://arxiv.org/html/2602.06451)).
- **Limitations.** Modality imbalance. Failure under distribution mismatch is not analysed in detail.
- **Relevance.** Its explicit cross-*dataset* term is what our unstained-vs-stained cell populations need.

### 3.5 Bridge CorrNet (Rajendran et al., NAACL 2016)

- **Setting and method.** The earliest explicit formulation I found. It learns a common representation of V1, V2 and V3 from only V1-V3 and V2-V3 parallel data ([abs](https://arxiv.org/abs/1510.03519)).
- **Evaluation.** Multilingual document classification and image→language retrieval via a pivot language ([abs](https://arxiv.org/abs/1510.03519)). Comparison to direct-pair training not verified **[unverified]**.

---

## 4. Few-anchor / semi-supervised alignment (OT, GW, relative representations)

These matter for (i) aligning G to L with "only" 15.7k pairs and (ii) bringing **unpaired clinical smears** into the G encoder's space.

### 4.1 SOTAlign (Roschmann, Krzakala, Mazelet, Bouniot, Akata; ICML 2026)

- **Method.** Aligns frozen unimodal encoders (DINOv3, NV-Embed-v2) in two stages ([HTML](https://arxiv.org/html/2602.23353v1), [ICML](https://icml.cc/virtual/2026/poster/65966)):
  1. A **linear teacher** (Procrustes, CCA, or contrastive linear) is fit on few pairs.
  2. The student is trained with **SigLIP on pairs + α·KLOT** on unpaired batches. KLOT is the KL between entropic OT plans of teacher and student affinities, KL(OT_ε*(K*) ‖ OT_ε(K)). Its explicit gradient (OT_ε(K) − OT_ε*(K*))/ε* avoids unrolled Sinkhorn: about 100× less memory and 50× faster.
- **Assumption.** The unimodal geometries are already compatible ("we posit", Platonic Representation Hypothesis), so the teacher's relational structure transfers to unpaired data.
- **Evaluation.** COCO/Flickr zero-shot retrieval and zero-shot classification.
  - 10k pairs + 1M unpaired: COCO T2I/I2T 26.5/34.1 vs SAIL 21.0/27.4; ImageNet 46.1 vs 38.2.
  - Matches SAIL with about 4× fewer pairs.
- **Failure modes.**
  - **All methods fail at 100 pairs.**
  - Gains depend on encoder quality (r = 0.83 between representational similarity and MeanR@1).
  - Unpaired data far from the paired distribution helps less.
- **Relevance.** 15.7k G-L pairs is in SOTAlign's working regime (10³-10⁴ pairs). Unpaired clinical smears + unpaired L could refine the G-L map. SOTAlign used frozen encoders and vision-language only, so biology is untested.

### 4.2 STRUCTURE (Gröger et al., NeurIPS 2025)

- **Method.** Regulariser that preserves each unimodal space's multiscale neighbourhood structure: JS divergence between similarity distributions, using random-walk powers. Aligns the layer pair with the highest mutual-kNN similarity instead of the last layer ([HTML](https://arxiv.org/html/2506.16895v1)).
- **Results.** 80k COCO pairs; average relative gains of 51.6% (classification) and 91.8% (retrieval) across 24 benchmarks; helps with as few as 1,000 samples ([HTML](https://arxiv.org/html/2506.16895v1)).
- **Limitations.** Authors say biology is untested.
- **Relevance.** A cheap geometry-preserving regulariser for the G encoder so it does not collapse onto L's geometry. CyCLIP's in-modal consistency, matching image-image to text-text similarity, is the same idea ([CyCLIP](https://arxiv.org/abs/2205.14459)).

### 4.3 S-CLIP (Mo et al., NeurIPS 2023)

- **Method.** Semi-supervised CLIP for specialist domains with few captions: OT-based caption-level pseudo-labels plus keyword-level partial-label learning ([NeurIPS](https://neurips.cc/virtual/2023/poster/71829), [arXiv](https://arxiv.org/abs/2305.14095v1)).
- **Results.** Remote sensing: +10% zero-shot classification over CLIP, matching supervised CLIP with 3× fewer pairs (per search summary of the abstract).
- **Relevance.** The OT step prevents pseudo-labels from collapsing onto a few targets, which applies to pseudo-marker assignment for unpaired clinical G.

### 4.4 ASIF and relative representations

- **ASIF.** Represents each input by its similarities to anchor pairs in its own modality (top-k = 800, exponent p = 8, 1.6M CC12M pairs). It reaches 18% ImageNet zero-shot with only 10k pairs and no training, but is weaker than CLIP/LiT at scale ([ASIF text](https://huggingface.co/papers/2210.01738.md)).
- **Relative representations** (Moschella et al., ICLR 2023). Anchor-similarity coordinates are invariant to isometries and rescalings and enable zero-shot stitching ([abs](https://arxiv.org/abs/2209.15430)).
- **Relevance.** A training-free baseline for G→M: represent a G cell by its similarities to the 15.7k anchor G cells, carry those weights to the anchors' L embeddings, then to M via kNN over L-M cells. This is bridge-integration-style "cells as dictionary atoms" ([Hao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517)) **[derivation]**.

### 4.5 Gromov-Wasserstein / unsupervised alignment, and why few anchors beat none

- **GW for word-embedding alignment** (Alvarez-Melis & Jaakkola 2018). Compares intra-space distance structures and is "purely unsupervised" ([abs](https://arxiv.org/abs/1809.00013)).
- **Where it fails** ([Søgaard et al. 2018](https://arxiv.org/abs/1805.03620)):
  - Unsupervised induction degrades for morphologically rich languages.
  - It also degrades for different-domain corpora and different embedding algorithms.
  - A weak seed signal (identical strings) helps.
- **Few anchors beat none** ([Vulić et al. 2019](https://arxiv.org/abs/1909.01638)):
  - Fully unsupervised methods "yield zero BLI performance for 87/210 pairs".
  - Weakly supervised methods with 500-1000 seed pairs are never beaten by unsupervised ones.
- **Seeded graph matching / QAP** (Maniparambil et al., CVPR 2024). With 320 seeds, it matches 500 queries at 72.3% (COCO, CLIP-V) without training ([HTML](https://arxiv.org/html/2401.05224)). Accuracy drops as the query/seed ratio grows and under noise.
- **Single-cell analogues.**
  - SCOT: GW alignment of unpaired multi-omics, self-tuned via the GW distance, evaluated with FOSCTTM (fraction of samples closer than the true match) on co-assayed data ([bioRxiv](https://www.biorxiv.org/content/10.1101/2020.04.28.066787v2.full.pdf+html), [OTT tutorial](https://ott-jax.readthedocs.io/tutorials/quadratic/200_gromov_wasserstein_multiomics.html)).
  - Pamona: partial GW, can use priors (cell-type labels or known correspondences), but needs the number of shared cells as input ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8696097), [AMSS summary](https://amss.cas.cn/kyjz1/202204/t20220429_6441174.html)).
- **Lessons for us.**
  - We have 15.7k G-L anchors, so we never need unsupervised GW for G-L.
  - Use (fused/partial) OT only for (a) clinical-smear → BSCCM-G domain adaptation and (b) diagnostics.
  - FOSCTTM on held-out G-L and L-M pairs is the standard per-hop metric.

### 4.6 Platonic Representation Hypothesis (Huh et al., ICML 2024)

- **Claim.** Representations converge across models and modalities, measured with a mutual-kNN alignment metric ([abs](https://arxiv.org/abs/2405.07987), [HTML](https://arxiv.org/html/2405.07987)).
- **Caveat** (from the paper's own limitations section): "Two different models cannot converge to the same representation if they have access to fundamentally different information"; alignment is capped by shared information ([HTML](https://arxiv.org/html/2405.07987)).
- **Use.** Measure mutual-kNN between G and L embeddings on held-out pairs as a diagnostic of how much information they share.

---

## 5. Losses: SigLIP vs CLIP, false negatives, continuous/partial targets

### 5.1 SigLIP vs InfoNCE (CLIP)

- **Loss.** SigLIP: L = −(1/|B|) Σ_ij log σ(z_ij(t·x_i·y_j + b)), with z_ij = ±1. It "processes every image-text pair independently". Initialise t' = log 10 and b = −10 ([SigLIP text](https://huggingface.co/papers/2303.15343.md)).
- **Small batches.** Sigmoid beats softmax "by a large margin" when the batch is under 16k (Table 5: 51.5 vs 47.7 at 512). It is "increasingly robust to all kinds of added noise". Batch-size gains saturate around 32k ([SigLIP text](https://huggingface.co/papers/2303.15343.md); [abs](https://arxiv.org/abs/2303.15343)).
- **For us** **[derivation]**:
  - G-L batches will be small (23×128×128 L stacks plus RGB), which favours sigmoid.
  - Per-pair independence makes it trivial to (i) give **soft targets** z_ij ∈ [0,1] for near-identical cells, (ii) **mask** pairs whose relation is unknown (different antibodies, no co-observed marker), and (iii) allow multiple positives.
  - SOTAlign also uses SigLIP on its paired term ([HTML](https://arxiv.org/html/2602.23353v1)).

### 5.2 False negatives and soft targets

- **The problem.** Random negatives can share the anchor's label, and correcting for this improves representations ([Debiased CL, Chuang et al. 2020](https://arxiv.org/abs/2007.00224)). False-negative elimination or attraction gives consistent gains ([Huynh et al.](https://arxiv.org/abs/2011.11765)).
- **X-CLR** ([HTML](https://arxiv.org/html/2407.18134v1)).
  - Replaces one-hot InfoNCE targets with s_ij = softmax_j(G_ij/τ_s) from a similarity graph built from captions or labels.
  - Results: +16.8% over CLIP on ImageNet when trained on CC3M; its advantage is largest in the low-data regime.
  - Lower τ_s recovers SimCLR. High τ_s destabilises training, and quality depends on the graph.

### 5.3 Continuous labels: y-aware InfoNCE and Rank-N-Contrast

- **y-aware InfoNCE** (Dufumier et al. 2021).
  - L = −Σ_k [w_σ(y_k,y_i)/Σ_j w_σ(y_j,y_i)] · log[exp f(v_i,v_k) / (1/n) Σ_j exp f(v_i,v_j)], with an RBF kernel on continuous metadata.
  - σ→0 recovers InfoNCE; σ→∞ puts everything in one class.
  - Beats SimCLR even at N = 100 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2106.08808)).
- **Rank-N-Contrast** (Zha et al., NeurIPS 2023).
  - For anchor i and positive j, the negatives are S_ij = {k : d(y_i,y_k) ≥ d(y_i,y_j)}, for a generic label distance d.
  - The paper used L1 distance, and angular distance for 2-D gaze vectors.
  - Theorem: near-optimal loss implies embeddings ordered by label distance.
  - Lowers error by 5.8-11.7% averaged across baselines on 4 benchmarks, and is more robust to corruption, spurious targets and data reduction ([HTML](https://arxiv.org/html/2210.01189v2)).
  - Multi-dimensional labels work through any distance d. Partially observed labels are not treated in the paper.

### 5.4 Recommended handling of partially observed marker vectors in a contrastive objective [derivation]

1. **Masked label distance.** d_ik = (1/|O_i∩O_k|) Σ_{m∈O_i∩O_k} |ỹ_im − ỹ_km|, computed on per-marker standardised (e.g. arcsinh/logicle) intensities over co-observed markers O_i∩O_k. If O_i∩O_k = ∅, the pair is **unknown**: weight 0 in both numerator and denominator (SigLIP mask), or excluded from S_ij in RnC.
2. **Batch construction.** Stratify batches by antibody group so most pairs are comparable. Single-antibody batches give a dense 1-D label. Mixing in the 37k all-marker cells provides full-vector comparisons.
3. **Soft targets.** Use target w_ik = exp(−d_ik²/2σ²) with σ set to the measurement noise of each marker, estimated from replicate or isotype control if available. Use it as the SigLIP target, or as the y-aware positive weight. RnC is the right choice when ordering matters more than calibrated similarity. Note that strict ranking over-penalises noise-level differences, so treat differences below the noise floor as ties.
4. **Positive definition for L-M.** The same-cell pair keeps target 1. Other cells get w_ik ≤ 1. Do not let w_ik for ~identical marker values push L embeddings of morphologically different cells together. This is why the L-L geometry should also be preserved (STRUCTURE / KLOT style regulariser) **[derivation]**.

---

## 6. Multimodal VAEs and missing modalities

### 6.1 MVAE (Wu & Goodman, NeurIPS 2018)

- **Posterior.** q(z|X) ∝ p(z)·∏ q̃(z|x_i). A missing modality is simply dropped from the product. Training uses a **sub-sampled** objective: the full-set ELBO, each single-modality ELBO, and k random subsets ([ar5iv](https://ar5iv.labs.arxiv.org/html/1802.05335)).
- **Weak supervision.** Only a fraction of the data is paired.
  - MNIST: 0.1% paired → 0.284 accuracy, 0.5% → 0.859, 2% → 0.939, 100% → 0.968.
  - "equals a fully supervised network even with two orders of magnitude less paired examples" on FashionMNIST ([ar5iv](https://ar5iv.labs.arxiv.org/html/1802.05335)).
- **Failures.** Very low pairing fails, reconstructions are blurry, and training only on complete data leaves the unimodal experts untrained ([ar5iv](https://ar5iv.labs.arxiv.org/html/1802.05335)).
- **PoE critique.** MMVAE argues PoE experts "hold the power of veto", so overconfident experts dominate ([ar5iv](https://ar5iv.labs.arxiv.org/html/1911.03393)).

### 6.2 MMVAE (Shi et al., NeurIPS 2019)

- **Posterior.** Mixture-of-experts, q(z|x_1:M) = (1/M) Σ q(z|x_m).
- **Coherence evaluation.**
  - Pretrained digit classifiers on cross-generations: MNIST→SVHN 86.4%, SVHN→MNIST 69.1%; MVAE near chance on these.
  - Linear latent probes.
  - CCA for CUB image-caption.
  - Source: [ar5iv](https://ar5iv.labs.arxiv.org/html/1911.03393).
- **Cost.** M² decoder passes.

### 6.3 MoPoE (Sutter et al., ICLR 2021)

- **Posterior.** A mixture over all 2^M subsets of the PoE of each subset. MVAE and MMVAE are special cases. The joint posterior is available for any observed subset ([ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02470)).
- **Evaluation.** Coherence via pretrained classifiers, latent logistic regression, and importance-sampled likelihood on PolyMNIST, MNIST-SVHN-Text and CelebA ([ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02470)).

### 6.4 "On the limitations of multimodal VAEs" (Daunhawer et al., ICLR 2022)

- **Theorem.** For mixture-based models with modality sub-sampling, E[log p(x)] ≥ L_S + Δ(X,S), with Δ = Σ_A ω_A H(X_{\A} | X_A) ≥ 0. Modality-specific information that cannot be predicted from a subset caps the ELBO ([ar5iv](https://ar5iv.labs.arxiv.org/html/2110.04121)).
- **Evidence.** MMVAE and MoPoE trail unimodal VAEs by more than 60 FID points on PolyMNIST and about 30 on CUB, and no model was both high-quality and coherent on CUB.
- **Practical warnings.** FID and likelihood hide coherence failures, and coherence needs ground-truth shared-factor annotations ([ar5iv](https://ar5iv.labs.arxiv.org/html/2110.04121)).
- **For us.** G→L or G→M *generation* will be an average, which is fine for E[M|G] and wrong for per-cell detail.

### 6.5 MMVAE+ (Palumbo et al., ICLR 2023)

- **Method.** Separate shared and modality-specific latents. Cross-modal generation samples the private part from a (learnable) prior. Improves quality without losing coherence and is robust to latent dimensionality and regularisation hyperparameters ([ICLR](https://iclr.cc/virtual/2023/poster/11084), [multivae docs](https://multivae.readthedocs.io/en/stable/models/multimodal_vaes/mmvae_plus.html)).
- **Not verified.** Numbers not verified **[unverified]**.

### 6.6 Partially observed feature vectors

- **Partial VAE (EDDI, Ma et al. ICML 2019).**
  - Uses a "permutation invariant set function encoding" of observed (identity embedding e_d, value x_d) pairs, combined by concatenation (PN) or multiplication (PNP) and summed.
  - The ELBO is computed only over observed variables.
  - Zero-imputation "introduces bias when the data are not missing completely at random" and confuses missing with true zeros ([ar5iv](https://ar5iv.labs.arxiv.org/html/1809.11142)).
  - This is the right encoder for M.
- **totalVI (Gayoso et al., Nat Methods 2021).**
  - Integrates CITE-seq datasets with **different or non-overlapping antibody panels** and imputes absent proteins. Missing panels are identified per batch.
  - Caveat: imputation "is somewhat reliant on how well the datasets mix in the latent space", assuming shared subpopulations ([search summary of bioRxiv/scvi-tools docs](https://www.biorxiv.org/content/10.1101/2020.05.08.083337v1.full.pdf), [scvi-tools](https://docs.scvi-tools.org/en/stable/user_guide/models/totalvi.html)).
  - Our "one antibody per cell" M is an extreme panel-mosaic.
- **MultiVI (Ashuach et al.).** Jointly models paired (multiome) and single-modality cells in one latent space ([bioRxiv abs](https://www.biorxiv.org/content/10.1101/2021.08.20.457057v1)). Mechanism details not verified **[unverified]**.

---

## 7. Missing-modality robustness (train/test-time absence)

- **ModDrop** (Neverova et al.). Randomly drops whole modality channels during training so the fused model predicts from any subset ([abs](https://arxiv.org/abs/1501.00102)).
- **Ma et al. CVPR 2022.** Multimodal transformers are not robust to missing modalities, the best fusion strategy is dataset-dependent, and they propose an automatic fusion search ([abs](https://arxiv.org/abs/2204.05454)).
- **ShaSpec** (Wang et al., CVPR 2023) ([HTML](https://arxiv.org/html/2307.14126v2)).
  - One shared encoder for all modalities plus modality-specific encoders.
  - A missing modality's feature is the **mean of the available modalities' shared features**.
  - Auxiliary losses: distribution alignment (shared features must not reveal the modality) and domain classification (specific features must).
  - Gains of more than 3-5% on BraTS2018.
- **GMC** (Poklukar et al., ICML 2022). Contrasts each modality-specific embedding with the joint embedding of the complete observation, so **training needs complete tuples** ([ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03390)). The same holds for higher-order objectives such as **Symile**, which captures total correlation; its missing-data handling reportedly adds missingness indicators ([arXiv](https://arxiv.org/html/2411.01053v1)) **[unverified detail]**.
- **Survey.** Wu et al. ([arXiv 2409.07825](https://arxiv.org/html/2409.07825v4)).
- **Implications for us** **[derivation]**:
  - Since no cell has G, L and M together, GMC, Symile and GRAM-style joint objectives cannot be trained for the full triple.
  - ShaSpec-style "fill the missing modality with the shared feature of the available one" is exactly what a pivot hub does at inference (G → shared → M head).
  - ModDrop over the 23 illumination channels is useful if downstream users have only a subset of L patterns.

---

## 8. Pivot translation in NLP: error compounding and how it was beaten

- **Explicit pivoting compounds errors.** In two-step pivot decoding, "translation errors made in the first step ... will affect the second step" ([Chen et al. 2017](https://ar5iv.labs.arxiv.org/html/1705.00753)).
- **Implicit zero-shot is worse than explicit pivoting** ([Johnson et al. 2017](https://ar5iv.labs.arxiv.org/html/1611.04558)).
  - Pt→Es: 21.6/24.8 BLEU vs 30.9 pivot (direct NMT 31.5); Es→Ja: 9.1 vs 18.0, roughly a 50% drop.
  - Zero-shot sentences often have embeddings separated from the trained directions.
  - Embedding distance correlates with BLEU (r = −0.42).
- **Why zero-shot fails, and the fix** ([Arivazhagan et al. 2019](https://ar5iv.labs.arxiv.org/html/1903.07091)).
  - Cause: the encoder is not pivot-invariant ("no guarantee that it will discover language invariant representations").
  - Fix: an explicit alignment loss, either a cosine pull between pooled encodings of parallel sentences or an adversarial loss.
  - Result: de→fr 17.0 → 25.9 vs pivot 26.3.
- **Spurious correlations** ([Gu et al. 2019](https://ar5iv.labs.arxiv.org/html/1906.01181)).
  - Zero-shot also suffers from spurious correlations that produce off-target output.
  - Back-translation through the pivot creates synthetic direct pairs: ZS+BTTP 33.1 vs pivot 31.5 vs supervised 34.4.
- **Teacher-student** ([Chen et al. 2017](https://ar5iv.labs.arxiv.org/html/1705.00753)).
  - Assumption: "parallel sentences have close probabilities of generating a sentence in a third language". Our analogue: a G image and its paired L image give the same marker distribution, i.e. M ⟂ G | L.
  - The student is trained on src-pivot pairs to match the pivot→tgt teacher, avoiding two-step decoding.
  - Word-level sampling performs best: +3.0 BLEU over the pivot baseline, and +5.6 with little source-pivot data.
- **Cross-modal distillation** (Gupta, Hoffman & Malik). Same idea for vision: match a labelled RGB network's pool5 features on unlabelled RGB-D pairs. Depth detection mAP 34.2 → 41.7 with 10k pairs. Transfer at higher layers degraded with limited pairs ([ar5iv](https://ar5iv.labs.arxiv.org/html/1507.00448)).
- **Lesson for G→M** **[derivation]**:
  - (a) Don't chain two separately trained regressors at inference (G→L-embedding→M-head); the M-head will see off-manifold G embeddings (the modality gap / "different embedding" effect).
  - (b) Instead distil the teacher's marker *distribution* p(M|L) into a G student on the 15.7k G-L pairs. This is back-translation-style pseudo-labelling of G cells with marker labels.
  - (c) Add an explicit alignment term (cosine/SigLIP) between G and L embeddings so a shared head is also usable, as Arivazhagan's fix does.

---

## 9. Cycle consistency: useful regulariser, unsafe evidence

- **CycleGAN.** Adversarial mapping alone is "highly under-constrained", and cycle loss narrows it ([ar5iv](https://ar5iv.labs.arxiv.org/html/1703.10593)). Stated failures:
  - geometric changes;
  - out-of-distribution content ("has not seen images of horseback riding");
  - label permutations (tree↔building);
  - a gap to paired methods.
- **Steganography.** CycleGAN can satisfy cycle consistency by hiding the source in imperceptible high-frequency signals ([Chu et al. 2017](https://arxiv.org/abs/1712.02950)).
- **Hallucination.** Distribution-matching losses can **hallucinate features** when target-domain class proportions differ, and translated images should not be used for direct interpretation ([Cohen et al. 2018](https://arxiv.org/abs/1805.08841), abstract only).
- **CyCLIP.** Cross-modal and in-modal symmetry regularisers for CLIP, +10-24% zero-shot ([abs](https://arxiv.org/abs/2205.14459)).
- **For us** **[derivation]**:
  - Never use "G→M predicted marker distribution matches the M population" as a *training* loss. It would impose the stained donors' cell-type proportions on the unstained G donors (the Cohen failure).
  - Never cite round-trip consistency (G→L→G) as evidence the G-M link is right (the steganography failure).
  - Use in-modal relational consistency (CyCLIP/STRUCTURE) only as a regulariser.

---

## 10. How the literature evaluated unpaired links, mapped to G-M

| # | Protocol | Used by | G-M analogue |
|---|---|---|---|
| E1 | Held-out benchmark where the "emergent" pair *does* have ground truth, never used in training | ImageBind, LanguageBind, C-MCR, Ex-MCR, EmergentBridge, BrokenBind (links above) | Not directly possible for G-M. Build **simulated triangles** with ground truth (E2). |
| E2 | Take fully observed data and **pretend** it is unpaired; ground truth known | Bridge integration: multiome split, Jaccard/AUC/kNN purity ([Hao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517)); StabMap multi-hop simulations on PBMC multiome ([Ghazanfar et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10869270/), search summary); MVAE shuffled pairs ([ar5iv](https://ar5iv.labs.arxiv.org/html/1802.05335)) | **Marker-split triangle:** on the 37k all-marker cells, define X = markers of spectral groups {1,2} and Y = groups {3,4}. Train X-L on cell set A and L-Y on set B, then evaluate X→Y on set C where truth exists. **Channel triangle:** X = a subset of L channels (e.g. DPC), pivot = remaining L channels. These calibrate how much the *machinery* loses per hop; they do not test G-specific information. |
| E3 | Label-mediated zero-shot (text prompts as class anchors) | ImageBind ([HTML](https://arxiv.org/html/2305.05665v2)) | Predict markers on G, apply the **same marker gates** used for the 10k labelled cells, and compare with **morphological** cell classes on G, which a hematologist can annotate on Giemsa. Report a confusion matrix and canonical-marker checks per morphological class. |
| E4 | Indirect check against canonical biology when no ground truth exists | Hao et al. checked label transfer against canonical loci ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517)) | Per morphological class, expected lineage markers (panel-dependent) must be high and others low. Also check that predicted marker *distributions* on G are plausible. Use this as a diagnostic only, never as a loss (§9). |
| E5 | Confidence collapse when the bridge lacks a population | Hao et al.: pDC removed from bridge → prediction score 0.907 → 0.514; ≥50 bridge cells per type suffice ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517)) | Count G-L pairs per cell type; rare types (e.g. basophils) may fall below ~50. Run a **leave-type-out** test (drop one type from G-L training) and require calibrated uncertainty to rise. |
| E6 | Per-hop retrieval with known correspondences | FOSCTTM in SCOT ([OTT tutorial](https://ott-jax.readthedocs.io/tutorials/quadratic/200_gromov_wasserstein_multiomics.html)), R@k in all binding papers | FOSCTTM/R@k on held-out G-L pairs. Marker-regression R²/AUC on held-out L-M cells. |
| E7 | Embedding overlap as a predictor of zero-shot quality | Johnson et al. (r = −0.42 distance vs BLEU) ([ar5iv](https://ar5iv.labs.arxiv.org/html/1611.04558)); Platonic mutual-kNN ([HTML](https://arxiv.org/html/2405.07987)) | Mutual-kNN between G and L embeddings on held-out pairs. Domain-classifier AUC between L-embeddings of the G-cell pool and the M-cell pool (covariate shift of the pivot). |

**E8, error bounds for G→M without same-cell G-M data [derivation].**

- **Setup.** Let T(L) be the teacher's marker prediction and g(G) the student's.
- **Bounds.**
  - Upper bound (always valid): RMSE(g,M) ≤ RMSE(g,T) + RMSE(T,M).
  - Lower bound: RMSE(g,M) ≥ |RMSE(T,M) − RMSE(g,T)|.
  - Classification analogue (union bound): P(g≠M) ≤ P(g≠T) + P(T≠M).
- **Where each term is measured.** RMSE(g,T) on held-out **G-L pairs** (same cell). RMSE(T,M) on held-out **L-M cells**, which must transport to unstained cells; check with E7.
- **Exact decomposition.** If M ⟂ G | L and T = E[M|L], then MSE(g,M) = MSE(g,T) + MSE(T,M) exactly, because the cross-term E[(M−T)(T−g)] = E[(E[M|L,G] − E[M|L])(T−g)] = 0.
- **What stays untestable.** The cross-term, i.e. marker information in G that L lacks. Report it as the residual assumption. This gives a reportable *interval* for G→M accuracy.

---

## 11. Recommendation for G-L-M

### 11.1 Architecture: an L-hub with frozen pivot, a G student, and a masked-marker set encoder

**Stage 0: pivot encoder (L).**
- Self-supervised pretraining (DINO/MAE-style) on all ~413k L stacks.
- Pretraining on all cells gives the strongest possible pivot; ImageBind showed emergent quality rises with pivot-encoder strength and that SSL DINO features carried emergent links better than supervised DeiT ([HTML](https://arxiv.org/html/2305.05665v2)).
- Keep illumination channels as input channels. ModDrop over channels is optional ([abs](https://arxiv.org/abs/1501.00102)).

**Stage 1: L-M binding (~350k cells).**
- **Marker encoder.** A set/PNP encoder over (marker-id embedding, spectral-group embedding, value) tokens ([Partial VAE](https://ar5iv.labs.arxiv.org/html/1809.11142)), no zero-filling. Optionally make it a **PoE over observed markers** with a prior expert ([MVAE](https://ar5iv.labs.arxiv.org/html/1802.05335)), so uncertainty shrinks as more markers are observed. The 37k all-marker cells train the full product.
- **Losses:**
  - (a) SigLIP-style sigmoid L↔M with masked soft targets (§5.4).
  - (b) A **probabilistic L→M teacher head**: heteroscedastic Gaussian or quantile loss per marker, computed only on observed entries.
  - (c) Rank-N-Contrast or y-aware term with masked label distance, if the marker-ordering signal needs strengthening ([RnC](https://arxiv.org/html/2210.01189v2), [y-aware](https://ar5iv.labs.arxiv.org/html/2106.08808)).
  - (d) A geometry-preserving regulariser on L (STRUCTURE/KLOT), so marker supervision does not erase morphology structure that G needs ([STRUCTURE](https://arxiv.org/html/2506.16895v1), [SOTAlign](https://arxiv.org/html/2602.23353v1)).
- Fine-tune L lightly or with LoRA. Then **freeze L for stage 2**: single pivot copy, no per-branch towers ([Fusion Embedding](https://huggingface.co/papers/2607.18666.md)).

**Stage 2: G-L binding (15.7k pairs) plus distillation.**
- **G encoder.** Initialise from an SSL model on BSCCM-G **and unpaired clinical smears**, or project an existing hematology/pathology foundation model into the L hub (Ex-MCR leaf→base, [HTML](https://arxiv.org/html/2310.08884)).
- **Losses:**
  - (a) SigLIP G↔L on same-cell pairs. It suits small batches; optionally add soft negatives from frozen-L similarity, X-CLR style ([HTML](https://arxiv.org/html/2407.18134v1)).
  - (b) **Teacher-student distillation**: KL(p_T(M | L_i) ‖ p_S(M | G_i)) on every pair, for *all 8 markers*, even though the cell was never stained ([Chen 2017](https://ar5iv.labs.arxiv.org/html/1705.00753), [Gupta 2016](https://ar5iv.labs.arxiv.org/html/1507.00448)).
  - (c) A pull-only L2 / cosine term to close the modality gap, so the shared M head and retrieval work across G and L ([C-MCR](https://arxiv.org/html/2305.14381v2), [Arivazhagan](https://ar5iv.labs.arxiv.org/html/1903.07091)).
  - (d) A SOTAlign-style KLOT term on **unpaired clinical smears + unpaired L**, with the stage-2 linear/SigLIP model as teacher. This is the route to clinical transfer, but expect gains only if clinical smears are close to the BSCCM-G distribution ([SOTAlign](https://arxiv.org/html/2602.23353v1)).
- **Not recommended:**
  - Adversarial distribution matching of predicted markers ([Cohen](https://arxiv.org/abs/1805.08841)).
  - Cycle losses as evidence ([Chu](https://arxiv.org/abs/1712.02950)).
  - Mixture-based multimodal VAEs as the main model ([Daunhawer](https://ar5iv.labs.arxiv.org/html/2110.04121)).
  - Full-tuple objectives (GMC/Symile), which need G-L-M triples we do not have ([GMC](https://ar5iv.labs.arxiv.org/html/2202.03390)).

**Shared embedding (goal 1).** z_L from the frozen hub, z_G from the G student (aligned by SigLIP + gap-closing), and z_M = the marker set/PoE encoder's output in the same space. This makes the hub ImageBind-shaped, with L as the image.

**Non-parametric baseline, always run alongside.** kNN in z_L: for each G cell, take its paired L embedding (or z_G), average observed marker values of its k nearest L-M neighbours per marker, and use the kNN variance as uncertainty. This is the ASIF / bridge-dictionary analogue ([ASIF](https://huggingface.co/papers/2210.01738.md), [Hao](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517)). If the learned G student does not beat this on E2/E8, the extra machinery is not justified.

### 11.2 Partially observed markers in the contrastive objective (summary)

- Use masked co-observed distance. Pairs with no co-observed marker are "unknown" and get weight 0 for both positive and negative.
- Use soft sigmoid targets w = exp(−d²/2σ²), with σ set from the measurement noise floor.
- Stratify batches by antibody group and enrich with all-marker cells.
- Use RnC with ties below the noise floor where ordering matters.
- Supervise markers directly with a probabilistic head on observed entries. Contrastive alignment gives the *embedding*; the marker *prediction* comes from the head and the distillation.

### 11.3 Evaluation plan

Report each hop separately (E6). Then report:
- E2 synthetic triangles, as a calibration of the transitive machinery;
- the E8 interval for G→M RMSE/AUC;
- E3/E4 morphology-mediated checks;
- E5 leave-type-out confidence;
- E7 pivot covariate shift.

State the residual untestable assumption explicitly: M ⟂ G | L, plus teacher-error transportability.

---

## 12. Unverified / not read in full

- **FreeBind, UniBind, MultiVI, Bridge CorrNet:** abstract-level only.
- **MMVAE+:** numbers not read.
- **Symile:** missing-data mechanism from a secondary summary.
- **Cohen et al.:** experimental details (tumour add/remove) not read.
- **StabMap:** full text blocked by captcha; the multi-hop and simulation claims come from search summaries of the abstract and PMC page.
- **SigLIP label-noise:** details beyond the quoted conclusion not read.
- **The E8 bounds and every [derivation]:** my own analysis, not taken from a paper; it should be reviewed.
- **Cell-type counts:** whether rare cell types reach ≥50 G-L pairs is a question about BSCCM, not verified here.
