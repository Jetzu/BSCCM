# N088 lit C: aligning cell images with molecular readouts, and across image domains (slides, draws)

Scope: cross-modal autoencoders, contrastive image-molecule models, unpaired OT/GW single-cell alignment, semi-supervised OT,
and domain generalisation across slides and batches. Focus: two observations. (1) Shared embeddings retrieve the same cell WITHIN a slide
(R@10 .38-.41) but not ACROSS slides (.017-.038; chance .0018/.0010). (2) Distribution-level alignment (DANN, AdaBN, OT, CORAL, mean
shift) never helped a regression target across blood draws.
Checked against context.md and the ideas/declined lists of N002, N047, N059, N066 and N079 (N002 already covered trimodal pivot alignment,
SigLIP/SOTAlign, CLOOME/MolPhenix/CellCLIP, Platonic/JEPA). Sources were looked up 2026-11-24 (WebSearch/WebFetch, Europe PMC REST, arXiv API).
Evidence tags:
- [abs] = abstract or proceedings page;
- [full] = full text read;
- [N002] = number taken over from the N002 strand-C report, where it was checked;
- [sec] = secondary source only;
- [derivation] = my own inference.
No paper below is cited from memory alone.

## (i) Annotated references (34, grouped)

### A. Cross-modal autoencoders and image -> omics translation
1. **Yang, Belyaeva, ... Uhler. Multi-domain translation between single-cell imaging and sequencing data using autoencoders.** Nat Commun
   12, 2021. https://www.nature.com/articles/s41467-020-20249-2 (PMC7782789) [full]
   - Method: one autoencoder per modality; the latents are matched with an adversarial (JS-divergence) loss. Optional anchors can be
     added: a label-conditioned discriminator, a shared-class classifier, or a paired-sample anchor loss.
   - Result: on RNA/ATAC benchmarks, 25 % of the pairs gives about DCCA-with-all-pairs performance. The image <-> RNA application has no
     held-out evaluation ("evaluation is based on the full dataset used for training").
   - The authors state the latent coupling is non-unique ("multiple solutions that can result in the same observed data
     distributions") and needs prior knowledge.
   - Map to us: adversarial marginal matching is our refuted DANN family. The parts that work are the anchors, which for us means
     distillation and class anchoring (K040).
2. **Image2Omics. Multi-omics prediction from high-content cellular imaging with deep learning.** arXiv 2306.09391, 2023/24.
   https://arxiv.org/abs/2306.09391 [abs]
   - Method: PAIRED multiple-instance learning from images to bulk transcriptome/proteome of macrophages.
   - Result: only 18.7 % / 13.4 % of transcripts and 8.5 % / 14.0 % of proteins (M1/M2) are significantly predictable.
   - Map to us: even with pairs, image -> molecule decoding is sparse and per-feature weak. The same holds in our sandbox (within-type
     r .15-.20, K046).
3. **uniPort (Cao, Gong, Hong, Wan). A unified computational framework for single-cell data integration with optimal transport.** Nat
   Commun 13:7419, 2022. https://www.nature.com/articles/s41467-022-35094-8 [abs]
   - Method: a coupled VAE plus minibatch UNBALANCED OT, and it needs some shared features.
   - Map to us: the unbalanced form addresses composition differences, which balanced OT ignores. It still aligns marginals, and does
     not separate readout shift from composition shift.

### B. Contrastive image-molecule models (CLIP-style for cells)
4. **CLOOME (Sanchez-Fernandez, ... Klambauer).** Nat Commun 2023. https://www.nature.com/articles/s41467-023-42328-w [abs + N002]
   - Method: InfoLOOB/CLIP between Cell Painting images and molecules.
   - Result: the correct image among about 2,000 candidates is found >70x better than random [N002].
   - Map to us: the images are well-level aggregates, so cell-level nuisance averages out. We match single cells, where it does not.
5. **MolPhenix (Fradkin, ..., Beaini). How molecules impact cells: unlocking contrastive phenomolecular retrieval.** NeurIPS 2024.
   https://arxiv.org/abs/2409.08302 [abs + N002]
   - Result: 77.3 % top-1 % recall on active molecules, 8.1x over the previous SOTA zero-shot.
   - Gains come from (a) a FROZEN pretrained phenomics encoder (CLOOME active-unseen .076 -> .466 with Phenom1 embeddings [N002]),
     which the authors credit with "reducing the impact of batch effects", and (b) soft similarity targets (S2L .910 vs SigLIP .776
     [N002]).
   - Map to us: lock the encoder, use soft targets. Both were done in N045/N054, and the cross-slide collapse persisted. So the
     bottleneck is not the loss.
6. **PhenoProfiler (Li, Zhang, Song et al.).** Nat Commun 2025, doi 10.1038/s41467-025-67479-w; arXiv 2502.19568 [abs, Europe PMC]
   - Result: up to +20 % accuracy/robustness over SOTA on about 400k images / 8.4M cells.
   - The key ingredient is a "phenotype correction strategy" that expresses each treatment RELATIVE to controls.
   - Map to us: reference-relative representations need a reference population in every batch. That is inductively unavailable for a
     new blood draw (see D/E).

### C. Unpaired and weakly paired single-cell alignment (OT / GW / manifold) and when it fails
7. **MMD-MA (Liu, Huang, Singh, Vert, Noble).** WABI 2019. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.WABI.2019.10 [abs]
   - Method: MMD + distortion penalty, with no correspondences.
8. **UnionCom (Cao et al.).** Bioinformatics 36:i48, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7355262 [abs]
   - Method: matches intra-dataset distance matrices.
   - Later critique (TopoGAN, bioRxiv 2022.04.27.489829 [sec]): UnionCom and MMD-MA were tested on data with perfect 1:1
     correspondence, and performance "drops significantly" on realistic data without it.
9. **SCOT (Demetci et al.).** J Comput Biol 2022, GW-OT. https://pmc.ncbi.nlm.nih.gov/articles/PMC8812490 [N002]
   **SCOTv2.** RECOMB 2022 / J Comput Biol 29(11) 2022. https://www.biorxiv.org/content/10.1101/2021.11.09.467903 [abs]
   - SCOTv2 exists because balanced GW mis-aligns datasets with disproportionate cell-type composition, and fixes it with unbalanced GW.
   **Pamona (Cao et al.).** Bioinformatics 2022 [N002]
   - Partial GW; needs the number of shared cells.
   - Map to us: in N054, GW recovered only 1.3-2.0 % of identities even for the SAME LF cells (chance .39 %), which this literature
     predicts. GW matches the dominant geometry of each space, and in our embeddings that geometry is slide/FOV nuisance plus lineage
     (see E).
10. **MaxFuse (Chen et al.).** Nat Biotechnol 42:1096, 2024. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11638971/ [abs]
    - Method: iterates fuzzy smoothing over each modality's own kNN graph, CCA co-embedding and matching, so it uses ALL features of
      each modality when the linked features are weak.
    - Result: 20-70 % relative improvement over prior methods under weak linkage.
    - Map to us: the right template for BSCCM <-> S-BIAD452 in MARKER space, where 5 markers are shared (CD3, CD19, CD56, CD45, CD14).
11. **MatchCLOT (Gossi, ..., Rapsomaniki).** Brief Bioinform 24:bbad130, 2023. https://pubmed.ncbi.nlm.nih.gov/37122067/ [abs]
    - Method: contrastive co-embedding from PAIRED training data, then entropic OT as approximate bipartite matching.
    - Result: +26.1 % over the top method; +28.9 % over the best NeurIPS-2021 competition modality-matching score.
    - Map to us: SOTA matching is pairs-trained. OT only cleans up the assignment.
12. **Labeled GW-OT (Ryu, Bunne, Pinello, Regev, Lopez).** AISTATS 2025 (PMLR 258). https://arxiv.org/abs/2405.00838 [abs]
    - Existing AE/GW alignment "may perform poorly" when cell states are not clearly distinguishable. Adding shared labels
      (perturbation) inside GW fixes it.
    - Map to us: label-conditioned matching is the generic fix, which is what class anchoring (K040) and N087-F do.
13. **KPG-RL keypoint-guided OT (Gu et al.).** NeurIPS 2022. https://papers.neurips.cc/paper_files/paper/2022/hash/6091c5644d73637e3cccdcab52a7031f-Abstract-Conference.html [abs]
    - Method: a few known pairs constrain the plan (mask + relation-to-keypoint preservation).
    - Map to us: semi-supervised OT helps when pairs are SCARCE. We have 15.7k pairs, and SOTAlign (the same family) was flat (K035).

### D. Why distribution alignment fails, especially for regression
14. **Zhao, Tachet des Combes, Zhang, Gordon. On learning invariant representations for domain adaptation.** ICML 2019.
    https://proceedings.mlr.press/v97/zhao19a.html [abs]
    - Gives an information-theoretic lower bound on the joint error of ANY invariant-representation method when the label distributions
      differ.
    - Matching marginals is insufficient under conditional shift.
15. **Tachet des Combes, Zhao, Wang, Gordon. Domain adaptation with conditional distribution matching and generalized label shift.**
    NeurIPS 2020. https://arxiv.org/abs/2003.04475 [abs]
    - Fix: align CLASS-CONDITIONAL distributions with importance weights for the class proportions.
    - Result: large gains over DANN/JAN/CDAN when label proportions mismatch.
16. **Gulrajani & Lopez-Paz. In search of lost domain generalization.** ICLR 2021. https://arxiv.org/abs/2007.01434 [abs]
    - Result: no DomainBed algorithm beats a tuned ERM by more than 1 point under equal model selection.
    - Model selection is part of the method.
17. **Koh et al. WILDS.** ICML 2021. https://arxiv.org/abs/2012.07421 [full, RxRx1 table]
    - RxRx1 (Cell Painting, held-out experimental batches): ERM ID 39.8 vs OOD 29.9.
    - CORAL 28.4, Group DRO 23.0, IRM 8.2; the methods "generally fail to improve over models trained with ERM".
18. **Miller et al. Accuracy on the line.** ICML 2021. https://proceedings.mlr.press/v139/miller21b.html [abs]
    - OOD accuracy is linearly tied to ID accuracy across architectures and training choices.
    - Camelyon17 (pathology) is a documented weaker case.
19. **Chen, Wang, Wang, Long. Representation Subspace Distance for domain adaptation regression (RSD).** ICML 2021.
    https://proceedings.mlr.press/v139/chen21u.html [abs]
    - Classification is robust to feature scaling, regression is not, so distribution alignment can hurt regression. RSD aligns
      orthogonal bases instead.
    - DARE-GRAM (CVPR 2023, arXiv 2303.13325 [abs]) adds explicit scale preservation.
20. **Adachi, Yamaguchi, Kumagai, Hamagami. Test-time adaptation for regression by subspace alignment (SSA).** ICLR 2025.
    https://arxiv.org/abs/2410.03263 [full]
    - Regression features live in <100-d subspaces (UTKFace 2041 valid dims -> 76).
    - Naive whole-feature alignment falls below the source model: R^2 .333 vs .406 on SVHN->MNIST; BN-adapt .318 vs .605 on California
      housing.
    - SSA (align only output-significant PCs, weighted) gives .511 vs .406 and .639 vs .605; oracle .874/.729.
    - Explicitly assumes COVARIATE shift only.
21. **Niu et al. Towards stable test-time adaptation in dynamic wild world (SAR).** ICLR 2023. https://arxiv.org/abs/2302.12400 [abs]
    - BatchNorm is "a crucial factor hindering TTA stability" under mixed shifts, small batches and imbalanced label shift.
    - Group/LayerNorm are more stable but not fail-safe.
22. **Zhao, Liu, Alahi, Lin. On pitfalls of test-time adaptation (TTAB).** ICML 2023. https://proceedings.mlr.press/v202/zhao23d.html [abs]
    - Hyperparameter and model selection for TTA is "exceedingly difficult".
    - OOD-generalisation augmentation practice can hurt TTA.

### E. Batch and site signatures in image embeddings, and how image-based profiling corrects them
23. **Arevalo, ..., Carpenter, Singh. Evaluating batch correction methods for image-based cell profiling.** Nat Commun 15:6516, 2024
    (Europe PMC abstract; preprint PMC10516049 [full]).
    - Setup: 10 scRNA-seq methods on JUMP Cell Painting.
    - Result: Harmony and Seurat-RPCA are in the top three in all 5 scenarios. Sphering (TVN-like) needs negative controls in every
      batch and did not beat the baseline in multi-lab/multi-microscope scenarios (preprint).
    - Every method except sphering needs batch labels and recomputes on the full data, i.e. it is transductive.
24. **Ando, McLean, Berndl. Improving phenotypic measurements in high-content imaging screens (TVN).** bioRxiv 10.1101/161422, 2017. [abs]
    - Method: whiten the embeddings on negative-control wells, which down-weights nuisance axes.
25. **Haslum et al. Metadata-guided consistency learning for high content images (CDCL).** MIDL 2023, PMLR 227 (2024).
    https://arxiv.org/abs/2212.11595 [full]
    - Diagnosis: off-the-shelf SSL fails across batches because "batch effects dominate the signal". On RxRx1-HUVEC with batch-held-out
      splits, kNN accuracy is DINO 14.3 vs supervised 51.4.
    - Fix: CROSS-BATCH POSITIVES (same treatment, different batch) + per-domain centring of DINO targets: kNN 53.6, Z-norm kNN 63.1 vs
      supervised 53.5.
    - After full fine-tuning the advantage shrinks to noise (MoA 1-NN 22.6 vs 21.8 ImageNet, SD 4).
26. **Howard et al. The impact of site-specific digital histology signatures on deep learning model accuracy and bias.** Nat Commun
    12:4423, 2021. https://www.nature.com/articles/s41467-021-24698-1 [abs]
    - CNNs identify the submitting site despite colour normalisation and augmentation.
    - Site-mixed train/validation splits inflate accuracy. Fix: site-preserved splits.
27. **Kömen, Marienwald, Dippel, Hense. Do histopathological foundation models eliminate batch effects?** AIM-FM workshop @ NeurIPS 2024,
    arXiv 2411.05489 [abs]
    - Hospital signatures persist in every model tested, are not removed by stain normalisation, and "dominate distances in feature
      space".
28. **de Jong, Marcus, Teuwen. Current pathology foundation models are unrobust to medical center differences.** arXiv 2501.18055, 2025
    [abs]
    - Across 10 FMs, the centre is predicted more accurately than tissue or cancer type. Only one model has a Robustness Index above 1,
      and only slightly.
    - Errors come mostly from same-centre confounders.
29. **Haghverdi, Lun, Morgan, Marioni. Batch effects in scRNA-seq corrected by matching mutual nearest neighbors (MNN).** Nat Biotechnol
    36:421, 2018. https://www.nature.com/articles/nbt.4091 [abs]
    - Matches batches through mutual nearest neighbours, so it needs only a shared subpopulation and no equal composition.
    - Assumes the batch effect is roughly orthogonal to the biological subspace.
30. **Luecken et al. Benchmarking atlas-level data integration (scIB).** Nat Methods 19:41, 2022. https://www.nature.com/articles/s41592-021-01336-8 [abs]
    - Batch removal trades off against biology conservation; scaling pushes methods toward over-correction.

### F. Shortcuts in contrastive and paired alignment, and the tools against them
31. **Chen, Luo, Li. Intriguing properties of contrastive losses.** NeurIPS 2021. https://arxiv.org/abs/2011.02803 [abs]
    - Feature suppression: "a few bits of easy-to-learn shared features can suppress, and even fully prevent, the learning of other
      sets of competing features."
    **Robinson et al. Can contrastive learning avoid shortcut solutions? (IFM).** NeurIPS 2021. https://arxiv.org/abs/2106.11230 [abs]
    - Harder instance discrimination trades features off; implicit feature modification reduces suppression.
32. **Bleeker, Hendriksen, Yates, de Rijke. Demonstrating and reducing shortcuts in vision-language representation learning.** TMLR 2024.
    https://arxiv.org/abs/2402.17510 [abs]
    - With an injected shared shortcut, contrastive image-text models learn "mostly" the shortcut.
    - Latent target decoding and IFM help "only partly".
    - Map to us: a contrastive LF<->Giemsa model need not encode marker-relevant morphology if an easier shared signal identifies the
      pair (cf. K035: same cell found ~20x chance, little marker information).
33. **Wu, Mosse, Zhuang, Yamins, Goodman. Conditional negative sampling for contrastive learning (ring negatives).** ICLR 2021.
    https://arxiv.org/abs/2010.02037 [abs]
    - Negatives drawn from a conditional "ring" (hard but not too hard): +2-5 pts linear evaluation.
    **Dwibedi et al. NNCLR.** ICCV 2021. https://arxiv.org/abs/2104.14548 [abs]
    - Nearest neighbour from a support queue as positive: ResNet-50 linear 71.7 -> 75.6.
    **Belrose et al. LEACE: perfect linear concept erasure in closed form.** NeurIPS 2023. https://arxiv.org/abs/2306.03819 [abs]
    - Provably removes all LINEAR information about a concept with minimal change.
    - Oracle-LEACE (concept known at inference) = the residual of X regressed on Z.

### G. Domain generalisation recipes for microscopy and robust fine-tuning
34. **Microscopy/pathology DG**
    - **Tellez et al.** MedIA 2019, https://arxiv.org/abs/1902.06543 [abs]: colour augmentation should always be used; normalisation
      alone adds little.
    - **Jahanifar et al.** DG in computational pathology, survey, ACM CSUR 2025, arXiv 2310.19656.
    - **Zamanitajeddin et al.** benchmark of 30 DG algorithms, arXiv 2409.17063 [abs]: SSL pretraining and stain augmentation
      consistently beat the dedicated DG algorithms.
    - **MixStyle** (Zhou et al., ICLR 2021, https://arxiv.org/abs/2104.02008 [abs]): mixes instance feature statistics ACROSS source
      domains, so it needs at least 2 training domains.
    - **DSBN** (Chang et al., CVPR 2019, arXiv 1906.03950 [abs]): a BN branch per domain.
    - **Salehi et al. AE-CFE** (Marr lab, MICCAI 2022, https://arxiv.org/abs/2207.00501 [abs]): WBC autoencoder features with a domain
      loss across three smear datasets.

    **Robust fine-tuning** (keep the pretrained features instead of aligning domains):
    - **LP-FT** (Kumar et al., ICLR 2022, https://arxiv.org/abs/2202.10054 [abs]): over 10 shift datasets, full fine-tuning is +2 % ID but
      -7 % OOD vs linear probing; linear-probe-then-fine-tune is +1 % ID, +10 % OOD vs fine-tuning.
    - **WiSE-FT** (Wortsman et al., CVPR 2022, https://arxiv.org/abs/2109.01903 [abs]): weight-space interpolation of pretrained and
      fine-tuned models, +2-10 pts OOD on ImageNet shifts at no extra training cost.
    - **SWAD** (Cha et al., NeurIPS 2021, https://arxiv.org/abs/2102.08604 [abs]): dense weight averaging, +1.6 pts average OOD on
      DomainBed.

Not re-cited (already covered in N002/N059): SigLIP/SOTAlign/STRUCTURE, CellCLIP, DinoBloom cross-scanner collapse, CytoNorm/cyCombine/CytoVI,
DeepIFC, Infinity Flow.

## (ii) Diagnosis: why the cross-slide collapse happens, and what usually fixes it

### Collapse of LF<->Giemsa retrieval across slides
1. **The within-slide number is probably inflated by a shared-nuisance shortcut** [derivation, testable at zero GPU cost; see P1].
   - What the two modalities share: the LF and Wright images of one cell come from the same slide position. They share FOV-level and
     position-level nuisance: focus/z-tilt, coverslip or medium thickness, illumination fall-off, background/debris and local density.
   - Project evidence that both modalities carry position:
     - LF features predict slide position at r .92 (K007).
     - Position alone predicts autofluorescence on Giemsa cells at r .72 (N012).
     - A linear probe identifies the slide at .92 accuracy from Giemsa (DinoBloom) features and .99 from LF features (N011).
   - The arithmetic is unfavourable. The paired slides hold 10,015/5,697 cells in 2,774/1,962 FOVs, i.e. 3.6/2.9 cells per FOV (N045
     report). The within-slide validation gallery is about 279 held-out but spatially INTERLEAVED FOVs (~1,000 cells).
   - So a head that only identifies the FOV, or its slide neighbourhood, would already reach a large R@10. Chance is ~.01, and perfect
     FOV identification would give ~1.
   - Two further facts point the same way:
     - The N045 checkpoints were chosen on exactly this within-slide R@10, a selection rule that rewards the shortcut (refs 16, 22).
     - SigLIP beats CCA 2.5x within the slide (.41 vs .16) but only 1.8x across slides (.038 vs .021). The nonlinear head buys mostly
       slide-specific matching.
   - Contrastive theory predicts this outcome:
     - easy shared features suppress harder ones (31);
     - injected shared shortcuts are learned "mostly" (32);
     - site signatures dominate the distances of even large foundation models (26-28).
2. **Across slides the shortcut map is invalid.** Each slide has its own position -> nuisance field, and only one Wright-stained slide
   exists per draw. Leftover retrieval of 10-20x chance is what coarse lineage/size matching yields.
   - This matches K035/K038: the aligned space finds the same cell well above chance but carries little marker information.
   - The cross-slide number is the honest one. The within-slide number should not be used to select or compare alignment methods.
3. **Why OT/GW/SOTAlign could not repair it.**
   - GW and manifold alignment match each space's dominant geometry, which here is nuisance plus lineage (refs 9, 27-28).
   - Balanced OT mis-assigns when compositions differ (SCOTv2).
   - These methods were validated mostly with perfect correspondences (TopoGAN critique of MMD-MA/UnionCom).
   - Few-anchor OT helps only when pairs are scarce (13), and we have 15.7k.
   - The successful methods all bring outside supervision: pairs (MatchCLOT), labels (labeled GW, Yang's anchors) or targets (MolPhenix
     soft targets). In our case that is the distillation of LF outputs, which is already the best bridge.
4. **What usually fixes it.**
   - Make the shortcut useless: negatives that share the nuisance (same-FOV or same-region "ring" negatives, ref 33), spatially
     blocked validation (26), and linear erasure of known nuisance variables (LEACE, ref 33).
   - Give positives that do NOT share it: cross-batch positives (CDCL 14.3 -> 53.6 kNN, ref 25) or mutual nearest neighbours across
     batches (29).
   - Reference-based correction (TVN, sphering, Harmony, CytoNorm) works in Cell Painting only because every batch contains biologically
     identical controls (23-24). BSCCM has no such cross-draw reference, and using the held-out draw's images is transductive.

### Distribution alignment never helped the regression target across draws
1. **The dominant draw shift is in P(y|x), the readout, not in P(x).** N059 found:
   - draw-1 HLA-DR labels track autofluorescence;
   - the grp:CD45 deficit sits on one dim slide;
   - the autofluorescence slope flips sign between draws.

   Input-side alignment cannot change P(y|x). Zhao 2019 gives the bound (14), and SSA states that it assumes covariate shift only (20).
2. **Composition (label) shift between isolations.**
   - Marginal matching (DANN, CORAL, balanced OT, AdaBN) forces the two draws' lineage mixtures to coincide, which moves cells across
     class boundaries (14).
   - The documented fixes are class-conditional with re-estimated class weights (GLS, ref 15) or neighbour-based (MNN needs no equal
     composition, ref 29).
   - BN-statistics TTA is unstable precisely under label shift (21), consistent with AdaBN -.005..-.011 (N026).
3. **Regression is scale-sensitive and low-rank.**
   - Whole-feature alignment perturbs the few output-relevant directions and rescales them. It goes below the source model on standard
     regression benchmarks (19-20).
   - The CORAL/OT variants tried here were all whole-space.
4. **ERM plus better ID accuracy is what transfers.**
   - DomainBed/WILDS: no DG/DA method beats ERM by >1 pt; CORAL is below ERM on RxRx1 batches (16-17).
   - OOD tracks ID (18).
   - This matches the project's history: what moved the GOAL were labels, encoders and ensembles, never alignment.
   - Among DG levers, the ones with consistent evidence:
     - pretraining/SSL plus augmentation (34);
     - preserving pretrained features (LP-FT/WiSE-FT);
     - flat-minimum weight averaging (SWAD).

     None has been tried here (train_lf.py has no EMA, SWA, LP-FT or interpolation; the encoder is fine-tuned at full lr,
     enc-lr-mult 1.0).
5. **One structural asymmetry is worth exploiting.** All slides of ONE draw come from ONE cell suspension, so their compositions match
   in expectation.
   - Within the training draw, slide-to-slide differences are pure nuisance, and invariance to them is safe (no Zhao-2019 trap).
   - The current SSL's positives are two views of the same crop, which share slide/FOV/position nuisance. That retains the nuisance (25;
     N002-F "not good views").
   - Cross-slide positives within the training draw are the inductive analogue of CDCL.

## (iii) Experiment proposals, ranked by expected value

GOAL arithmetic: 22 cells (11 targets x 2 directions), seed noise .0027, registered bar = control + .0054 (single model) or > .6916
(ensemble). Screens use 1 seed x 2 directions; a second seed only if the screen passes. Compute estimates come from the project's
measured costs (r50 GOAL fine-tune ~.7 H100-h per direction; r50 60-ep SSL ~5 H100-h per draw, N078/N085). Everything is on the
preempt queue and far below the 25 GPU-h approval line.

### P1 (rank 1). FOV/position-shortcut audit of LF<->Giemsa alignment, then nuisance-conditional alignment and distillation
- **Objective:** LF->Giemsa cross-slide transfer (secondary: honest metric for all alignment work).
- **Hypothesis:**
  - Within-slide R@10 ~.40 is mostly FOV/position identification from nuisance shared by both modalities.
  - Removing that shortcut lowers the within-slide number but raises cross-slide retrieval and the information in the bridge.
- **Recipe:**
  - Step 0 (CPU, 48 cores, ~1-2 h). Use the N045 v2 embeddings, or recompute frozen LF SSL-r50 and DinoBloom-B Giemsa features on the
    15.7k pairs. For the SigLIP-pairs and CCA heads:
    - (a) Split within-slide R@10 into FOV-hit@10 (any cell of the query's FOV in the top 10) and within-FOV R@1 (chance 1/n_FOV, about
      .28-.34).
    - (b) Position-only retrieval: cross-fit (blocked by FOV) a ridge/GBM regressor from each modality's embedding to slide (x, y),
      plus an FOV focus proxy (FOV-mean LF brightfield sharpness). Retrieve by distance between the predicted positions.
    - (c) Nuisance erasure, then refit the heads:
      - FOV-mean centring per modality: subtract each FOV's mean embedding, computed from that FOV's own images, so it is
        inductive per FOV;
      - LEACE on (x, y) polynomial features.

      Score within-slide R@10, cross-slide R@10 and transferred autofluorescence r (v5c truth) in both directions.
  - Step 1 (only if the shortcut is confirmed). Retrain the SigLIP head on erased features with:
    - same-FOV and same-region (radius <= 2 FOVs) negatives up-weighted ("ring" negatives);
    - batches built as 32 whole FOVs;
    - checkpoints selected on a SPATIALLY BLOCKED validation (contiguous slide regions held out instead of random FOVs).

    Then re-run the TT6c/hub Giemsa student with targets distilled from the nuisance-erased LF teacher features. Report cross-slide
    autofluorescence r and MLL23/PBC contrasts against the TT6c/hub baselines.
- **Expected effect** (P(shortcut confirmed) ~.6):
  - within-slide R@10 .40 -> .10-.25;
  - cross-slide R@10 .03 -> .04-.06;
  - distilled-student cross-slide autofluorescence r +0 to +.04;
  - no GOAL effect.
  - The main value is a corrected selection metric for all Giemsa work.
- **Compute:** Step 0 is 0 GPU-h. Step 1 is ~0.3 H100-h for the heads plus ~2 H100-h for the students in both directions, ~2.5 H100-h
  in total.
- **Kill:**
  - Stop after Step 0 if position-only R@10 < .10 AND within-FOV R@1 >= 2x chance, i.e. no shortcut.
  - Kill Step 1 if cross-slide R@10 rises < 1.3x in both directions AND student cross-slide autofluorescence r gains < +.03.
- **Overlap with N087:** none. N087-F matches Giemsa to marker profiles class-conditionally; it can consume P1's erased Giemsa
  embeddings.

### P2 (rank 2). Robust fine-tuning for draw transfer: LP-FT + WiSE-FT interpolation + SWA/SWAD weight averaging
- **Objective:** GOAL.
- **Hypothesis:**
  - Full-lr fine-tuning (enc-lr-mult 1.0, 10 epochs) distorts the in-domain SSL features toward draw-specific solutions.
  - Keeping them (LP-FT), interpolating in weight space (WiSE-FT), or averaging along the trajectory (SWAD) moves the model off the
    ID/OOD line in the OOD direction (refs 18, 34).
- **Recipe:**
  - LP stage: fit the GOAL head (same architecture, all 11 targets + aux) on FROZEN SSL-r50 features of the training draw. Reuse the
    N087-A dumps and their best linear head.
  - FT stage: train_lf.py with the head initialised from LP (new flag --head-init), enc-lr-mult .3, otherwise the n051_v10_base recipe.
  - In the same run, keep an SWA average of the weights from 30 % of the steps to the end. Recalibrate BN on training-draw images only
    (inductive).
  - WiSE-FT: theta_a = (1-a) theta_LP + a theta_FT with a in {.25, .5, .75}. Recompute BN statistics on the training draw.
  - Choose a by leave-one-slide-out validation INSIDE the training draw, never on the test draw.
  - Arms (seed 0, both directions): control (existing n051_v10_base), LP-FT, LP-FT+SWA, WiSE-FT(a*). Then score each passing arm as a
    member of the N084 ensemble.
- **Expected effect:**
  - single model +.002-.008 GOAL;
  - ensemble +.001-.004, since weight averaging partly duplicates ensembling.
  - The literature gains (+10 % OOD for LP-FT, +2-10 pts WiSE-FT, +1.6 SWAD) come from broad pretraining. Our SSL is in-domain, so the
    margin is smaller. P(pass .0054) ~.25.
- **Compute:**
  - 3 fine-tune runs x 2 directions x .7 = ~4.2 H100-h, plus WiSE/SWA evaluation ~.3, so ~4.5 H100-h;
  - the second seed adds the same.
- **Kill:**
  - Gate first on N087-A. If the frozen-SSL decoders lose > .03 GOAL to fine-tuning AND their in-distribution -> cross-draw drop is no
    smaller than fine-tuning's, there is no effective robustness to preserve: run SWA only.
  - Kill if the best arm is below control + .0054 and its ensemble is <= .6916.
- **Overlap with N087:** depends on A/D1 (frozen features, projector features) for the LP stage. Not a duplicate: N087 never
  fine-tunes from an LP head and never interpolates.

### P3 (rank 3). Cross-slide positives in in-domain SSL (CDCL/NNCLR/MNN analogue inside the training draw)
- **Objectives:** GOAL; LF->Giemsa; frozen-feature decoders.
- **Hypothesis:**
  - The current SSL positives (two augmentations of one crop) share all slide/FOV/position nuisance, so the encoder keeps it (position
    r .92, slide probe .99).
  - Positives that pair a cell with its biological twin on ANOTHER slide of the same draw remove nuisance without the composition
    confound. Same suspension means same composition, so Zhao-2019 does not apply.
  - Expect a large effect on frozen features (CDCL 14 -> 54 kNN) and a small one after fine-tuning (CDCL MoA within SD).
- **Recipe:**
  - Continue each training-draw SSL-r50 checkpoint for 15 epochs (train_ssl_lf.py, arm aug, training-draw rows only, Giemsa cells
    excluded) with an added term:
    - maintain a 64k projector-output queue with slide ids (NNCLR);
    - for each anchor take its nearest neighbour from a DIFFERENT slide, kept only if MUTUAL within the top 5 (MNN);
    - loss += .5 x invariance(z_anchor, stopgrad(z_partner)), ramped in over 2 epochs;
    - keep 2-view invariance + SIGReg;
    - subtract a per-slide running mean from the projector outputs before the cross-slide term (CDCL domain centring).
  - Mechanism check (CPU), before any fine-tune:
    - slide-ID linear-probe balanced accuracy within the training draw;
    - position r;
    - 3-class lineage kNN accuracy on frozen features.
  - Then:
    - the N087 frozen-decoder pipeline on the new features;
    - one GOAL fine-tune per direction (n051_v10_base);
    - an ensemble-member score;
    - optionally P1's cross-slide Giemsa retrieval with the new LF features.
- **Expected effect:**
  - frozen-feature GOAL +.01-.03;
  - fine-tuned single model +0-.005;
  - as a decorrelated ensemble member +.002-.004;
  - cross-slide Giemsa R@10 x1.2-1.5.
  - P(GOAL bar) ~.15. Higher if P2 shows that preserving SSL features matters, since the two stack: P2 on top of P3 features.
- **Compute:**
  - SSL continuation 15 ep ~1.3 H100-h per draw x 2, plus 2 fine-tunes x .7, so ~4 H100-h;
  - full 60-ep retraining (~12 H100-h) only if the screen passes.
- **Kill:**
  - Stop before fine-tuning if slide-probe accuracy falls < 10 % relative, or lineage kNN accuracy drops > 1 pt.
  - Kill if frozen-feature GOAL gains < +.01 AND fine-tuned GOAL < control + .0054.
- **Overlap with N087:** none. N087-D changes the projector bottleneck, not the positives. Not the refuted DANN: no adversary, no
  cross-draw data, conditional (neighbour) not marginal matching.

### P4 (rank 4). Nonparametric anchor transfer in marker space (MaxFuse/MNN-style) for joint multi-marker decoding
- **Objective:** multi-marker decoding.
- **Hypothesis:**
  - The MSE head outputs conditional means and erases co-expression (CD4-CD8 within T: true -.81, single-stain -.07; K046).
  - Matching each cell's predicted shared-marker vector and image neighbourhood to cells of an all-marker reference, then imputing the
    joint profile from matched reference cells, restores co-expression without hand-coded rules. This is a learned prior from external
    joint data, which is allowed.
- **Recipe:**
  - Sandbox: S-BIAD452, N080 protocol, each donor held out; reference = the validation donor's joint labels, as in N087-B.
  - Inputs: N080 'single'-regime predictions and features.
  - Matching:
    - (1) Smooth predictions over the test cells' own image-feature kNN graph (k = 15, MaxFuse fuzzy smoothing).
    - (2) MNN matching (k = 30) to the reference on the shared-marker vector plus predicted coarse type.
    - (3) Imputed joint = mean of the matched reference vectors.
    - (4) Output = w x imputed + (1-w) x direct per marker, with w fitted on the validation donor.
  - Metrics: registered 36-pair error, CD4-CD8 within T, within-type r, macro r.
  - BSCCM port only if the sandbox passes: reference = S-BIAD452 on the 5 shared markers after training-draw quantile mapping.
    Report MM3 and co-expression plausibility. BSCCM cannot score joint truth (K045).
- **Expected effect:**
  - pair error .229 -> .20-.22;
  - CD4-CD8 within T -.07 -> -.3..-.5;
  - macro r +-.01;
  - GOAL ~0.
- **Compute:** CPU (<1 h) plus existing dumps; < .5 H100-h.
- **Kill:** pair error not <= .21, or macro r loss > .01.
- **Overlap with N087:** HIGH. This is the nonparametric arm of N087-B (parametric PCA/AE prior) and an alternative to E (flow decoder).
  Run it in the N087-B harness with the same metrics, not as a separate item.

### P5 (rank 5, diagnostic; protocol flag). Regression-aware, class-conditional test-time subspace alignment (SSA) on frozen features
- **Objective:** close or open the "image-side covariate shift" route for the GOAL.
- **Hypothesis:**
  - Earlier CORAL/OT/AdaBN failures could be artefacts of whole-space, marginal alignment (refs 19-21).
  - SSA restricted to the output-significant PCs, done per predicted lineage (GLS-style class-conditional, ref 15), is the strongest
    covariate-only correction.
  - If it is still < +.003, the remaining shift is in the readout (N059) and image alignment is closed for good.
- **Recipe:**
  - On the N087 dumps (fine-tuned r50 2048-d, both directions):
    - training-draw PCA K = 100;
    - weight each PC by its head sensitivity;
    - align the test-draw per-PC mean and variance (closed-form diagonal Gaussian), marginal and per predicted 3-class lineage, with
      class weights re-estimated by EM;
    - re-score the GOAL.
- **Protocol:** this uses unlabelled held-out-draw images at TEST time. It is in the same class as AdaBN (N026), so it is a diagnostic
  only and not a GOAL attempt without a user ruling.
- **Expected effect:** marginal -.005..0 (as AdaBN); class-conditional 0..+.003.
- **Compute:** CPU ~30 min, 0 GPU-h.
- **Kill:** if < +.003, write "image-side covariate alignment closed (whole-space, subspace, class-conditional)" to KNOWLEDGE.
- **Overlap with N087:** uses the A dumps only.

### Considered and NOT proposed
- Harmony/Scanorama/TVN/sphering/CytoNorm on the test draw: they need test-batch data or biologically identical controls (23-24, N059).
  This is transductive; sphering also failed in the multi-lab JUMP scenarios.
- More unpaired OT/GW (SCOT/SCOTv2, Pamona, UnionCom, MMD-MA, uniPort) or few-anchor OT (KPG-RL): the pair count is not our bottleneck
  (N002-F); GW and SOTAlign already failed (K035, K038); the literature failure modes match ours (refs 7-9, 13).
- Yang-2021-style adversarial latent coupling: non-unique by the authors' own statement, and it is DANN in disguise (K018).
- MixStyle/DSBN/stain-style randomisation for Giemsa: they need several training domains (one stained slide per draw), and the failure
  is not colour (Macenko/augmentation flat, N010/N011).
- More contrastive-loss variants (soft targets, SigLIP vs CLIP): MolPhenix's gains are already in N045/N054, and they do not touch the
  shortcut.
