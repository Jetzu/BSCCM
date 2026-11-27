# BSCCM overview figures: what each panel shows and says

**Common to all figures.**
- Bars show the mean. Whiskers are 1 SD. Hollow rings are the individual values (seeds, donors, teachers or slides, as stated per figure).
- Stars come from two-sided paired t-tests against the grey reference bar, Holm-corrected within each panel. "ns" means p ≥ .05.

**GOAL.** GOAL is the main score of the project.
- Task: predict each cell's surface-marker level from its label-free images (multi-illumination LED-array microscopy).
- Training and test: train on one blood draw, test on the other.
- Score: Pearson r between predicted and measured level, averaged over 11 targets:
  - 8 single markers: CD16, CD14, CD45, CD3, CD123, CD56, HLA-DR, CD19;
  - 3 pooled antibody panels: CD3/CD19/CD56, CD45, CD16;
  - then averaged over both directions (draw 1 → 0 and 0 → 1).
- Target: .70. Seed noise: SD .0027.

---

## Figure 1: Progress of the GOAL over the project (`fig1_progress`)

**a. Single networks.** Each ring is the best single-network result of one notebook item (mean over seeds where several were run). The step line is the best so far.
- Colour marks the label version the value was scored on: v5 (N023–N050) or v10 (N051 onwards). Each change of label version re-bases all numbers, so a jump between colours is not model progress.
- On v5, single networks rose from .640 (N023) to .653 (N040, noisy-student pseudo-labels).
- On v10 they rose from .673 (N051) to .677 (N063, two-part output head with per-field-of-view offsets, seeds 2–3).
- About 15 later single-network ideas all landed at .662–.676.

**b. Multi-network results.** Each ring is an item's primary (planned) ensemble or composite.
- On v10 the GOAL record is .6862 (N053: ResNet-50 base + 2× zoom + Cytoland encoder; replicated with a fresh seed at .6864).
- Later combinations reached .690–.691, including the pre-registered N084 at .6911, all below the improvement threshold of .6916 (record + 2 × noise).
- **N085 set the new record, .6967:** a pre-registered equal-weight ensemble of 8 members. Caveat: 6 of the 8 were chosen after their scores on these two draws were known (together .6929); the 2 new ConvNeXt-S members add +.0038, below 2 × noise.
- Descriptive secondary combinations reached up to .6929 (N065); they are not counted.
- The dashed line is the .70 target.

**Says:** most of the gain came from better labels (v5 → v10) and ensembling diverse encoders. Single networks have stayed flat at about .67 since N051, and the best ensemble is about .01 short of .70.

## Figure 2: What was tried: single-network changes vs a matched control (`fig2_attempts`)

Every panel shows the change in GOAL against the same ResNet-50 control (in-domain self-supervised pretraining), compared seed by seed (control .6730 / .6683). Rings are seeds.

**a. Output heads.**

| Head | Change in GOAL |
|---|---|
| HL-Gauss classification head | −.012 |
| two-part head (positive / negative) | +.005 |
| panel head | ±.000 |
| Pearson-correlation loss | +.002 |

**b. Losses and sampling.**

| Change | Change in GOAL |
|---|---|
| per-field-of-view offsets | +.003 |
| pair loss | +.004 |
| no autofluorescence target | +.001 |
| field-of-view-balanced sampler | −.007 |

**c. Labels and supervision.**

| Change | Change in GOAL |
|---|---|
| consensus labels | −.008 |
| label separation | ±.000 |
| censored loss + 20 epochs | −.005 |
| auxiliary fluorescence-image decoder | −.002 |
| multi-marker fine-tune | −.003 |
| symmetry augmentation | −.001 |

**d. Encoders and data.** ConvNeXt-S with in-domain pretraining −.003, Cytoland −.005. Single-seed arms, so no test:

| Arm | Change in GOAL |
|---|---|
| Cytoland at full learning rate | −.027 |
| DINOv3-B | −.022 |
| DINOv3-S | −.024 |
| DinoBloom-B | −.018 |
| ResNet-50 with 4× longer pretraining | −.002 |
| + ImageStream external data | −.006 |

**Says:** no change to the head, loss, labels or encoder moved a single network beyond seed noise; every test is ns. Generic pretrained vision encoders are clearly worse than in-domain self-supervised pretraining.

## Figure 3: Where the remaining error is (`fig3_gap`)

**a. Per-target r.** Mean over both directions for four models:
- ResNet-50 and ConvNeXt-S, each the mean of seeds 0 and 1;
- the N053 ensemble;
- the N084 ensemble.

Bright lineage markers are predicted well: CD16 about .90, CD14, CD3, CD45 and the pooled panels about .70–.85. Four dim markers carry most of the error: CD19 .35, HLA-DR .47, CD56 .52, CD123 .56. The ranking is the same for every model.

**b. Per direction (N084 ensemble).** Draw 0 → 1 (trained on the smaller draw) is weaker overall: .679 vs .703. The main losses are on CD123 (.54 vs .65), HLA-DR and the pooled CD45 panel.

**c. Learning curve at matched training steps (N082, seed 0).**
- 50 % of training labels, 10 epochs: .646.
- 50 %, 20 epochs: .658.
- 100 %, 10 epochs: .673.

At the same number of training steps, twice the labelled cells gives +.016 (.673 vs .658). Twice the steps on the same half gives only +.012 (.658 vs .646). The model is limited by labels, not by optimisation.

**d. Label agreement.** Agreement between the v10 labels and an independently derived label version (local background subtraction) on the held-out cells. Rings are directions.
- CD19 agrees only .32 and CD56 .71, so their labels are unstable.
- The other targets agree at .81 or more.
- High agreement does not prove the labels are correct: CD123 agrees .89 but is still predicted at only .54–.65.

**Says:** the error is concentrated in a few dim markers. For CD19 and CD56 the labels themselves are unreliable, and more labelled cells would help more than longer training.

## Figure 4: Decoding all markers of a cell jointly (`fig4_multimarker`)

**a. BSCCM pooled-panel score (MM3).** MM3 is the mean r of the three pooled antibody panels. ConvNeXt-S (in-domain pretraining) scores .771 vs .758 for ResNet-50.
- The gain (+.0135) cleared the pre-registered bar (+.007) on both seeds.
- With only 2 seeds the paired t-test is ns.

**b–f** use the external imaging-flow-cytometry dataset S-BIAD452.
- It has 274,050 cells from 3 donors, with 9 markers measured on every cell.
- The models see only the brightfield and side-scatter channels.
- Three training regimes are compared:
  - **all 9 markers** labelled on each cell;
  - **one marker per cell**, as in BSCCM's single-stain slides;
  - **one marker per cell + pooled panels** on 10 % of cells.
- Each donor is held out once; rings are donors.

**b. Mean r over the 9 markers.** All 9 markers .596, one marker per cell .526, + pooled .542.
- The cell-type oracle reaches .814. It gives each cell the average marker profile of its true (gated) cell type.

**c. Per-marker r.**
- All-marker training beats single-marker training on every one of the 9 markers. The largest gains are on CD4 (.52 vs .36) and CD15 (.86 vs .77).
- CD8, CD56 and CD19 are worst in every regime (.26–.43).
- The oracle's CD4/CD8 values are high partly because T-cell subsets are gated by CD4 and CD8 themselves.

**d. Correlations between marker pairs within a population** (averaged over donors).
- CD4–CD8 within T cells: measured −.81. Predicted −.36 (all markers), −.07 (one marker), +.16 (pooled).
- CD3–CD19 within lymphocytes: measured −.73. All regimes predict −.83 to −.89.
- CD3–CD56 and CD19–CD56: weak, and similar across regimes.

**e. Pre-registered co-expression error.** Mean |predicted − measured| correlation over all 36 marker pairs on whole donors: all markers .189, one marker .229, pooled .237 (ns).

**f. Within-cell-type r.** Pearson r inside each gated cell type: .20 / .18 / .15.

**Says:** labelling every cell with all markers gives better per-marker decoding (+.07). It also reproduces co-expression better, most clearly the CD4/CD8 exclusivity that single-stain training loses. No regime differs significantly with 3 donors. The bigger limit is that these images mostly encode cell type: decoding within a cell type is weak in every regime, and models fall short even of the cell-type oracle.

## Figure 5: Giemsa → markers via the label-free bridge (`fig5_giemsa`)

Students that read Giemsa-stained blood images are trained from the label-free model. They are evaluated on the external MLL23 clinical smear set, as AUROC over pre-registered fine cell-type contrasts. In panels a–b, rings are 3 label-free teacher models, each the mean of its 2–3 student seeds; tests are paired by teacher.

**a. Distillation target.**
- Distilling the teacher's latent "hub" representation gives .681 AUROC. Distilling its marker outputs gives .568.
- The teacher-paired test gives p = .07.
- Under the pre-registered rule (mean gain .113 > seed noise floor .074) the hub counts as better. The per-teacher gains were +.061, +.109 and +.169.

**b. Anchoring strength for output distillation** (class-anchor weight .5 / 2 / 8): .568 / .604 / .627. This is a trend only, ns; the pre-registered rule says no difference.

**c. Three-way latent designs (N054).** Same-cell autofluorescence r on the 2 held-out slides (rings).
- Output distillation .658, label-free hub .650, direct map .647.
- Two-space design with label-free flow .642, with Giemsa flow .625.
- Unpaired Gromov–Wasserstein alignment .32.

**Says:** distilling a shared latent representation transfers better to clinical Giemsa smears than distilling marker outputs. The two-space design does as well as the hub. Alignment without paired cells fails.

---

## Figure 6: Architecture of the best model (`fig6_architecture`), written for ML readers

**Task.** Per-cell multi-output regression.
- Input: one cell imaged with a programmable LED-array microscope. There are 23 illumination patterns (brightfield, darkfield rings at increasing angles, half-circle phase-contrast patterns and single LEDs) plus one computed phase image, stacked into a 24 × 128 × 128 tensor.
- Output: 13 regression targets (protein levels); 11 of them are scored.

**Main difficulty: sparse labels.** Training uses a masked MSE over the observed targets only.
- Cells from single-stain slides have 1 protein target plus 1 background (autofluorescence) target.
- Cells from pooled-panel slides have 4 (draw 0) or 3 (draw 1) pooled targets and no background target.

**Evaluation.** Train on one blood draw, test on the other (a domain shift). Score = Pearson r, averaged over 11 targets and both directions.

**Model: an ensemble of 4 separately trained members.** Each member is an image encoder → global pooling → linear heads; test predictions are averaged.
1. ResNet-50 (23.6 M parameters, 2048-d features), full 128 px view.
2. ResNet-50, 2× centre zoom.
3. ConvNeXtV2-T (27.9 M, 768-d), initialised from the encoder of a published microscopy image-to-image model (Cytoland). Its 1-channel stem is repeated to 24, it has a learned per-channel gain, and it is fine-tuned at 0.1× the learning rate.
4. ConvNeXt-S (49.5 M, 768-d).

**Self-supervised init (members 1, 2, 4).**
- 2 augmented views (random scale and shift), invariance loss plus the SIGReg Gaussian regulariser.
- Runs on unlabelled cells of the training split only.
- 40 epochs for ResNet-50; 60–120 epochs for ConvNeXt-S.

**Heads and fine-tuning.**
- Linear regression head (13 outputs).
- Auxiliary 3- and 10-class cell-type heads (cross-entropy, weight 0.5). Their labels exist only for pooled-panel cells of draw 0, and they are not used at test time.
- Fine-tuning: 10 epochs, AdamW, lr 5e-4, batch 256, ±4 px random shift. No rotations or flips, because the illumination channels are direction-dependent.

**Score.** This 4-member model scores GOAL .6911 (pre-registered, N084), just below the .6916 improvement threshold. The current record, N085 (.6967), averages these members' architecture family more widely: the 3 N053 members, three composite models (separate heads for pooled and single targets) and two ConvNeXt-S models.
- The GOAL record, N053, is the first three members (.6862; .6864 with the Cytoland member re-trained on a fresh seed).
- A 6-member combination registered in N065 as a descriptive secondary analysis reached .6929; it is not counted toward the GOAL.

## Figure 7: Architectures compared (`fig7_architectures`)

**a. Single encoders.** Same data, labels and heads for every encoder; rings are seeds; paired t-tests against ResNet-50 for the arms with 2 seeds.

| Encoder | Initialisation | GOAL |
|---|---|---|
| ResNet-50 | our self-supervised pretraining | .671 |
| ResNet-50, 4× longer pretraining | our self-supervised pretraining | .671 |
| ConvNeXt-S | our self-supervised pretraining | .668 (ns) |
| ConvNeXtV2-T | published microscopy model (Cytoland) | .666 (ns) |
| ConvNeXtV2-T, full learning rate | published microscopy model (Cytoland) | .646 |
| ViT-B | blood-smear photos (DinoBloom) | .655 |
| ViT-B | natural images (DINOv3) | .651 |
| ViT-S | natural images (DINOv3) | .649 |

The arms also differ in fine-tuning learning rate, input resolution (ViTs at 224–256 px) and pooling. This panel compares complete recipes; it does not isolate the pretraining source.

**b. Ensembles** (mean of member predictions):
- ResNet-50 alone: .674.
- Full view + 2× zoom pair: .680. Both ResNet-50 arms were trained on the cf150 label variant; trained on v10 labels they score .673 and .679.
- Adding a third member:
  - a ViT: .680–.682;
  - Cytoland: .686, the registered record.
- Adding ConvNeXt-S as a fourth member: .691.
- All 8 members of the N085 record: .697 (6 of them chosen with knowledge of their test-draw scores).

Seed noise SD is .0027.

**Says:**
- In this comparison, CNNs initialised by self-supervised pretraining on our own unlabelled cells are the best single encoders. Generic ViTs, even with nearly 4× more parameters, are .015–.021 lower, with the recipe caveats above.
- Among our self-supervised CNNs the architecture hardly matters.
- Ensembles gain most from members with a different pretraining source (Cytoland) or architecture (ConvNeXt-S). ViT members add almost nothing.

---

## Figure 8: Aligning two image modalities of the same cells (`fig8_alignment_crossmodal`), written for ML readers

**Setup.** The same blood cells exist in two image modalities:
- label-free microscope stacks (LF), on which the marker predictor is trained;
- photos of the same cells after Giemsa staining (GS), the format hospitals use.

There are 15.7k registered cell pairs on 2 slides. The aim is to make LF-trained predictions available from GS images. Every test is on a held-out slide; rings are the two slides (b1: 5,697 cells; b0: 10,015 cells), one seed each.

**a. Exact-cell retrieval GS → LF on the held-out slide (R@10).** Chance is .0018 / .0010; dashed line .0018.
- Linear map (ridge regression GS → LF features fitted on the pairs): .013.
- CCA: .017. Procrustes: .016.
- SigLIP contrastive: .030.
- SOTAlign (SigLIP plus optimal transport on unpaired clinical GS and LF cells): .028.
- Across slides, every method stays at .01–.04.
- Within a slide (held-out fields of view, chance .010–.016), the same SigLIP and SOTAlign models reach R@10 .38–.40.
- The field-of-view audit (N092) ruled out a shortcut: trained on pairs of different cells from the same field, the same model reaches only R@10 .02-.03 (chance .01-.02), while true pairs pick the partner within its field .77-.85 of the time (chance .23-.28). So the embedding does learn cell identity; it breaks under the shift from one slide to another.

**b. Transferring a per-cell signal.** The signal is a scalar measured on both slides (autofluorescence); the score is r against its measured value on the held-out cells. Each bar is a complete pipeline.
- Embedding plus ridge readout: .21–.40 (CCA .21, SigLIP .28, SOTAlign .33, Procrustes .40). The linear map gives .57.
- GS students trained to regress the LF model's outputs (output distillation) or its latent: .66 (.715 / .601) and .65 (.751 / .549).
- A two-space model whose final alignment step is unpaired Gromov–Wasserstein: .32 (.18 / .46). Its GS encoder was itself trained with pairs.
- The pipelines differ in more than the alignment step. For example, the student pipelines' LF training also saw unstained cells of the held-out slide.

**c. Learning from set-level supervision with the pairs hidden.** The LF → label map is learned only from matched sets of cells. Per-cell r on a new blood draw is plotted against set size; the dashed line is the fully paired readout (.561).
- This is one draw, with features from an encoder that also saw the test draw.
- Sets of the same cells recover most of the signal at 4–16 cells (.55 / .50) and decay towards .10 at 2,048.
- 1-D OT collapses beyond 64 cells.
- Sets of different cells carry no per-cell information (r ≈ 0).

**Says:**
- Joint embeddings learn cell identity within a slide (confirmed by the N092 field-of-view control) but do not survive a new slide.
- For carrying a per-cell signal from LF to GS, directly distilling the LF model into a GS network beat every embedding-plus-readout pipeline we ran.
- Without paired cells, only small sets of the same cells give useful supervision.

## Figure 9: Domain-shift methods (`fig9_alignment_domain`)

**a–b. Clinical smears.** A Giemsa-reading network trained on our cells is applied to public clinical smear datasets from other labs, stains and cameras. The score is the AUROC of pre-registered cell-class contrasts; 0.5 is chance.

**a. PBC dataset.** Rings are the 8 contrasts; paired t-tests against no adaptation.
- No adaptation: .41.
- Test-time whitening: .44.
- Domain-adversarial training (DANN on unlabelled PBC images): .46 at λ = .1, .55 at λ = .3.

All are ns, and individual contrasts range from .11 to .90.

**b. MLL23 dataset.** Rings are 3 teacher models; paired t-tests by teacher. All arms use class anchoring: a cross-entropy on 3 common classes from labelled PBC images.
- Output distillation: .568.
- Post-hoc alignment of the clinical features:
  - unbalanced OT .562, mean shift .571, CORAL .571 (ns);
  - single teachers move by up to −.038;
  - these arms decode an aligned mean embedding rather than averaging decoded predictions.
- Stronger anchor loss (× 4 / × 16): .604 / .627.
- Distilling the LF latent instead of the outputs: .681, and .704 with stronger anchoring.

All are ns at n = 3 teachers. By the pre-registered rule, latent distillation counts as better (mean gain .113 > seed floor .074) and anchor strength makes no difference.

**c. LF model on a new blood draw.** Change in per-cell r against the same unadapted model.
- DANN: +.0025 / −.0015 for the two directions.
- AdaBN (BatchNorm statistics re-estimated on 20k unlabelled images of the new draw): −.005 to −.011 on all 4 runs.

**Says:**
- Unsupervised distribution-level alignment (DANN, whitening, OT, mean shift, CORAL, AdaBN) gave no consistent gain on any of the three shifts. One exception: a mean shift on the latent-distillation student passed its noise floor on 4 PBC contrasts (+.053 vs .048) but lost .031 on MLL23.
- The largest gains came from distilling the teacher's latent and from stronger class anchors, and even those are ns at 3 teachers.

## Figure 10: Decoders, autoencoders and marker priors (`fig10_decoders_autoencoders`), N087

**a. How many dimensions do the predictions need?** A linear decoder with a rank-k bottleneck on frozen features. On the fine-tuned ResNet-50 features, rank 1 gives .458, rank 4 .617, rank 8 .656 and rank 12 .664; the model's own head (dashed) is .673. Self-supervised features that were not fine-tuned saturate near .59. The predictions use about 8–12 dimensions, so a very low-dimensional bottleneck would cost accuracy.

**b. Frozen features with an MLP decoder** (rings are the 2 directions):

| Frozen features | GOAL |
|---|---|
| Self-supervised ResNet-50, not fine-tuned | .603 |
| 144 hand-made per-channel image statistics (no learning) | .607 |
| Autoencoder latent, 16 / 64 / 256 dims | .617 / .624 / .594 |
| Fine-tuned model, own head | .673 |

No decoder (ridge .670, MLP .661, kNN .664) beats the fine-tuned model's own linear head.

**c. Clustering of 10,137 held-out draw-0 cells against 10 gated cell types** (kNN balanced accuracy, chance .10):
- Autoencoders .342–.346 beat self-supervised features (.306). K-means NMI shows the same order: .38–.39 vs .34.
- Simple image statistics do as well (.354, NMI .351).
- So the autoencoder's clustering advantage is mostly brightness, scatter and contrast, not a better representation of cell identity.
- The self-supervised features are deliberately isotropic and cluster worst.

**d. Learned marker priors in the ImageStream data** (9 markers on every cell; rings are 3 held-out donors). The measure is the registered co-expression error over 36 marker pairs:

| Training | Co-expression error |
|---|---|
| One marker per cell | .229 |
| + linear prior (4 dims) | .270 |
| + autoencoder prior (6 dims) | .212 |
| All 9 markers, 1/9 of the cells (same label count) | .174 |
| All 9 markers, all cells | .189 |

- Both priors were fitted on another donor's full labels.
- The autoencoder prior is lower than single-marker training on all 3 donors (paired t, Holm p < .05), but it misses the pre-registered bar of ≤ .21.

**Says:**
- The decoder is not the bottleneck; fine-tuning the encoder is what matters.
- Autoencoders mostly capture brightness and scatter.
- A marker prior borrowed from another donor helps co-expression only slightly; measuring the markers jointly is what really helps.

## Figure 11: Can the models separate cell lineages? BSCCM (`fig11_lineages_bsccm`), N097

BSCCM never measures two single markers on the same cell; CD3 and CD19 sit on different slides. Lineages are therefore defined on each marker's own slide of the **test** blood draw:
- A cell is "positive" if its measured label lies above the valley between the two peaks of the label distribution.
- For the rare lineage markers it must also lie above median + 3 MAD. Without this, 60% of draw-1 CD19-slide cells would count as B cells.

The model is the N085 record ensemble (GOAL .6967).

**a. Predicted marker profile of each true lineage** (mean predicted z-score, averaged over both directions):
- T cells get high predicted CD3, and NK cells get high predicted CD56.
- Monocytes get high CD14 and HLA-DR; CD123+ cells get high CD123 and HLA-DR.
- B cells get high CD19 and low CD16 and CD14.
- T cells also get somewhat raised CD19/CD56, and NK cells raised CD3. The three lymphocyte lineages share a "lymphocyte" signal.

**b, c. Predicted CD3 vs predicted CD19 for true T cells (blue) and true B cells (red).**
- Train draw 1 → test draw 0 (b): T and B cells lie on opposite arms. B vs T AUROC is .95 (39 B, 89 T).
- Train draw 0 → test draw 1 (c): 67 of 692 true T cells get high predicted CD19 and higher predicted HLA-DR. They are spread over 63 fields of view and have clearly positive CD3 labels, so this is a model error, not a label artefact. Only 10 B cells exist in this test set, and AUROC is .70.

**d. Does each marker detect its own positives?** AUROC on the marker's own slide, train draw 1 → 0 / train draw 0 → 1:

| Test | AUROC |
|---|---|
| CD3+ vs CD3− | .99 / .86 |
| CD19+ vs CD19− | .83 / .87 |
| CD56+ vs CD56− | .91 / .94 |
| CD14+ vs CD14− | .98 / .97 |
| CD16+ vs CD16− | .97 / .97 |
| CD123+ vs CD123− | .87 / .86 |
| B vs T (predicted CD19 − CD3) | .95 / .70 |

**e. Are predicted CD3 and CD19 anticorrelated?**
- Over all cells: no, r +.23 / +.18. Both are lymphocyte markers, so both are higher in lymphocytes than in granulocytes and monocytes.
- Within gated lymphocytes: yes, r −.54 (draw 0, the only draw with gated lymphocytes).
- Within true T + B cells: −.49 in train draw 1 → 0, and +.01 in train draw 0 → 1.

**f. B vs T AUROC for the record ensemble and each of its 8 members:**
- Every member separates B from T in train draw 1 → 0 (.92–.95).
- In train draw 0 → 1, members score .63–.76.

**Says:**
- Lineage identity is decoded well for the large, morphologically distinct populations: granulocytes (CD16), monocytes (CD14) and T cells (CD3).
- CD3 and CD19 come out anticorrelated within lymphocytes, as expected, but not over all white blood cells.
- T vs B separation works in one direction (.95) but drops to .70 in the other, where about 10% of T cells look B-like to the model.

## Figure 12: CD3–CD19 exclusivity where every marker is measured: ImageStream sandbox (`fig12_lineages_sandbox`), N097

The ImageStream sandbox (S-BIAD452: brightfield and side-scatter images) measures all 9 markers on every cell. Each of the 3 donors is held out once (N080 ResNet-50 models). Cell types come from gates on the held-out donor's own measured labels.

**a–c. Lymphocytes of held-out donor 0, CD3 vs CD19.**
- Measured (a): T cells (CD3 high, CD19 low) and B cells (CD3 low, CD19 high) form two separate clouds.
- Predicted, both training regimes (b, c): one continuous anti-diagonal band. The model places each cell on a single "T ↔ B" axis instead of predicting two populations.

**d. r(CD3, CD19) within lymphocytes** (rings are donors):
- Measured: −.63 / −.88 / −.68.
- Predicted: −.84 to −.93 in both regimes, so the predictions are *more* anticorrelated than the truth.

**e. B vs T AUROC** (predicted CD19 − CD3):
- All-marker training: .74 / .94 / .46. One-marker training: .81 / .90 / .51.
- The anticorrelation is present, but a cell's position on the axis only partly tracks its true lineage.

**f. Accuracy of the predicted CD3 and CD19 over all held-out cells:**
- CD3: r .58–.89.
- CD19: r .53 / .66 / .09 with all-marker training. On donor 2 the model barely predicts CD19, so B vs T there is at chance.

**g–i. Mean z-score of every marker per gated type**, measured (g) vs predicted with all-marker (h) or one-marker (i) training:
- The predictions reproduce the granulocyte / monocyte / lymphocyte split.
- They compress the lymphocyte subtypes (CD4 T, CD8 T, B, NK) toward each other.

All stars are paired t-tests, Holm-corrected; with 3 donors, power is low.

**Says:** the exclusivity of CD3 and CD19 is learned as a direction in the output space, even too strongly. Assigning individual lymphocytes to T or B is limited by how well CD19 itself is predicted, and that is donor-dependent.

## Figure 13: CD4 vs CD8 T cells in the ImageStream sandbox (`fig13_cd4_cd8_sandbox`), N097

CD4 and CD8 exist **only in the external ImageStream sandbox** (S-BIAD452). BSCCM has no CD4 or CD8 stain. All 9 markers are measured on every sandbox cell, and each of the 3 donors is held out once (N080 ResNet-50 models). T cells and the CD4 T / CD8 T subsets come from gates on the held-out donor's own measured labels.

**Axis units:** "intensity, z-score" = (x − mean) / SD of the marker's asinh-transformed intensity over all held-out cells, computed separately for measured and predicted values. 0 is an average cell; +2 is 2 SD above average. Correlations are always marked r.

**a–c. T cells of held-out donor 0, CD4 vs CD8.** Measured (a): two mutually exclusive populations, CD4 T (CD4 high, CD8 low) and CD8 T (CD8 high, CD4 low). Predicted (b: all-marker training; c: one-marker training): one diffuse cloud.

**d. r(CD4, CD8) within T cells** (rings are donors):

| | Donor 0 | Donor 1 | Donor 2 |
|---|---|---|---|
| Measured | −.75 | −.81 | −.86 |
| All-marker training | −.85 | −.43 | +.20 |
| One-marker training | −.05 | −.50 | +.33 |

The exclusivity is reproduced on one donor only; on donor 2 the predicted correlation even has the wrong sign.

**e. CD4 T vs CD8 T AUROC** (predicted CD4 − CD8): .62 / .75 / .54 with all-marker training and .50 / .68 / .62 with one-marker training. Close to chance.

**f. Accuracy over all held-out cells:** predicted CD4 r .60 / .71 / .23, CD8 r .28 / .16 / .40 (all-marker training).

Paired t, Holm; with 3 donors, power is low.

**Says:** CD4 and CD8 T cells look almost the same in these label-free images, so the models can tell T cells apart from other cells but not CD4 T from CD8 T cells. That is consistent with the literature (DeepIFC CD8 r .41).

## Figure 14: How well can every pair of cell lineages be separated on BSCCM? (`fig14_lineage_separability`), N097

**Scale:** each tile compares two lineages A and B, for example T (CD3) vs B (CD19).
- Every cell gets one score from the *predicted* markers of the record ensemble: z(predicted marker of A) − z(predicted marker of B). For T vs B that is predicted CD3 − predicted CD19.
- The tile shows the **AUROC** of this score: the probability that a randomly drawn A cell scores higher than a randomly drawn B cell.
- Reading guide:

  | AUROC | Meaning |
  |---|---|
  | .5 | Coin flip, not separable |
  | .7 | Weak |
  | .8 | Moderate |
  | .9 | Good |
  | 1.0 | Perfect |

- No classifier is fitted, so the matrix is symmetric. Darker = better separated.

Each lineage is the positive cells of its marker on its own single-stain slide of the test draw. Panels:
- **a.** Train draw 1 → test draw 0: T vs B .95.
- **b.** Train draw 0 → test draw 1: T vs B .70. This test draw has only 10 B cells, and about 10% of T cells look B-like to the model (Figure 11).
- **c.** Mean of a and b: T vs B .82. All pairs range from .82 to .99; monocyte vs T and NK vs monocyte are ≥ .98.

**Slide control:** the two lineages of a pair come from different slides. Scoring all cells of one slide against all cells of the other with the same score should give .5, since every slide samples the same blood draw. The result is mean |AUROC − .5| = .04 (range .40–.54), so slide effects are small.

**Says:** monocytes, granulocytes and lymphocytes separate almost perfectly. T vs B is the weakest pair, and its strength depends on the direction.

## Figure 15: Measured vs predicted lineage-defining markers on BSCCM (`fig15_marker_correlation`), N097

Each panel shows the cells of one marker's single-stain slide in the test draw, scored with the record ensemble. Blue is train draw 1 → test draw 0 ("1→0"); orange is train draw 0 → test draw 1 ("0→1").

**Axes:**
- x: the measured label (asinh-transformed fluorescence, z-scored per slide). The vertical stack at the left is the clipped background floor.
- y: the prediction, on the same scale.

**Legend:** Pearson r over all test cells of the slide, with n. The display shows a random 800 cells per direction.

| Marker | r (1→0 / 0→1) |
|---|---|
| CD3 | .81 / .77 |
| CD19 | .36 / .36 |
| CD56 | .49 / .58 |
| CD14 | .85 / .83 |
| CD16 | .91 / .89 |
| CD123 | .63 / .60 |
| HLA-DR | .52 / .45 |
| CD45 | .73 / .83 |

These are the per-marker r values that enter the GOAL.

**Says:**
- CD3, CD14 and CD16 positives form a separate high cloud that the model predicts well. These are the T-cell, monocyte and granulocyte populations.
- CD19 (B cells) and CD56 (NK cells) are rare and dim: most positives get only slightly raised predictions, which is why r stays at .36–.58.

## Figure 16: CD3 vs CD19 on the same cells: where is the anticorrelation? (`fig16_cd3_cd19_anticorrelation`), N097

Figure 15 plots each marker against itself (measured CD3 vs predicted CD3). That is an accuracy plot, so it must rise. Anticorrelation is a property of two *different* markers on the same cell. This figure plots CD3 against CD19.

**a–c. ImageStream sandbox**, lymphocytes of held-out donor 0; both markers are measured on every cell. Legends give r for all three donors.
- Measured (a): T cells (CD3 high, CD19 low) and B cells (CD19 high, CD3 low) form two clouds. r = −.63 (donors 1, 2: −.88, −.68).
- Predicted, all-marker (b) or one-marker (c) training: one anti-diagonal band, r −.84 to −.93. That is more anticorrelated than measured.

**d–f. BSCCM, record ensemble.** No BSCCM cell has both CD3 and CD19 measured, so only predictions can be plotted.
- **d. All test cells:** r +.23 / +.18, positive. Both are lymphocyte markers, and most white cells (granulocytes, monocytes) are low on both.
- **e. Gated lymphocytes** (draw-0 all-marker slide, 371 cells): r −.53.
- **f. True T and true B cells** of test draw 0: r −.49. B cells sit at high predicted CD19 and low CD3, T cells the other way.

**Says:**
- Within lymphocytes the models do produce the expected CD3–CD19 anticorrelation.
- Over all white cells the correlation turns positive, because the lymphocyte vs non-lymphocyte difference dominates.

## Figure 17: Lineage separability in the ImageStream sandbox (`fig17_lineage_separability_sandbox`), N097

Same score and AUROC scale as Figure 14, on the external ImageStream dataset; all 9 markers are measured on every cell. All T cells are merged into one T lineage (CD3). Values are the mean of 3 held-out donors. Panel a uses all-marker training, panel b one-marker training.
- With the measured markers, every pair is 1.00 by construction.
- Myeloid vs lymphoid pairs: .85–1.00.
- Within lymphocytes:

  | Pair | All-marker (a) | One-marker (b) |
  |---|---|---|
  | T vs B | .72 (donors .75 / .94 / .46) | .74 (.81 / .91 / .52) |
  | B vs NK | .72 | .79 |
  | T vs NK | .79 | .79 |

## Figure 18: AUROC matrix with values, BSCCM (`fig18_auroc_matrix_bsccm`), N097

Figure 14 and Figure 17 now also show the values in every tile (user 11-27); Figure 18 is the same BSCCM matrices. The label under each matrix says what it shows:
- **a.** BSCCM, train draw 1 → test draw 0.
- **b.** BSCCM, train draw 0 → test draw 1.
- **c.** Mean of a and b.

Each tile compares the row lineage with the column lineage, using predicted markers from the record ensemble:
- Every true cell of either lineage gets the score z(predicted marker of the row lineage) − z(predicted marker of the column lineage).
- AUROC is the fraction of (row cell, column cell) pairs in which the row cell scores higher. .5 = coin flip, 1 = perfect.
- The lineages are the measured positives of each marker on its own single-stain slide of the test draw.

Cells per lineage (a / b): T 89 / 692, B 39 / 10, NK 34 / 112, monocyte 334 / 273, neutrophil 504 / 1,733, basophil/pDC 50 / 49.

**Train/test split, exactly:** BSCCM contains two separate blood draws (the dataset's "batch 0" and "batch 1").
- **Train draw 1 → test draw 0:** train on all 269,283 draw-1 cells; score on the 13,113 draw-0 cells in the held-out 10% of fields of view (FOV split 2).
- **Train draw 0 → test draw 1:** train on all 127,946 draw-0 cells; score on the 26,322 draw-1 cells in FOV split 2.
- The other 90% of the test draw is never used, not even for self-supervised pretraining.
- The Wright-stained cells (10,015 in draw 0, 5,697 in draw 1) are excluded from both.
