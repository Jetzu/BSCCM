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
- Later combinations reached .690–.691, including the pre-registered N084 at .6911. All stayed below the improvement threshold of .6916 (record + 2 × noise).
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
