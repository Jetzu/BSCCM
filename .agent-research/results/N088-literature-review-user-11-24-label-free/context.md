# Context for the N088 literature review (BSCCM project, read first)
DATA. BSCCM (Berkeley Single Cell Computational Microscopy; Pinkard et al.): ~400k human white blood cells, each imaged label-free (LF)
with a programmable LED-array microscope: 23 illumination patterns (brightfield, darkfield rings, half-circle DPC, single LEDs) + a computed
phase image = 24 x 128 x 128 per cell. Fluorescence surface-marker labels: 8 single markers (CD16, CD14, CD45, CD3, CD123, CD56, HLA-DR, CD19)
from SINGLE-STAIN slides (each cell has only ONE marker measured) + pooled-panel slides (CD3+CD19+CD56 in one dye, CD45, CD16 ...; 3-4
pooled targets per cell) + an autofluorescence target. Two blood draws (batches). Giemsa-stained photos exist for ~15.7k of the same cells
(registered pairs); external clinical Giemsa sets PBC and MLL23. External sandbox: S-BIAD452 ImageStream imaging flow cytometry (274k cells, 3
donors, brightfield + side scatter, 9 markers measured on EVERY cell).
TASK / GOAL. LF image -> per-cell marker levels. GOAL = Pearson r over 11 targets, train on one blood draw, test on the other (domain
shift), mean of both directions. Best single network .6766, best pre-registered ensemble .6911, record .6862, target .70, seed noise .0027.
WHAT IS KNOWN (do not re-propose without a new angle):
- Encoders: in-domain SSL (2 views, invariance + SIGReg) ResNet-50 is the best single encoder; ConvNeXt-S SSL same; Cytoland (ConvNeXtV2,
  FCMAE + virtual staining) slightly lower; DINOv3/DinoBloom ViTs -.015-.021. Ensembles of diverse encoders +.015.
- Decoders/heads all flat: HL-Gauss, two-part zero/positive, 8-factor panel latent head (shared factors for singles + pooled), Pearson loss,
  fluorescence-image auxiliary decoder, censored loss, label-separation head, consensus labels, symmetry augmentation, pair loss.
- Label-limited: 2x labelled cells +.016 at matched steps; CD19/CD56 labels unstable (label-version agreement .32/.71); field-of-view level
  label error caps the GOAL (within-FOV scoring .72).
- Joint decoding sandbox (S-BIAD452): all-9-marker labels per cell .596 macro r vs one-marker-per-cell .526; cell-type oracle .814; within
  cell type r only .15-.20; single-stain training loses CD4/CD8 exclusivity.
- Alignment: LF<->Giemsa shared embeddings (CCA, Procrustes, SigLIP, SOTAlign/KL-OT, unpaired GW) retrieve cells within a slide (R@10
  .38-.40) but collapse across slides (.02-.04); distillation (Giemsa net regresses LF outputs/latent) transfers best; DANN, AdaBN, OT,
  CORAL, mean shift never gave a consistent gain; class anchoring helps clinical transfer. Set-level supervision works only with small sets
  of the SAME cells.
- Running now (N087): rank-k / MLP / kNN decoder sweep on frozen features; marker autoencoder prior (PCA/AE, k=2-6) in the sandbox; image
  autoencoder (k=16/64/256) for clustering; SSL projector (2048-2048-2048-256 MLP) features and projection-bottleneck fine-tuning.
CONSTRAINTS. Inductive (no held-out draw/donor labels or images in training, except unlabelled SSL on the training draw); no hand-coded
lineage/marker-exclusivity rules (learned priors from external jointly-measured data are OK); compute: Helma H100s, preempt queue,
anything > 25 GPU-h needs user approval; academic use, non-commercial licences OK. Prior reviews: results/N002-*/, N047-*, N059-*, N066-*,
N079-* (ideas.md, agents/) -- read their ideas/declined lists and do not repeat them.
