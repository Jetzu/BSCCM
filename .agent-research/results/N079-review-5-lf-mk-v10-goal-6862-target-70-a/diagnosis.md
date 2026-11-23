# N079 diagnosis (review 5; 3rd review without progress)

Gap: registered .70 - .6862 = .0138 (descriptive 6-way .6929). Single-model control .6706; every re-processing of BSCCM's own
measurements since N066 is flat or worse (attempts.md).

Instruments (reused + new):
1. K043 (N066): the b10 direction already scores .7036; the b01 deficit sits in draw-1 evaluation labels (replicate slide .88 vs .58,
   FOV-level background disagreement); a label consensus is not a better TRAINING target (N067 worse), separation adds nothing (N075).
2. NEW (n079_joint_consistency.csv, user priority): on held-out 'all'-slide cells the SINGLE-marker outputs, linearly composed (5-fold
   FOV cross-fit on the test cells -- diagnostic), reproduce each measured POOLED multi-marker channel as well as the dedicated pooled
   head: mean r .778 vs .783 (CD3/19/56 .804/.797 vs .812/.795; CD45 .796/.664 vs .779/.660; CD16 .857/.750 vs .868/.784). All 8 singles
   .803, head + singles .810 (recalibration on test-draw labels, not inductive; assumption breaker's inductive cross-output stack +.001).
   -> the per-cell marker profile decoded from single stains is JOINTLY CONSISTENT with the only joint measurement BSCCM has. Joint
   coherence is not the bottleneck; per-marker accuracy is.
3. N076: training only on the multi-marker slides changes nothing (MM3 -.0012) -> the multi-marker cells are already fitted.
4. K033 learning curve (v5): each doubling of labelled cells +~.015-.02 GOAL (weak4 +.05) -> the model is DATA-limited on the weak
   markers. BSCCM labels are exhausted (all training-draw rows used; inductive rule bars the other draw).
5. External ceiling (N077, DeepIFC at 60x, 527k PBMC/WBC, pixel-trained): CD19 r .61, CD56 .53, CD45 .90, CD14 .88, CD3 .87 vs ours
   CD19 .32-.35, CD56 .49-.58, CD45 .73-.83, CD14 .83-.85, CD3 .77-.80 -> sizeable headroom on CD19 at higher resolution / more data.
6. Pending: N074 (dense fluorescence-image supervision: a richer per-cell signal from the SAME data), N078 (in-domain SSL 4x longer /
   ConvNeXt-S: stronger representation; the SSL loss was still falling at epoch 40).

Bottleneck hypothesis: LF->marker decoding on BSCCM is limited by the INFORMATION in its labelled set (few positive cells per weak marker
per draw, one marker per cell) plus evaluation-label noise (unreachable); the model already extracts what these labels support and
decodes a jointly consistent profile. Progress needs NEW information: (a) more/other labelled cells with ALL markers per cell
(S-BIAD452: 9 markers on every cell, ImageStream BF/SSC; user now allows external data), (b) a richer per-cell training signal (N074),
(c) a stronger pretrained representation (N078, external label-free pretraining).
Falsifiers: if adding S-BIAD452 as a second stem (shared trunk + heads) gives no gain on BSCCM CD19/CD56/CD3/CD14/CD45 even though it
adds ~4.3k B / 3.5k NK cells with full panels, the modality gap (60x flow BF/SSC vs 20x LED array) blocks transfer and the data-limit
reading only holds within-modality; if N074/N078 are flat, representation/supervision richness on the same data is also exhausted.

## Corrections after the Codex red-team (agents/codex_diag_redteam.md), reproduced first-hand
- WITHDRAWN "jointly consistent profile / joint coherence is not the bottleneck": a CELL-TYPE-ONLY predictor (FOV-cross-fitted pooled
  means of the predicted 3-class type) already reaches .7299 on b01 vs composed singles .7372 and the pooled head .7462 (n079_rt_repro.csv;
  Codex .7299/.7372). Pooled channels on BSCCM measure mostly cell type; 4 of the 6 entries are singleton channels (CD45, CD16) and the
  CD3+CD19+CD56 channel observes one projection, so co-expression WITHIN a type is NOT testable on BSCCM at all.
  -> Joint per-cell decoding can only be validated on data with all markers measured on the same cell (S-BIAD452).
- WEAKENED "data-limited": the K033 learning curve doubled gradient steps along with labels (train_lf.py steps = epochs x len(dtr),
  verified) -> data quantity and optimisation length are confounded. N078 (longer SSL) and a step-matched learning curve separate them.
- DeepIFC is a reference, not a ceiling (frozen PBMC, separate network per marker, different targets); S-BIAD452 NK counts are NKT
  subsets (2,214 / 1,318), not 3.5k NK cells. IFC pixel = 0.1 um^2 (.33 um/px; confirmed: IDEAS area 139 um^2 = 42 px BF diameter).
Revised hypothesis: on BSCCM the per-marker readout plateau is real but its cause (information vs optimisation/representation) is open;
the user's priority -- joint per-cell decoding -- is UNMEASURABLE on BSCCM and needs an all-marker dataset both to train and to test.
