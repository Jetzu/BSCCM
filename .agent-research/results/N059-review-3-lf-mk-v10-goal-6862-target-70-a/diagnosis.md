# N059 diagnosis: where the gap to .70 lives (n059_diag.py -> n059_diag.csv; best ensemble N048 base+zoom2+Cytoland = .6862)

Gap = .70 - .6862 = .0138 (5 x noise). Ceiling / oracle tests on the SAME GOAL cells, v10 labels:
1. Draw shift is ~0 on AVERAGE: the cross-draw ensemble equals an in-distribution single model trained on both draws (FOV split,
   n039_ceil_fov_v5_mt0, v5-trained) at .6862 vs .6862. Direction-specific: b10 ens .700 vs in-dist .676; b01 ens .672 vs .696.
2. Calibration is NOT the gap: oracle per-slide affine recalibration of predictions (uses test labels) .6890 (+.003).
3. FOV-level label error is still present: within-FOV r .726 (+.040) for the ensemble, .720 in-distribution.
4. Pearson .686 vs Spearman .531: the score is carried by bright positive tails; ranking within the negative bulk is poor.
5. Per-target (ens / in-dist): CD19 .34/.25 and .33/.36, CD56 .48/.48 and .56/.56 -> weak for BOTH models = label/biology-limited;
   HLA-DR .53/.57 and .44/.58, grp:CD45 .75/.85 and .65/.71, grp:CD16 .85/.88 (b10) -> a DRAW-SHIFT deficit concentrated in
   HLA-DR and the pooled-panel groups (the in-distribution model reads them better); CD123 ens .65/.53 >> in-dist (v5 labels).
   Closing only the HLA-DR + grp:CD45 + grp:CD16 shift deficits to in-distribution level would add ~+.016 -> ~.702.
Bottleneck hypothesis: the remaining reachable gap is a draw-specific READOUT of HLA-DR and the pooled-panel groups (antibody lot /
dye-mixture / unmixing differences between the two isolations; K029 autofl slope flip, K019 batch-1 pooled QC), not image
features or calibration; the weak dim markers (CD19, CD56) are label-limited for any model.
Falsifier: if a model that sees held-out-draw IMAGES only (no labels) cannot reduce the HLA-DR/grp gap, and the in-distribution
advantage on those targets comes from seeing held-out-draw LABELS, the gap is a label-readout difference that an inductive LF
model cannot close by construction -> target .70 would require label-side work (per-draw unmixing re-derivation) or is a ceiling.
Caveat: the in-distribution model is v5-trained (cofactor q10) -> its CD3/CD123/CD56 numbers are not comparable; groups and
CD45/CD16/CD14/HLA-DR cofactors (150/500) differ less.

## Corrections after the Codex diagnosis red-team (codex/diag_rt/diag_redteam.md), reproduced first-hand
- 19/22 scores come from ONE slide -> the per-slide affine oracle cannot change Pearson there: "calibration is not the gap" was
  vacuous. A cross-fitted MONOTONE (isotonic, 5 FOV folds, uses target-draw labels) recalibration oracle gives .6907 (+.0045);
  HLA-DR b10 .526 -> .591, grp:CD45 b10 .753 -> .778, grp:CD16 b10 .853 -> .873 (reproduced; Codex's piecewise-linear oracle .6904).
  -> part of the HLA-DR / group deficit is a nonlinear amplitude distortion of the readout, not ranking (HLA-DR Spearman > Pearson).
- "Draw shift ~0 on average" is a cancellation: in-dist minus ensemble per-target contributions +.02325 / -.02323 (reproduced).
- The in-distribution reference differs in labels (v5), data size (318k vs 269k/128k), SSL (transductive n007) and sees held-out
  labels; data-matched 128k in-dist scores .6646 (< the cross-draw ensemble). It cannot isolate a label-readout cause.
- The .70 bridge (+.017 when importing only the in-dist wins) is a selective hybrid, not a demonstrated reachable predictor.
- CD19/CD56 "label-limited for any model" is too strong: label count 50->100 % still lifted weak targets +.035 (K033).
Revised hypothesis: the residual gap mixes (i) nonlinear draw-specific amplitude distortion of HLA-DR and pooled groups (oracle
+.0045 total, up to +.065 on HLA-DR b10), (ii) finite label count on weak markers, (iii) FOV-level label error (.04 within-FOV);
none alone closes .014, and (i) is unreachable inductively unless the model's output scale can be made draw-invariant.

## Ceiling measurements (n059_ceilings.py -> n059_m1.csv, n059_m4.csv; outsider M1/M4)
- M1 ensemble asymptote: greedy forward selection over 22 existing v10 members (TEST-selected = optimistic) peaks at .6922 (N=5),
  fit r_inf = .693 -> no ensemble of the current recipe family reaches .70. Ensembling is closed.
- M4 rare-event leverage: share of held-out label variance in the top 1 / 2 / 5 % of cells: CD123 .88-.92 / .91-.94 / .94-.96
  (18-30 cells per direction carry the target); HLA-DR .34-.51 / .46-.65; CD19 .36-.39 / .48-.49; CD56 .31-.38 / .45-.50; strong
  markers CD16 / grp:CD16 <= .05-.12 (top 1 %). -> the weak targets are few-shot rare-positive problems: their r is set by whether
  ~20-70 bright cells per direction are ranked on top and given the right amplitude; per-target metric noise is large there.
