# N038 diagnosis: where the gap (.67 - .6527 = .017) lives

## Ceiling test D1b (same architecture, identical cells) -- d1b_same_arch.py -> d1b_same_arch.csv
Same SSL encoder (n007 r50), same 10-epoch fine-tune; FOV-split model trained on BOTH draws (n015_sslinit_fov, i.e. it HAS seen the
test draw, 318k cells) vs batch-held-out single model (n023_v5_b10/b01, other draw only), scored on the SAME split-2 cells of the test
draw, v5 labels, common-11:
  draw 0 test: in-distribution .6552 vs held-out .6437 (gap .0115); draw 1 test: .6612 vs .6409 (gap .0203); overall .6582 vs .6423.
D1 (n024 r18 FOV vs the N031 GOAL ensemble, same cells): r18 in-distribution .6457/.6741 vs ensemble .6612/.6443 (d1_fov_vs_batch.csv).
=> Seeing the test draw is worth ~.016 for the current model family. A PERFECT removal of the draw shift (single model) would land at
~.658; with the +.010 ensemble gain at ~.668 -- i.e. the target .67 sits AT the in-distribution ceiling of the current predictor.
The "FOV ~.71 vs batch .634" gap quoted in GOAL target_source compared different cell/target sets; on identical cells it is ~.016.

## Where the mean is lost (per target, D1b in-distribution / held-out)
Lineage markers are at their ceiling with ~0 shift cost: CD14 .85-.88, CD16 .89-.91, CD45 .71-.88, grp:CD3/19/56 .77-.79.
Four targets drag the mean, in-distribution too: CD19 .28/.47, CD56 .41/.39, CD123 .41/.53, HLA-DR .61/.60 (held-out .28-.56).
Raising each of these four by +.10 would add ~+.036 to the 11-target mean (= 2x the gap); the draw shift is concentrated in the same
targets (HLA-DR +.05/.15, CD19 .00/.10, CD56 .05/.01) plus grp:CD45 draw 0 (.125).

## Bottleneck hypothesis
The GOAL is bounded by PER-CELL information about lymphoid subsets and rare/dim populations (CD19 B cells, CD56 NK, CD123
basophils/pDC, HLA-DR on monocytes/B) -- either (a) image-limited: 128-px single-cell LF crops at the current resolution / model do
not resolve them, or (b) label-limited: dim antibodies, autofluorescence-dominated unmixing and the position artefact (K007: position
alone predicts CD19 .84) leave low label reliability. Both cap in-distribution r, so neither draw-shift work (DANN, AdaBN refuted)
nor training-label hygiene (N037 flat) can reach .67. Evidence for (b): N035 post-hoc, the label of CD19 is only r .32 (draw 0) /
.56 (draw 1) predictable from the cell's own fluorescence integrals (FOV-CV ridge), CD56 .54/.48, vs CD16 .97 (label mostly NOT the
cell's own photons after detrend+unmixing); K004 within-class r .08-.45; K020 tail/focal helps in-distribution only.

## What would falsify it
1. Label reliability ceiling: a spectral split-half (or bimodality/mixture) reliability estimate for CD19/CD56/CD123/HLA-DR labels
   >= .85 => labels are reliable, the image model is the bottleneck (then resolution/context/model changes are the lever).
   Reliability <= r^2-compatible (sqrt(rel) ~ current in-distribution r) => the GOAL target is unreachable and must be re-scoped.
2. An in-distribution model with much more information (full LED stack at native resolution, larger context, larger backbone)
   reaching >> .66 on the same FOV cells => the current predictor, not the data, is the ceiling.
3. If the draw-shift part (.016) closes with a method that never sees test-draw images in training (GOAL constraint is inductive:
   no test-draw images in pretraining; treat self-training on test images as transductive and report it only as a control),
   e.g. physics-based LED/illumination augmentation or draw-invariant input normalisation, the GOAL reaches ~.668.
Metric is not the limit: noise .002 vs gap .017.

## REVISION after the diagnosis red-team (codex/diag_rt/diag_redteam.md), reproduced first-hand (d1c_fov_models.py -> .csv)
The ".658 in-distribution ceiling" is WITHDRAWN. D1b compared models with different training labels (v4 vs v5 for the grp targets:
n015's draw-1 common-11 is .709 on v4 vs .661 on v5 with unchanged predictions), cohorts and sample counts, and both arms used the
transductive n007 encoder; it is a confounded net difference (+.032 / -.016 contributions cancel), not a ceiling. Reproduced
counterexample on the same split-2 cells, v5 common-11: N022 tail05 FOV (tail weighting .5 + focal 10-gate head, v4 labels) .6800
(draw 0 .6866, draw 1 .6733) vs N022 base FOV .6589 -- almost all from CD123 (.417 -> .646, .531 -> .676), HLA-DR +.03/.005.
The SAME change trained batch-held-out (n022_tail05_b10, b1to0) scores draw 0 .6306 vs base .6343 (v5 common-11): the in-distribution
CD123/rare-population gain does NOT transfer to the new draw (K020 said the same on the legacy metric).

Revised bottleneck hypothesis: the GOAL is limited by DRAW-SPECIFIC learning of the rare/fine populations. What the predictor learns
for CD123 (basophils/pDC), HLA-DR and lymphoid subsets from one draw is partly that draw's appearance (imaging/donor), so
objectives that sharpen those populations help in-distribution (+.02) and are neutral-to-harmful across draws. Lineage markers
(CD14/CD16/CD45) transfer. HLA-DR has a positive in- vs cross-draw gap in both draws (bootstrap CIs [.025,.084], [.113,.178],
red-team). CD19/CD56 are low in EVERY recipe (.28-.47 / .36-.42), so for those two an information/label limit remains possible but
unmeasured (no valid reliability estimate exists; the N035 ridge numbers are not reliability).
Falsifiers: (1) a draw-invariance mechanism trained only on the training draw (between-slide replicate nuisance, physics-based
illumination/defocus augmentation, physical-unit inputs) lifts held-out CD123/HLA-DR while tail weighting stays in-distribution-only
-> confirmed; (2) if tail05 + such invariance transfers its +.02 to the held-out draw, the GOAL moves past .67; (3) if nothing lifts
CD19/CD56 in-distribution either, those two are information/label-limited and the target should be re-scoped per target.

## REVISION 2 -- the metric's labels are the main limit (assumption breaker agents/assumption_breaker.md + agents/ab/*, reproduced
first-hand by d2_within_fov.py -> d2_within_fov.csv)
- Same GOAL ensemble, same cells, scored WITHIN fields of view (label and prediction demeaned per FOV): .7211 vs pooled .6527
  (b10 .7262 / b01 .7160). CD19 .30 -> .48 / .40 -> .63, CD56 .38 -> .56 / .36 -> .53, CD3 .56 -> .77 / .60 -> .74.
- The v5 LABELS of CD19/CD56/CD3 carry FOV-level variance far above a permutation null (excess .45/.46, .45/.49, .39/.34 for
  b10/b01 test draws); the PREDICTIONS carry none (excess -.02..+.02). CD16/CD45/CD14/HLA-DR/CD123 labels: <= .13.
- Cause (agents/ab/e_bgmodel.py/.csv; inspected, numbers not re-run): the authors' unmixed values use a SMOOTHED background
  (`background_lowess`); in the dim dye channel shared by CD3/CD19/CD56 the background error (background - background_lowess) has FOV
  excess .94-.97 and correlates .87/.83 with the v3 CD19 label; dye values with the authors' PER-CELL background have FOV excess .03.
  Our smooth position detrend (v4) cannot remove FOV-scale offsets; v5 keeps them.
- So ~35-50 % of the CD19/CD56/CD3 label variance is non-cell background-estimation error that no LF model can (or should) predict;
  the frozen pooled GOAL caps those targets near r_within x sqrt(1 - f). Reaching .67 on v5 labels would require ~60-90 % of the
  remaining draw gap closed (assumption breaker) -- the eval, not the predictor, is now the binding limit.
- Second artefact: draw-1 single HLA-DR (and CD19/CD14) labels within lymphocytes run AGAINST their own dye (r -.19) and with the
  authors' autofluorescence estimate (-.56); in-distribution models learn it, an inductive model cannot (agents/ab/h_hladr.out;
  [unverified by me]).
- Metric uncertainty: FOV-bootstrap SD of the GOAL .0077 (agents/ab/c_metric_boot.json); v5 seed pairs differ .0037-.0038 ->
  "flat" should be judged with paired differences, not the .002 noise alone.
Bottleneck (final): label/eval -- FOV-level background-estimation error in the dim-dye targets (+ draw-1 autofl artefact) -- plus
a genuine, smaller, draw-specific generalisation gap for rare populations (CD123, HLA-DR; revision 1).
