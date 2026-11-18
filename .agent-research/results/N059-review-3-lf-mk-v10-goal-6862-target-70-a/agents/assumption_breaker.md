# N059 assumption breaker (read-only review; CPU checks only)

Scripts + outputs (scratch, numbers repeated inline below):
`/scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/n059ab/` -> `s2.py`..`s13.py`,
`s2_per_target.csv`, `s3.log`, `s4_composition.csv`, `s5.log`..`s13.log`, `s8_n053_gain_by_item.csv`, `s10_decouple.csv`.
All on the GOAL cells, v10 labels (`targets_v5cx_v10.npz`), GOAL ensemble = n048_v9_base + n048_v9_zoom2 + n053_cytoland (.6862),
in-distribution reference = `runs/n039_ceil_fov_v5_mt0` (as in n059_diag.py). Every run < 10 s CPU.

## Verdict on the N059 diagnosis
- Right: the HLA-DR / pooled-group deficit exists and is not image features, composition, calibration or label-pipeline (v5 vs v10).
- Wrong in three places:
  1. "Calibration is NOT the gap" (oracle per-slide affine +.003) is a vacuous test: Pearson is affine-invariant and 19/22 items have
     one test slide (n059_diag.csv n_slides). A cross-fitted 12-output LINEAR readout of the ensemble's own outputs (5 FOV folds,
     held-out labels) gives **.7042 (+.0180)**; the same readout on the in-dist model gives .6875 (+.0013); isotonic per-target
     +.0032 (s2_per_target.csv, s13.log). The deficit is a cross-target REMIX of existing outputs, not missing image features.
  2. "Draw-specific readout" is partly a LABEL artefact and partly SLIDE-specific, not draw-specific:
     - HLA-DR: the draw-1 label carries a same-cell autofluorescence coupling of opposite sign (label-only test, r(HLA-DR label,
       same-cell autofl label) in lymphocytes **-.606 draw 1 vs +.093 draw 0**; whole slide -.195 vs +.375; s7.log). The in-dist
       model learns it from same-slide labels (its lymphocyte HLA-DR prediction correlates -.857 with the ensemble's autofl head,
       s6.log). Removing the same-cell autofl component from the test label (diagnostic only) shrinks the in-dist advantage
       .140 -> .055 (b01) and .043 -> .017 (b10) (s7.log). The draw-1 HLA-DR stain is near floor: raw p99/p50 1.30 vs 3.72,
       mono/gran median 1.01 vs 1.35 (s6.log), i.e. the same failure mode that got grp:CD123/HLA-DR/CD14 masked in batch 1 (K019).
     - grp:CD45 b01: the whole deficit sits on the 5x-dimmer draw-1 rep0 slide (rep0 ens .572 vs in-dist .656; rep1 ens .876 vs
       in-dist .869). A readout fitted on rep1 does not transfer to rep0 (.590 vs own-slide .669) and rep0 -> rep1 hurts
       (.758 vs raw .876) (s3.log). The two slides of ONE draw differ as much as the draws do.
  3. "Draw shift ~0 on average" (.6862 = .6862) is a coincidence of mismatched comparators. Matched cross-draw control n023_v5
     (same r50, same transductive n007 SSL init -> it has also seen held-out images, same v5 labels, 10 ep) scores .6602 (b10
     .6641, b01 .6563) vs in-dist .6862 -> matched single-model shift cost **.026** (s3.log; runs/n023_v5_b{10,01}/result.json).
     The ensemble + better training labels paid for it.
- Reachable inductive gap: the diagnosis' +.0172 (sum of HLA-DR/grp:CD45/grp:CD16 deficits / 22) minus the autofl-coupling part of
  HLA-DR (.111) and the rep0-slide part of grp:CD45 b01 (.054) leaves **<= +.0097 -> ~.696 even if every remaining deficit closes
  fully** (arithmetic on n059_diag.csv + s7.log). .70 cannot come from these targets alone under the inductive GOAL.

## The four confounds asked about
| candidate explanation of the in-dist advantage on HLA-DR / grp:CD45 / grp:CD16 | test (run) | result | verdict |
|---|---|---|---|
| sees held-out-draw IMAGES (transductive n007 SSL) | n023_v5 = same init, cross-draw | HLA-DR .486/.450 vs .569/.580; grp:CD45 .725/.641 vs .850/.708; grp:CD16 .831/.781 vs .881/.803 (s3.log) | NOT images |
| more labels (318k vs 269k/128k cells) | n039 mt128000 (128k cells, matched to b0to1 127,946) | HLA-DR .552/.563, grp:CD45 .837/.700, grp:CD16 .882/.801 (s2_per_target.csv) | NOT label count |
| sees held-out-draw LABELS | autofl-residualised test label; per-slide readout transfer | HLA-DR advantage 60-70 % = same-cell autofl coupling; grp:CD45 b01 = rep0 slide only | YES - and mostly label artefact |
| v5 vs v10 label pipeline | r(v5,v10) labels; score on v5 | grp items r .975-.993, HLA-DR .942/.991; on v5 labels HLA-DR b01 ens .440 vs in-dist .606 (s2) | NOT pipeline (focus targets) |
| slide count / offsets | within-slide r | b01 grp:CD45 within-slide .654 = pooled .654 (n059_diag.csv) | not offsets; one bad slide |
| cell-type composition shift | predicted-class fractions; class-wise affine readout; class-shift proxy | lym fraction .18-.25 on single slides, .363 vs .229/.254 on 'all' slides; class-affine readout .6857 (no gain); within-class r carries the gap (b10 grp:CD45 true gates lym .112 vs .300, gran .052 vs .243) (s4.log) | NOT composition |

## Assumption inventory (status after this check; cheapest overturning test)
Data
- D1 two draws = two independent new isolations; per target the test set is ONE slide (19/22 items) -> draw = slide = stain lot.
  Test: per-slide readout transfer within draw 1 (run): rep0<->rep1 does not transfer (above). BROKEN for the pooled panel.
- D2 each stain worked in both draws. Test: raw p99/p50 per draw (run, s6.log): HLA-DR 3.72/1.30, CD14 2.46/1.38, CD16 13.5/3.0,
  CD45 8.3/2.9, CD123 1.50/1.28 (near floor in both). BROKEN for draw-1 HLA-DR/CD14 and CD123 everywhere.
- D3 cell-type composition comparable. Run (s4): holds; not the driver.
- D4 input preprocessing is inductive. `led_norm_stats.py` takes per-slide background scale from 300 cells of EVERY slide incl. the
  held-out draw, and a global p50 flat field (label-free test-time normalisation). Untested; cheapest test: recompute scale/flat
  field from training-draw slides only and re-evaluate one checkpoint (~0.1 GPU-h). Low prior of mattering (AdaBN-type changes hurt).
Labels
- L1 labels measure the antibody, not autofluorescence. Run (s7): same-cell autofl coupling flips sign between draws for HLA-DR
  (lym -.606/.093), CD14 (lym -.567/.025), CD45 (all .611/.026), CD3 (.350/-.303), CD19 (gran -.745/-.399). BROKEN (extends K029).
- L2 v10 vs v5 does not drive the in-dist comparison: holds for focus targets; BROKEN for CD3/CD56/CD19 (r(v5,v10) .66-.89, s2),
  where the v5-trained in-dist model is handicapped. It is also far below the ensemble on CD123 b10 (.365 vs .650; on v5 labels
  .407 vs .633) -> the in-dist model is not a per-target ceiling.
- L3 gates (cls10) are independent truth: they are the authors' gates on the same pooled fluorescence -> circular for grp targets.
- L4 FOV-level label error (K028) caps CD19/CD56 pooled r: still true (within-FOV +.20/+.15, n059_diag.csv).
Representation / loss / model
- R1 the remaining error is predictor error. BROKEN: a linear remix of the ensemble's 12 outputs reaches .7042 while the in-dist
  model gains nothing from the same remix (s2). The information is in the outputs; the new draw's label definition differs.
- R2 both directions optimise the same objective. BROKEN: batch 1 has no gates, so b1to0 never trains the class heads
  (cls10 balanced acc .151 b10 base vs .431 in-dist; result.json) - asymmetric, cannot be fixed without held-out labels.
- R3 ensemble members fail differently: no (residual r .91-.99, K036/K039).
Metric / evaluation
- E1 noise sd .0027 is the right yardstick. Single-model per-item seed |diff| mean .0151, max .124 (CD123 b01: Cytoland s0 .503
  vs s1 .379) -> one item moves the GOAL by .0056 (s8.log). 59 % of the N053 gain (.0034 of .0058) is the two CD123 items, a
  near-floor, tail-carried target (Pearson .65/.53 vs Spearman .28/.11, n059_diag.csv; class eta^2 .02-.03, s9.log). Fragile.
- E2 the in-dist model is a reachable ceiling. BROKEN (v5-trained, transductive SSL, single model, learns same-slide artefacts).
- E3 oracle per-slide affine measures calibration. BROKEN (vacuous for Pearson; see verdict 1).
- E4 GOAL is a test set. It is the development set for ~59 items on 2 slides per target; no untouched third draw exists.

## Ideas tested on CPU here (falsified - do not run on GPU)
- Inductive autofl de-coupling (subtract the TRAINING draw's label-only autofl slope x the ensemble's autofl head):
  GOAL .6862 -> .6845 (alpha .5) / .6787 (alpha 1); HLA-DR b01 +.068 but CD45 b10 -.086, CD19 b01 -.118 (s10.log). Net negative.
- Masking the dim rep0 'all' slide in b1to0, class-level mechanism: shifting b10 grp predictions by the training-draw rep1-vs-pooled
  class means gives grp:CD45 .753 -> .753, grp:CD16 .853 -> .849 (s11.log). Any rep0 effect must be within-class.

## Pre-mortem: "Feb 2027, still < .70 - why?"
1. The gap lived in the held-out labels: draw-1 HLA-DR autofl coupling and the dim rep0 'all' slide are unknowable inductively;
   the remaining reachable part (<= .0097) was smaller than .0138 from the start.
2. n = 2 draws, one test slide per target: every "draw effect" was a slide/stain-lot effect; fixes tuned on those slides.
3. Movement came from CD123 tail luck (seed spread .124 on one item) and averaged out on re-runs; the 5-sd bar was never robustly met.
4. CD19/CD56 (4 of the 6 weakest items) are capped by FOV-level label error and could not move pooled r.
5. Model-side levers stayed at +.002-.004 each (ensemble saturation K039, image-side nulls K032), needing 4+ independent wins.
6. The only route that clears .70 (new-draw readout, .7042) needs held-out labels, and nobody asked the user to rule on it.

## Ideas (>= 3)
1. **New-draw few-shot readout (needs a user ruling; it breaks the inductive GOAL).** Mechanism: fit a ridge-shrunk 12-output
   linear readout of the frozen ensemble on n labelled cells per target from the new isolation. Moves the bottleneck because the
   residual shift is a linear remix of existing outputs (s2: .7042; in-dist +.0013 only). Expected: GOAL .70-.705 at ~500 labelled
   cells/target; label-efficiency (s3.log, gain vs raw, 20 reps): HLA-DR +.048/+.088, grp:CD45 +.077/+.036, grp:CD16 +.029/+.001
   at 500 cells; negative below ~100-200 cells, so shrinkage toward identity is required (CD3 b10 n=629 and CD19 b01 n=722 lose
   without it). Cheapest falsifier: pre-registered n in {200, 500}, FOV-disjoint, 20 reps, full GOAL mean; kill if < .700 at 500.
   0 GPU-h, ~10 CPU-min; build ~60 lines. It is also the realistic deployment for a new blood isolation.
2. **Training-label QC of the dim draw-1 rep0 'all' slide (b1to0 only).** Mechanism: rep0 grp:CD45 raw asinh sd .86 vs 1.65/1.63,
   r(CD45,CD16) -.485 vs -.704/-.677 (s4.log); a draw-0-trained model already reads rep1 at in-dist level, so rep0 labels are the
   outlier teaching the b10 model a wrong within-class CD45 (true-gate within-gran r .052 vs .243). Precedent: N023 masking +.10/+.05
   on grp items (K019). The CPU class-mean proxy is null, so expected gain is 0 to +.004 (grp:CD45 b10 .753 -> <= .85). Falsifier:
   one r50 base b1to0 member with rep0 grp labels masked, 1 seed, ~0.4 GPU-h (n048_v9_base_b10 took 1440 s); kill if single-model
   grp:CD45 b10 < .76 (current .732). Build: ~10 lines (mask in make_targets_v5 or a slide-mask flag in train_lf.py).
3. **CD123 variance control (pre-registered, >= 3 seeds).** Mechanism: CD123 items are tail-carried with the largest seed spread
   (.124) and carried 59 % of the last gain; bag CD123-only predictions over cheap extra seeds and/or use a CD123-only tail weight
   (N041 tail05 gave CD123 +.031 but CD19 -.033 when global). Moves the most volatile item toward its seed mean, not the mean
   up. Expected +.002-.004, or exposes N053 as partly luck. Falsifier: 3 extra b01 r50 base seeds (~0.5 GPU-h, 565 s each);
   kill if bagging them into the ensemble leaves CD123 b01 <= .528 (current) or the item's seed sd is < .02 (then CD123 is not
   the noise source). Build: per-target tail-weight vector, ~5 lines.
4. **Slide adapters for the pooled panel (b1to0).** Mechanism: shared output head plus per-training-slide 12x12 remix adapters
   (rep0, rep1), shared head used on the new draw; absorbs the non-transferable per-slide remix (s3.log) instead of averaging it in.
   Expected +.001-.003 (3 grp items only). Falsifier ~0.4 GPU-h; build ~40 lines in train_lf.py. Lowest priority.
- Decision the review should put to the user: (a) allow idea 1 (few-shot protocol), or (b) keep the inductive GOAL and accept
  ~.69-.696 as the realistic ceiling on two draws, or (c) revisit the BSCCM-coherent extra download to get a third draw that can
  tell draw effects from slide effects.
