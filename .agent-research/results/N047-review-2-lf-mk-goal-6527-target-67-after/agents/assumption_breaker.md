# N047 review 2: assumption breaker

Claim under test: "the GOAL (common-11, v5, batch-held-out, best .6527, target .67) is saturated for inductive per-cell predictors, and the
gap is eval-label noise".

All analyses ran on CPU on login node helma6, each under 1 min except one: the first a4 run used too many threads and was killed. The rerun took
10 s with 4 threads. Everything ran with `apptainer exec ... bsccm.sif python3 -I`. Scripts, csv and stdout are in `/tmp/n047_ab/` on helma6.
**The integrator should copy `/tmp/n047_ab/*.{py,csv,out}` to `agents/ab/`.** Below, `ab/<file>` means `/tmp/n047_ab/<file>`.

"HO10" is the 10-model inductive ensemble from d1_big_ensemble.py. Its pooled GOAL is .6571 (b10 .6648, b01 .6494). "IN128" is the N039
in-distribution r50 FOV model trained on v5 with 128k cells, data-matched to b01.

## 0. Bottom line

1. **(a) The FOV offsets are NOT image-visible.** That part of "label noise" survives. Its cause is mis-stated in K028, and most of it is
   created by our own label transform.
   - **Not image-visible.** No FOV-level LF quantity predicts the per-FOV residual offset of CD19/CD56/CD3. I tried cell count, cell size, the
     FOV-mean of all 13 predicted outputs (composition), focus z, and the border (background) brightness of all 23 LED channels plus DPC.
     - Within-draw 2-fold CV R² is at most .03, and mostly negative.
     - Fitting on the other draw (inductive), R² is at most .02.
     - Adding the inductive prediction changes the GOAL by -.020 (count/size/composition) and -.004 (LED border, mean over 16 single
       target × draw entries).
     - Sources: ab/a_fov_offsets.csv, ab/a3_led_border.csv.
   - **Mechanism correction.**
     - At FOV level the v5 offsets do not track the dye-channel lowess background error: r .03/-.10 for CD19, -.18/.03 for CD56, -.04/.02 for
       CD3.
     - They do track the authors' FOV-level autofluorescence estimate (CD19 -.41/-.49, CD56 -.38/-.36, CD3 -.29/-.29) and the 690- background
       (CD19 -.42/-.47). Source: ab/a2_offset_source.csv.
     - Re-applying the recovered unmixing (fidelity .978-.999) with per-cell background in **one** channel never removes the CD19 offset. The
       best single channel is 690-, which takes FOV excess .545/.494 down to .404/.267. Per-cell background in **all** channels gives
       .045/.019 (ab/a4_channel_attrib.csv).
     - So the offset is multi-channel background error amplified by an ill-conditioned unmixing (coefficients up to -607 on 426-446). It is
       not a dim-dye-channel error.
   - **Our transform makes it large.** v3/v5 single labels are asinh(x/q10), with q10 the slide's 10th percentile. That q10 is .056 for CD3
     in draw 1, .84 for CD3 in draw 0, and .17/1.59 for CD56. A near-log scale like that blows up the additive offsets of the negative
     majority. With a fixed cofactor of 50 (same detrend, z-scoring and cells), FOV excess falls sharply (ab/a5_cofactor.csv):

     | target | FOV excess, v5 | FOV excess, cofactor 50 |
     |---|---|---|
     | CD3 | .44/.38 | .10/.05 |
     | CD56 | .53/.57 | .36/.27 |
     | CD19 | .57/.51 | .53/.49 (barely changes) |

   - **The same saved predictions then score above .67.** My q10 replication matches v5 (.6533 vs .6527). Changing only the single-label
     cofactor to 50 gives N031 ensemble .6828, HO10 .6870, single n031_base .6716, IN128 .6838. With a per-slide q50 cofactor, N031 scores .6758
     (ab/a6_cofactor_goal.csv). **This is a re-base, not progress** (N023 precedent).
2. **(b) The gap is not label noise. It is a low-dimensional, draw-specific readout.** FOV offsets cost every model the same: the oracle
   leave-one-out FOV correction would give .756 (ab/a_fov_offsets.csv), and IN128 reaches .6705 on draw 1 against the same labels. The b01
   gap (IN128 - HO10, same cells) is .0211 over 11 targets and .0327 excluding CD123.
   - **A readout remap closes it.** I fitted a per-target ridge remap over the ensemble's own 13 heads on test-draw labels (oracle, 2-fold FOV
     cross-fit, 10 seeds). It closes 91 % of the gap excluding CD123 (.0327 -> .0029), and raises HO10 to .6784 on draw 1 and .6860 on draw 0.
     IN128 gains nothing (-.001/+.004), so the in-distribution readout is already optimal (ab/b2_readout.csv).
   - **The remap is draw-specific in sign.** The autofl head weight is +.57 for HLA-DR in draw 0 and -.69 in draw 1. For CD19 it is +.33 and
     -1.00 (ab/b3_remap_weights.csv).
     - The within-class slope of the label on the authors' autofl, draw 0 / draw 1: HLA-DR +.05/-.60, CD14 -.01/-.80, CD19 -.45/-1.20. CD3 and
       CD56 are stable at -.87/-.79 and -.87/-1.00 (ab/c2_autofl_debias.csv).
     - Draw-1 stains are 2.5-3.9x dimmer in v5 units for HLA-DR, CD14 and CD45 (k ratio, ab/c3_unstained_slopes.csv). A dimmer stain amplifies
       a fixed unmixing residual.
     - An oracle of one scalar per target (the autofl-head weight) is already worth +.0134 GOAL.
   - **Quantified share of the b01 gap (sum over 10 targets excluding CD123, .327):**
     - Oracle within-class autofl residualisation of the test labels removes the whole single-target part: .235 -> -.035. HLA-DR alone goes
       .141 -> .027. That is 83 % of the summed gap (ab/b_draw_gap.csv, arm B).
     - The grp part (.092, 28 %) is per-slide:
       - grp:CD45: the whole gap sits on the 5x-dimmer rep0 slide (.572 vs .651); rep1 has none (.875 vs .867).
       - grp:CD3/CD19/CD56: the gap sits on rep1 (.837 vs .893).
     - **The "CD19 at 35 % dose" explanation is refuted.** Subtracting λ·CD19-head never helps: λ = 0 is best on rep1, and on rep0 the best gain
       is +.004 (ab/b2_readout.out, dose check).
3. **(c) The ensemble is saturated for every member type I could test.** Residual correlation within the 10 members is .971. The
   different-family inductive members are r18 supervised on v4 labels (N013) and r50 scratch (N014). Their residuals correlate .94 with the
   ensemble's, and they are weaker (.615/.625).
   - With the mixing weight chosen on the other direction, the GOAL goes .6568 -> .6563/.6560. Low-pass (N042) and lc50/lc25 (N040) are also
     +0.
   - Output shape (the cheap stand-in for a classification or rank head):
     - isotonic E[y|p] fitted on the other draw: -.034;
     - oracle cross-fit isotonic: -.002;
     - rank-gauss: -.106, because Pearson rewards the tails.
   - Source: ab/c_ensemble.csv.
   - **Every zero-label readout correction fails:**

     | correction | GOAL change |
     |---|---|
     | remap stacked on the source draw | -.038 |
     | remap transferred across directions | -.045 |
     | source-draw autofl debias | -.003 / -.005 |
     | coefficient read from the new draw's UNSTAINED negative-control slide | -.0017 |

     Sources: ab/b3_remap_weights.csv, ab/c2_autofl_debias.csv, ab/c3_unstained_control.csv.

**Verdict.** "Saturated for inductive per-cell predictors" survives every cheap test I could run. "The gap is eval-label noise" is wrong as
stated. The FOV noise is a ~10 % tax on all models, not the gap, and for CD3/CD56 we impose most of it ourselves with q10. The gap is a
draw-specific readout: an autofluorescence coefficient that flips sign or scale between draws, plus per-slide grp labels. Zero-label methods
cannot learn it, but about 250-500 labelled cells per marker from the new isolation would. The two levers are both metric decisions for the
user: a cofactor re-base, or k-shot calibration.

## 1. Assumption ledger: assumption · cheapest overturning test · result

| # | Assumption | Test (file) | Result |
|---|---|---|---|
| A1 | FOV offsets are not image-visible | FOV count/size/composition/focus, 23-LED+DPC border -> offset; within-draw CV + inductive (a_fov_offsets, a3_led_border) | **Holds**. R² ≤ .03; GOAL change ≤ 0 |
| A2 | Offsets = dye-channel `background_lowess` error (K028) | FOV corr with each channel's bg error; per-channel per-cell-bg swap through the recovered unmixing (a2, a4) | **Broken**. Multi-channel error via unmixing; dye-only fix leaves .44-.49 |
| A3 | Label noise is intrinsic to the data | asinh cofactor sweep with v4 detrend replicated (a5, a6) | **Broken for CD3/CD56**. q10 cofactor of .06-1.6 creates it; cofactor 50 gives N031 .6828 on the same predictions |
| A4 | The .017 gap is label noise | in-dist vs held-out on identical cells, oracle readout remap (b_draw_gap, b2) | **Broken**. FOV tax is common to all models; the gap is a 14-parameter draw-specific readout (91 % closed) |
| A5 | Draw-1 definitions = CD19 dose + autofl strata | λ·CD19-head sweep, autofl residualisation, per-slide scoring (b_draw_gap, b2 dose check) | **Half-broken**. Autofl explains the singles (83 % of the summed gap); dose does not; grp gaps are per-slide (dim rep0 CD45) |
| A6 | Ensemble saturated | residual correlation, other-family members, isotonic, rank-gauss (c_ensemble) | **Holds** for all available member types |
| A7 | Draw-specific readouts cannot be learned zero-shot | source stacking, cross-direction, source debias, unstained negative control (b3, c2, c3) | **Holds**. All are -.002 to -.045 |
| A8 | "Best .6527" vs "target .67" is a well-posed gap | metric re-derivations (a6) | **Fragile**. The same predictions span .6527 (q10) to .6828 (cofactor 50); the target was set on q10 labels and never confirmed |

## 2. Ideas (constraints in declined.md respected: labels-only, no position or fluorescence input, inductive SSL, no DANN/AdaBN)

**I1. v5f: fixed-cofactor single-antibody labels (labels-only re-base, needs the user's go).**
- *Mechanism.* Replace asinh(x/q10_slide) with asinh(x/c) for the 8 single-antibody labels. Choose c by a **label-only, pre-registered rule**,
  for example the smallest c whose training-draw split-0 FOV excess is ≤ .15 for every single target. Never choose it on model r. Detrend,
  z-scoring and masks stay as they are.
- *Why it moves the bottleneck.* It removes the self-inflicted part of the FOV tax: CD3 FOV excess .44/.38 -> .10/.05, CD56 .53/.57 ->
  .36/.27.
- *Expected gain.* Re-base GOAL for the same predictions is .6828 for N031 and .6870 for HO10 (c = 50). This is reported separately and is
  not progress. A retrain on v5f labels adds roughly 0 to +.01 per-cell skill.
- *Cheapest falsifier.* Scoring is already done (0 GPU-h). The QC would be r(v5f, v5) for the lineage markers ≥ .95, the CD3 positive mode
  intact, and the own-dye sign. The retrain to separate per-cell skill from the metric change costs ~1.3 GPU-h (base + zoom2, both directions).
- *Build.* 0.5 day: one cofactor option in make_targets_v3 plus QC.
- *Risk.* Forking paths. The rule must be fixed before scoring, and it is a different label for CD3/CD56 (r with v5 .71-.80).

**I2. k-shot isolation calibration as a co-primary metric (protocol re-scope, needs the user's go).**
- *Mechanism.* Freeze the ensemble. Per target, fit a ridge readout over its 13 heads on k labelled cells from a random half of the new
  isolation's FOVs, and score on the other half.
- *Why it moves the bottleneck.* The b01 gap is a low-dimensional readout whose autofl sign flips between draws. It cannot be learned
  zero-shot (A7), but it can be learned from a few cells.
- *Expected gain.* Draw 0 / draw 1, mean over 11 targets (ab/b2_fewshot.csv):

  | k labelled cells per marker | 100 | 250 | 500 | 1000 |
  |---|---|---|---|---|
  | gain | -.044 / -.015 | +.005 / +.020 | +.016 / +.022 | +.015 / +.026 |

  Equivalent to about .675-.68.
- *Cheapest falsifier.* 0 GPU-h, already done. Next is a pre-registered k and FOV split, rerun once.
- *Build.* 0.5 day.
- *Note.* This is biologically honest: a lab stains a few hundred cells per new donor. It does not count as zero-label GOAL progress.

**I3. Draw-consistent autofluorescence coefficient in the single-antibody labels (labels-only re-base, narrow, distinct from v8).**
- *Mechanism.* Re-derive HLA-DR, CD19 and CD14 single labels so the within-LF-class label-autofl slope matches across draws. Fit the slope on
  each draw's training-split cells, then pre-register: "|slope| ≤ .1 in both draws, lineage contrasts unchanged".
- *Why it moves the bottleneck.* Oracle residualisation of the test labels removes 83 % of the summed b01 gap. A label-free route through the
  unstained control does not work (c3: -.0017), so this has to be a label re-derivation.
- *Expected gain.* On the frozen v5 GOAL, ~0 (the labels are frozen). On re-based labels, the b01 draw gap .0211 -> about -.005 (arm B); the
  absolute gain is small (HLA-DR draw 1 about +.03).
- *Cheapest falsifier.* 0 GPU-h: rescore saved predictions on the candidate labels after QC.
- *Build.* 1 day CPU.
- *Risk.* The user already rejected v8. This touches only one coefficient per (marker, draw), but it still changes eval labels.

**I4 (frozen GOAL, inductive, low prior). A free extra arm in the I1 retrain: score the v5f-trained models on frozen v5 too.**
- *Mechanism.* Training on less-amplified CD3/CD56 labels removes gradient noise from training labels that are 38-57 % FOV offsets. The N040
  label-count curve (+.035 on weak targets for 2x labels) suggests the weak targets are limited by training labels.
- *Expected gain.* 0 to +.004 on frozen v5. It overlaps Codex idea 3 (within-FOV contrast loss); run one of the two, not both.
- *Cost.* Included in I1's 1.3 GPU-h.

## 3. Pre-mortem: three reasons we still miss .67

1. **We optimise against a transform artefact.** The q10 cofactor manufactures 38-57 % FOV noise in CD3/CD56. Per-cell gains are taxed and
   invisible. The re-base that shows .68 is filed as "not progress", and the project sits at .653-.657 indefinitely (a6 vs attempts.md).
2. **Each new isolation needs its own readout.** The autofl coefficient flips sign or scale between draws because draw-1 stains are 2.5-3.9x
   dimmer, and the grp gap sits on single slides. Every zero-label fix I tested is negative (-.002 to -.045). With n = 2 draws, any correction
   is fitted to the one flip we have observed.
3. **Statistics and forking paths.** The FOV-bootstrap SD of the absolute GOAL is .0077 (N038 c_metric_boot.json), so the .017 gap is about
   2 SD. The target was set on q10 labels and never confirmed by the user. If the cofactor (I1) or k (I2) is chosen after looking at r, the
   user will rightly reject it as metric gaming. Pre-register label-only selection rules.

## Files (`/tmp/n047_ab/` on helma6; copy to `agents/ab/`)

| file | content |
|---|---|
| a_fov_offsets.py/.csv/.out | FOV offsets vs LF FOV features (count, size, composition, focus), CV and inductive; oracle LOO |
| a2_offset_source.py/.csv/.out | FOV-level decomposition: v5 offset vs v3, trend, dye bg error, authors' autofl, 6-channel bg |
| a3_led_border.py/.csv/.out | 23-LED + DPC border brightness per FOV vs offsets |
| a4_channel_attrib.py/.csv/.out | recovered unmixing with per-cell bg swapped per channel; FOV excess after replicated detrend |
| a5_cofactor.py/.csv/.out, a6_cofactor_goal.py/.csv/.out | asinh cofactor sweep; GOAL-equivalent under cofactor q10/50/q50 |
| b_draw_gap.py/.csv/.out, b2_readout.py/.csv/.out, b2_fewshot.csv | draw-gap decomposition, robust 13-head remap, CD19 dose check, k-shot curve |
| b3_remap_weights.py/.csv/.out | remap weights (autofl sign flip), cross-direction and source-stack transfer |
| c_ensemble.py/.csv/.out | residual correlations, other-family members, isotonic, rank-gauss |
| c2_autofl_debias.py/.csv/.out, c3_unstained_control.py/.csv/.out, c3_unstained_slopes.csv | inductive autofl debias; unstained negative control |
