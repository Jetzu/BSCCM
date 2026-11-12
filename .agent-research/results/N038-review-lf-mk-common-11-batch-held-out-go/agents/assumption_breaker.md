# N038 review: assumption breaker (GOAL = LF->MK common-11, batch-held-out, v5 labels; best .6527 N031; target .67)

All analyses: CPU, login node helma6, < 1 min each, `apptainer exec ... bsccm.sif python3 -I`. Scripts, csv and stdout are in
`/tmp/n038_ab/` (helma6-local /tmp; the brief limited me to /tmp and this file, so **the integrator should copy `/tmp/n038_ab/*.{py,csv,out,json}`
into `results/N038-.../agents/ab/`**). Paths below are given as `ab/<file>`, meaning `/tmp/n038_ab/<file>`.
Predictions scored: N031 inductive ensemble (n031_{base,zoom2}_{b10,b01}, reproduces GOAL .6527 exactly, ab/c_metric.out),
n023_v5 single models, in-distribution FOV models n015_sslinit_fov (r50, **trained on v4 labels**) and n024_r18_v5_fov (r18, v5 labels).

## 0. Bottom line

1. **The diagnosis' ceiling is mis-measured.** n015 (the D1b "in-distribution" model) was trained on v4 labels. On the batch-1 grp targets it scores
   .679/.648/.711 against v5 labels but .886/.834/.846 against the v4 labels it was trained on. Swapping its grp numbers for the v5-trained r18 FOV model
   gives a single-model in-distribution value of **.6743** (not .658). Even the weaker r18 alone scores .6599 (ab/f_extra.out, ab/f_d1b_ceiling.csv).
   So .67 is below the in-distribution ceiling, not at it. The single-model draw gap is ~.032 (.674 vs n031 base .6425).
2. **Part of that ceiling comes from draw-specific label artefacts, which an inductive model cannot learn and should not learn.**
   Within draw-1 lymphocytes, the single-antibody HLA-DR label is *anti*-correlated with its own dye channel (r -.19) and with the authors'
   autofluorescence estimate (r -.56). In draw 0 the same stratum gives +.88 and +.03. CD19 and CD14 behave the same way in draw-1 lymphocytes
   (-.53 and -.58 with autofluorescence). In-distribution models learn this pattern: their HLA-DR prediction correlates -.73 with autofluorescence and
   reaches within-lymphocyte r .71. The draw-0-trained ensemble gets .10 (ab/h_hladr.out, ab/g_extra.out). Regressing autofluorescence out of the labels
   shrinks the single-target draw gap from .0169 to .0108 (HLA-DR d1 .153 -> .115; CD19 d1 .088 -> .057; ab/i_autofl_resid.out). After removing it,
   fully closing the draw gap would land the ensemble at roughly .672 (best-of-two upper bound). Reaching .67 then needs **~60-90 % of the
   artefact-free draw gap closed**, depending on how much an ensemble gains in-distribution.
3. **CD19/CD56/CD3 are label-limited, and the cause is identified.** About half of their v5 label variance is FOV-level offset
   (excess over a permutation null: CD19 .51/.33, CD56 .46/.56, CD3 .34/.30 for draw 0/draw 1). CD16, CD45, CD14, HLA-DR and CD123 are at .00-.13
   (ab/b_reliability.csv). The offsets come from the authors' smoothed background (`background_lowess`):
   - The dye channel computed with it has FOV excess .87/.86 for CD19. With the authors' per-cell `background` instead, it is .03/.03, and our
     border-ring integral gives .02.
   - `background - background_lowess` in the dye channel correlates r .87/.83 with the v3 CD19 label (ab/e_bgmodel.csv).
   - The model does not predict these offsets: prediction FOV-share equals the null (ab/c_metric.csv).
   - Scored within FOVs, the same ensemble reaches GOAL-equivalent **.721** (CD19 .49/.63, CD56 .56/.53, CD3 .77/.73) against pooled .6527.
4. **The metric is noisy and tail-driven exactly where the gap is.**
   - Absolute GOAL: FOV-bootstrap SD .0077, 95 % CI [.6365, .6659] (ab/c_metric_boot.json).
   - CD123 and CD19 carry about 78 % of that sampling variance (per-target SD .098/.072 and .040/.090).
   - For CD123, the top 1 % of cells supply 89 %/92 % of the covariance (Spearman .27/.11 vs Pearson .615/.459). For CD19 the share is 54 %/78 %.
   - Seed noise on v5 is a difference of .0038/.0037 between seeds, and the paired test-sampling SD of a model difference is .0030. A flat zone of
     ±.004 is too tight; ±.008-.009 is about 2 sigma (ab/f_extra.out).
5. The diagnosis cites N035's "CD19 label only .32 predictable from the cell's own photons" as evidence that labels are the limit. That is right in
   direction but wrong in mechanism. The v3 label *is* the authors' photons: ridge on their 6 channels gives r .94/.94. N035 got a low value because
   the fm_prep integrals lack the authors' FOV background offsets: the v3 label from fm_prep integrals reaches only .27/.37 (ab/b_reliability.csv).

## 1. Assumption ledger: assumption · cheapest overturning test · result

### Data
- **D1. Two draws are a valid test of a "new isolation", and b10/b01 are symmetric replicates.** Test: compare the two directions. **Broken.**
  - Training size differs: b10 has 269k cells, b01 128k.
  - Only draw 0 has gates, so only b01 gets the cls3/cls10 auxiliary CE. The b10 class head is untrained: on draw 0 its class-oracle r is
    .11-.27, against .58-.85 on draw 1 (ab/c_metric.out).
  - grp:CD123/HLA-DR/CD14 is masked in draw 1.
  - With n = 2 draws there is no estimate of between-isolation variance.
- **D2. The FOV-vs-batch gap measures draw shift.** Untested confound: data quantity. The in-distribution model trains on ~318k cells against
  128k for b01, so the draw-1 test confounds shift with 2.5x data. Cheapest test: a FOV-protocol run with `--max-train 128000` (~0.5 GPU-h).
  Not run.
- **D3. Slides within a draw sample the same blood.** Test: predicted-class composition per slide, and reweighting draw-1 test cells to
  draw-0 composition. **Holds roughly.** Per-target r moves by -.02 to +.03 (grp:CD45 .655 -> .683, CD3 .602 -> .581) and the mean effect is small
  (ab/f_extra.out). The 'all' slides differ more (lym/gran/mono .36/.47/.17 in d0 vs .23/.55/.21 in d1).
- **D4. Per-slide input normalisation (global flat field plus a per-slide × channel background scalar from the test slide's background ring) removes
  draw optics differences.** Not tested directly, and AdaBN hurt (N026). Cheapest test: per-channel draw difference of normalised cell-free ring
  statistics (5 min CPU).

### Labels
- **L1. The authors' unmixed value measures binding per cell.** **Broken for CD19/CD56/CD3** (FOV-offset fractions as in bottom line 3, traced to
  `background_lowess`). It holds for CD16/CD45/CD14 and largely for HLA-DR d0 and CD123.
- **L2. The v4 position detrend removes the artefact and nothing else.** **Partly broken.**
  - It removes the smooth part (CD19 v3 FOV excess .87 -> v5 .51) but leaves FOV-specific offsets.
  - It turns censored floor values, a point mass, into position noise: CD3 d1 is 45 % censored with censored label SD .26. Collapsing censored
    test labels to a constant raises the GOAL .6527 -> .6550 (ab/g_extra.out).
- **L3. A v5 label means the same thing in both draws.** **Broken** for single HLA-DR/CD19/CD14 in draw-1 lymphocytes (bottom line 2). N023 masked
  only the batch-1 *pooled* group. The single-antibody batch-1 labels have the same dim-dye unmixing problem inside lymphocytes.
- **L4. The label is the cell's own photons (N035 says no).** Overturned as stated (bottom line 5).
- **L5. asinh(x/q10) plus z per (batch, marker) is a neutral scale.** Pearson ignores the affine part, but asinh leaves CD123 with skewness 9.8/8.9,
  so the metric rewards a handful of basophils (section 3).
- **L6. Label noise does not depend on the input, so training on noisy labels is unbiased.** **Broken** for the draw-1 artefact, which is
  LF-predictable through autofluorescence (FOV-model prediction vs autofluorescence -.73). It holds for the FOV offsets (prediction FOV-share is at
  the null).

### Representation, loss, model
- **R1. A single 128-px crop with no rotation or flip augmentation is enough.** Partly tested: zoom gives +.002-.003 complementarity (K023), and
  within-FOV r is high (.72). Untested: a flip with channel permutation (DPC L<->R swap, phase sign), about 1 GPU-h.
- **S1. Equal-weight masked MSE plus CE is Pearson-optimal for every target.** The MSE optimum is the conditional mean. For CD123 that is
  tail-dominated, and tail weighting helped in-distribution only (K020). The CE term exists in b01 only (D1).
- **M1. The predictor is near what the data allow** (attempts.md premise). **Broken**: in-distribution single model ≥ .674 vs .6425 held-out (§4).
  The predictor has headroom. The remaining question is how much of it is draw-specific label artefact rather than draw shift.

### Metric and evaluation
- **X1. Pooled Pearson per (target, draw), one slide each, is a stable per-cell skill measure.** **Broken** for CD123 and CD19 (section 3).
- **X2. A GOAL change below .004 is real.** **Too tight** (bottom line 4).
- **X3. The D1b ceiling of .658, and ".67 sits AT the ceiling".** **Overturned** (bottom line 1 and §4).
- **X4. The per-target gaps in diagnosis.md are signal.** Two are not:
  - CD123 -.147/+.067 and CD19 .00/+.10 are within 1-1.5 bootstrap SD (CD123 .07-.10, CD19 .04-.09).
  - The CD123 d0 "held-out beats in-distribution" anomaly (.615 vs .41/.30) is a roughly 18-basophil tail statistic.
- **X5. Inductive.** The constraint holds. SSL excludes the held-out batch. The only test-slide statistic used is the label-free background-ring
  scalar (led_norm_stats.py). That is acceptable but should be stated.

## 2. Label reliability: CD19/CD56/CD123/HLA-DR vs CD16/CD14 (requested)

| method (file) | CD19 d0/d1 | CD56 | CD3 | CD123 | HLA-DR | CD14 | CD16 | CD45 |
|---|---|---|---|---|---|---|---|---|
| FOV-offset share of v5 variance, excess over permutation null (b_reliability.csv) | .51/.33 | .46/.56 | .34/.30 | .03/.07 | .03/.13 | .06/.12 | -.01/.00 | -.02/.02 |
| -> reliability upper bound 1-f | .49/.67 | .54/.44 | .66/.70 | .97/.93 | .97/.87 | .94/.88 | ~1 | ~1 |
| -> pooled-r cap for a perfect per-cell model, sqrt(1-f) | .70/.82 | .74/.67 | .81/.84 | .99/.96 | .99/.93 | .97/.94 | 1 | 1 |
| current ensemble pooled r (c_metric.csv) | .30/.40 | .38/.36 | .56/.60 | .62/.46 | .58/.45 | .86/.83 | .90/.90 | .71/.85 |
| current ensemble within-FOV r | .49/.63 | .56/.53 | .77/.73 | .67/.51 | .57/.48 | .87/.88 | .92/.90 | .72/.85 |
| dye channel FOV excess, authors' `background_lowess` vs per-cell `background` (e_bgmodel.csv) | .87/.86 vs .03/.03 | .63/.41 vs .01/.00 | .34/.10 vs .00/.00 | .03/.18 vs .00/-.01 | .03/.23 vs .02/.00 | .11/.34 vs .06/.00 | .03/.02 vs .02/.01 | .01/.28 vs .01/.01 |
| 2-component GMM on the v5 label: positive fraction, d′ (b_reliability.csv) | .24/.90, .60/.15 (no B-cell mode) | unstable | unstable | .06/.05, 8.0/4.9 | .35/.26, 2.9/1.1 | .20/.12, 12.9/7.1 | .48/.59, 7.8/10.1 | .26/.47, 3.9/6.5 |
| same, on an independent border-ring-background relabel (d_ringlabel.csv) | **.026**/.47, **8.5**/1.8 | .07/.37 | .11/.14 | .24/.42 | .74/.44 | .20/.12 | .54/.63 | .29/.43 |
| label tracks its own dye channel within lymphocytes (h_hladr.out) | d0 +.36, d1 **-.13** | | | | d0 +.88, d1 **-.19** | | | |

Attenuation logic: with a technical FOV-offset fraction f, pooled r ≈ within-FOV r × sqrt(1-f). This predicts CD19 .34/.51 (observed .30/.40),
CD56 .41/.35 (.38/.36) and CD3 .62/.61 (.56/.60). So the pooled-vs-within gap is explained by the offsets.

**Known biases:**
- *FOV excess.* It treats all excess FOV-level variance as technical. Real FOV-scale biology, such as clumps or local composition, would be
  miscounted. Prediction FOV-share at the null argues against image-visible clustering. FOVs hold only 5-10 cells of a given slide, so the excess
  formula is approximate. And 1-f is only an **upper** bound on reliability: per-cell noise inside FOVs is not measured.
- *Background comparison.* The per-cell `background` is noisier per cell. It removes the offsets but my quick re-unmix with it ("loc" label) is
  worse for CD19 (ensemble .24 vs .30) and better for CD3 (.69 vs .56) and HLA-DR d1 (.58 vs .45). Mean over 16 single target × draw: .590 vs .609
  (e_bgmodel.out). Label quality depends on the unmixing, not only on the background.
- *GMM.* It assumes Gaussian components on the asinh-z scale. Graded expression (HLA-DR, CD56dim) violates that, so the type-only ceiling is not a
  reliability.
- *Ring relabel.* It is crude: aperture r < 32, dye spectrum from the authors' gates, 2 autofluorescence PCs. The d1 unmixing failed for CD19 and
  CD56 (d′ < 2).
- *Spectral split-half.* I tried a three-indicator reliability on autofluorescence-regressed channels 500-550 / 585-625 / 627-673 for the
  CD3/CD19/CD56 dye. It gives .92/.93 for CD19, .66/.89 for CD56 and .97 for CD3, **but it is not informative**. On unstained cells the same
  residuals correlate .78/.83 (shared non-dye nuisance), so the estimate is inflated and bounds nothing (b_reliability.csv, spec_*).

**Verdict:** labels for CD16/CD45/CD14 are reliable. CD16/CD14 pooled r (.83-.90) sits near their caps (.94-1.0). CD45 d0 (.71, within-FOV
.72) has a per-cell deficit, not a label one. CD19/CD56/CD3 are
label-limited to a pooled cap of about .67-.84 by background-model offsets. HLA-DR d1 (and CD19/CD14 d1 in lymphocytes) carry a draw-specific
unmixing artefact. CD123 is FOV-clean, but its pooled r is a tail statistic. Within FOVs the predictor is decent (.72 GOAL-equivalent), so the
image is not the only cap. Diagnosis options (a) image-limited and (b) label-limited are both partly true, and (b) has a specific, fixable cause.

## 3. What the pooled-Pearson metric rewards (c_metric.csv, f_extra.out, g_extra.out)
- **A few outliers.** GOAL Spearman mean .524, winsorised (1/99 %) .636, dropping the top-1 % covariance cells .584.
  - CD123: r .615 -> .214 (d0) and .459 -> .095 (d1) without the top 1 %.
  - CD19: .299 -> .172 and .395 -> .133.
- **Unpredictable FOV offsets** in CD19/CD56/CD3 labels penalise every model equally. Within-FOV GOAL is .721.
- **Censored floors:** +.0023 if collapsed. Small.
- **Composition between draws:** ±.02-.03 per target, ~0 on the mean. Small.
- **Two pooled slides (grp, d1):** grp:CD45 per-slide r .87 (rep1) vs .57 (rep0, the 5x-dimmer slide), pooled .655. The dim slide's label quality
  sets the number.
- **Affine invariance per (target, draw):** calibration shifts between draws cost nothing. Only non-affine shift and label-semantics shift (L3)
  register.
- **Draw-1 artefacts:** the metric rewards predicting -autofluorescence in draw-1 lymphocytes for HLA-DR, CD19 and CD14 (L3). An inductive model
  can only lose there.

## 4. Re-estimated ceiling (f_d1b_ceiling.csv, i_autofl_resid.csv)

| estimate | value |
|---|---|
| diagnosis D1b (n015, v4-trained) | .6582 |
| r18 v5-trained FOV model alone | .6599 |
| n015 with the 6 grp numbers taken from the v5-trained r18 | **.6743** |
| per-target best of the two FOV models (upward biased) | .6764 |
| held-out N031 ensemble | .6527 |
| held-out single model (n031 base) | .6425 |

The artefact-free draw gap is about .019 (ensemble vs best in-distribution single model). This combines the single targets after autofluorescence
residualisation (.0108 mean over 16) with the grp targets (mean .042 over 6, uncorrected). The draw-1 lymphocyte artefact is not removable for grp
from these files.

The largest low-noise gaps:

| target × draw | gap | bootstrap SD |
|---|---|---|
| HLA-DR d1 | .115 after residualisation | .024 |
| grp:CD45 d0 | .088 | .015 |
| grp:CD45 d1 | .056 | .013 |
| HLA-DR d0 | .048 | .023 |
| grp:CD16 d0 | .039 | .016 |
| grp:CD3/19/56 d1 | .039 | .010 |

Missing control: a v5-trained r50 SSL-init FOV model. It costs ~0.5 GPU-h and would replace this hybrid with a single measurement.

## 5. Ideas (constraints in declined.md respected: labels-only cleaning, no position feature, no fluorescence input, inductive SSL, no DANN/AdaBN)

**I1. Measure the ceiling properly before any method work.** It decides whether .67 is reachable.
- *Mechanism:* the n015 recipe (SSL-init r50, 10 ep) retrained on v5 under the FOV protocol, plus a data-matched FOV run (`--max-train 128000`). Score
  both on the identical split-2 cells against v5 and against autofluorescence-residualised labels.
- *Why it moves the bottleneck:* it separates the draw gap into shift, data quantity (D2) and draw-1 label artefact (L3). Only shift is closable
  inductively.
- *Expected:* in-distribution ~.674-.68. Draw-1 HLA-DR/CD19 advantage shrinks after residualisation. If the data-matched run loses ≥ .01, part of
  the "shift" is data and is not closable.
- *Falsifier:* if the v5-trained r50 FOV model is < .665 on identical cells, .67 is unreachable without a label re-base. Cost 2 runs × ~0.5 GPU-h
  = **~1 GPU-h**. Build: trivial (existing flags).

**I2. Background-correct relabel (v8) for single-antibody targets, plus a draw-1 own-dye-sign QC.** Labels only. User: "clean LABELS only".
- *Mechanism:* recompute each channel as `(total_raw - background)/shading`, i.e. the authors' per-cell background instead of `background_lowess`.
  Or use the fm_prep border-ring integrals with a smaller aperture. Unmix single-dye slides with per-draw dye spectra and an unstained-cell
  autofluorescence basis. Mask (as in N023) any slide × fluorescence-defined stratum whose label correlates negatively with its own dye channel
  (HLA-DR d1 lymphocyte stratum: -.19).
- *Why it moves the bottleneck:* it removes 33-56 % non-cell variance from the CD19/CD56/CD3 labels. It also removes the LF-predictable draw-1
  artefact that inflates the in-distribution reference and that the metric rewards.
- *Expected gain:* on the frozen v5 GOAL, ≈ +0 to +.005, because the offsets are not input-predictable (N037 analogue +.001). Scored on v8, the same
  model would move toward its within-FOV level (≤ .72). That is a **re-base, not progress** (N023 precedent): report it separately, with the target
  re-derived and confirmed by the user.
- *Falsifier:* 0 GPU-h. v8 must pass criteria **pre-registered before any model scoring**:
  - FOV excess ≤ .05 for every single target;
  - a CD19 positive mode of 1-6 % with d′ ≥ 3 in **both** draws;
  - r(v8, v5) ≥ .95 for CD16/CD45/CD14;
  - a non-negative own-dye correlation in every stratum.

  If no unmixing passes, drop the idea. My crude versions pass the FOV criterion but fail draw-1 bimodality. Then rescore saved predictions
  (0 GPU-h) and run one GOAL retrain on v8 scored on v5 (~1.5 GPU-h).
- *Build:* ~1 day CPU.

**I3. Shared-dye structured head (CD3 + CD19 + CD56 = grp:CD3/CD19/CD56).** The N023 spectra are the same dye (cos .997-1.000).
- *Mechanism:* the head outputs three non-negative abundances per cell in linear units (invert the asinh/z). Singles supervise their own component;
  'all' cells supervise the sum. Apply the same idea to CD123/HLA-DR/CD14 in draw 0.
- *Why it moves the bottleneck:* the per-cell CD19/CD56 deficit may be **few positives**, not image information. A CD19 slide has ~2.6 % B cells
  (ring relabel, d0), i.e. ~330 B cells per draw-0 slide and ~200 in draw 1. The sum constraint lets ~37k 'all' cells plus ~38k CD3 cells inform the
  B/NK split.
- *Expected:* CD19/CD56 +.02-.05 -> GOAL +.004-.009.
- *Falsifier first:* a learning curve. Retrain the base arm in both directions with CD19/CD56 training labels subsampled to 50 % (~1 GPU-h). If
  CD19/CD56 r drops < .015, positives are not the limit, so kill I3.
- *Build:* ~1 day. The detrend makes the linear-units inversion approximate, so document it.

**I4 (cheap, small). Per-slide reliability weighting of batch-1 'all' grp labels.** The IDEAS.md entry is open.
- *Mechanism:* down-weight or mask the dim rep0 'all' slide (grp:CD45 per-slide r .57 vs .87) in b10 training.
- *Why:* it targets the largest low-noise draw gap, grp:CD45 d0 (.088, SD .015), and grp:CD16 d0 (.039).
- *Expected:* +.001-.003 GOAL.
- *Falsifier:* one b10 base run with rep0 grp labels masked (~0.5 GPU-h). Build: one flag.

## 6. Pre-mortem: "three months later we did not reach .67"

1. **We optimised inside label noise.**
   - The gap (.017) is about 2 SD of the test-sampling uncertainty of the absolute GOAL (SD .008).
   - CD123 and CD19, whose pooled r rests on roughly 7-20 bright cells and carries 78 % of the sampling variance, swung each attempt.
   - CD19/CD56/CD3 were capped by background-model FOV offsets that no image model can predict.
   - Real per-cell progress, visible within FOVs (.72), never showed up in the pooled v5 number, and ±.004 "wins" were noise.
2. **The reference we chased was partly artefact.** The in-distribution ceiling we tried to approach included the draw-1 single-antibody unmixing
   artefact (HLA-DR/CD19/CD14 lymphocytes, anti-correlated with their own photons) and 2.5x more training data in the b01 direction. Inductive,
   draw-invariant methods, already flat or negative historically (DANN, AdaBN), could close at most ~.019. Reaching .67 needed 60-90 % of that.
3. **n = 2 draws, an unconfirmed target, and an asymmetric protocol.** Every method was tuned on the same two directions. b10 had no class
   supervision, b01 had half the data. The .67 target came from mismatched cell and target sets (FOV .71 vs batch .634) and was never confirmed by
   the user or re-derived when the ceiling estimate changed. With label repairs (v5 -> v8) pending, the goal could not be scored stably.

## Files (all `/tmp/n038_ab/` on helma6; copy into results/N038-.../agents/ab/)

| file | contents |
|---|---|
| c_metric.py / .csv / .out / _boot.json | GOAL reproduction, Spearman/winsor/top-1 %, FOV decomposition, class structure, FOV bootstrap |
| b_reliability.py / .csv / .out | FOV excess, unstained control, background model, spectral split-half, GMM, label vs photons |
| d_ringlabel.py / .csv / .out | independent border-ring-background relabel scored with ensemble and FOV models |
| e_bgmodel.py / .csv / .out | `background_lowess` vs per-cell `background`, background-model error vs label |
| f_extra.py / .out, f_d1b_ceiling.csv | seed pairs, bootstrap SDs, paired difference SD, D1b ceiling correction, composition reweighting, censored |
| g_extra.py / .out | censored collapse, HLA-DR per-class structure, variance shares |
| h_hladr.py / .out | draw-1 lymphocyte label vs autofluorescence and own dye channel |
| i_autofl_resid.py / .csv / .out | draw gap on autofluorescence-residualised labels |
