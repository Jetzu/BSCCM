# Outsider review (N047): reaching mean r .67 on LF->MK common-11, or showing the score is the limit

Sources read: `notebook/GOAL.md`, `N047/attempts.md` only. Everything else is general BSCCM knowledge.
Statements about pipeline internals (loss scale, augmentations, metadata) are **assumptions**, marked [assume].
Written 2026-11-17 (Tue Nov 17 19:28 CET).

## 0. Bottom line

* The model axis looks saturated. Ten-model ensemble .657 and noisy student .6564 (+.0037). Round 2 was worse,
  and every attempt since N038 has stayed at or below +.004. Reaching .67 needs another +.0136 to +.0173.
  That is 5 to 6 seed-noise sd, and about 4x the best paradigm gain so far.
* My main suspicion is that the **target is mis-set**, not that the models are exhausted. The target was
  "close half of FOV ~.71 vs batch .634", computed before the re-base. If N039's "v5-trained FOV model
  .678" is the within-draw, FOV-held-out score on the **same** common-11 v5 scorer (that is my reading), then
  the remaining within-vs-cross gap is .678 - .6527 = .025. In that case .67 asks us to close about 70 % of
  that gap. Part of the gap cannot be predicted from images by construction: N038 showed FOV background
  error and labels whose definition is specific to each draw.
* Do this first: run a CPU-only decomposition (section 1, roughly 0 GPU-h, about 1 day). It tells us whether
  .67 is reachable at all, and if so which idea in section 2 to fund.
  Only section 2 idea A has a credible path to the full +.017. Its gain would partly be measurement-artefact
  modelling, so it must be reported as such.

## 1. Is the score the limit? How to show it (Step 0)

**Key identity.** Let m_test(x) = E_test[y|x], the conditional mean in the held-out draw on the scoring scale.
For any predictor f, cov(f,y) = cov(f,m_test). It follows that

    r(f, y) = corr(f, m_test) * r_max,   r_max = sd(m_test)/sd(y)  (= corr(m_test, y))

The GOAL is therefore **(transfer factor) x (ceiling)**:
* The transfer factor is how well a model trained on the other draw tracks the test draw's own conditional mean.
* The ceiling is how much of the test label the image determines at all, given label noise, FOV background
  and biology that LF imaging cannot see.
No image model can raise the second factor.

**S1. Measure both factors (the centrepiece).**
* Train the standard recipe *within* each draw with 2 FOV-disjoint folds, cross-fitted. Every test cell then
  gets an out-of-fold prediction m_hat_test. This is for diagnosis only and never for selection.
* Per target and direction, compute:
  - r_within = r(m_hat_test, y), which estimates the ceiling for this model class;
  - T = corr(f_cross, m_hat_test), the transfer factor, computed label-free.
* If mean r_within is about .678, then .67 needs T of about .988 averaged over 22 cells. That means near-perfect transfer.
* Targets with T >= .97 have nothing left to gain from transfer. Stop working on them.
  Targets with low T are where work should go.
* Cost: 4 single-model trainings. That is about 4x the N040 single-model cost, roughly 4-8 H100-h if one model
  takes 1-2 h. If N039's FOV models already cover both draws with out-of-fold predictions, the cost is 0.
* This is a *soft* bound (relative to the model class). A better architecture lifts r_within as well.

**S2. Hard, model-free ceiling: label reliability.** For each target and draw, r_max <= sqrt(Rel), with
Rel = 1 - var_noise / var(y). Three estimates:
- (a) **Noise floor from the unstained slide** of the same draw, run through the identical unmixing. var(y | true 0)
  gives shot/readout noise, autofluorescence leakage and background error for cells that truly carry no marker.
- (b) **Split-channel reliability.** Most fluorophores emit into 2 or more of the 6 channels. Unmix the marker twice
  from disjoint channel subsets, correlate the two per-cell estimates, then apply Spearman-Brown.
  This gives true per-cell test-retest reliability without new data.
- (c) **FOV background variance.** Take the spread of per-FOV medians of biologically negative cells (gated by
  LF cell type), minus sampling variance. This gives sigma_b^2 / sigma_y^2.
Batch 1's pooled assay already failed QC because of crosstalk. I expect b0to1 (scored on batch-1 labels) to be
label-capped. Report the ceiling **per direction**.

**S3. Oracle bounds on what each class of fix could gain.** Use test labels to *bound* only, never to choose.
- O1, per-target global isotonic recalibration: the upper bound for any monotone-scale or calibration fix.
- O2, per-slide affine fit: the cost of staining/titre differences between slides.
- O3, per-FOV offset, leave-one-cell-out: the cost of FOV background error. Then r_O3 is about
  sqrt(r^2 + sigma_b^2 / sigma_y^2), which caps idea A.
- O4, latent-state mean swap. Use K of about 20 k-means states of the ensemble embedding, fit on the training draw.
  Compare predicting mu_k^train with predicting mu_k^test. The difference is the cost of non-affine,
  draw-specific label definitions, which no inductive method can recover.
- O5, composition swap pi_train -> pi_test. This caps idea D.

**Decision rule.** Add up what a label-free method could plausibly recover:
- about half of O1 and O3;
- 0 of O2 and O4;
- all of O5 if TTA is ruled allowed;
- the low-T targets from S1.
If that sum is below +.017, .67 cannot be reached on this metric. Re-target, and change what is reported.

**S4. Uncertainty that matters.** The seed sd of .0027 is the right noise for *paired model comparisons*.
"Predicts in a NEW isolation" rests on n = 2 draws, so show both directions separately, not only their mean.
Add a FOV-cluster bootstrap CI on the GOAL, resampling FOVs rather than cells, because cells within a FOV
share the background error.

**What I would report instead, if the limit is confirmed:**
1. The GOAL as now defined, kept for continuity, with its FOV-bootstrap CI and both directions.
2. **Transfer efficiency** per target: r_cross / r_within(test draw), equal to T times the ratio of model to
   ceiling. This is the question the GOAL actually asks. New target: mean efficiency >= .97, or half the
   *measured* gap, (.6527 + .678)/2, about .665, if .678 holds.
3. **Within-FOV Pearson and Spearman** (FOV-demeaned, pooled). These are immune to FOV background error and to
   monotone label-definition shifts between draws.
4. **Disattenuated r** = r / sqrt(Rel_test), from S2: the fraction of the measurable signal that is explained.
5. **Biology readouts.** Gating AUROC/AUPRC for marker+ cells, with the gate set by a 2-component GMM on each
   draw's own labels, plus per-slide composition error in percentage points (immunophenotype fractions).
   This is what a user of a label-free immunophenotyper consumes.

## 2. Ideas that move the GOAL as defined

### A. FOV-context "measurement model" head (unconventional)
* **Mechanism.** Predict the *measured* marker as biology plus artefacts that can be seen in LF. Model it as
  y_i = s_i + sum_j kappa(d_ij) s_hat_j - beta * g(FOV).
  - The sum term is optical spillover from bright neighbours, using predicted marker levels of nearby cells.
  - g(FOV) is background bias. If background is a per-FOV statistic of the fluorescence image (my assumption),
    dense FOVs or FOVs with many bright cells inflate it, so the labels of every cell in that FOV are pushed
    down in a predictable way.
  - Inputs are cell positions and counts per FOV [assume available in metadata], plus the predicted markers
    of the other cells.
  - kappa and beta are fit on the training draw. A 2-4 parameter physics form per target transfers better
    than a free set-transformer.
* **Why it moves r.** N038 found FOV-level background error in v5. Any part of it that is predictable adds
  directly: r' = sqrt(r^2 + q * sigma_b^2 / sigma_y^2). Getting from .6564 to .67 needs
  q * sigma_b^2 / sigma_y^2 of about .018 on average (from .6527: .023).
* **Expected gain.** Between 0 and +.015, with high variance. It is capped by O3: if O3 adds less than .03,
  this idea alone cannot reach .67. It is the only idea here that adds *information* rather than squeezing the
  same information harder.
* **Cheapest falsifier.** 0 GPU-h, a few CPU hours. Take existing ensemble predictions.
  - On the training draw, regress residuals (y - f) on FOV features: cell count, sum and max of predicted
    marker, distance-weighted neighbour predicted marker, distance to the nearest predicted-bright cell.
  - Use gradient boosting plus the 3-parameter physics fit.
  - Apply to the test draw and score.
  - Kill the idea if the GOAL gain is below +.003.
  - Also check within-FOV r. If the GOAL rises but within-FOV r does not, the gain is artefact modelling.
    Report it that way, which also proves the labels carry predictable artefacts. That evidence serves section 1.
* **Build cost.** About 1 day, plus about 2-4 H100-h for a small set head on frozen embeddings, if the
  falsifier passes.
* **Caveat.** This raises the score partly by modelling the instrument and pipeline, not biology. Report it on a
  separate line and get a user ruling before counting it toward the GOAL.

### B. Pearson-optimal readout: scoring-scale calibration plus shift-mimicking stacking (cheap)
* **Mechanism.** The maximiser of Pearson r is E[y|x] on the **scoring scale**. Some setups break this:
  - training on a transformed label (log/asinh) [assume, unknown];
  - training with a robust or Huber loss, or with tail weighting;
  - averaging members with equal weights.
  In each case the output is a monotone-distorted or sub-optimally pooled version of E[y|x].
  The fix has three parts:
  - (i) a per-target isotonic map, fit on held-out FOVs **of the training draw**;
  - (ii) NNLS stacking weights over all existing members (N031, N040 students, N041), fit on a
    *shift-mimicking* validation split. Hold out the slides or imaging sessions inside the training draw whose
    image statistics are furthest from the rest, not random FOVs;
  - (iii) choose a model per target with the same proxy.
* **Why it moves r.** It removes calibration and pooling losses that no retraining fixes. Today's selection uses
  random-FOV validation, which rewards draw-specific features.
* **Expected gain.** +.001 to +.004. It is bounded by O1, and by an oracle that picks the best existing model
  per target.
* **Falsifier.** 0 GPU-h. Compute O1 and the oracle per-target best-of-existing on the test predictions.
  If both together add less than +.004, drop the idea.
* **Build.** Hours.

### C. Physics-equivariant augmentation and TTA by permuting LED channels
* **Mechanism.** Rotating or flipping a cell image is only physically valid if the LED-pattern channels are
  permuted the same way. A 90 degree rotation turns DPC-left into DPC-top, and a single off-axis LED maps to
  its rotated position.
  - Find the subgroup of D4 under which the 23 patterns are closed, using the known LED coordinates.
  - Build the permutation table.
  - Use it for training augmentation and for 4-8x test-time averaging.
  - [assume] Today the pipeline either does no rotations (losing up to 8x data efficiency on rare-population
    markers with few positives) or rotates without permuting channels (training on images that are physically
    impossible).
* **Why it moves r.** Rare markers (CD19, CD56, CD123) are limited by having few positive examples. TTA also acts
  as a free ensemble that is closer to invariant.
* **Expected gain.** +.001 to +.003 from TTA alone; +.003 to +.006 with retraining, if this is not already done.
  It is about 0 if the pipeline already does it.
* **Falsifier.** About 0.2 H100-h. Run inference-only equivariant TTA on the existing ensemble in both
  directions. Kill below +.001, but retrain anyway if the current augmentation turns out to be naive rotation.
* **Build.** Half a day for the permutation table and tests; one retrain cycle afterwards.

### D. Test-time composition (prior) adaptation by EM — needs a user ruling
* **Mechanism.** E_test[y|x] = sum_k p_test(k|x) mu_k. Ambiguous cells, such as lymphocytes where morphology
  cannot separate T, B and NK, are dominated by the training draw's prior. Two isolations differ in their
  neutrophil:lymphocyte ratio and their T:B:NK split.
  Saerens-style EM (2002) on the test draw's **unlabeled** LF images re-estimates pi_test. That is about 20
  numbers per slide, with no training on test images.
* **Expected gain.** +.000 to +.004, concentrated in CD3, CD19, CD56 and CD16. It is capped by O5.
* **Falsifier.** About 0.3 H100-h to extract embeddings, then CPU work. Run O5 first; if it adds less than .003,
  kill the idea.
* **Ruling.** The GOAL bans the held-out draw from *pretraining*. Per-slide prior estimation at prediction time
  is test-time adaptation (TTA): it can be deployed, because every new isolation comes with its unlabeled LF
  images. Report it on a separate "deployable-TTA" line unless the user rules that it counts.

### E. Transductive noisy student (classic UDA self-training) — same ruling as D
N040 showed that pseudo-labels work even from training-draw cells only (+.0037 for the ensemble, +.0103 for a
single model). Self-training on the **test draw's** images is the standard UDA step, and it adapts the head to
the target feature distribution.
* **Expected gain.** +.003 to +.008.
* **Falsifier.** One round, one direction, one seed: about the cost of one N040 student.
* **Caveat.** N031's transductive *SSL* did not help (.6512 vs .6527), so the prior is moderate.

## 3. Recommended order and odds
1. Step 0 (section 1). S2 and S3 are CPU-only and take about 1 day. S1 is 0 GPU-h if N039 already has
   out-of-fold models, otherwise about 4-8 H100-h.
2. Falsifiers for B and C (under 0.5 H100-h) and the CPU falsifier for A, run in parallel.
3. If A passes, build it (about 2-4 H100-h) and report it both as a GOAL line and as evidence of a label artefact.
   Ask the user to rule on D and E, without blocking on the answer.
4. If Step 0's decision rule says the gain is out of reach, re-target to transfer efficiency, or to half of the
   measured gap (about .665), and switch reporting to the five-item list in section 1.

My rough odds of reaching .67 on the current inductive metric are about 15-25 %, and almost all of that rests on
idea A. B and C together most likely land at .659-.662.
