# N066 outsider report (read only GOAL.md + attempts.md), 2026-11-19

## What the two files say to an outsider

- Need metric >= .70. 6-way (descriptive) = .6929 = mean(b10 .7036, b01 .6822). With b10 fixed, b01 must reach >= .6964 (+.014).
  **The whole gap is in one direction: b01 (train on draw 0).**
- Draw 0 has 2.6-4.8x fewer labelled cells for the affected targets (pooled 10.2k vs 27.1k, CD3 6.5k vs 31.3k, CD16 9.4k vs 27.9k).
  The per-target asymmetry has the same sign (grp:CD45, grp:CD16, CD3 favour training on draw 1). Simplest reading: **b01 is
  label-starved, not readout-limited.** All five flat attempts changed the readout (head, loss, ensemble member) on the same
  labelled cells. None of them increased the effective training signal per target in draw 0.
- Back-of-envelope: if the .0214 b10-b01 gap comes from about 3x labels, the slope is about .0135 on b01 per doubling of effective
  labels. Doubling the effective draw-0 signal gives about +.0135 on b01, or +.007 on the metric. If b10 also gains, the metric gets
  to about .70. So the levers should multiply effective supervision, not reshape the head.

## Check first (0-0.6 GPU-h, before building anything)

C1 (0 GPU). Regress per-target (b10 - b01) on log(n_draw1 / n_draw0) for all 11 targets. Use the numbers you already have.
  If CD45 and CD56 (the targets that reverse) also have reversed label ratios, the label-starvation story holds. If they don't,
  draw 1 is simply a harder test set (prevalence or label noise), and ideas 1-2 below matter less than idea 3.
C2 (0.5 GPU-h). Learning curve: train b10 on draw 1 subsampled per target to draw-0 counts (1 seed). If b10 drops by >= .015,
  the asymmetry is label count. If it barely moves, the gap is on the test side. Read it against a within-draw-1 CV score.
C3 (0 GPU, minutes). Physics equivariance test on the current best model. Rotate or flip a held-in crop by 90/180/270 degrees AND
  permute the illumination channels to match (see idea 2), then correlate the predictions with the unrotated ones. A correlation
  below about .97 means the symmetry has not been learned and idea 2 has room. Also record what augmentation the recipe uses
  now: none, naive rotation (channels NOT permuted, which is physically wrong for DPC and half-annulus darkfield), or correct.
C4 (0 GPU). Cross-slide prediction marginals. Within one draw, all per-antibody slides are aliquots of the SAME blood, so the
  predicted CD3 (and every other target's) distribution should be the same on every slide of that draw. Run the existing model
  on all slides of the training draw, and on the test draw for diagnosis only. Compare the predicted marginals (mean, positive
  fraction, quantiles) per slide. A spread beyond sampling error means slide nuisance (focus, illumination, density) leaks into
  the predictions. Each target is labelled on essentially ONE slide per draw, so its head has only ever seen that slide's
  nuisance. That is a cross-draw failure mode that no readout change fixes.

## Ideas

### 1. Change the training signal: dense ensemble-teacher labels on ALL training-draw cells (panel completion + distillation)
Mechanism: the training draw has about 10-20x more imaged cells than labelled cells per target, because every other slide is
unlabelled for that target. Run the existing per-direction ensemble on every cell of the TRAINING draw only (fully inductive).
The teacher is the E3c/6-way ensemble trained on that draw. Write soft labels for all 11 targets. Train a student on real labels
(weight 1) plus teacher labels (weight about 0.3-0.5) with strong augmentation (noisy-student). The unit of supervision changes
from "one slide labels one marker" to "every cell carries all 11 targets".
Why it moves the metric: it attacks b01's label starvation directly, and the student absorbs the ensemble's gain (.690 vs a .678
single model) into each member. A later ensemble of students then stacks on top of that, instead of re-buying the same variance
reduction.
Expected: b01 single student around ensemble level (+.010 vs single control), and an ensemble of 3 students +.003-.008 on b01
over the current ensemble. Metric +.002-.006 alone; more if combined with idea 2.
Falsifier: b01 only, 1 seed. Teacher inference on about 200k cells is about 0.1 GPU-h, student training 0.5 GPU-h. Kill the idea
if the student is < control single + .006 on b01.
Build: about half a day (inference dump plus a masked soft-label loss in the existing trainer).

### 2. Physics-correct dihedral symmetry: rotate/flip with LED-channel permutation, plus 8x TTA
Mechanism: cells have no preferred orientation, but the 24 LED-array channels are direction-specific (DPC L/R/T/B,
half-annulus darkfield). A 90-degree rotation of the crop is the same as rotating the LED pattern, so it is a linear map on
channels. DPC_Left goes to DPC_Top and so on. Half annuli that are missing are linear combinations under incoherent
illumination, for example Top = Full - Bottom and Left = Full - Right. Verify this from the BSCCM channel metadata and exposure
normalisation. Apply all 8 D4 elements as augmentation, and average over all 8 at test time.
Why: it gives up to 8x physically valid views of every labelled cell, which is exactly what the label-starved b01 direction
lacks. It also removes a hidden inconsistency if the recipe currently rotates without permuting channels.
Expected: metric +.004-.009 if there is no rotation augmentation now. +.002-.005 if naive rotation is used now. About 0 if it is
already correct (C3 tells you which). TTA alone on existing models is a free +.001-.003 if C3 shows they are near-equivariant.
Falsifier: TTA-only on existing members (about 0.05 GPU-h, no training), then one b01 training run with D4+permutation
(0.5 GPU-h).
Build: 2-4 h, mostly getting the channel geometry right. Include a unit test: rotating 4 times by 90 degrees must give the
identity, and Full - Bottom must match Top on real data within noise.

### 3. Change the decomposition: same-blood marginal-invariance loss across slides (slide-invariant features)
Mechanism: per target, labels come from one slide, but the blood is shared by all slides of a draw. Add a loss on unlabelled
cells of the training draw. For each target, the distribution of predictions (quantile/sliced-Wasserstein or MMD on predicted
values) must match across all slides of the training draw, including the labelled slide. Optionally do the same for the
penultimate features (CORAL across slides). The model then cannot encode slide nuisance into a target's output. Inter-draw shift
is largely slide-type nuisance, so the model is trained to be invariant to exactly what changes at test.
Why: it is a dense, label-free constraint on 10x more cells. It targets the "one slide per head" confound that readout changes
cannot see. It uses no held-out images, so it is fully inductive.
Expected: conditional on C4. If C4 shows slide-dependent marginals: +.003-.008 on the metric, largest on targets with big
direction asymmetry. If C4 is flat: about 0. Skip it then; C4 costs nothing.
Falsifier: C4 first (0 GPU), then one b01 run (0.5 GPU-h).
Build: about 1 day (slide-balanced batch sampler plus a quantile-matching loss).

### 4. Direction-specific recipe chosen by leave-one-slide-out (LOSO) inside the training draw (pre-registered)
Mechanism: b01 has 3-5x fewer labels, so its optimal capacity, regularisation, epochs and ensemble weights differ from b10's.
Choose them per direction with a pre-registered rule: hold out one training-draw slide's cells (spatially or by slide), never
the test draw. Then stack the members with per-direction, per-target weights fitted under the same rule.
Expected: +.002-.004. It mainly harvests ensemble value that equal weighting leaves on the table.
Falsifier: refit weights on existing predictions (0 GPU). One extra b01 run with stronger regularisation (0.5 GPU-h).
Build: a few hours.

### 5. (Lower priority) Physics-normalised inputs
Per-FOV flat-field (divide by empty-background intensity per channel), log/asinh on darkfield channels (small dynamic range,
carries granularity), and an optional DPC-derived phase channel. Check per-channel background and cell-interior intensity
histograms per draw first (0 GPU). Only build if the draws differ by gain or offset. Expected 0 to +.005.

## Suggested order and budget
C1, C3, C4 (0 GPU, same day) -> C2 + TTA-only (about 0.6 GPU-h) -> ideas 1+2 combined in ONE b01 screen (0.6 GPU-h) against a
same-seed control. If b01 >= +.010 single-model: run both directions with 3 students each (about 3 GPU-h), then pre-registered
LOSO stacking (idea 4). Total < 6 GPU-h. Add idea 3 only if C4 is positive.

Constraints respected: every lever uses only training-draw images and labels. Teacher labels, marginal matching and LOSO
selection never touch the held-out draw. TTA is inference-only. I deliberately avoided hand-coded lineage or exclusivity
structure (a user ruling exists against it). Test-time BN or prior adaptation on held-out images is a grey zone and is not
proposed.
