# Outsider review: building the LF->MK common-11 batch-held-out predictor from zero

Scope: I read only `notebook/GOAL.md` and `N038.../attempts.md`, plus the BSCCM background facts in the brief. Every GPU-hour figure
below is an estimate from FLOP counts (ResNet-50 at 128 px is ~1.4 GFLOP forward, ~0.3x the 224-px cost). None was measured.
Assumptions are marked **[assume]**.

## 0. What the two files imply

- Best .6527 (N031), noise .002, target .67. That needs +.017, about 8 noise units. FOV-held-out r is ~.71, so roughly .06 is lost
  when the test draw changes. Reaching .67 means recovering about 30 % of that loss.
- N037 (cleaner, cross-fitted training labels) gave +.001. Noise in the training labels is not the bottleneck.
- **The metric is free under draw-global shifts.** Per target, Pearson r is computed inside the test draw, so it does not change
  under any single affine map of the predictions. A draw-wide offset or gain, which is what most domain-adaptation fixes correct,
  costs nothing. Only effects that change the ranking inside the test draw cost r:
  - (a) per-slide offsets or gains inside the test draw. One target pools its single-stain slide with the pooled-panel slide(s).
  - (b) shifted or warped features, so the head trained on the training draw reads the wrong direction.
  - (c) features that lose information under the new imaging conditions.
  - (d) real donor or isolation biology that was never seen in training. With one training draw, (d) is mostly irreducible.
- Quantitative handle for (a): suppose a fraction f of a target's test-draw prediction variance is slide-mean nuisance that is
  uncorrelated with the labels. Then pooled r = r_clean * sqrt(1 - f). With f = 5 %, removing it gives .6527 / .975 = **.670**.
  A per-slide nuisance of only a few percent of the variance is therefore worth the whole remaining gap.
- The attempts log says all 5 attempts assumed the per-cell predictor is "near what the data allow", and nothing tested that
  assumption. The .71 vs .65 gap is evidence against it unless the gap turns out to be biology (d).

## 1. Diagnose before building (<= 1 GPU-h total, mostly CPU)

These use the held-out draw only for analysis. Nothing is trained or pretrained on it.

| id | measurement | decides |
|----|-------------|---------|
| D1 | On N031 test predictions, per target: f = var(slide means of predictions) / var(predictions); within-slide r vs pooled r | size of (a); upper bound for idea 1 |
| D2 | Freeze the N031 encoder. Fit a ridge head on test-draw labels with FOV-grouped CV. **Diagnostic only, never reported, never used for selection.** | r_probe ~.71: features transfer, head misreads (b), go to ideas 1/4. r_probe ~.65: features lose info (c), go to ideas 2/3 |
| D3 | Logistic draw classifier on (i) background-pixel channel statistics, (ii) cell-only pixels, (iii) N031 penultimate features; same for slide ID **within** one draw | is the draw signature imaging or biology? Does the training draw contain the same nuisance axes? |
| D4 | Stratify test r by per-cell nuisance proxies: focus (BF Laplacian variance, DPC sharpness), background level, cell area; compare against the training-draw range | whether idea 3 has a target |

Also report a per-target gap table (FOV r minus batch r for each of the 11 targets). Effort should go where the gap is, not to the mean.

## 2. Ideas

### Idea 1: slides are technical replicates, so learn the nuisance subspace inside the training draw and remove it (inductive; unconventional here)

- **Mechanism.** **[assume]** All slides of one draw are aliquots of the same isolated cell suspension. They differ only in
  antibody and imaging session. Then any feature direction along which slide distributions differ *inside the training draw* is
  technical nuisance (focus, illumination, exposure, any stain-induced LF artefact), not biology.
  - 1a, post-hoc RUV/TVN: from training-draw features, take the top-k between-slide directions (LDA on slide ID), project them
    out of the penultimate features, and refit the head. At test time, apply the same fixed projection. No test images are needed.
  - 1b, end-to-end: during supervised training, add a slide-ID adversary on the trunk (gradient reversal, or an MMD/HSIC penalty
    between slides), using training-draw slide IDs only. Add MixStyle, which swaps feature statistics between training slides.
- **Why it moves the number.** Draw-to-draw imaging differences come from the same physical knobs as session-to-session
  differences within a draw. Removing the directions those knobs move attacks (a) directly (the sqrt(1-f) term) and also (b). It
  also removes any stain-leak shortcut, such as fluorophore absorbance visible in the LF channels, which helps robustness at no cost.
- **Expected gain.** If D1 gives f ~ 3-5 % on the bad targets and the nuisance subspace captures most of it: +.010 to +.017 on
  the mean, which is the formula above. If slide ID within the training draw is not decodable from N031 features (AUC < .6): about 0.
- **Cheapest falsifier.** Run 1a on frozen N031 features: <0.5 GPU-h (feature extraction for ~400k crops) plus CPU ridge. Also
  measure what share of the train-to-test draw feature-mean shift lies in the training-slide nuisance subspace. **If the share is
  < 20 % and 1a moves GOAL < .004, drop the idea.**
- **Build cost.** 1a takes half a day. 1b takes about a day, plus ~1 GPU-h per direction per seed to retrain.
- **Risk.** If the slides of a draw were *not* one suspension (for example, different hold times), the between-slide subspace
  contains biology. Check the metadata first. Weight slides by inverse cell count.
- **Variant that needs a user ruling.** This one is transductive at inference only: no labels, no gradients, no pretraining.
  Normalize features or predictions per test slide using that slide's own unlabeled statistics, or give the net K random crops of
  the same slide as calibration context through a set encoder with FiLM. It works like per-plate normalization in Cell Painting.
  Report it separately and do not count it toward GOAL until ruled.

### Idea 2: feed physical quantities, not raw LED counts

- **Mechanism.** Divide each LED channel by its own cell-free background in the same FOV (flat-field). Then compute:
  - BF absorbance, -log(I/I0)
  - DPC asymmetry (I_L - I_R)/(I_L + I_R) on both axes
  - quantitative phase, by Tikhonov deconvolution with the known DPC weak-object transfer functions (from LED geometry and NA)
  - darkfield normalized by the BF background

  These quantities are in radians or optical density. They do not depend on LED power, exposure or camera gain, and the DPC ratio
  cancels LED-intensity drift to first order. Phase approximates dry mass and granularity, which separate leukocyte classes.
- **Why it moves the number.** A CNN on raw intensities is free to key on absolute brightness, background level and per-LED power,
  and all of these differ between sessions. Normalizing removes that axis before the network ever sees it. This attacks (c).
- **Expected gain.** +.005 to +.015 if D2 shows information loss and D3 places the draw signature in intensity/background
  statistics. About 0 if the draw stays equally decodable from normalized cell pixels, because then the shift is biology.
- **Cheapest falsifier.** Preprocessing runs on CPU (FFT deconvolution of 128-px crops). Then train raw-input vs physical-input
  arms supervised from scratch, so the SSL stage does not have to be redone. Use both directions and 1 seed: ~4 GPU-h
  (2 arms x 2 directions x ~1 GPU-h). Compare batch r **and** the FOV-minus-batch gap. **Drop the idea if the gap does not
  shrink.** Re-run SSL with the new inputs only if it wins.
- **Build cost.** 1-2 days. The WOTF needs the LED positions and NA.

### Idea 3: optical-forward-model domain randomization (unconventional)

- **Mechanism.** Under the weak-object approximation, every BF/DPC LED image is a linear transfer function of the cell's
  absorption mu and phase phi. The transfer function is set by the LED angle, the NA and the defocus/pupil aberration.
  1. Recover (mu, phi) for each training cell with idea 2's deconvolution.
  2. Re-render the BF/DPC channels under randomly perturbed optics: defocus +-1-2 um, LED-position jitter (array misalignment),
     NA +-5 %, per-LED gain, and Poisson-Gaussian camera noise.
  3. Darkfield is second-order and outside the WOTF, so it gets only gain/blur/noise jitter.
  4. Train on the re-rendered views, and use them as positive pairs in the SSL stage.
- **Why it moves the number.** One draw covers a narrow range of focus and alignment, so the net cannot learn invariance to them,
  and the other draw lands off that manifold. Generic per-channel photometric augmentation, which is probably what "SSL(aug)"
  does, is not physically consistent across the 23 channels. A real defocus changes BF, DPC and DF together in a specific coupled
  way that independent per-channel blur cannot reproduce. Physics-coupled augmentation covers exactly the plausible imaging
  nuisance, and it attacks (b) and (c).
- **Expected gain.** It depends on D4. If test r drops in focus bins outside the training range and the test draw's focus
  distribution is shifted, bringing those cells back to in-range r is worth +.01 to +.02. If r is flat across focus bins, about 0.
- **Cheapest falsifier (before the full build).** (i) Run D4 on CPU. (ii) Do a sensitivity probe: re-render a 5k-cell
  training-draw subset at +-1 um defocus and measure the rank correlation between perturbed and original N031 predictions
  (~0.2 GPU-h). **If rho > .98 under physically plausible perturbations, the model is already invariant: drop the idea.**
  The full test is 2 arms x 2 directions, ~4 GPU-h.
- **Build cost.** 3-5 days, including a check that in-focus re-renders reproduce the real images. It reuses idea 2's code.

### Idea 4: densify the labels inside the training draw (cross-marker noisy student; inductive)

- **Mechanism.** Each training cell is labelled for one target, so ~10/11 of the (cell, target) matrix is empty, and unstained
  slides give no supervision at all. Have the current ensemble pseudo-label all 11 targets for every training-draw cell. Train a
  student on the real labels plus down-weighted teacher labels, with heavier noise (augmentation, dropout, stochastic depth, and
  idea 3's re-rendering if it exists). Do one round.
- **Why it moves the number.** The student gets ~10x more supervised pairs per target and has to fit a coherent joint marker
  profile per cell. For example, the CD3, CD19 and CD56 lymphocyte programs are mutually exclusive. That pushes the trunk toward
  cell-type structure, which is the part that transfers across donors. Noisy-student distillation is the documented route to
  robustness under shift (the ImageNet-A/C results). It also folds the 2-member ensemble into one model, which frees ensemble
  slots for idea 5.
- **Expected gain.** +.003 to +.01 for a single student over a single N031 member. Low variance, but not enough on its own.
- **Cheapest falsifier.** One round, both directions, 1 seed: ~2-3 GPU-h. **Pass** if the single student is >= the N031
  ensemble minus noise. **Fail** if it is <= a single N031 member.
- **Build cost.** Half a day.

### Idea 5: heterogeneous ensemble under shift

- **Mechanism.** N031 combines two views of one encoder (base + 2x zoom), so their errors are highly correlated. Add members with
  different failure modes:
  - the idea 2 physical-input net
  - a **frozen generic encoder** (DINOv2/v3 ViT-L on BF / DPC-x / DPC-y and DF 3-channel composites, features concatenated, ridge
    head). It never sees BSCCM in pretraining, so it is trivially inductive.
  - the idea 4 student

  Average the predictions after z-scoring them inside the test draw, which costs nothing because r ignores affine maps.
- **Why it moves the number.** Ensemble gain scales with how decorrelated the members' errors are on the shifted draw.
- **Expected gain.** +.005 to +.01 over the best member if pairwise error correlation is <~ .8. The zoom member's contribution
  to N026/N031 suggests ensembling already pays here.
- **Cheapest falsifier.** ViT-L feature extraction for 400k crops x 2-3 composites is ~1-2 GPU-h, plus a CPU ridge. Choose
  members by error correlation on **training-draw held-out FOVs**, never on test-draw labels.
- **Build cost.** Half a day.

## 3. Order of work and budget

1. **Day 1:** D1-D4 and idea 1a on frozen features, about 1 GPU-h. Then branch:
   - D1/D3 show slide nuisance (f large, slide ID decodable): go to 1b, then 4 and 5.
   - D2 shows information loss and D3 shows an intensity signature: go to 2, then 3.
   - D2 gives r_probe ~.65 and the draw is decodable from normalized cell pixels: the shift is biology. With one training draw the
     realistic ceiling may be below .67. Ideas 4 and 5 are the remaining levers, and the target should be renegotiated, since it
     is an unconfirmed Claude proposal.
2. All falsifiers together come to ~15 GPU-h, below the 25 GPU-h approval line.
3. Ideas 1, 2 and 4 are separable and can go into one attempt.

Out of bounds or already dead:
- test-time training or pseudo-labelling on the held-out draw (violates the constraint)
- generative or virtual-stain outputs for per-cell r (N034/N035 measured per-cell worse)
- more label cleaning (N037 was flat)

## 4. What I would do differently from attempts.md

1. **Every attempt measures GOAL.** 4 of the 5 were void because they optimized a downstream Giemsa student or changed the output
   form.
2. **Test the "predictor is near ceiling" assumption (D2) instead of building around it.** The FOV-vs-batch gap argues against it.
3. **Use the experimental design as free supervision.** Slides of one draw are replicates, so the variation between them is
   labelled nuisance (idea 1).
4. **Treat the 23 LED images as physics, not as generic channels** (ideas 2 and 3).
5. **Spend effort in front of the predictor** (inputs, invariances, density of supervision, ensemble diversity) rather than on
   cleaner labels or downstream students.
