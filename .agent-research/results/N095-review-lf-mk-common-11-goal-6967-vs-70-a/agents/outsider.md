# N095 outsider view: how I would solve it from scratch

Inputs read: attempts.md and declined.md only. GPU-h figures are in units of M = one ensemble member trained in one direction
(substitute your measured cost; I assume M is about 2-4 H100-h).

## Framing (the arithmetic decides where to look)
- GOAL = (b10 .7066 + b01 .6869)/2. Reaching .70 needs +.0033, and clearing noise (> .7021) needs +.0054. Coming from one
  direction alone, that is +.0066 / +.0108 in that direction.
- In-distribution, Pearson r is maximised by E[y|x] up to an affine map. Masked MSE already targets that, and ensembling
  (+.002 per member, now saturated) is the standard way to estimate it better. So representation work is nearly used up.
  The remaining headroom is the **shift**: within the b01 test draw, grp:CD45 scores .881 on replicate 1 and .578 on
  replicate 0. A 0.30 r gap between two slides of the same draw is bigger than any model change has achieved, and no
  attempt touched it.
- **Step 0 (0 GPU-h; it decides which idea below to run):** compare b01 replicate 0 with replicate 1 on four things.
  (a) Shift in the prediction distribution vs shift in the label distribution. (b) Input statistics: a focus metric (high-
  frequency energy of the phase image), per-LED-channel mean and sd, background level and slope. (c) Label co-expression
  on panel cells, e.g. CD45 vs CD3/CD14/CD16 scatter. (d) Per target, pooled r vs mean within-slide r, which measures how
  much r is lost to between-slide calibration. Routing: input outlier -> idea 2; systematic label corruption (compressed,
  saturated or misregistered) -> idea 1; neither -> ideas 3 and 4.

## Idea 1: labels are measurements, so model them (contradicts the shared assumption)
- **Mechanism.** Replace equal-weight masked MSE with a hierarchical Gaussian NLL for each training slide s and target k:
  y = a_sk * f(x) + b_sk + u_FOV + eps, with eps ~ N(0, sigma_sk^2). a and b are shrunk toward 1 and 0. u_FOV is a random
  intercept with a learned tau. sigma_sk is learned. All of these are fitted on the training draw only, and at test time
  the model outputs f(x).
- **Why it beats the current approach.** Today a corrupted slide, like a training-side twin of replicate 0, pulls f with
  full weight. Stain-scale differences between slides and FOV offsets (the residual FOV clustering: CD19 +.43, CD56 +.29)
  are fitted by f itself, so f partly learns to tell slides and FOVs apart from acquisition cues, which do not transfer.
  The nuisance terms take up that variance, and sigma_sk down-weights bad slides automatically.
- **Non-obvious payoff.** b10 trains on the draw that contains replicate 0. Its labels on that replicate are only about
  (.578/.881)^2, roughly 43%, as reliable as on replicate 1, so most of the gain should appear in **b10**, not b01.
- **Expected gain.** If the corruption were zero-mean noise, MSE would already be unbiased and the gain about .001. If it is
  systematic (saturation, scale compression, misregistration), expect +.003 to .008 on GOAL. Step 0(c) tells the two apart.
- **Leakage guard.** Never set slide weights from the reverse-direction model's predictions. That model was trained on
  the test draw.
- **Cheapest falsifier.** 0 GPU-h first: on b10 OOF or validation predictions within its own training draw, check whether
  r differs by slide by much more than noise. If it is flat across slides, drop the idea. Otherwise run 1 member x 2
  directions (2M) against its MSE twin.
- **Build cost.** About 60 lines (embedding tables keyed by slide and FOV id, NLL loss, priors); half a day.

## Idea 2: physics-consistent acquisition augmentation (the gap is in the input physics, not the representation)
- **Mechanism.** The 23 LED channels are views of one complex field under known illumination angles. Use the computed
  phase and amplitude plus an angular-spectrum forward model to re-render training channels under random nuisances drawn
  at the scale Step 0(b) measures between slides:
  - defocus of +/-1-3 um;
  - per-LED gain of +/-15%;
  - LED-position or tilt jitter, which shifts illumination k-vectors;
  - background slope and camera offset.

  Cheap version: random per-channel gain, a defocus-like blur, and a background ramp. The user already ruled "inputs
  keep their artefacts (augment instead)".
- **Why it beats the current approach.** Every member learned on one draw's acquisition statistics. SSL, encoder swaps
  and seeds all share that blind spot, which explains why ensembling saturates: the members' errors under shift are
  correlated. Stain normalisation and DANN were refuted, but on Giemsa smears, not on LED-array physics.
- **Expected gain.** If replicate 0 is input-shifted, recovering half its gap on grp:CD45 and HLA-DR (.45) in b01 gives
  about +.004 to .008. If the inputs are not outliers, about 0.
- **Cheapest falsifier.** About 0.3 GPU-h, no training. Apply the measured-size perturbations to validation images from
  the training draw and re-score the current E8. If r barely moves, or replicate 0 is not an input outlier, drop the idea.
- **Build cost.** Cheap version 1 day; full forward model 2-3 days; then 1 member x 2 directions (2M).

## Idea 3: score targets independently, but train them jointly (contradicts the "independent targets" half)
- **Mechanism.** In single-stain slides each cell carries 1 of 13 labels. Use E8 to pseudo-label all 13 targets for every
  training-draw cell. Train an anchored noisy student: real labels where they exist, teacher soft labels elsewhere, and
  strong augmentation (idea 2's if built). Add a consistency term under the **learned** external co-expression prior
  (allowed 11-24), used as a fixed, frozen likelihood on the 13-vector, not as a decoder.
- **Why it beats the current approach.** It gives 13 times denser supervision per image. A noisy student beats its
  teacher mainly under distribution shift (Xie 2020: large ImageNet-A/C gains). Each head also sees the cell's full
  lineage context, so weak targets like HLA-DR, which co-varies with the monocyte, B-cell and DC lineages, borrow strength
  from strong ones. This differs from the refuted "unanchored cross-slide students" because real labels anchor it, and
  from N087's frozen-feature decoders because it fine-tunes end to end.
- **Expected gain.** +.002 to .005 from adding the student to E8; more if paired with idea 2.
- **Cheapest falsifier.** 1 student in b01 (1M). If the student alone does not beat the best single member by more than
  2x noise, stop.
- **Build cost.** 1 day: pseudo-label dump plus a loss mask change.

## Idea 4: supervise the fluorescence image, read the scalar through the label pipeline
- **Mechanism.** Add a deterministic dense head that predicts the per-cell fluorescence crop (pixel loss, alongside the
  scalar heads). The scored prediction then becomes v10(predicted image): the exact label rule applied to the predicted
  image.
- **Why it beats the current approach.** A scalar regressor has to learn implicitly how the label pipeline handles
  membrane geometry, neighbour spill-over and background. A dense head gets about 16k targets per cell instead of 1, and
  the pipeline's spatial quirks are applied explicitly rather than learned. This is not the refuted generative
  flow-matching (N034/N044/N087): it is deterministic, auxiliary, and leaves the label definition unchanged.
- **Expected gain.** +.002 to .006, mostly on low-signal markers (HLA-DR, CD123, CD56).
- **Cheapest falsifier.** 1M in b01 with the dense head as an auxiliary loss only. If the scalar r does not move, drop it
  before building the v10 readout.
- **Build cost.** 1-2 days, since a UNet-style decoder on the existing encoder is enough.

## Idea 5: add donors, not members (short)
The real n is one training draw. External data are allowed (11-23), so add supervised marker labels from other donors as
multi-task auxiliary data. Candidates: BSCCM-coherent, if it comes from a different draw (0 GPU-h provenance check
first), and imaging-flow-cytometry WBC sets with CD labels (brightfield and darkfield). The member architecture stays the
same. Expected gain +0 to .01; this is high variance. Falsifier: 1 member x 2 directions (2M). Build: data wrangling, 2-4
days.

## Order
1. Step 0 (0 GPU-h).
2. Idea 1's per-slide r check (0 GPU-h).
3. Idea 2's stress test (about 0.3 GPU-h).
4. Run whichever of ideas 1 and 2 survives in a bundled attempt, plus idea 3 (about 3-5M in total, well under 25 GPU-h).
5. Ideas 4 and 5 only if those come back flat.

If Step 0 shows replicate 0 is label-limited and not input-shifted, b01's .687 is close to its label ceiling. The .70 must
then come from b10 (idea 1) and from cross-target sharing (idea 3), not from b01.
