# N079 outsider memo (read only GOAL.md + attempts.md)

## What I see from outside
- Best .6862 is reported as equal to the in-distribution reference. If that holds, the gap between draws is closed and the
  cap is (a) how much information the encoder pulls out of the 24 LED channels and (b) label reliability. More rounds of
  label re-processing (N067/N073/N075) cannot move either one. The 5 flat attempts fit this picture.
- Bar .6916 = best + 2 sd. Reaching .70 needs about +.014 over the best and +.03 over the single-model control (.6706). No
  single idea below is likely to do that alone. The plausible route is a clearly stronger and *different* single model
  plus an ensemble that gains from that difference ("ensembles saturate" means the members are too similar).
- The user's top priority (decoding all markers of a cell jointly) is a different objective from the metric. Per-marker MSE
  heads give a vector of marginal means. For an ambiguous T/B/NK cell that vector reads like "half CD3, half CD19", which
  is not a cell. Joint decoding needs an output that is a distribution over cell states. It also needs ground truth with
  all markers on the same cell, and BSCCM never has that. The external dataset is the only place it can be checked.

## Idea 1: a stronger and different encoder (metric lever; most likely to move GOAL)
- **Mechanism.** Fully fine-tune a large pretrained ViT (DINOv2/v3 ViT-B, natural-image pretraining). Label-free BF/DF
  images look like grayscale natural images, so this pretraining should carry over. Input: upsample 128 to 224 and feed a
  physical 3-channel rendering (Brightfield, DPC-LR-minus-RL phase proxy, sum of the high-angle DF LEDs) through the
  *unchanged* pretrained patch embedding. Add a zero-initialised 24-channel conv adapter in parallel, so the full LED
  information can enter without destroying the pretrained stem. Run a cheap CNN control next to it: ResNet-50 with the
  stem's max-pool removed and stride 1 (a 4x larger final map at 128 px; ResNet-50 at 128 px ends at 4x4, which probably
  blurs small lymphocyte texture).
- **Why.** None of the 5 attempts changed the representation. N078 (in-domain SSL, ConvNeXt-S) is pending and is the
  in-domain version of this test. This idea is the out-of-domain, large-pretraining version, and its errors should be less
  correlated with the ResNet ensemble.
- **Expected gain.** Single model +.005 to +.015 over .6706. Adding it to the existing ensemble: +.004 to +.008 over .6862.
  Only this combination has a realistic shot at the .6916 bar.
- **Cheapest falsifier.** b1to0 only, 1 seed. ViT-B costs about 2 GPU-h and the stem-modified ResNet-50 about 0.7 GPU-h.
  Kill the ViT if single r < control + .004 **and** if adding it to the ensemble gives < .6890.
- **Build cost.** About half a day: timm/DINOv2 weights, the rendering function, the adapter.

## Idea 2: a latent cell-state model as the unit of prediction (decomposition and training signal; serves joint decoding)
- **Mechanism.** Predict a posterior over about 12 latent states (neutrophil, eosinophil, basophil, classical and
  non-classical monocyte, B, CD4 T, CD8 T, NK, NKT, pDC/DC, other), plus a small per-state residual, instead of 11
  independent regressions. Each marker has a per-state, per-draw-calibrated emission p(y_m | t) (Gaussian or
  two-component, in v10 space).
  - Loss for a single-antibody cell: -log sum_t q(t|x) p(y_obs | t, r(x)).
  - Loss for an 'all'-slide cell: the likelihood of each pooled channel, with pooled mean = sum of the component markers'
    means. Do this in raw intensity space, because asinh does not add linearly.
  - So every label, whichever marker it is, trains the same state posterior. A CD3-low cell is evidence for "not T", and
    that evidence feeds the CD4, CD8 and CD56 predictions as well.
  - Metric output = posterior mean (identical in form to today's output, so it is directly comparable).
  - Joint output = samples, or MAP state plus residuals. These are coherent vectors, with exclusivity of CD3/CD19/CD14 and
    CD4/CD8 only inside CD3+.
- **Use of the external dataset.** Its 9-marker joint data (7 markers overlap BSCCM: CD3, CD4, CD8, CD14, CD19, CD45,
  CD56) define the states and initialise the emission means, covariances and state frequencies. BSCCM-only markers
  (CD16, CD123, HLA-DR) get their emissions from BSCCM single-slide mixture fits. CD15 and Siglec8 separate neutrophils
  from eosinophils, which BSCCM has no marker for.
- **Why.** This is a different training signal (pooled likelihood and a shared discrete latent) and a different unit of
  prediction (the cell state, not the marker). It is also the natural form for the user's priority.
- **Expected gain.** Metric about +.000 to +.008. I would not sell it on the metric. Joint coherence should improve a lot:
  fewer impossible marker vectors, better MM3 and better reproduction of pairwise marker distributions on 'all' slides.
- **Cheapest falsifier.** First, 0.3 GPU-h in the sandbox (Idea 3a). Then one BSCCM b1to0 run, about 0.6 GPU-h. Keep it if
  the metric is within 1 sd of control and the joint metrics improve. Kill it if the metric falls by more than 2 sd.
- **Build cost.** 1 to 1.5 days: the mixture head, the pooled likelihood, and initialising emissions from external gates.

## Idea 3: the external imaging-flow dataset, used concretely (new measurements)
**3a. Joint-decoding sandbox (do this first; it answers the user's question with ground truth).**
- Train on the external images while *simulating BSCCM's regime*. Each training cell reveals only 1 random marker out of
  9, and about 10% of cells reveal only pooled sums that mimic the 'all' slides: (CD3+CD19+CD56), plus CD45 alone.
- Compare three models:
  - (i) per-marker heads (today's design);
  - (ii) Idea 2's latent-state model;
  - (iii) an oracle trained with all 9 markers per cell, which gives the ceiling and shows how much one-marker-per-cell
    training costs.
- Optional (iv): a masked-marker transformer that predicts p(y_S | x, y_T). It also supports "measure 1 to 2 markers,
  impute the rest" at inference.
- Evaluate on a held-out donor (or a held-out acquisition file if there are no donor IDs) with full joint truth:
  - per-marker r;
  - per-cell r across the whole vector;
  - state accuracy;
  - CD4+CD8+ and CD3+CD19+ double-positive rates against the truth;
  - 2-D Wasserstein distance for the CD4 vs CD8 and CD3 vs CD56 pairs.
- **Cost.** About 0.15 GPU-h per ResNet-18 run (about 98k cells at roughly 64 px), about 1 GPU-h for the whole grid.
- **Build cost.** About 1 day, mostly data loading and gating.
- **Falsifier.** If (ii) does not beat (i) on vector r or on impossible-vector rate, drop Idea 2 before spending any BSCCM
  compute on it.

**3b. Cross-modal supervised pretraining (new labelled cells for data-poor markers).**
- External gives about 98k labels *per marker*. One BSCCM draw gives roughly 20k per marker.
- Render BSCCM into the shared modality: the 'Brightfield' channel maps to ImageStream BF (typically Ch01/Ch09), and the
  sum of the highest-angle DF LEDs maps to SSC (typically Ch06). Verify the channel indices. Resample both to the same
  um/px: ISX 60x is 0.33 um/px; take BSCCM's value from its metadata. Centre and pad to the same field.
- Train a 2-channel ResNet-50 on both datasets with:
  - per-dataset BatchNorm (domain-specific BN);
  - shared heads for the 7 overlapping markers, each with a per-dataset affine calibration (Pearson ignores affine
    differences, and fluorophores and cytometers differ);
  - CD15/Siglec8 heads trained on external data only, CD16/CD123/HLA-DR heads on BSCCM only.
- Then inflate the first conv layer to 24 channels: copy the BF filters to the BF/DPC LEDs and the DF filters to the DF
  LEDs, scaled by 1/n. Fine-tune as usual.
- **Cheapest falsifier (0.3 GPU-h).** Train on external data only and predict the BSCCM held-out draw zero-shot from the
  2-channel rendering. Also train a BSCCM-only 2-channel model (0.5 GPU-h) as the reference. If zero-shot r on
  CD3/CD19/CD14/CD45 is below about 0.2, or below a third of the BSCCM 2-channel reference, the domain gap is too large.
  The likely culprits are suspension vs slide, 60x vs BSCCM NA, and differing fixation. In that case keep 3a and drop 3b.
- **Expected gain if the transfer works.** +.003 to +.010, concentrated on CD56/CD4/CD8, which have the most headroom
  (N067 shows CD56 at about .50). Joint pretraining also gives a diverse ensemble member.
- **Build cost.** 1 to 2 days: rendering, resampling and the two-domain loader.

## Idea 4: measure the ceiling before spending more GPU (0 GPU-h, about 2 h CPU)
- Estimate per-marker label reliability from the fluorescence crops. Split-half the pixels (an odd/even checkerboard sum
  gives two independent shot-noise draws), apply Spearman-Brown, and take r_max = sqrt(reliability).
- This is a lower bound on the noise, because it does not capture staining variability.
- Report mean r / r_max per marker. If the gap to the ceiling is concentrated in 2 to 3 markers, point Ideas 1 and 3b at
  them, for example with loss weights or extra capacity. If the mean ceiling is below about .75, tell the user that .70 is
  near the physical limit and that the effort should go to joint decoding (Ideas 2 and 3a).

## Order I would run
1. Idea 4: 0 GPU-h.
2. Idea 1 (2.7 GPU-h) and 3a (1 GPU-h) in parallel.
3. The 3b zero-shot gate (0.8 GPU-h).
4. Idea 2 on BSCCM only if 3a supports it.

The whole screen is about 5 GPU-h. The 3-seed confirmation comes only after a single-seed result clears .6916 as part of
an ensemble.
