# N066: change the training evidence

The registered gap is .0138 (.6862→.70); the descriptive six-way gap is .0071 (.6929→.70). b10 already reaches .7036; b01 is .6822. Holding b10 fixed requires **+.0142 on b01**, or +.1562 summed across its eleven correlations. Priorities are b01 grp:CD45 .660 versus b10 .779, grp:CD16 .784 versus .868, and HLA-DR .445 versus .525 ([diagnosis](../../diagnosis.md), [per-slide measurements](../../n066_perslide.csv)). These deficits are not all recoverable: N064 establishes no numerical total-noise ceiling.

**Common protocol.** Gains below are speculative absolute GOAL increments against a matched fixed ensemble; zero remains plausible and gains overlap. Keep evaluation Y, masks, rows and common-11 fixed to v10. `train_lf.py --targets` supplies BOTH training and evaluation labels, so separate training arrays first. Fit maps, spectra, statistics and selection exclusively on the source draw, with source-FOV cross-fitting. Reuse N064 geometry/curation logic, not its two-draw fitted artifacts. Inference uses individual LF cells without adaptation. No held-out images enter training. Costs assume one Helma H100, r50 .5 GPU-h/direction, Cytoland 1.1. Confirm survivors with two seeds × two directions (2 GPU-h/r50 arm plus controls); freeze ensemble size/weights before GOAL scoring. Progress bar: >.6916; goal: ≥.70. No training was run here.

## 1. Background consensus becomes the training truth — CONTRADICTS the training signal

**Mechanism.** Reconstruct source-only corner-background labels through the same unmixing/cofactor/curation pipeline. Train on a source-standardised 50:50 v10/local blend for CD56, HLA-DR and the three valid pooled groups; retain v10 elsewhere. Add a third estimator, e.g. a robust background plane on non-cell pixels, only if source-FOV validation supports it. Preserve bright-tail contrast. CD19 starts at zero local weight.

**Bottleneck / gain.** Existing predictions score .6996 against consensus versus .6929 against v10: **diagnostic re-scoring, not progress**. CD56 consensus r rises .039/.033, b01 HLA-DR .023 and grp:CD45 .011. CD19 instead falls .348→.312 and .328→.286 ([alternative-label measurements](../../n066_altlabels.csv)). These locate partially independent background errors that might poison training. Recovering .02–.04 on four to six direction-target entries suggests **+.003–.008**, optimistically; evaluation error remains.

**Cheapest falsifier / build.** CPU reconstruction/QC, then one b01 r50: **.5 GPU-h incremental**, 1.0 including fresh control. Reject if targeted frozen-v10 correlations fail to improve jointly or positive-tail signal collapses. **1–2 days**. This changes training labels only and does not repeat v8 as an evaluation re-base.

## 2. Allocate gradients using measured background uncertainty

**Mechanism.** Retain v10 training targets. Estimate per-FOV, per-target uncertainty from LOWESS/corner/plane disagreement, using other cells or pixel regions where possible. Apply bounded inverse-uncertainty weights, normalised within target and source intensity strata, with a positive minimum weight. Preserve rare bright examples and report effective independent FOV count.

**Bottleneck / gain.** Residual FOV share exceeds its permutation null by .46/.43 for CD19, .29/.30 for CD56, and b01 .21 for grp:CD45 and .17 for HLA-DR ([FOV measurements](../../n066_fovresid.csv)). Many cells can repeat one background mistake. Unlike the flat FOV sampler/random offsets, these weights have a measured fluorescence-quality anchor ([declined directions](../../declined.md)). Expect **+.001–.004**: fewer misleading gradients, balanced against scarce labels on draw 0's sole pooled slide. Disagreement is uncertainty, not proven error; shared bias can evade it.

**Cheapest falsifier / build.** **0 GPU-h** CPU test: within source intensity strata, disagreement must predict source-OOF residual magnitude beyond cell count. Otherwise stop. If supported, **.5 GPU-h** b01 weighted run. **0.5–1 day** after reconstruction.

## 3. Morphology microcohorts replace individual noisy examples — CONTRADICTS the unit of supervision

**Mechanism.** Build tight source-LF neighbourhoods separately per measured assay, taking at most one labelled cell per FOV. Predict the cohort's robust measured-label mean from an anchor cell, retaining an ordinary-cell loss to preserve within-cohort biology. Cross-fit neighbourhood construction and choose cohort size on source FOVs. Labels come from actual fluorescence, not teacher predictions; deployment still predicts individual cells.

**Bottleneck / gain.** This averages different fields before fitting. Source draw 0 has 10,210 pooled cells versus 27,074 in draw 1 ([attempts](../../attempts.md)); doubling labels lifted weak-target r by .035 ([KNOWLEDGE, K033](../../../../notebook/KNOWLEDGE.md)). More efficient supervision could recover **+.002–.006**, corresponding to .01–.03 across several weak direction-target entries. No lineage gates/exclusivity are imposed. N034's large/random bags lost information (K026); small morphology-matched, cross-FOV cohorts are the specific untested distinction.

**Cheapest falsifier / build.** **0 GPU-h** with cached inductive features: compare source-OOF cohort and individual-label regressions, including rare-tail preservation. Stop if variance reduction merely attenuates biology. Then **.5 GPU-h** b01. **1–2 days**. Without cached features, budget extraction separately.

## 4. Teach background-invariant fluorescence geometry as privileged information

**Mechanism.** From source fluorescence and LF-derived cell masks, compute membrane/interior contrast, radial profiles and boundary-aligned coarse maps in actually stained channels. Spatial differences cancel constant background. Add a small discriminative auxiliary decoder during LF training, retain the v10 scalar head, then discard the decoder.

**Bottleneck / gain.** Scalar targets discard spatial evidence separating cell-bound signal from background. CD56 background-model agreement is only .666/.757 ([alternative-label measurements](../../n066_altlabels.csv)), while pixel reliability is .9999 (K041). This adds measured supervision rather than another scalar-loss variant. Expect **+.001–.005**, mainly weak-marker representation gains; invisible slide offsets remain. This develops the previous review's parked descriptor idea ([N059, E](../../../N059-review-3-lf-mk-v10-goal-6862-target-70-a/ideas.md)).

**Cheapest falsifier / build.** CPU: require reproducible marker-specific spatial contrast under source-only background perturbations. Then one descriptor auxiliary b01 run, **.5–.7 GPU-h**; reject if frozen-v10 weak-target r stays flat. **2–3 days**. Avoid full fluorescence generation, whose new-draw scalar accuracy was worse (K027).

## 5. Fit fluorescence acquisition rather than thirteen curated outputs — CONTRADICTS the decomposition and what the loss sees

**Mechanism.** Train LF-derived molecular factors through an assay-conditioned forward model of the six measured fluorescence channels. Separately measured background/shading enters ONLY the training decoder. Single-antibody slides constrain their assayed component plus autofluorescence; pooled slides constrain measured mixtures. Source unstained cells constrain autofluorescence/background, never invented marker-zero labels. Constrain spectra from source measurements, retain a v10 anchor head, and export its individual-cell LF prediction.

**Bottleneck / gain.** N062 panel factors still saw the same scarce curated labels and were flat (.6704). Raw channels/controls add measurements, including readouts discarded by scalar clipping. Shared factors might transfer single-assay information into the 10,210-cell pooled cohort while separating background before asinh. That imbalance and the .080 HLA-DR direction deficit motivate **+.002–.007**; the flat panel result is strong negative evidence ([attempts](../../attempts.md)). This cannot correct unseen draw-specific stain effects.

**Cheapest falsifier / build.** CPU source-FOV forward-model fit: reject unstable spectra or background that absorbs bright-cell signal. Then frozen-inductive-feature decoder fitting (**≤.1 GPU-h**), requiring source-OOF v10 pooled/HLA-DR improvement before end-to-end **.7–1 GPU-h** b01. **3–5 days**. N064 `pool_model`/`curate` guide implementation; derive source-only calibrations afresh.

**Order:** screen 1, then 2; pursue 3 if effective label count remains limiting, and 4/5 for richer supervision. Confirm with fresh seeds and FOV-bootstrap differences on frozen v10. b01 grp:CD45 replicate r .578 versus .881, despite similar prediction SD .84, warns that much of the deficit is evaluation error ([per-slide measurements](../../n066_perslide.csv)); none of these ideas guarantees .70.
