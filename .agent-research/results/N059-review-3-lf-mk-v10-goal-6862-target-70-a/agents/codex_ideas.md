# N059: five experiments against the remaining bottleneck

Keep `targets_v5cx_v10.npz`, common-11 masks, GOAL cells and scoring fixed. Baseline .6862; required gain .0138; noise SD .0027. N055's .6902 misses its .6916 evidence bar, so further ensemble growth is a weak bet ([attempts](../../attempts.md), [declined](../../declined.md)).

[Diagnosis](../../diagnosis.md) identifies readout shift, not missing image resolution: HLA-DR plus pooled CD45/CD16 account for about +.016 of potential recovery. Summing their exact in-distribution deficits in [the CSV](../../n059_diag.csv) gives +.0173; this is a comparator, not an attainable ceiling, since that model saw both draws and used v5 training labels. CD19/CD56 have essentially no in-distribution advantage. Within-FOV scoring reaches .7261, but that diagnostic uses test labels and cannot be deployed.

All estimates below are speculative **absolute GOAL gains over .6862**, after integration into the existing predictor; they are not additive. Every fit, threshold, teacher and hyperparameter choice uses only the source draw. Hold source FOVs aside for screening, then refit on permitted source rows. Held-out-draw images enter only final inference; its labels enter only fixed scoring. No test-draw adaptation. Costs assume one Helma H100, ~.5 GPU-h per r50 direction; build costs are person-days.

1. **Make the loss compare cells within a FOV — contradicts independent cellwise MSE.**

   Mechanism: mix ordinary masked MSE with same-target, same-FOV pair-difference loss, `[(p_i-p_j)-(y_i-y_j)]²`. Shared additive FOV offsets cancel. Sample FOVs uniformly; retain absolute loss to preserve between-FOV biology and bright-tail amplitudes. Inference remains one LF cell to one fixed-v10 prediction, without a context head or test-FOV correction.

   Why/gain: CD19 pooled→within-FOV r is .342→.543 / .328→.480; CD56 .484→.574 / .564→.646 [CSV]. Those entries contribute .0239 of diagnostic headroom. Recovering 15–30% through less noisy training suggests **+.004–.007**. Held-out label error remains irreducible; centering cannot deliver the .726 oracle.

   Cheapest falsifier: source-FOV blocked frozen-feature heads, **≤.05 GPU-h**; kill if improved within-FOV r comes with worse ordinary pooled r. Then one fixed mixture both directions, **1 GPU-h**. Build **.5–1 day**. This changes what the loss sees rather than repeating N037's flat FOV-context head.

2. **Share molecular factors across single and pooled assays — contradicts independent output coordinates.**

   Mechanism: replace CD45/CD16 single/pooled heads with two shared LF-derived molecular factors and small assay-specific nonlinear decoders. Fit observed source labels, constrain pooled residual capacity, and preserve other outputs. Single and pooled z scores are different coordinates: do not force numerical equality. No lineage/exclusivity priors or synthetic positive labels.

   Why/gain: single CD16 transfers at .907/.890, pooled CD16 only .853/.787; pooled CD45 is .753/.654 versus in-distribution .850/.708 [CSV]. Sharing reliable single-marker supervision could stop pooled heads relearning panel shortcuts. Their shift budget is .0089; recovering one-third to three-quarters suggests **+.003–.007**. Source labels must identify useful shared factors; unseen assay mixtures remain a risk.

   Cheapest falsifier: source-FOV validation while subsampling pooled labels, testing whether adding real single-assay supervision improves pooled predictions; frozen-feature fit **≤.1 GPU-h**. Kill on pooled-accuracy loss or no transfer. Fixed end-to-end pair **1 GPU-h**. Build **1–2 days**.

3. **Stop contaminated autofluorescence supervision from steering marker features — contradicts treating every measured target as useful.**

   Mechanism: keep v10 marker labels unchanged, but remove co-stained-slide autofluorescence loss from shared-encoder training. First remove that auxiliary loss entirely; if useful, restore source-unstained-only supervision with a separate adapter. This changes training supervision, not the benchmark labels.

   Why/gain: `train_lf.py` averages regression across all K outputs, including autofl. `make_targets_v6.py` documents co-stain contamination; K029 reports a draw-dependent autofl slope sign flip. A conflicting auxiliary task could bias HLA-DR/panel features. Recovering 15–30% of their .0173 deficit suggests **+.002–.005**, with zero gain plausible if features already separate it.

   Cheapest falsifier: matched b0→b1 r50 run with autofl weight zero, **.5 GPU-h**; kill if the HLA-DR/panel sum fails to improve without harming common-11. Reverse direction adds **.5 GPU-h**. Build **.25–.5 day**. The v6 GS work did not establish this GOAL ablation.

4. **Predict observed intensity bins and conditional amplitudes — contradicts a single linear continuous readout.**

   Mechanism: for HLA-DR/CD19/CD56, learn source-defined intensity-bin probabilities and within-bin conditional means; output their probability-weighted mean in original v10 units. Use observed labels, fixed quantile bins and a final MSE term. Do not infer latent positivity, labelling efficiency or FP/FN rates. The output remains continuous, not sample percent-positive.

   Why/gain: HLA-DR Spearman exceeds Pearson (.627 versus .526; .510 versus .440), suggesting nonlinear amplitude errors. CD19 ranks poorly (.077/.136); aggregate Pearson .686 versus Spearman .531 shows bright-tail dependence [CSV]. Explicit bulk/tail ambiguity could help sparse-label readouts. A .02–.04 rise on these six entries gives +.005–.011; discount uncertainty to **+.002–.006** expected.

   Cheapest falsifier: frozen features, source-FOV blocked comparison against an equally sized MLP head, **≤.1 GPU-h**; kill unless pooled Pearson improves beyond simple nonlinearity. Fixed end-to-end pair **1 GPU-h**. Build **.5–1 day**. Unlike N022 tail weighting, observations retain their frequency; unlike N049, no unidentified prevalence/error mixture is fitted.

5. **Use fluorescence spatial structure as privileged supervision — contradicts reducing each measured cell to one scalar.**

   Mechanism: from source LF–fluorescence pairs already in public BSCCM, extract deterministic membrane-ring/core contrast, within-cell intensity quantiles, and cell-versus-border signal. Train LF auxiliary heads for these alongside unchanged v10 targets; discard them at inference. Start with actually stained CD19/CD56/HLA-DR cells. No off-panel pseudo-labels, fluorescence generation, or shared-space alignment.

   Why/gain: fluorescence geometry might distinguish membrane signal from K028's smoothed-background error, providing richer supervision per scarce labelled cell. K033's 50→100% label curve raises weak-target r by .035; seek information per label rather than resolution (N042). A .01–.03 rise on six entries gives +.003–.008; use **+.002–.006** as a cautious forecast. Descriptors may prove no more LF-predictable than current labels.

   Cheapest falsifier: source-only, blocked-FOV descriptor probes on frozen LF features, **≤.1 GPU-h** after CPU extraction. Kill if cell-local descriptors contain no reproducible LF signal beyond scalar predictions, or only predict FOV background. One auxiliary recipe both directions **~1–1.5 GPU-h**. Build **2–3 days**. N039's failed background-corrected labels stay rejected; descriptors never replace scored labels.

Run 3 first for cost, then 1/2; advance 4/5 after their probes. Confirm survivors across two seed pairs with paired FOV-bootstrap differences; +.0054 is an evidence bar, not the .70 finish line. Preserve legacy-v5 reporting. Source-only learning cannot reconstruct arbitrary unseen antibody-lot readouts; these may all fail. Do not run the diagnosis's held-out-image-training falsifier: it violates the inductive constraint.

Evidence: [KNOWLEDGE K028–K034, K039](../../../../notebook/KNOWLEDGE.md); [harness](../../../../../scripts/train_lf.py); [encoder](../../../../../bsccm/encoders.py); [target construction v3](../../../../../scripts/make_targets_v3.py), [v5](../../../../../scripts/make_targets_v5.py), [v6](../../../../../scripts/make_targets_v6.py).
