# N047 review 2: proposals after the frozen-v5 plateau

The best next experiment is a **source-only audit of the readouts already learned**, followed, if justified, by **training without demonstrably contradictory source supervision**. Neither should be sold as a likely route to .67. The evidence supports a plateau for this predictor family against these labels; it does not establish an information-theoretic ceiling for LF.

This report proposes **six new experiments** and **three alternative GOAL definitions**. Ideas 3–5 explicitly contradict the current practice of fitting absolute cell labels and adding supervision to essentially every model. All gains are hypotheses; zero or negative gain remains plausible for every proposal. No jobs, GPU computation, CPU training, repository edits, or data edits were performed. Only this requested report is written in the designated output directory; analysis scratch is under `/tmp/codex_n047/`.

## Evidence, scope, and novelty

The frozen score is the equal average of the 22 target-by-direction Pearson correlations: common-11, `targets_v5.npz`, original masks and opposite-draw split-2 cells. N031 is .6527; the ten-model aggregate is .6571; .67 requires **+.0173 over N031 or +.0129 over the ten-model aggregate**. The estimated seed SD is .0027; a +.0054 change is the stipulated flat-zone boundary, not a confidence interval for unseen isolations. The two available isolation directions do not estimate population-level donor uncertainty.

N047's within-FOV scores are .7205 and .7259 for N031 and the ten-model aggregate. N039's v5-trained reference is .678 with both draws available but .659 at matched 128k training count. Those are reference performances, not proven ceilings. Its large asymmetric transfer deficit includes different assay definitions. N042 argues against missing high-frequency detail as the next lever. N040's ensemble improvement does not compound in N046. These facts substantially lower the prior probability of another architecture, augmentation, or pseudo-label iteration succeeding.

I read GOAL.md; KNOWLEDGE.md, particularly K001/K019/K023/K028; N047 attempts, binding declined directions, and diagnosis; the N038 synthesis and underlying Codex/outsider/assumption-breaker proposals; and `scripts/train_lf.py`, `scripts/rescore_targets.py`, and `bsccm/gpu_norm.py`. The ideas below do **not** reopen v8/v8b, input cleaning, DANN, AdaBN, private weak-marker encoders, reconstructive SSL, physics augmentation, noisy-student rounds, photon auxiliary supervision, correlation/mixture losses, or shared-dye additive heads. In particular, the repeated-antigen proposal below concerns CD45/CD16 across assays, not the prior CD3+CD19+CD56 sum proposal.

**Selection detail that matters:** the batch-held-out branches in `train_lf.py` train on all eligible source-draw splits. Existing supervised checkpoints have therefore already seen any apparent source split-2 validation labels. New readout tuning requires genuinely held-out source FOV predictions, obtained by refitting with those FOV labels excluded. Reusing in-sample source predictions would make these cheap tests misleading. Existing source-only SSL initialisations may be reused; no held-out-draw SSL is permitted.

### One new, read-only diagnostic

I aligned global row IDs across all ten cached models and compared their two outputs for each repeated antigen on the **same evaluated cells and same v5 target mask**. No weights, transformations, models, or selectors were fitted. The calculation took less than one second of execution in a uv environment at `/tmp/codex_n047/ideas_venv/`; script and CSV are `/tmp/codex_n047/ideas_readout_audit.py` and `ideas_readout_audit.csv`.

| Ten-model predictions; evaluated target | Direction | Own output r | Other assay's output r | Fixed raw half-average r |
|---|---|---:|---:|---:|
| grp:CD45 | b10 | .7567 | .7967 | .7885 |
| grp:CD45 | b01 | .6574 | .6669 | .6665 |
| grp:CD16 | b10 | .8280 | .8459 | .8523 |
| grp:CD16 | b01 | .8077 | .7781 | .7983 |
| single:CD16 | b10 | .9042 | .8304 | .8891 |

Replacing only both pooled-CD45 outputs by their fixed half-averages would contribute roughly `(.03183 + .00910)/22 = .00186`. That is **post-hoc arithmetic, within the flat zone**, not a validated new GOAL result. The opposite effects for CD16 rule out indiscriminate head averaging. Even after averaging ten models, an output named for an assay is not always its best readout.

## Six experiments on the unchanged GOAL

Costs below are planning estimates in **summed H100 GPU-hours**, including both directions unless stated. They are not benchmarks. Build cost is experienced implementer time. A short screen can reject its tested implementation; it cannot prove a whole mechanism impossible.

For ideas 1 and 2, first create one shared source-only out-of-fold cache: two source-FOV folds, both directions, base model, approximately **1–2 GPU-h**, plus up to .2 for inference. Reserve **3 GPU-h** including IO and control refits. This cost is shared, not charged again for each readout. A two-scale version roughly doubles the fitting work and is unnecessary until a screen survives.

| Idea | Hypothesised incremental gain over .6571 | Cheapest decisive screen after shared cache, where needed | Build |
|---|---:|---:|---:|
| 1. Source-only nonlinear scalar readout | 0–.003 | .02–.08 GPU-h | .5–1 day |
| 2. Repeated-antigen consensus | 0–.003; .005 only with broader correction | .03–.10 GPU-h | 1 day |
| 3. Within-FOV contrast supervision | 0–.006 | .8–1.6 GPU-h | 1–2 days |
| 4. Censor contradictory source supervision | 0–.006 | 0 GPU-h audit, then .8–1.6 | 1–2 days |
| 5. Bag source FOVs instead of recipes | 0–.004 | 1.6–3.2 GPU-h | 1–2 days |
| 6. Translation integration at inference | 0–.002 | .1–.3 GPU-h | .5 day |

Ranges describe plausible useful outcomes rather than positive expected values. Add .0044 to compare a candidate built on the ten-model reference with N031, but that inherited difference is not evidence for the new mechanism. Do not add gains across ideas. None individually has evidence for supplying the missing .0129.

### 1. Correct nonlinear readout bias without another encoder

**Mechanism.** Fit a strongly regularised monotone piecewise-linear map `g_j(p_j)` from source out-of-fold predictions to each observed v5 target. Use very few source-quantile knots, shrink toward the identity, and bound tail slopes. Compare against the identity on independent source FOVs, then freeze the map for the opposite draw. Begin with the weak targets; retain an identity option for every target. The encoder and linear marker heads remain fixed.

**Why this bottleneck.** A linear head may have systematic tail compression after sparse MSE training and ensembling. Ten models can share that nonlinear bias. Pearson is unchanged by a positive affine transform, but can improve under an appropriate nonlinear conditional-mean correction even when cell order is unchanged. This tests information already present in the output, rather than another detail-preserving architecture or another correlation loss. It cannot predict independent FOV offsets or an unseen draw's assay map.

**Expected gain.** A .01–.02 correction on two to four target-direction entries supplies only .0009–.0036 in the 22-entry mean. Budget 0–.003. Transfer of the source response curve is the weak assumption, especially for HLA-DR; rare-tail distortion could make the result worse.

**Cheapest falsifier.** After the shared cache, .02–.08 H100-h for tiny GPU readout fits and scoring. Reject if independently held-out source FOVs show no repeatable curvature benefit, or if all improvement comes from a handful of extreme cells and disappears when the source FOV split changes. Freeze knots, regularisation, and identity choices before any new opposite-draw scoring. Bootstrap FOVs, and report the top-1%-removed diagnostic alongside the unchanged primary score.

**Build and constraints.** .5–1 day. No target-draw moments, ranks, labels, calibration, or parameter updates. No winsorising evaluation labels. This is prediction calibration, not a v5 re-base. It does not duplicate N038's proposed training-loss/head replacement.

### 2. Make two readouts of the same antigen inform one another

**Mechanism.** For CD45 and CD16, use the pair of already-predicted single/pool outputs to construct assay-specific readouts. The first screen is a source-selected convex combination, with identity as the default; a later version can use source-only monotone maps before combination. Fit each destination assay using only its own observed source labels. At inference both inputs are LF model outputs, never measured fluorescence. Do not enforce equality of the two v5 labels: they were processed and measured differently.

**Why this bottleneck.** The diagnostic above isolates a residual readout mismatch on relatively reliable markers. It offers a route around weak-marker label noise without trying to reproduce FOV background errors. The single-antigen head can encode transferable antigen biology while its pooled counterpart overfits assay-specific source relationships. This is useful even when both predictions are highly correlated: the ten-model CD45 output correlation on b10 pooled cells is .9484, yet their label correlations differ by .0400.

**Expected gain.** The two pooled-CD45 half-averages give a post-hoc scale of .00186. The b10 pooled-CD16 half-average adds another .00110, but its b01 counterpart loses .00042. Expect 0–.003 after legitimate source selection; .005 would require a stronger cross-assay correction than the present evidence supports. There is no route to .67 from this arithmetic alone.

**Cheapest falsifier.** .03–.10 H100-h after the shared source cache. Ask whether source-held-out FOVs consistently choose useful departures from the identity and whether the choices survive source slide/replicate blocking where both assays have support. If source validation selects mappings that fail transfer, stop; do not pick different weights using the table above. Compare with scalar calibration from idea 1 to establish added information from the second output.

**Build and constraints.** About 1 day. Preserve the common-11 outputs and original masks. Missing labels remain missing. This needs no same-cell single/pool assay pairs. The held-out diagnostic has already exposed these benchmark directions, so a result remains development evidence requiring fresh training seeds, not an untouched confirmatory test.

### 3. Fit within-FOV cell contrasts rather than absolute noisy levels

**Contradiction of current practice:** the source label's absolute value need not be the right unit of supervision even though it remains the frozen unit of evaluation.

**Mechanism.** For a target with diagnosed FOV offsets, train on differences of cells from the same source FOV: `[(p_i - p_k) - (y_i - y_k)]²`. Implement this by within-FOV centring inside the loss, with FOV-balanced sampling and a small ordinary-MSE anchor. Retain ordinary supervision for clean targets. Use original v5 values; create no new label file. Select one contrast/anchor ratio on source FOVs and freeze it.

**Why this bottleneck.** Under `y_if = z_if + a_f + noise`, pair differences cancel the source offset `a_f` exactly. This changes gradient variance and the effective supervision of CD19/CD56/CD3, without requiring N039's failed re-unmixing. N037 fitted cleaner absolute labels; this tests an objective that analytically removes intercepts during learning. Differences do not remove non-additive assay errors, and subtraction on the transformed v5 scale only cancels offsets additive on that scale. The existing predictions' near-null FOV variance makes a large benefit unlikely: they may already ignore the offending signal.

**Expected gain.** If within-FOV r improves .02–.04 for three targets in both directions, attenuation by label nuisance could leave about .003–.008 in the pooled mean; plan 0–.006 after ensemble redundancy. A within-FOV-only gain is not GOAL progress and cannot close the label-noise cap.

**Cheapest falsifier.** .8–1.6 H100-h for paired ordinary/contrast source-trained screens in both directions, using the same sampler and eligible inductive init. Reuse a compatible matched control when available. Reject if source-held-out within-FOV r barely changes, or if the frozen candidate adds less than .002 to the ten-model ensemble without a clear reduction in error on the affected targets. For the stronger bottleneck challenge, require >.0054 pooled improvement; compare its within-FOV improvement with the pooled improvement as specified in N047 diagnosis.

**Build and constraints.** 1–2 days. FOV membership groups the training loss only; coordinates, slide/FOV IDs, and assay metadata never enter the predictor. Input artefacts remain. Pair counts are not independent sample counts: resample FOV blocks. Keep an anchor because genuine FOV composition differences should not automatically be discarded.

### 4. Remove contradictory supervision even if it reduces label count

**Contradiction of current practice:** the next useful labels may be fewer, better-defined source observations, rather than denser pseudo-supervision.

**Mechanism.** Audit source-only target-by-assay-by-broad-morphology strata for a physical contradiction: a v5 target moving against its antibody-bearing dye while following autofluorescence, as previously observed for HLA-DR/CD19/CD14 in one draw. Require support across multiple source FOVs and a negative-control comparison. Mask the affected target loss in the implicated source stratum; keep those cells for other observed targets and source-only SSL. Fit no new unmixing and do not replace their values with v8 or pseudo-labels. Start with HLA-DR because its assay semantics remain a diagnosed deficit.

**Why this bottleneck.** A wrong observation map can teach a stable but non-transferable biological association. More pseudo-labels can propagate it; source MSE can reward it. Removing an identified wrong-sign source mapping tests a mechanism different from FOV detrending or new image detail. It will not fix a wrong mapping that exists only in the held-out draw, which likely accounts for much of the bad b01 direction. It may therefore improve the easier direction only, or do nothing.

**Expected gain.** Correcting .04–.10 on one HLA-DR target-direction entry is .0018–.0045 GOAL; an additional affected entry might bring the effect toward .006. Plan 0–.006, with a substantial chance of a loss because the excluded stratum contains real biological information. Do not extrapolate the entire N039 draw gap as recoverable.

**Cheapest falsifier.** First, a source-label/photometry audit: 0 GPU-h, minutes of array summaries; no model fitting. If the required contradiction is absent or cannot be separated from spectral cross-talk, stop. A paired original-mask/censored-mask screen costs .8–1.6 H100-h. Compare with a random mask matched for assay, FOV, morphology, and label count. Reject if censoring gains no more than random deletion, loses rare-positive recall, or adds nothing to the ten-model reference.

**Build and constraints.** 1–2 days. Fluorescence is used exclusively to audit source supervision, never as LF input. The selection rule must be fixed before inspecting opposite-draw photometry. Do not select cells by agreement with the current predictor, which would erase hard biology. This changes a training loss mask, not frozen evaluation labels or eligibility; it neither revives v8b nor needs a GOAL re-base. Record every removed source target entry.

### 5. Diversify training sets at the FOV level, not just model recipes

**Contradiction of current practice:** each member need not consume all available source labels. More correlated supervised fits can be less useful than fits whose source measurement errors differ.

**Mechanism.** Partition source FOVs into assay-stratified blocks. Fit complementary leave-block-out models, preserving rare-marker support and comparable optimisation exposure; average them with weights fixed using source out-of-fold predictions. First try two complementary omissions, then at most four if evidence warrants it. Resample whole FOVs, not individual cells, because the diagnosed nuisance is shared within FOVs.

**Why this bottleneck.** N047's ten models mostly inherit the same source supervision. Recipe variation need not diversify the errors induced by a few influential source FOVs or rare cells. FOV bagging asks whether correlated training-label influence is behind the ensemble plateau. It does not make independent test-label noise predictable. N040's positive label-count curve and the weak FOV dependence already present in predictions are evidence against a large benefit; this should be gated cheaply.

**Expected gain.** If a genuinely removable prediction-error variance component is 1%, `r / sqrt(.99) - r` at r=.657 is about .0033. That arithmetic assumes signal covariance is retained; it is not a measured noise fraction. Plan 0–.004. Loss of positives or signal can overwhelm any variance reduction.

**Cheapest falsifier.** Two omitted-block fits per direction plus an exposure-matched full-source control: approximately 1.6–3.2 H100-h, depending on shared controls and subset size. Reject if source-FOV validation shows that the omissions increase bias or give no additional residual diversity compared with equal-compute full-data seeds. Only then score the source-selected bagged blend on frozen opposite-draw cells. Report target tails separately so diversity from missing rare populations is not mistaken for useful diversity.

**Build and constraints.** 1–2 days for a provenance-preserving row sampler and ensemble manifests. All SSL and supervised rows remain source-only and non-Giemsa. No opposite-draw tuning of blocks or weights. Keep all test cells; there is no evaluation subsampling. This is a training-data intervention, not the previously proposed heterogeneous-encoder ensemble.

### 6. Integrate over cell-centering uncertainty at inference

**Mechanism.** Average each frozen model's predictions over a small fixed set of translations within the existing training augmentation support, for example centre plus four axial shifts of two pixels. Translate the already-normalised LF/phase tensor together, exactly as training does, then apply the model's usual full-view or zoom path. Keep BN in evaluation mode. Select the translation grid on honest source-held-out predictions.

**Why this bottleneck.** Training translation augmentation does not mathematically enforce translation invariance, especially before a fixed central zoom crop. A small number of cells with unusual centring can dominate weak-marker Pearson covariance. Integrating predictions attacks that finite predictor bias without finding new optical detail; it remains compatible with N042's low-pass result. It cannot alter assay semantics or infer missing FOV offsets.

**Expected gain.** Plan 0–.002 overall. For example, a .02 improvement confined to CD123 in one direction contributes .00091. It is an inexpensive check, not a plausible standalone solution to .67.

**Cheapest falsifier.** .1–.3 H100-h for a fixed translation grid on cached-checkpoint source validation and both evaluation directions, subject to IO. Reject if prediction variance under translations is negligible, source-held-out r does not improve, or apparent gains reverse after removing the most influential 1% for the diagnostic. Do not increase the grid after opposite-draw results are known.

**Build and constraints.** .5 day. No test-time fitting, BN updates, dataset-level statistics, rotations, LED permutations, input denoising, or new image channels. This is within-cell deterministic inference. Compare with spending the same inference budget on an extra existing ensemble member.

## Three proposals to re-scope the GOAL itself

These are **user decisions**, not changes implemented by this report. None creates a gain on frozen-v5 common-11. Unlike review 1, these do not simply nominate within-FOV r, v8 labels, a reliable-target subset, or reliability-normalised r as a replacement.

### R1. Measure marker information beyond broad morphology in a new isolation

**Proposed claim and measurement.** Replace the absolute mean-r threshold with evidence of incremental marker discrimination among morphologically comparable cells. Define a coarse morphology partition using source-only LF features, without the evaluated marker labels; freeze its construction before evaluation. Within each opposite-draw FOV and morphology stratum, measure pairwise concordance of predicted and v5 marker order, counting ties as half. Aggregate prespecified strata/FOVs with fixed weights, then average targets and directions. Compare the marker predictor with a source-trained broad-morphology-only reference and a within-block label-permutation null. Report all common-11 results; do not select a favourable marker subset. Use the same partition for every model compared within a direction.

**Reviewer argument.** High global r for CD16/CD14 may largely identify neutrophils and monocytes. That is weaker than the fine marker information needed to justify the LF pivot. Conditional concordance asks the sharper biological question and is invariant to any shared monotone label transform within a matched block. Same-FOV matching suppresses common offsets while morphology matching suppresses trivial lineage separation. It is a different estimand from within-FOV Pearson: the objective is added marker ordering beyond a fixed morphology reference, not higher pooled association after centring.

**Failure modes and falsifier.** Fluorescence noise still corrupts pair ordering; conditioning does not manufacture truth. Sparse rare populations can leave too few usable blocks, and morphology partitions may hide meaningful signal or incompletely control lineage. Before adopting, report pair support, stratum balance, FOV-block confidence intervals, and permutation behaviour. Reject this GOAL if support is insufficient in either direction or if scores mainly reflect separation between the morphology bins. Do not count all cell pairs as independent.

**Gain, cost, constraints.** Frozen-GOAL gain: none. Expected scientific benefit: distinguish broad-lineage prediction from finer marker information without new acquisition. Pilot on saved outputs: 0 GPU-h with a fixed pre-existing eligible morphology reference; allow .2–.5 H100-h if source feature extraction is needed. Build 1–2 days. Requires explicit adoption of the new estimand. It still cannot validate GS→MK: K001 has no paired GS-marker ground truth, and CIA remains an assumption.

### R2. Measure useful predictions at declared coverage, with abstention

**Proposed claim and measurement.** Define the LF pivot as a system that can decline unsupported cells. Use source out-of-fold errors to build a label-blind deployment selector from LF model disagreement and distance to source feature support. Freeze its thresholds on the source. On the opposite draw, measure error and r on accepted cells, actual coverage, and error on rejected cells. An example specification to ratify is at least 80% overall coverage and at least 50% coverage in each prespecified rare/morphology subset, with improvement over a baseline selector at comparable coverage. Retain the unchanged all-cell GOAL as a mandatory companion. Use one common cell-acceptance rule across targets where feasible, and expose per-assay coverage.

**Reviewer argument.** A deployment pipeline benefits from knowing when its LF pivot is unsupported. A forced prediction for every unusual cell may be the wrong objective under isolation shift. Explicit coverage constraints prevent a high score obtained merely by discarding difficult lymphoid or rare-marker cells; baseline comparisons on the same accepted cells distinguish selection from better prediction. The claim becomes useful, limited prediction on observed support, rather than an unqualified .67 average for an arbitrary isolation.

**Failure modes and falsifier.** Shared FOV measurement errors can be invisible to LF and to ensemble disagreement; a confident model can still be wrong. Source-calibrated intervals have no guaranteed coverage after an isolation shift. First ask whether the frozen selector orders opposite-draw errors better than random selection, and whether its advantages survive within rare subsets. If uncertainty is unrelated to error, or useful performance requires rejecting most rare cells, this is not a viable re-scope. Do not tune the selector against held-out outcomes.

**Gain, cost, constraints.** Frozen all-cell gain: none; accepted-set r is a new, selection-dependent quantity. Expected benefit is a defensible risk/coverage trade-off, not a promised numerical r increase. Source-cache reuse plus selector fitting costs roughly .1–.3 H100-h; build 1–2 days. Requires user approval of coverage and subset requirements. No new data, fluorescence input, test adaptation, or evaluation-label filtering is authorised by this proposal.

### R3. Replace the arbitrary .67 threshold with a reproducible two-isolation improvement claim

**Proposed claim and measurement.** Treat BSCCM as a fixed two-isolation benchmark. Define success by a preregistered improvement over the **ten-model .6571 reference at a stated inference/training budget**, not crossing .67 once. One concrete candidate rule is mean paired improvement >.0054, positive paired FOV-bootstrap lower confidence bound, and no direction losing >.0054; repeat with at least three independent training-seed pairs and report every target-direction entry. Set the final rule and compute budget before new outcomes. Use matched source-label counts as a secondary transport analysis, because N039 shows that training quantity changes the apparent reference substantially.

**Reviewer argument.** The .67 target was never user-confirmed and arose from a mismatched FOV comparison. A seed-selected crossing would not establish new-isolation reliability. A fixed-budget improvement over a strong reference, with an explicit asymmetry check, is an interpretable algorithmic claim. It also prevents a small single-model win already absorbed by the ten-model ensemble from being counted again. This is a change in the success criterion, not a claim that .6625 is biologically sufficient.

**Failure modes and falsifier.** Bootstrap intervals cover FOV/cell sampling conditional on these draws, not new donors; three seeds measure optimisation variability only. If gains are seed-unstable or entirely one-directional, the claim fails. Even success warrants the phrase “on these two isolations,” not “for new blood isolations generally.” Without additional acquisition, a population-level isolation-generalisation claim cannot be repaired by statistical resampling.

**Gain, cost, constraints.** No automatic score change. Expected benefit is a reproducible, properly delimited result rather than a nominal milestone. Reanalyse saved reference predictions at 0 GPU-h; budget roughly 3–6 H100-h to replicate a surviving ordinary-size candidate across three seed pairs, adjusted for actual member count. Build .5–1 day plus experiment work. User must ratify the revised claim and threshold; frozen GOAL.md stays untouched.

## Recommended bounded portfolio

1. Build the shared honest source-FOV prediction cache, then test scalar calibration and repeated-antigen consensus. Reserve **3.2 H100-h** total. Their limited upside is justified by their low incremental cost and the direct diagnostic.
2. Audit source supervision contradictions before spending GPU time. If present, prioritise idea 4; otherwise test idea 3. Reserve **1.6 H100-h** for one matched screen. These are alternative first bets, not a bundle that hides causality.
3. Test translation integration for **.3 H100-h**. Run FOV bagging only if source error influence/diversity warrants it; reserve **3.2 H100-h**.
4. Reserve **6 H100-h** for replication and IO of the most promising survivor. Total proposed first portfolio cap: **14.3 H100-h**; stop and re-estimate before exceeding it. The binding aggregate approval line remains 25 GPU-h. All future GPU work would use SLURM preempt; no jobs were submitted here.

Every candidate must retain non-Giemsa source-only training/SSL, unchanged real LF input artefacts and cached normalisation, no position features, unchanged evaluation cells/masks, and no target-draw fitting. Report N031 and the ten-model comparator, both directions, pooled and within-FOV r, influential tails, and paired FOV uncertainty. Keep legacy 12-target v4 and the failed pooled CD123/HLA-DR/CD14 target separately reported; neither can silently disappear because this review focuses on common-11.

**Decision proposed:** pursue the small readout audit and one source-supervision test; discuss R1–R3 before commissioning another broad training portfolio. If no candidate moves the ten-model reference outside the .0054 flat zone, the evidence supports changing the scientific claim or stopping this optimisation target, rather than treating another .001–.004 as progress.

## Local evidence used

- [Frozen goal](../../../../notebook/GOAL.md) and [knowledge ledger](../../../../notebook/KNOWLEDGE.md): K001/K019/K023/K028 and label/training restrictions.
- [N047 attempts](../../attempts.md), [binding declined directions](../../declined.md), [diagnosis](../../diagnosis.md), and [ten-model per-target scores](../../d1_big_ensemble.csv).
- [N038 synthesis](../../../N038-review-lf-mk-common-11-batch-held-out-go/ideas.md), its underlying idea documents, and FOV/noise diagnostics.
- [N039 matched reference](../../../N039-label-v8-re-derive-single-antibody-marke/n039_ceiling_per_target.csv), N040 learning curves, and N042 low-pass scores.
- Predictor/scorer implementation: `/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py`, `scripts/rescore_targets.py`, and `bsccm/gpu_norm.py`.
