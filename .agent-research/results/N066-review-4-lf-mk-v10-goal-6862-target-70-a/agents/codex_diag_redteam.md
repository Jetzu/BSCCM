Verdict: **MAJOR overclaim in causal identification, the inductive ceiling, and the training prediction.** The measured effects reproduce; most simple alternatives tested here fail. Background-related discrepancy is supported; which label is erroneous remains unidentified.

Commands actually run (code and outputs in `/tmp`; supplied Apptainer command failed because FUSE was unavailable):

```bash
/tmp/n066rt_env/bin/python -I /tmp/n066_rt/audit.py > /tmp/n066_rt/audit.log 2>&1
/tmp/n066rt_env/bin/python -I /tmp/n066_rt/followup.py > /tmp/n066_rt/followup.log 2>&1
/tmp/n066rt_env/bin/python -I /tmp/n066_rt/bootstrap.py > /tmp/n066_rt/bootstrap.log 2>&1
```

**A/B/C** below denote these scripts/logs. A asserts identical cell ordering across ensemble members and reconstructs the original weighting, masks, and joins. Target-level results: `/tmp/n066_rt/{alt,fov,slides,decomposition}.csv`. No existing repository file was edited.

**(a) Consensus: simple controls fail to explain it; independence remains unidentified.** [A–C]

Means reproduce **v10 .692940, local .665236, consensus .699629**; directions **.703633/.682248 → .708481/.690777**. Joint FOV bootstrap, stratified by slide with shared resamples across targets: gain CI **[.00229,.01164]**, **1000** resamples [C]. This concerns the changed evaluation target.

“Any correlated second noisy variable improves r” is false. For standardized labels, a=r(P,Y), b=r(P,L), c=r(Y,L), consensus r=(a+b)/sqrt(2+2c). Construct L*=cY+sqrt(1−c²)ε with ε orthogonal to Y and P: it exactly matches Y–L correlation but carries no additional predictive information. Consensus falls to **.672193** [C].

Further controls:

* Shuffle local labels within **50** v10-quantile strata, **200** permutations: consensus **.676316** [A]. This approximately preserves nonlinear mapping/conditional noise while destroying paired identity; ties and finite strata limit the control.
* Monotonic quantile-map v10 to the local marginal: **.687532**. Winsorize v10 at its **99th** percentile: **.671139** [A]. These distribution/tail controls do not reproduce the aggregate gain.
* For Δ=z(local)−z(v10), add only half the FOV-mean Δ: **.700493**; only half the cell-centered Δ: **.691646** [B]. The aggregate gain largely resides in FOV differences. These are evaluation-label diagnostics.

**MAJOR identification failure:** B constructs L*=cY+βV+ε, where V is standardized P residualized against Y, β matches observed P–L correlation, and orthogonal noise matches remaining variance. It reproduces consensus **exactly .699629**, without another fluorescence measurement. Assuming Y is error-free gives the identical gain through model-aligned distortion of L. This deliberately predictor-dependent counterexample establishes non-identifiability, not a usable labeling method. The controls establish paired information, not independent background error or biological truth.

Source inspection shows local and v10 labels share fluorescence foreground/calibration. LF uses LED/DPC images, so literal fluorescence-pixel reuse as model input is unsupported. Morphology-related background bias remains possible.

**(b) Composition: no major falsification; MODERATE slide confounding for pooled targets.** [A]

A restricts residual permutations to slide, predicted cell type, and slide × prediction cluster × target-prediction decile. Clusters use LF predictions only. Original FOV excess versus the last control:

| Target | b10 | b01 |
|---|---|---|
| CD19 | .453 → .448 | .435 → .433 |
| CD56 | .297 → .296 | .298 → .295 |
| CD14 | — | .287 → .299 |
| HLA-DR | — | .174 → .167 |
| grp:CD45 | — | .204 → .201 |

Observable composition does not explain these large effects. Predicted types are imperfect proxies; finer biological composition remains unexcluded.

Slide conditioning does explain substantial pooled excess: b01 grp:CD16 **.114 → .040**, grp:CD3/CD19/CD56 **.149 → .090**. A draw-wide shuffle includes slide offsets in the FOV effect. Remaining clustering of **label minus model** still includes missing image signal and model bias; it is not a causal label-error fraction.

**(c) Range-only explanation rejected; error attribution unresolved.** [A,B]

For b01 grp:CD45, replicate **1/0** reproduces r **.88125/.57759**, label SD **.73122/1.19284**, prediction SD **.83619/.83827**. Covariance is slightly larger on the worse slide: **.53883/.57755**. Extra label variance arithmetically depresses r, but its origin is unknown. OLS residual SD **.34562/.97374** measures signal missing from this predictor, not necessarily label error.

Common-overlap weighting over **20** pooled label-quantile bins matches label SD to **.60233/.59692**, yet r remains **.80829/.49240**. Matching prediction distributions gives **.88158/.56857**; matching prediction clusters gives **.88243/.56442**. Spearman **.81855/.59598** retains the gap. Marginal range/monotonic tail scaling alone is insufficient. Equal prediction SD does not prove equal populations or image-reachable information.

**(d) MAJOR: training gain and ceiling do not follow.**

Every consensus score changes evaluation labels while holding predictions fixed. None estimates training on consensus and evaluating against frozen v10. With conditionally unbiased independent training-label errors, both choices have the same population regression signal up to affine scale; only finite-sample variance may improve. Shared or morphology-dependent bias can instead hurt the v10 objective. Size/sign require matched training runs. Label-count gains do not identify background noise as their cause.

Failure of one acquisition-context residual regressor establishes neither unpredictability from LF images nor a ceiling near the claimed **.70–.71**. No total-label reliability or bound on image–label predictability was identified. The defensible result is a real, largely FOV-level alternative-label effect and a persistent replicate deficit, with causal source and training usefulness unresolved.
