# N034 re-review after F1–F8 fixes

**Verdict: MAJOR revisions required. The reported image-evaluation averages reproduce, but they are development-set results, do not establish calibrated uncertainty, and the set benchmark does not justify the broad impossibility or registration conclusions.** No new runtime BLOCKER was reproduced. A strict fluorescence-unseen new-draw claim would still retain the original F1 blocker; the revised, fluorescence-assisted split-2 protocol is substantially cleaner.

Reviewed the working tree on `bsccm-data-model`, 2026-11-11. This report is the only repository write. No source, notebook, job, checkpoint, or result array was changed; no commit, GPU, SLURM, training, or neural inference was performed. Host NumPy was installed only under `/tmp/$USER/codex-N034b/python`. Existing N034 is the review record; no new notebook item was created because the brief explicitly permits only this report.

Paths below are relative to `~/BSCCM` unless absolute. Abbreviations:

- **Item**: `.agent-research/notebook/items/N034-lf-fluorescence-marker-images-by-conditi.md`.
- **R**: `.agent-research/results/N034-lf-fluorescence-marker-images-by-conditi/`.
- **W**: `/hnvme/workspace/v181bb10-bsccm`.
- **E**: `W/runs/n034/`; evaluation sources are each directory's `eval.json` and `per_cell.npy` dictionary, keyed `set|stain|label`.

## 1. Disposition of F1–F8

| Finding | Status | Evidence and remaining scope |
|---|---|---|
| F1: scored new-draw calibration cells | **PARTIAL** | `scripts/fm_eval.py:76` now restricts **both** cohorts to clean split 2. All saved per-cell records in all nine full/sweep evaluations satisfy this restriction. Gain fitting/acceptance uses split 0/1 (`scripts/fm_prep.py:61`, `:64`, `:67`), so the original scored-cell overlap is fixed for A. However, each new draw still supplies fluorescence for its own gain/label calibration (`scripts/fm_prep.py:58`; `scripts/make_targets_v3.py:16`, `:28`, `:35`; `scripts/make_targets_v4.py:19`). This supports independent scored FOVs **conditional on new-draw fluorescence calibration**, not a fluorescence-unseen draw. The item discloses calibration at `:62`. B still scores **all** new-draw splits (`scripts/n034_setmatch_sim.py:59`), including the label-calibration cells; see below. |
| F2: harmful gain fit without held-out acceptance | **RESOLVED** for the specific defect | Smooth Huber fit and split-1 acceptance replace the HGB model (`scripts/fm_prep.py:54`, `:64`, `:67`). Stored `W/fm_prep.npz['gain_info']` reproduces **72/132** accepted, median variance ratios **.978030 fit / .976954 split 1 / .982938 split 2**, and **4/72 = 5.56%** worse on split 2. These match Item `:59`. This validates the stated empirical safeguard, not an independent optical/biological decomposition. `a_c_batch=1` and the split-0 clean restriction are present (`scripts/fm_prep.py:74`). |
| F3: unclean v5 label lineage | **PARTIAL** | A defaults to v5c in preparation, training, and evaluation (`scripts/fm_prep.py:33`; `scripts/fm_train.py:18`; `scripts/fm_eval.py:24`). Current target hash agrees with every full/sweep evaluation and its recorded training configs. But “targets_v5c everywhere” (Item `:62`) is false: **B hardcodes v5** (`scripts/n034_setmatch_sim.py:18`). Its autofl label and numbers retain the old lineage; singles are unchanged in the stored comparison. |
| F4: trivial controls and marginal/spatial yardsticks | **PARTIAL** | Template is fitted on real clean training-draw split-0 images (`scripts/fm_eval.py:59`, `:66`); LF-shuffled sampling is implemented (`:99`). Per-channel disk correlation, regression-quintile W1 and CRPS are present (`:49`, `:54`, `:116`). Their arithmetic checks out on stored arrays. The optimized samples nevertheless fail rank calibration; quintile W1 is a coarse distribution check, and the shuffled control has minor fixed-point contamination. No cap-location/spatial-sensitive validation was added. Item `:110` appropriately admits the spatial limitation, but “hypothesis (2) confirmed” overstates the calibration evidence. |
| F5: unequal optimizer budgets | **RESOLVED** | Full `cfm_v2` and `reg_v2` recorded configs both have **30 epochs, 215,838 rows, batch 128, seed 0**, and the same schedule parameters. Both `metrics.jsonl` files finish epoch 30 at **50,610 steps**. Validation losses reproduce Item `:71`: CFM .307088 @5 → .300880 @15 → .299684 @30; regression .067204 @5 → .066346 @15 / .066316 @20 → .066481 @30. This is one matched-budget screen, not seed replication. |
| F6: smoke skipped label scoring | **RESOLVED** | `R/smoke2.sbatch:15` uses 80 cells/stain, K=3 and 300 readout-fit cells. `W/runs/n034_smoke2/eval/eval.json` contains **12 label records per set**, spanning all ten stains, with every requested scoring key, including CRPS, conditional W1 and rank histogram. The QC-masked pooled target is absent. No additional GPU smoke was run here. |
| F7: averaging space and false ceiling | **RESOLVED** in the implemented definitions | Documentation distinguishes the target-space image mean and scalar readout mean (`scripts/fm_eval.py:5`, `:10`, `:11`); implementation uses `r_avgread` and `r_realref` (`:112`, `:123`), and marginal SD now averages K sample columns (`:127`). Item Plan `:16` still says “ceiling” and `:18` retains the scalar-teacher analogy; those statements should be qualified. Finite-K guided samples are not automatically draws from the true conditional posterior. |
| F8: artifact identity / resume | **PARTIAL** | Mode and prep identity are asserted (`scripts/fm_eval.py:33`); trainer saves/checks LF artifact fingerprints on resume (`scripts/fm_train.py:37`, `:40`). But evaluation instantiates the **current** `GPUNorm` (`scripts/fm_eval.py:29`) without comparing current LF files against those fingerprints (`bsccm/gpu_norm.py:17`). Its only current-artifact assertion is prep. Targets may change without an explicit rescoring override, and evaluation seed/readout-fit rows/weights are not saved (`scripts/fm_eval.py:74`). **No present mismatch found:** current prep, v5c and all three LF artifact hashes match the saved full-run configs. |

## 2. New or newly measurable findings

### N1 — MAJOR: sampler selection uses the reported new-draw test

**Evidence:** Item `:83`, `:85`, `:88`, `:90`, `:92`, `:106`; `scripts/fm_eval.py:76`, `:129`.

The sequence explicitly inspects new-draw deficits, launches steps/CFG sweeps, inspects new-draw improvements, launches round 2, and promotes CFG 2–3 using those results. All eight sweeps reuse identical scored rows, real readouts and regression readouts; I checked their arrays exactly. Reusing rows makes the sweep comparison fair, but does not preserve an untouched test after choosing its sampler.

The difference is material: new-draw CRPS **.765498** at s16/g1 becomes **.569711** at s48/g3 on those same cells. Report the latter as a **development result**. A failure scenario is choosing guidance that compensates draw-0 intensity shift while worsening a future draw. This is visible already in the cohort preference: inbatch CRPS is **.331369** at s48/g2 but **.382772** at s48/g3; newdraw improves from **.596509** to **.569711**.

**Verification: REPRODUCED** selection history and stored-array identity; optimism on a future draw is **REASONED**, not estimated. Choose sampler/readout settings using training-draw validation, freeze them, and evaluate a fresh reserved cohort/draw. Seed 1 in the sweep is a sampling/readout-fit seed, not a second trained-model seed; both networks remain training seed 0.

### N2 — MAJOR: improved distribution scores are being promoted into calibration/dispersion claims that the ranks reject

**Evidence:** `scripts/fm_eval.py:122`, `:126`, `:127`; Item `:88`, `:93`, `:107`, `:109`, `:128`; `R/n034_sampler_sweep.csv:86`, `:110` (the corresponding 12-label groups).

At the promoted settings, the mean fraction of real readouts outside the K=4 ensemble range is:

| Sampler | Inbatch | Newdraw | Exchangeable continuous-draw reference |
|---|---:|---:|---:|
| s48/g2 | .508333 | **.783333** | .400000 |
| s48/g3 | .569167 | **.768333** | .400000 |
| s96/g3 | .579583 | **.764583** | .400000 |

Thus the better CRPS is real, but calibrated per-cell uncertainty has **not** been demonstrated. Item's hypothesis (2) explicitly includes a calibrated rank histogram (`:19`); it cannot be called confirmed at `:109` without retaining this failure.

Also, `sd_ratio_sample` is **marginal across cells**, averaged over sample columns (`scripts/fm_eval.py:127`). It is not posterior width. At s48/g3/newdraw, marginal SD ratio is **.923701**, while mean within-cell SD / real marginal SD (`sharpness`) is only **.246300**. Conversely, marginal ratio >1 at g4 does not by itself establish over-dispersed conditional uncertainty. Bias, missing cell-specific signal and spread all affect the rank failures. Calling this calibrated uncertainty, or diagnosing conditional over-dispersion from the marginal ratio, would change a downstream uncertainty interpretation.

**Verification: REPRODUCED** directly from `ro_s`/`ro_real`, not rounded histograms. CRPS implementation itself is correct for the empirical ensemble; this is a conclusion defect, not a sign or normalization bug. Keep “lower empirical CRPS than the deterministic baseline,” and report the calibration failure alongside it. CFG>1 also changes the learned conditional sampling law (`scripts/fm_common.py:164`); it is not solely a numerical-integration correction.

### N3 — MAJOR: B changes bag locality and sample budget together with pairing

**Evidence:** `scripts/n034_setmatch_sim.py:27`, `:33`, `:57`, `:64`, `:68`; Item `:46`, `:49`, `:113`.

`same` uses all selected LF cells, sorted by FOV within slide. `other` globally permutes each slide, splits it into **disjoint halves A/B**, and uses only A as LF and B as labels. Its bags have no matched FOV/composition stratum. Paired ridge uses the whole selected training set. Consequently the contrasts change **pairing, number of supervised LF cells/bags, and locality** simultaneously.

Reconstructing the exact seeded bag builders on stored metadata, CD16 at n=16 gives:

| Arm | LF cells used | Bags | Mean distinct FOVs / LF bag |
|---|---:|---:|---:|
| same | 20,000 | 1,250 | 3.1968 |
| other | 10,000 | 625 | 15.9472 |

This is a valid experiment about this particular independent slide-wide bag construction. It is not a controlled estimate of pairing loss at otherwise matched bag composition/budget, nor a test of different cells grouped within a shared spatial or observable stratum. A scenario with informative region-level mixtures is intentionally destroyed by randomizing `other` slide-wide.

**Verification: REPRODUCED** bag membership/count/locality; generalization to other bag designs is **REASONED**. Add paired/same baselines using the same LF half, a random-bag `same` arm, and a disjoint-cell shared-stratum arm if the conclusion is about set-level matching broadly. The mathematical limitation for truly independent samples from one unvarying population remains sensible; the experiment does not establish that every useful grouping in BSCCM has that property.

**Pair leakage check:** No direct pairing leak was found. The A/B halves are disjoint across all retained bags, not merely within each bag. `set_learn` sorts label multisets and assigns by current prediction order (`scripts/n034_setmatch_sim.py:42`, `:47`); label-row order is not supplied to the fit. Shared slide membership is intended weak supervision. At n=1, `same` reduces exactly to paired ridge: maximum CSV correlation difference over all targets/sets is **4.44e-16**. No claim that `other` accidentally trains on its own cell labels is warranted.

### N4 — MAJOR: failure of this nonconvex learner is not failure of information in the bags

**Evidence:** `scripts/n034_setmatch_sim.py:40`, `:43`, `:45`, `:49`; Item `:48`, `:50`, `:112`.

Only one alternating sort/ridge algorithm, 30 iterations and eight starts is tested. Its training objective chooses a solution without demonstrating optimization sufficiency or comparing another estimator that uses exactly the same observed bag information. The bag-size decay can contain solver/objective failure as well as information loss.

**Reproduced synthetic counterexample using the exact source functions**, with no neural training: 16,384 independent 32-dimensional standard-normal LF vectors, linear labels plus .1-SD noise, 64 same-cell bags of size 256, independent 4,096-cell test set, RNG seed 123. Results:

- True-pair ridge: **r = .999849**.
- Source `set_learn`, 8 starts × 30 iterations: **r = .032092**.
- Ridge on **observable bag-mean X and bag-mean y only**, lambda `10/256`: **r = .990750**.

Bag means are functions of the label multiset and require no within-bag registration. The source learner can therefore look like chance despite strong identifiable per-cell signal in its supplied bags. This does **not** show that actual BSCCM large bags or independent A/B bags attain .99; it invalidates interpreting this learner's failure as an information-theoretic ceiling.

Minimal reproduction (NumPy only, run from repo root; extract functions to avoid executing the script's data-loading/training top level):

```python
import ast, types, numpy as np
from pathlib import Path
m = ast.parse(Path('scripts/n034_setmatch_sim.py').read_text())
nodes = [q for q in m.body if
    (isinstance(q, ast.FunctionDef) and q.name in ('ridge', 'set_learn')) or
    (isinstance(q, ast.Assign) and any(isinstance(t, ast.Name) and
                                     t.id == 'pred' for t in q.targets))]
g = dict(np=np, a=types.SimpleNamespace(starts=8, iters=30))
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<bench>', 'exec'), g)
rng = np.random.default_rng(123)
X = rng.standard_normal((16384, 32)); Xt = rng.standard_normal((4096, 32))
w = rng.standard_normal(32)
y = X @ w + .1 * rng.standard_normal(len(X))
yt = Xt @ w + .1 * rng.standard_normal(len(Xt))
b = np.arange(len(X)).reshape(-1, 256)
_, wo = g['set_learn'](X, y, [(i, i) for i in b])
wb = g['ridge'](X[b].mean(1), y[b].mean(1), lam=10/256)
for name, ww in [('paired', g['ridge'](X, y)), ('OT', wo), ('bag-mean', wb)]:
    print(name, np.corrcoef(g['pred'](Xt, ww), yt)[0, 1])
```

**Verification: REPRODUCED on synthetic data; untested on BSCCM feature fits.** Narrow B to this algorithm/design, and test bag-mean or other weak-supervision baselines before making the wet-lab or impossibility recommendation.

### N5 — MAJOR for B's clean independent-cohort claim: old labels and all-split testing remain

**Evidence:** `scripts/n034_setmatch_sim.py:18`, `:19`, `:59`; Item `:62`; upstream calibration references under F1/F3.

B did not receive A's v5c/split-2 fixes. Depending on target, **79.1–81.1%** of its newdraw rows are split 0, which supplied their own label transforms. Its six single-marker labels match v5c, but the batch-0 clean unstained autofl rows used here differ from v5c by **RMS .230596**. This is why a blanket “v5c everywhere / calibration cells never scored” cannot describe all of N034.

Its frozen `n007_ssl_aug` features are also transductive: that encoder saw test-draw images, as already established in `.agent-research/notebook/items/N031-inductive-ssl-control-re-pretrain-the-ss.md:7`. Loading those exact features is explicit at `scripts/n034_setmatch_sim.py:19`. This is unlabelled pretraining, not cell-label pairing leakage, and does not automatically explain the within-benchmark differences. It does mean B is not an inductive new-draw experiment under the N031 rule.

**Verification: REPRODUCED** label differences and split fractions from stored arrays; encoder provenance verified from N031's record, not independently retrained here. Rescore B with v5c/split 2 and state its feature-pretraining protocol. The size/sign of any resulting r change was not measured.

### N6 — MINOR: evaluator robustness and control/provenance gaps

**Evidence:** `scripts/fm_eval.py:74`, `:82`, `:112`, `:116`, `:117`, `:122`; `scripts/fm_figure.py:52`.

- A permutation is not a derangement. The comment “ANOTHER cell” is false for fixed points. Reconstructed counts: v2 **11/4,000 inbatch, 7/4,000 newdraw**; seed-1 sweep **12/2,000 and 10/2,000**. These are slightly paired controls, not catastrophic leakage. No current headline reversal is demonstrated. Derange or disclose the small contamination.
- Conditional W1 is invariant to permutations **within** the five regression-readout strata. On actual s48/g3/newdraw CD3 arrays, adversarially reversing predicted mean ranks against real readouts inside each stratum leaves conditional W1 **exactly .510730**, while correlation with the real readout falls **.530742 → .214029**. It supports quintile-level fidelity, not conditioning on all LF information. CRPS and per-cell correlations provide complementary checks; none should be replaced by this W1 alone.
- Tied regression outputs can produce empty strata. The mean over all five bins at `:117` then includes empty-array W1 → NaN. A constant/collapsed regression readout is a concrete trigger. Distribution metrics also ignore `ok` after its correlation use. **These do not affect the current full/sweep numbers:** every stored `ok` is true and strata have exactly 80 cells (v2) or 40 cells (sweeps). They are latent robustness defects; silent NaN-dropping summaries could hide a failed target in a future run.
- Evaluation seed, batch size, readout sample size, fit rows and ridge coefficients are absent from saved metadata. I reconstructed seed 0 for v2 and seed 1 for sweeps from the current selection code/arrays, but this should not require reverse inference. Save them and assert LF artifact identity at evaluation; see F8. The shuffled control supplies only one sample per cell, so it has no equal-K CRPS comparator. Applying ensemble-row permutations offline confirms the conditional CRPS gain survives an equal-K shuffled check: s48/g3/newdraw **.569711** versus **.951333** mean over 100 stain-matched permutations (seed 340). This is a stored-readout diagnostic, not a new image-generation run.

**Verification:** fixed points, within-stratum permutation, populated strata/masks, collapsed-bin NaN and offline shuffled-ensemble result **REPRODUCED**. The constant-regression counterexample gives bin counts `[0, 0, 0, 0, 20]` and conditional W1 `NaN`. Identity mismatch scenarios are **REASONED**, with no current mismatch.

### N7 — MINOR: montage “+/−” are selected brightness extrema, not marker calls

**Evidence:** `scripts/fm_eval.py:95`; `scripts/fm_figure.py:21`, `:22`, `:33`.

The montage retains only the first six sorted evaluation rows per stain, then chooses the channel most variable **among those six test images** and labels its brightest/dimmest cells “+/−”. There is no marker-positive threshold or gate. For example, the v2 unstained “autofl.−” cell has curated autofl **+.452161**, above the shown “+” cell's **+.144657**. The CD3 display channel changes from channel 2 in v2 to channel 4 in the promoted montage. Neither behavior changes quantitative scores, but the display is not a reproducible, prespecified marker-positive/negative spatial test.

**Verification: REPRODUCED** from both montage arrays; inspected `R/fig_n034_s48_g3.png`. Label images “brighter/dimmer example,” disclose the six-cell selection and physical channel, and retain the item's caveat that conditional cap/membrane fidelity remains unshown. No accusation of numerical metric cherry-picking follows from illustrative montage selection alone.

## 3. Numerical reconciliation

Every numeric field in **both** `R/n034_eval_v2.csv:1` and `R/n034_sampler_sweep.csv:1` matches its corresponding evaluation JSON field exactly (maximum parsed numeric discrepancy **0**). Recomputed empirical CRPS and conditional W1 in v2, s48/g2, s48/g3 and s96/g3 agree to **≤2.22e-16**. The formulas are standard empirical W1 and empirical-ensemble CRPS, respectively; no reversed CRPS sign, incorrect pairwise factor, accidental pooled-cell CRPS, or bad quintile indexing was found. The CRPS finite-K V-statistic is the score of the empirical forecast distribution; changing K changes its finite-ensemble behavior, so compare v2 K=8 against the sweeps K=4 with that qualification. The eight sweeps themselves use the same K, rows and readout fits.

Macro means over the 12 reported labels reproduce the item's central A numbers:

| Run/set | r_avgread | r_reg | W1 flow/reg | Quintile W1 flow/reg | CRPS flow/reg |
|---|---:|---:|---:|---:|---:|
| v2 newdraw | .439816 | .599371 | .752520 / .695718 | .791973 / .739200 | .738774 / .778349 |
| v2 inbatch | .508387 | .694840 | .344754 / .222130 | .392050 / .298846 | .413683 / .383771 |
| s48/g2 newdraw | .547721 | .611540 | .553719 / .710977 | .598682 / .740864 | .596509 / .776406 |
| s48/g2 inbatch | .557168 | .664016 | .161914 / .247134 | .214767 / .311498 | .331369 / .386654 |
| s48/g3 newdraw | .544719 | .611540 | .493290 / .710977 | .550734 / .740864 | .569711 / .776406 |
| s96/g3 newdraw | .547513 | .611540 | .487764 / .710977 | .545546 / .740864 | .565198 / .776406 |

Sources: Item `:73`, `:85`, `:91`; CSV v2 data begin at `:2`/`:14`; sweep s48/g2 data begin at `:74`/`:86`, s48/g3 newdraw at `:110`. The item’s other round-1/round-2 macro averages also agree to displayed precision. “Beats every distribution score” (Item `:89`) is true **for the three macro averages**, not every label: at s48/g2/newdraw, flow wins W1 on **11/12**, quintile W1 on **10/12**, CRPS on **10/12**. CD3 is worse on all three (`R/n034_sampler_sweep.csv:90`).

The set CSV's principal same-cell means also reproduce:

| Mode / n | Newdraw mean r |
|---|---:|
| paired / 1 | .562200 |
| same / 1 | .562200 |
| same / 4 | .521513 |
| same / 16 | .431925 |
| same / 64 | .191079 |
| same / 256 | .028943 |
| same / 2048 | −.020379 |

Source: `R/setmatch/setmatch_sim.csv:1`, grouped by `mode,n,test`. The headline **93% / 77%** is the ratio of macro means (.927630 / .768276), not a uniform per-marker recovery guarantee.

Three prose corrections matter:

1. **“77–99%” practical registration recommendation is not true across targets.** At n=16, recovery relative to paired is CD16 **97.6%**, CD14 **97.5%**, CD45 **95.5%**, CD3 **84.7%**, HLA-DR **49.1%**, CD19 **21.6%**, autofl **5.4%**. CD19 already retains only **48.4%** at n=4 (`R/setmatch/setmatch_sim.csv:136`); autofl n=16 is **.023042 vs .429924 paired** (`:164`, `:158`). Limit the recommendation to demonstrated targets, this learner and this bag design; it does not establish Giemsa-image performance.
2. **“Other −.04..+.08 at every size” is slightly wrong even as a macro range.** Means at n=1,4,16,64,256,2048 are **−.016591, .004236, −.038390, .078310, −.053814, −.052656**. Nor is every marker zero: newdraw CD3 at n=64 is **.248397** (`R/setmatch/setmatch_sim.csv:100`); inbatch CD45 is **.337618** at n=64 (`:75`). Without bag/seed replication or a null distribution, “chance” is an interpretation, not a tested significance result. Signed averaging can also cancel directions; mean absolute r for `other` reaches **.108282** at n=2048.
3. **v2 “single-sample SD / real SD .41–.89 inbatch” is inaccurate as stated** (Item `:80`). Averaging K columns as implemented, the 12-label range is **.409092–1.201546**; even restricting to single-antibody targets it is **.409092–.815938**, not .89. Group CD16 is **1.201546** (`R/n034_eval_v2.csv:12`), and unstained autofl **.921894** (`:13`). Most markers are under-spread marginally, but not all.

## 4. Required integration plan and limits

No source changes are supplied by this read-only review. Before accepting the conclusions:

1. Amend Item/figures to distinguish development scores, lower CRPS, failed rank calibration, marginal spread and spatial fidelity. Correct the set-recovery and SD-range prose.
2. Freeze sampler/readout selection on validation before a fresh test. Preserve the current numbers as exploratory results; do not relabel reused test FOVs as untouched.
3. Bring B onto v5c/split 2, disclose or replace transductive n007 features, match LF/bag budgets and grouping, and add a bag-mean baseline before general claims about identifiability or registration requirements.
4. Harden `scripts/fm_eval.py` artifact checks/provenance, tied strata, masks and shuffled control; clarify brightness labels/channel selection in `scripts/fm_figure.py`. These are the real files an integrator would touch, plus the notebook/result summaries when new results exist. No rewrite of the already-correct CRPS/W1 arithmetic is called for.

Not checked: neural training/inference on CPU or GPU; actual checkpoint tensor identity or stochastic replay; independent optical registration; fluorescence unmixing/antibody specificity; full image spatial fidelity; new BSCCM set-learner fits; second training seed or opposite blood-draw direction; a fresh donor/FOV test; causal reasons for intensity bias; improvement magnitude after proposed protocol changes. No numerical inflation from F1's remaining assisted calibration, legacy B labels, or transductive SSL was estimated. Current artifact hashes agree; failure scenarios from missing guards are not evidence of a present corrupted run.
