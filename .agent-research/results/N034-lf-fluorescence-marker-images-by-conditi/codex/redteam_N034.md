# N034 red-team review

**Verdict: BLOCKED for an independent new-blood-draw claim.** Held-out-draw fluorescence participates in target calibration, including calibration on cells subsequently scored as test cells. Flow direction, CFG, and the basic transform algebra appear consistent; the main defects concern preprocessing and evaluation evidence.

Reviewed the working tree on `bsccm-data-model`, 2026-11-11. No repository code, notebook records, jobs, or checkpoints were changed. This report is the sole permanent output. No GPU execution, SLURM submission, or model training was performed.

## Findings

### 1. BLOCKER — “newdraw” includes cells used to fit its fluorescence correction

**Evidence:** `scripts/fm_prep.py:38-49,55`; `scripts/fm_eval.py:63`; upstream `scripts/make_targets_v3.py:16-42`, `scripts/make_targets_v4.py:19-21`, `scripts/make_targets_v5.py:27-29`.

`fm_prep` fits fluorescence-dependent gain models on split-0 cells of **every slide in both blood draws**, without restricting calibration to the training draw. Evaluation calls **all splits** of the other draw `newdraw`. In the default b1→b0 direction, 102,097 of 127,946 clean new-draw cells (79.80%) are split 0 and were eligible to fit their own image-target correction. Stored `gain_info` records 612,098 channel-cell fit observations in batch 0; these span six channels and are not distinct-cell counts.

Curated labels have the same overlap: v3 calibration and v4 position detrending use each draw's split-0 fluorescence, and v5 uses split-0 slide statistics. `make_targets_v3.py:3` explicitly describes calibration/evaluation separation as split 0 versus split 2. `fm_eval.py:63` discards that separation for `newdraw`.

**Concrete failure scenario:** A new draw's fluorescence is unavailable at deployment, but the benchmark removes its position dependence using that draw's measured fluorescence, partly fitted on the cells being scored. Evaluation uses an assisted, partly in-sample reference distribution and can become easier. It cannot establish generalization to a draw whose fluorescence has never participated in calibration. There is no direct held-out-cell inclusion in the optimizer's row set: the leakage is through reference construction. The magnitude of score inflation has not been measured.

`a_c` also uses all clean cells, including test cells (`fm_prep.py:55`), so training accesses test-derived statistics. **Qualification:** its current six medians equal the medians recomputed on batch-1 split-0 clean cells exactly. That access has no demonstrated numerical effect here and is not the principal blocker.

**Verification: REPRODUCED** from `targets_v5.npz`, `fm_prep.npz`, and the generating code. Current and training-only `a_c` both equal approximately `[5.9304, 5.9304, 4.4478, 4.4478, 7.4130, 2.9652]`.

**Resolution:** For a strict new-draw claim, freeze target calibration without new-draw fluorescence. For a fluorescence-assisted benchmark, disclose that assistance and use independent calibration cells/FOVs excluded from scoring. Do not call all other-draw rows independent test data.

### 2. MAJOR — The position-gain fit fails a held-out-FOV check on actual data

**Evidence:** `scripts/fm_prep.py:45-51`. Its only reported fit-quality statistic is correlation between trend and log integral on the same cells used for fitting. There is no held-out check or rejection of a harmful correction.

For `antibodies_CD16/batch_0/slide_replicate_0`, channel 0, using positive-integral clean cells:

| Quantity | Split 0, fitted | Split 2, held-out FOVs |
|---|---:|---:|
| Cells | 7,418 | 1,005 |
| corr(log gain, log disk integral) | 0.3429 | -0.0087 |
| var(log integral − log gain) / var(log integral) | 0.8898 | 1.1919 |

The correction reduces fitted-cell log-integral variance by 11.0%, but **increases held-out-FOV variance by 19.2%**. Other channels on this slide show the same discrepancy. Across slide/channel pairs, median gain/log-integral correlation is 0.264 on batch-0 split 0 versus 0.116 on split 2; batch 1 gives 0.169 versus 0.073.

**Concrete failure scenario:** Finite-sample cell-composition fluctuations or local acquisition structure are fitted and removed as though they were multiplicative optical gains. Fitted rows and unseen FOVs acquire different target distributions. A model can be penalized for predicting real variation or rewarded against a reference whose variation was attenuated in sample. Random placement supports detrending in principle; it does not establish that every fitted fluctuation is an optical gain.

**Verification: REPRODUCED**, without fitting a model or rereading image pixels. Computed these correlations and variance ratios from stored `integ`, `gain`, `slide`, `split`, and `hist_row`. This demonstrates correction generalization failure on this slide/channel. It does **not** establish which biological components were removed or that every correction is harmful.

**Resolution:** Validate correction on independent FOVs against gain=1, quantify residual position predictability and biological signal retention, and use disjoint/cross-fitted corrections where appropriate. The notebook's in-fit correlations do not validate the correction.

### 3. MAJOR — Legacy v5 labels reintroduce Giemsa cells through upstream calibration

**Evidence:** defaults at `scripts/fm_prep.py:29`, `scripts/fm_train.py:18`, `scripts/fm_eval.py:18`; readout labels at `scripts/fm_eval.py:22,55-59`. Upstream: `scripts/make_targets_v3.py:13-16,34-42`, `scripts/make_targets_v4.py:8-20`, `scripts/make_targets_v5.py:10-16,27-29`. Context: N029's clean-calibration decision.

Direct image-training, readout-fitting, and evaluation row filters exclude `hist_row >= 0`. However, default `targets_v5.npz` is the **unclean** lineage. Without `BSCCM_CLEAN=1`, upstream calibration includes 12,502 split-0 Giemsa cells, all on unstained slides. They affect autofluorescence fits and the unstained reference for grouped-marker calibration. Filtering rows after these fits cannot undo their influence.

Comparison with existing v5c on **non-Giemsa, masked-valid rows**:

| Target | RMS label change | Maximum absolute change |
|---|---:|---:|
| autofl | 0.07805 | 1.75571 |
| grp:CD3/CD19/CD56 | 0.04411 | 0.23031 |
| grp:CD45 | 0.04826 | 0.24243 |
| grp:CD16 | 0.04325 | 0.20014 |

All eight single-marker labels were identical in this comparison. The image loss does not read `T['Y']`: this concerns the curated-label yardstick/readout and the Giemsa-exclusion claim, rather than direct use of legacy scalar labels as image-training targets.

**Concrete failure scenario:** A report says Giemsa measurements entered no target/readout fit, but the reference labels and resulting ridge coefficients already depend on them. The clean-cohort correction established in N029 is lost.

**Verification: REPRODUCED** by counting upstream-eligible rows and comparing v5/v5c label arrays. No calibration was rerun.

**Resolution:** Use and fingerprint the clean target lineage for the yardstick, then rebuild dependent readout results. Keep the direct row exclusions, which are correct.

### 4. MAJOR — Spatial/distribution yardsticks admit strong trivial controls

**Evidence:** radial features and membrane ratio at `scripts/fm_eval.py:34-39`; whole-image pixel correlations at `scripts/fm_eval.py:76,81`; marginal W1 and pooled ranks at `scripts/fm_eval.py:95-107`; spatial-pattern hypothesis at N034 item lines 19-20.

**Actual-data control:** From the same 200-cell sample used for transform checks, averaged transformed images of available batch-1 split-0 cells separately by stain. Compared each fixed template with 10 batch-0 cells of that stain. Each template used 6-10 reference cells and **no evaluated cell's LF**. Mean `pix_r` over 100 new-draw cells was **0.5289**, including **0.8284** on the `all` stain. These small-sample diagnostics are not estimates of a fully fitted template baseline. They show substantial pixel correlation from shared channel/background/centered-cell structure alone.

The 24 radial integrals and “membrane” ratio are invariant to angular rearrangement within annuli. A synthetic left/right cap reversal gave **exactly identical radial features**. These features cannot establish cap location or membrane morphology. The fixed 32-56-pixel annulus is not a cell-specific membrane segmentation.

W1 and rank histograms are marginal checks pooled over cells. With their exact NumPy formulas, 400 standard-normal truths and eight independent standard-normal samples per cell (seed 34), an unconditional null gave:

- W1/sample SD = 0.0796 versus 0.8004 for a constant-mean regressor;
- sample spread ratio = 1.008;
- outside fraction = 0.2075 versus nominal 0.2222;
- `r_sample = -0.0128`.

**Concrete failure scenario:** An unconditional stain-matched generator beats regression on W1/spread and has a plausible rank histogram; a centered stain template scores well on pixel correlation; generated angular structures remain unrelated to the cell. Those successes would not establish conditional or spatial fidelity. Existing per-cell correlations can expose the unconditional null. This finding does **not** say the W1/rank formulas are wrong or that every metric would pass the null.

**Verification: REPRODUCED** using fluorescence mmap data for the template and synthetic NumPy counterexamples for angular invariance and calibration. No neural model was run.

**Resolution:** Include stain-template and LF-shuffled/unconditional controls on identical cells; assess calibration within LF-predictable strata. Support membrane/cap claims with spatially sensitive measurements or inspected paired images. Interpret marginal W1 as a marginal-distribution result.

### 5. MAJOR — Submitted CFM/regression runs have unequal optimization budgets

**Evidence:** N034 item lines 35-36; actual `runs/n034/{cfm,reg}/ckpt.pt` configs; LR schedule at `scripts/fm_train.py:45-46`; evaluator comparison guard at `scripts/fm_eval.py:29-31`.

Actual configs specify CFM **30 epochs**, regression **15 epochs**, both seed 0, 215,838 training cells, batch size 128, width 64, and the same initial LR. Planned budgets are 50,610 versus 25,305 batches. Cosine decay depends on total epochs, so their LR trajectories also differ. Evaluation checks only that training draws agree.

**Concrete failure scenario:** CFM improves over an insufficiently optimized deterministic baseline, or regression shrinkage is assessed before convergence. “Same network” alone cannot attribute a difference to stochastic generation. Unequal budgets may be justified if both methods demonstrably converge, but no completed convergence evidence was available during review.

**Verification: REPRODUCED** from checkpoint ZIP/pickle opcode metadata, without loading tensors; consistent with the notebook. Impact on final performance is **REASONED**, not measured.

**Resolution:** Use matched-update controls and/or establish each mode's convergence using training-draw validation, then compare fixed checkpoints on identical rows. Do not tune on `newdraw`. The one-seed run is a screen, as planned, not replicated evidence.

### 6. MAJOR — Smoke evaluation skipped every central label/readout metric

**Evidence:** N034 `smoke.sbatch:15` uses `--per-stain 24`; `scripts/fm_eval.py:88-89` skips labels with fewer than 50 evaluated valid rows. The N034 item calls the pipeline “end-to-end OK” at line 34.

**Concrete failure scenario:** Smoke evaluation exits successfully while `r_real`, `r_sample`, `r_mean`, `r_avgread`, label-readout W1, and rank-calibration results are all absent. Errors in the central scoring path survive the end-to-end claim.

**Verification: REPRODUCED** from `/hnvme/workspace/v181bb10-bsccm/runs/n034_smoke/eval/eval.json`: all 20 set/stain entries have image metrics and **none has a label/readout result**. Ridge feature extraction/fitting was invoked; application and scoring inside the label loop were not exercised. This is a verification gap, not proof of a hidden runtime error in the skipped block.

**Resolution:** Run an evaluation smoke with enough valid cells to enter the label loop and assert expected keys, including intentional omission of QC-masked batch-1 pooled CD123/HLA-DR/CD14. That GPU evaluation was not run under this brief.

### 7. MINOR — “Mean image,” posterior-mean interpretation, and “ceiling” need qualification

**Evidence:** `scripts/fm_common.py:37-42`; `scripts/fm_eval.py:6-9,36-39,80,90-102`; N034 item lines 17-18.

Let `U = asinh(corrected_counts/a_c)/3`, and `R` denote the radial-integral ridge readout. Implemented definitions are:

- `r_sample`: average over K of corr(`R(U_k)`, curated label);
- `r_mean`: corr(`R(mean_k U_k)`, curated label);
- `r_avgread`: corr(`mean_k R(U_k)`, curated label).

They are internally consistent. However, `R` applies `sinh`, integration, then `asinh`, so `R(E[U|LF])` generally differs from `E[R(U)|LF]`. Averaging target-space images is also different from averaging fluorescence counts. For target values 0 and 1 with `a_c=5.9304`, the linear-count mean is **29.7050 counts**, while inversion of their target-space mean gives **12.6275 counts**.

**Concrete failure scenario:** Differences between `r_mean`, `r_avgread`, and a scalar teacher are attributed to posterior modeling when they arise from nonlinear operation order. The transformed-MSE regressor estimates `E[U|LF]`, so target-space sample means are an appropriate comparison to that regressor. Equating this with the scalar teacher's conditional mean needs additional assumptions. The squared-correlation argument concerns the same scalar variable and its own conditional mean, not automatically `R(E[U|LF])`.

Also, `r_real` is empirical real-image reference agreement, not a mathematical upper bound: a predictor removing readout noise could correlate better with curated labels. `sd_ratio_sample` uses only sample index 0 (`fm_eval.py:101`), making it noisier than the K-averaged W1.

**Verification: REPRODUCED** for the nonlinear-mean counterexample; interpretations are **REASONED** from the formulas. No defect in `Prep.linear` itself was found.

**Resolution:** Name the averaging space; use `r_avgread` for scalar conditional-mean interpretation. Report linear-count image means separately if desired, and call `r_real` real-image reference agreement.

### 8. MINOR — Evaluation does not validate transform identity; resume omits LF artifact fingerprints

**Evidence:** training fingerprints at `scripts/fm_train.py:36-40`; evaluation at `scripts/fm_eval.py:22-31`; LF artifact loading at `bsccm/gpu_norm.py:17-21`.

Training saves/checks `prep_sha256` and `targets_sha256`. Evaluation loads current prep/targets without comparing saved hashes. It does not check checkpoint mode against `--cfm`/`--reg`, or matching transforms/cohorts for compared checkpoints. Resume does not fingerprint `cellmeta.npz`, `led_norm.npz`, or `backgrounds_p50.npy`.

**Concrete failure scenario:** Gain artifacts are corrected but an old model is silently evaluated against the new transform; or LF normalization changes between preempted segments while saved config still matches. Numbers then mix incompatible representations. A wrong-mode checkpoint also runs an incompatible inference convention.

**Verification: REASONED** from missing guards. **No current mismatch was found:** saved prep/target hashes in both full-run checkpoints match current artifacts. No file was replaced to provoke a failure.

**Resolution:** Check mode and artifact identity before scoring; save identities/configs in evaluation output. Extend resume checks to LF normalization. Explicitly named scalar-label rescoring can intentionally allow a target-file discrepancy while recording it.

## Other checks and qualifications

**LF normalization is test-specific and transductive.** `scripts/led_norm_stats.py:15-26` samples 300 cells from each slide for `slide_scale`, then 2,000 cells globally for `s_c`, without split/draw/Giemsa restrictions. `GPUNorm` selects each evaluated cell's slide scale at `bsccm/gpu_norm.py:30`. Reconstructing the source's seeded choices with current metadata gives 3,000 batch-0, 663 split-2, and 313 Giemsa cells among 6,600 slide-scale reference cells. The global scale sample contains 663 batch-0 and 75 Giemsa cells. Target and cellmeta slide arrays agree, validating the reconstruction's slide ordering. This is unlabelled input calibration, not fluorescence-label leakage; its performance effect was not measured. Thus strict “no test information anywhere” and universal Giemsa-exclusion claims also fail for LF preprocessing. Whether this normalization is acceptable must be specified in the deployment protocol. The optical flat-field's original acquisition provenance was not audited.

**Direct rows and label mapping:** `fm_train.py:32-34` restricts image training to clean training-draw split 0 and validation to clean same-draw split 1. `fm_eval.py:48-59,63-67` restricts readout fitting to clean training-draw split 0 and directly evaluates clean rows only. Stored ten stain names match `LAB`; all eight single-marker names are valid. Batch-1 `all` has zero valid `grp:CD123/HLA-DR/CD14` labels, so its readout is correctly absent, including b1→b0 where there is no valid training-draw fit. The other evaluated stain/label masks cover all clean cells of that cohort. Hence distribution/rank calculations using all rows rather than `ok` do not currently include partially masked rows in those reported cohorts.

**Target transform and integer handling:** On 200 seeded cells, 10 per stain per draw, recomputation from fluorescence mmap matched stored backgrounds and all four integrals exactly. Maximum border-noise discrepancy was 6.35e-7 counts from float precision. NumPy float32 forward/inverse algebra agreed within 0.00293 counts. `linear(target(fl,g))` correctly returns `(fl-bg)/gain`, as documented, rather than raw fluorescence. Sampled maximum raw value was 4095, with no sampled pixel above 32767; this is not a full-dataset range assertion. Independently, the int32 cast at `fm_common.py:26` safely represents the entire uint16 range before float conversion.

**Flow matching:** Source reasoning finds `(1-t)z + ty` and velocity `y-z` (`fm_train.py:73-78`) consistent with forward-time Heun integration, noise at 0 to data at 1 (`fm_common.py:151-167`). CFG training zeros LF and its presence flag while retaining stain; sampling's null does the same. Time embedding scales the same normalized time by 1000 in both paths. Regression training/inference both use zero state, zero time, full LF and stain. EMA lerp direction is correct; the architecture has no running BatchNorm buffers requiring separate updates. No flow-sign, CFG-null, integer-overflow, or EMA-sign defect was found by inspection.

**Translation and row ordering:** Normalization uses shift=0 before identical joint translation/replicate padding of LF and fluor (`fm_common.py:45-55`). RawSet and FluorSet index the same global row. Evaluation loaders use `shuffle=False` and ordinary ordered delivery (`vs_common.py:121-125`), preserving sorted `rows` order; a final short batch does not invalidate `bi*bs`. This verifies code-level indexing/augmentation, not independent optical LF/fluor registration.

**Metric definitions:** For equal-size scalar vectors, mean absolute difference after sorting is empirical W1. The code averages this across K sample columns and divides by real-readout SD. Strict-lower ranks give K+1 bins; `2/(K+1)` is the right extreme-bin rate for exchangeable continuous draws. Ties need randomized handling to preserve that assertion. Pooling ranks does not establish conditional calibration, as finding 4 shows. A fixed invocation has fixed seeds/order, but changing readout sample size advances the shared NumPy RNG differently and changes later evaluation cohorts. Default 400 cells per stain is a screen, not precision comparable to an all-cell teacher score; no uncertainty intervals are emitted.

**Resume/preemption:** By inspection, the saved cursor follows completed processing. `EpochBatches` regenerates an epoch permutation with a private seeded generator and starts at that cursor (`vs_common.py:241-255`); prefetch does not advance the saved training cursor. Model, EMA, optimizer, epoch, next batch, step, CPU/CUDA RNG states are restored. LR derives from restored step/config. Translation uses saved global Torch RNG; loader seeding uses a separate generator. Validation uses a fixed local noise generator and no condition dropout. Checkpoints use tmp+fsync+rename (`vs_common.py:35-41`), and signals request save/exit after processing. A checkpoint after an epoch's last batch but before finalization can resume an empty remainder and log zero partial-epoch training loss; this alone does not skip updates. No substantive cursor/LR/EMA resume defect was identified, subject to finding 8 and untested runtime behavior.

## Verification and reproducibility

The requested Apptainer invocation failed before Python started: `squashfuse_ll exited: fuse: device not found, try 'modprobe fuse' first`. Host Python lacked NumPy/Torch. Installed only **NumPy 2.5.4** into `/tmp/$USER/codex-N034/python`, keeping pip/cache/temp files under that permitted scratch directory. Checks used host `python3 -I` with an explicit scratch NumPy path and `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. No compute job was submitted. These bounded checks use small reductions and sequential mmap reads of 200 images, without model fitting; parallel scheduling would add overhead to this diagnostic.

Temporary scripts are `/tmp/$USER/codex-N034/{check_metadata,check_cells,check_gain,check_metrics}.py`. All material measured results are recorded above so the report does not depend on scratch retention. Essential reproduction recipes:

```python
# W = '/hnvme/workspace/v181bb10-bsccm'; T/P = targets_v5/fm_prep npz.
clean = T['hist_row'] < 0
# F1: count clean batch-0 rows by split; sum gain_info n_fit for batch-0 slides.
# F2, per slide/channel and split in (0, 2):
q = clean & (T['slide'] == slide_id) & (T['split'] == split) & (P['integ'][:, c, 0] > 0)
u = np.log(P['integ'][q, c, 0]); v = np.log(P['gain'][q, c])
print(len(u), np.corrcoef(u, v)[0, 1], np.var(u-v)/np.var(u))
# F3: compare v5/v5c Y on clean & mask[:, j], target by target.
# Image checks/template: default_rng(340), loop batch (0,1), then stain range(10),
# choose 10 clean rows per pair without replacement. Transform these images;
# average only selected b1 split-0 rows within a stain; score its 10 b0 rows.
# F4 null: default_rng(34), real=normal(size=400), samples=normal(size=(400,8)).
```

Source SHA-256 snapshot:

| File | SHA-256 |
|---|---|
| scripts/fm_prep.py | `5f75446fbf14c7cb251720463ec6067c3b1472fde12c17091f795594f46e8bcd` |
| scripts/fm_common.py | `61c3205b5962a00c6b8ef5de4882a9fa6d8dc4d32d1b3b0dd9466c1574540915` |
| scripts/fm_train.py | `0c7c36456ce25f867f0d3755b8ec19564986ed0d5e640d16687fbe85678b60a7` |
| scripts/fm_eval.py | `7a131f18a5803dc3315f7d43e06dc60ebf17a089709418c37e3578e18677f406` |
| fm_prep.npz | `87932288a744fc62f9e3e36e39e1541a1a6c2b33aec7c1589f17488cae2b9e7f` |
| targets_v5.npz | `676a9fe70aa547358b7f2a3c0405293d79214adec82a3474b9f19d398450c1bb` |

## Handoff and unchecked work

Integration order: resolve new-draw calibration and clean target lineage; validate position correction on disjoint FOVs; add evaluation identity/output checks; establish template/shuffle controls and matched/converged regression evidence; rerun scoring. Likely implementation files are `scripts/fm_prep.py`, `scripts/fm_train.py`, `scripts/fm_eval.py`, and, for strict inductivity, normalization construction in `scripts/led_norm_stats.py`. `fm_common.py` needs changes only if physical-count averaging is added. The notebook should record the estimand and review before conclusions are marked verified. This read-only review supplies no implementation changes.

**Not checked:** Torch/GPU execution, training convergence, full evaluation, real SIGTERM/restart equivalence, CUDA numerical stability, network sampling accuracy, or independent optical registration. No new HGB/ridge model was fitted. No causal estimate of score inflation or proof of biological signal removal was produced. Full-run conclusions await completed, corrected evaluation; the smoke outputs do not supply them. Resume was reviewed structurally but remains runtime-unverified here. Notebook updates and commits are left to the integration agent because this brief authorizes writing only this report.
