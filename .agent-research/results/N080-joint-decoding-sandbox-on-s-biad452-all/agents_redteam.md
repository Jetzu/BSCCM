# N080 red-team: integrated scripts/train_ifc_sandbox.py (Codex + Claude fold-stats block, lines 549-569)

Reviewer: Claude (red-team for Codex-written code). Reviewed 2026-11-24. Scratch: /tmp/n080_rt (analyze.py, oracle.py, suite/tests/test_rt_extra.py).
Real data: compensated intensities read from the 11 S-BIAD452 IDEAS FCS files with pack_sbiad452.read_fcs. These are all events, a superset of the zarr cells. No GPU, no model training.

## Verdict
No leakage into the trained model was found. The claim fails on the evaluation side. The test truth is standardised with the test runs' own statistics. The gate-based metrics are not biological gates. The "cell- vs cluster-level" metric cannot tell within-type decoding from cell-type knowledge.

## Verified OK (first-hand)
- **Tests:** Codex's 31 tests pass against the integrated script. I added 3 invariance tests and all 34 pass.
  - On single_pooled cells, hidden CD3 and CD19 were changed with their sum kept fixed, and hidden CD8 +5000. Run in both inverse and head modes, the model weights are unchanged (allclose 1e-5).
  - Test-donor images were multiplied by 7. The model weights are bit-identical.
- **New block:**
  - c_tr uses training-donor revealed entries only (`_tr_set & masks[:, j]`). Pooled cells contribute nothing.
  - Training and validation runs are z-scored from revealed entries only.
  - chan_scale comes from 5,000 training rows.
  - The block runs before fit_pool_calibration (line 570), so pooled calibration sees the new y.
- **Masks:**
  - Exactly 1 revealed marker per single cell. Per-marker counts on 250k rows are 24.8k-25.3k.
  - The pooled fraction is 0.1006 and is independent of the marker (0.098-0.103 per marker).
  - single and single_pooled share the same marker on non-pooled rows.
  - Masks are a stateless function of the row, so they are stable across resume and subsets.
- **kNN:** the bank is training rows only (`visible_y[tr]`, `masks[tr]`), one bank per marker. Pooled cells never enter it.
- **Donor split:** the regex maps sample1_early and sample1_late (likewise sample2) to one donor, so they are never split. train/val/test are disjoint.
- **Metrics on synthetic cases:** Frobenius = `||corr(P)-corr(Y)||_F` (2.6058 vs numpy 2.6058). Per-marker r, F1 (2/3 known case) and pair deltas are correct. Populations come from TRUE labels with training thresholds.

## MAJOR
1. **The test truth is z-scored per run with its own statistics (lines 559-564, `_is_test[k].all()` branch).** The model never sees it, but it is test-run statistics (the mean and SD of all 9 markers) used to define the target. "No test statistic used" is therefore false.
   - It barely matters for per-marker r within one run. It does affect r pooled over the early and late runs of donors 1 and 2, and all absolute metrics: RMSE, cosine, gate F1 and the DP rate.
   - Measured on FCS, run means differ from sample0 by up to 1.6 SD (CD56 -1.5 to -1.6 in all donor 1/2 runs; CD19 +1.1 to +1.3 in donor 2). Within-run SD ratios range 0.64-1.62.
   - A type-mean oracle (item 3) scores RMSE 0.57 vs 0.99 (test donor 0), 0.46 vs 0.82 (1) and 0.50 vs 0.59 (2), comparing test truth z-scored per run against test truth z-scored with training-donor statistics. Gate F1 also moves: B 0.31→0.45, NK 0.72→0.25.
   - Arm comparisons stay fair, because every arm uses the same target.
   - Fix: report both targets, or make the primary absolute metrics use training-donor statistics. Rewrite the claim as "per-run-standardised target (transductive)".
2. **The gates are fixed quantiles, not biology (lines 232-251).** "Positive" means above the training q75 for every marker, so every marker is forced to 25 % positive.
   - A 2-component GMM on the training donor (donor 2) gives these positive fractions: Siglec8 4.6 %, CD14 8 %, CD19 8 %, CD56 11 %, CD4 16 %, CD8 18 %, CD15 62 %.
   - On test donor 0, the q75 gates applied to the TRUE labels call 15.5 % eosinophils vs 1.5 % neutrophils (biologically inverted) and 19.5 % monocytes.
   - The true CD3+CD19+ rate is 0.06 %, a q75 artefact (with GMM gates it is 1.25 %).
   - Gate F1, the DP rate (the HistoPlexer check) and the coarse populations used for within-population co-expression are therefore not interpretable.
   - Fix: bimodal or valley thresholds fitted on training-donor revealed marginals, a CD45-dim granulocyte gate, and a sanity table of true-label population fractions in metrics.json.
3. **The within-population r does not test decoding beyond cell type (lines 292-304).** This is the N079 question. A type-mean oracle has zero within-type information: it assigns each test cell its training-donor fine-type mean (eos, neut, mono, CD4T, CD8T, T, B, NK), with types taken from its true labels. It scores:
   - overall macro r 0.77 / 0.87 / 0.83 (test donor 0/1/2);
   - "lymphocytes" within-population macro r 0.86 / 0.67 / 0.82;
   - CD3, CD19, CD4 and CD8 within lymphocytes 0.77-0.97.

   The 4 coarse quantile populations mix fine types, so between-type variance counts as "cell-level". Codex's test_cluster_correct_does_not_mean_cell_correct only covers a predictor built on the same 4 populations. The fine types above were defined with ad hoc quantile gates, so treat the oracle numbers as approximate. This blocks any claim of decoding beyond cell type.
   - Fix: add this oracle as a baseline arm (fine gates from item 2). Report the model minus the oracle for per-marker r and the key pairs within fine types.

## MINOR
4. **Stale documentation of the target.** setup.json (line 615) still says "Pack Y: per-sample z scores using all sample cells". The README "Calibration scope" section also says the pack Y and chan_scale are honoured. The fold statistics (c_tr, per-run mu/sd, chan_scale) are only printed (line 569). Write them to setup.json.
5. **Pooled loss weighting (line 429).** The loss is mean(single entries) + 1.0 × mean(pooled entries). With bs 256 there are about 230 single entries vs about 51 pooled entries, so each pooled entry weighs about 4.5x a single entry, varying with the batch's binomial pooled count. Also, single_pooled replaces 10 % of single cells rather than adding pooled ones, so single vs single_pooled confounds the loss of labels with the gain of pools. Document it, or normalise per entry.
6. **Validation pooled targets use the training donor's scale on raw intensities (lines 112-115).** Validation donor raw intensities carry donor offsets (CD56 about 1.5 SD in asinh), so single_pooled checkpoint selection partly measures batch effects.
7. **The cluster-level r is a Pearson over 4 population means, one of them the heterogeneous "other" (line 303).** It is nearly uninformative.
8. **The seed hash adds the seed without SplitMix's gamma (line 58).** Seed s+1 masks equal seed s masks shifted one row, which I verified. Per-cell assignments still differ, so this is harmless but not a fresh draw.
9. **Rotation swaps the shape of the zero padding.** Smoke-pack images have about 38 zero rows and about 56 zero columns, so 90° rotations produce training geometry never seen at test. [unverified impact]

## Not issues
- Using training-donor full truth for gate thresholds is evaluation-only.
- Validation selecting the checkpoint is fine.
- The resume cfg check and the stateless masks are fine.
- The FCS file for sample1_early part3 was truncated earlier but has been re-downloaded (2026-11-24 07:31).
