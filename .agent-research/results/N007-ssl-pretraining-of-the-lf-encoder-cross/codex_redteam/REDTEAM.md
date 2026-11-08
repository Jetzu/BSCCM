# N007 red-team review

Verdict: **FAIL**. One critical and four major correctness findings. Source files were read only. All source locations below are relative to /home/hpc/v181bb/v181bb10/BSCCM/.

## Numbered findings

1. **Critical — collapsed embeddings receive perfect retrieval scores.**

   **Locations:** scripts/train_bind.py:58-59; scripts/screen_gs_lf.py:22-24.

   Rank counts only candidates with similarity strictly greater than the paired candidate. Every tied candidate is consequently treated as rank one. **Failure:** 100 identical embeddings in each modality produce R@1=1.0, R@10=1.0, and medr=1.0 in both exact source functions, despite containing no pairing information. Chance recalls are 0.01 and 0.10. This can turn representation collapse into apparent successful binding/alignment, including MK retrieval.

   **Minimal fix:** sort candidates by descending similarity with a consistent deterministic tie break, then locate the paired candidate in that ordering. Compute both directions from actual candidate positions. Include constant embeddings as a metric control and report tie frequency.

2. **Major — the near-duplicate mask ignores pairs with different marker sets, usually comparing only autofluorescence.**

   **Locations:** scripts/train_bind.py:53-56, used at :92. The contract at :53 says “same observed marker set.”

   The overlap test checks set intersection, not equality. Every single-antibody cell also observes autofluorescence, so different antibody conditions intersect even without a shared surface-marker measurement. **Failure:** masks [CD14, autofl] and [CD123, autofl], with values [2.8784, -0.5238] and [-0.4069, -0.3228], are ignored at tau=0.25; neither surface-marker value is compared. On 1,024 real mk_tr rows sampled with NumPy default_rng(0), 111,830 ordered off-diagonal pairs were ignored, **103,718 with unequal observed sets**. An exact-source two-row Torch fixture reproduced the erroneous mask.

   **Minimal fix:** require equality of the entire observed-mask vector, then require all observed values to be within the threshold; preserve diagonal positives. If partial-set matching is intended, explicitly define that contract and prevent autofluorescence alone from establishing surface-marker equivalence.

3. **Major — supervised auto-resume can silently relabel old results as a different experiment.**

   **Locations:** scripts/train_lf.py:16, :70-74, :87-89.

   Checkpoints contain no configuration or eligible-row identity, and any existing out/ckpt.pt is loaded without validation. **Failure:** finish an eight-epoch --protocol b0to1 run, then invoke the same output directory with --protocol fov --epochs 8. The restored epoch is already eight, so training and evaluation are skipped; result.json is overwritten with **FOV arguments and row counts but the previous b0to1 metrics**. Changing protocols partway can also carry previously trained-on cells into the new evaluation population. Same-shaped channel reorderings are undetected.

   **Minimal fix:** save and validate protocol, channels/phase, architecture, target identity, training/evaluation rows, and schedule configuration before loading. Reject incompatible output reuse. Save/restore RNG states for reproducible continuation.

4. **Major — target calibration consumes evaluation labels, including the nominally held-out batch.**

   **Locations:** scripts/make_targets_v2.py:17-18, :22-25, :30-32, :36-39; consumed by scripts/train_lf.py:18-24 and scripts/probe_lf.py:152-161.

   Target quantiles, means, standard deviations, and unstained-reference statistics are fitted before splitting, using all rows of each batch. **Failure:** keeping training measurements [1,2,3] fixed while changing the sole evaluation measurement from 4 to 100 changes the training standardized targets from [-1.4758,-0.2806,0.5612] to [-0.8364,-0.5385,-0.3287]. Other evaluation changes also alter q10 and therefore the nonlinear asinh transform. In the actual data, batch-0 CD3 q10 is 0.8618657 using all rows versus 0.8389471 using training rows. Batch-1 calibration consumes 557,379 observed target entries even for b0to1 evaluation. Held-out labels therefore help define the task; batch-transfer scores assume oracle target calibration. Affine z-scoring alone does not change within-batch Pearson, but this does not remove nonlinear calibration dependence or effects on training and RMSE.

   **Minimal fix:** fit calibration on declared training or independent calibration rows, freeze parameters, and apply them to evaluation rows. For batch holdout, reserve independent calibration controls or report an explicitly oracle-calibrated protocol separately from held-out-batch generalization.

5. **Major — screen SigLIP assigns contradictory labels to repeated copies of a positive pair.**

   **Locations:** scripts/screen_gs_lf.py:31-33.

   Sampling uses replacement, but only the batch diagonal is positive. **Failure:** sampled indices [7,7] yield two copies of pair 7 in each modality, yet matching off-diagonal copies are labeled negative. This happens routinely at the advertised pair counts: an independent NumPy sampling check with 300 draws from 300 pairs produced 300 ordered duplicate off-diagonal comparisons, versus 300 diagonal positives. The learning curve confounds number of pairs with false-negative pressure.

   **Minimal fix:** sample without replacement using randperm, capped at the number of available pairs, or derive positive/ignored entries from equality of sampled global pair IDs.

6. **Minor — binding standardization includes the held-out GS slide.**

   **Locations:** scripts/train_bind.py:24-29.

   LF and GS means/standard deviations use all rows before train/test selection. **Failure:** changing an unlabeled held-out GS feature outlier changes standardized training inputs and therefore the learned adapter. All 5,697 batch-1 GS evaluation rows contribute. This is feature-level transduction, distinct from finding 4; the binding description calls the slide held out without declaring this preprocessing access.

   **Minimal fix:** fit GS statistics on training histology rows and LF statistics on the declared training union, then transform test features. Alternatively, explicitly label the protocol transductive and include an inductive control. The screen's all-row PCA fit is explicitly declared transductive in its own documentation.

7. **Minor — returned screen SigLIP maps fail when called.**

   **Locations:** scripts/screen_gs_lf.py:34, called at :49.

   This is an explicit execution blocker, not silent corruption. The no_grad context surrounds lambda creation, not subsequent execution. **Failure:** calling a returned map with ordinary float32 inputs raises “Can't call numpy() on Tensor that requires grad.” A CPU source-function smoke check reproduced it. The full screen cannot reach its final JSON/CSV once it enters this branch; finding 5 still applies independently to its loss.

   **Minimal fix:** detach mapped tensors before conversion to NumPy, or put the executed mapping bodies under no_grad/inference_mode.

## Checked and found correct

- **LF normalization, executed:** RawSet, LFNorm, and GPUNorm on CPU were compared on ten real rows: six random rows plus extrema of both position coordinates. All 23 channels plus phase, bf_dpc4 without phase, and reordered subset [22,1,0,18] plus phase agreed. Maximum absolute difference: **1.9073486e-6**. Flat-field offset subtraction, crop indexing, edge padding/clamping, slide/channel scales, log clipping, and phase concatenation matched. The int16 view is valid under the supplied uint16 <=4095 invariant. Translation follows normalization, preserving raw/flat-field registration.
- **Batched views, executed:** cross-illumination checks with 2, 3, 5, and 6 views on three samples gave disjoint exhaustive per-sample partitions, balanced sizes 12, 8, [5,5,5,5,4], and 4, and zeroed absent data. Augmentation-only dropout on 100 synthetic samples was 0.10521 with no empty view. Forcing all channels to drop exercised the fallback, retaining exactly one channel per view. Captured affine matrices had isotropic scale **0.90266–1.09987**, maximum output translation **5.99025 px**, and zero off-diagonal terms. Blur kernels, gain/noise, and presence planes are correctly indexed per sample/channel. Fixed blur radius three retains tiny Gaussian tails omitted by the reference's adaptive radius; this is semantic rather than bitwise output/RNG equivalence.
- **Feature alignment, source tracing plus metadata checks:** extract uses an ordered non-shuffling loader and contiguous offsets; all-feats supplies arange(N), so exported row equals global index. Sorted union/searchsorted maps probe subsets correctly. Histology exports retain global indices and hist_row. Real hist_row values are a bijection onto 15,712 histology records. Index and scalar-measurement parquet tables have aligned 412,941-row RangeIndexes.
- **Masks, classes, splits, actual data:** observed Y entries are finite; unobserved entries are zero, so masked multiplication does not conceal NaNs in this target file. Histology cells have no surface-marker masks; all 15,712 have autofluorescence masks. Gate column order matches the cls3 map and multiply gated cells are excluded. Every FOV has one split. Default GS training/evaluation cells occupy distinct slides, IDs 2 and 19.
- **Losses and metrics, inspection:** supervised MSE averages observed examples per target, then active targets. Evaluation applies target masks. Within-class Pearson centers both vectors within each class. The test-class-mean oracle is descriptive and does not feed learned heads; the probe explicitly documents test-truth use. Probe scaling is fold-local, primal/dual ridge equations include an unpenalized intercept, and alpha selection uses training folds only. Marker-set padding masks exclude unobserved keys and pooling excludes padded outputs. Shared heads receive normalized projections; distillation stops the LF teacher gradient; direct GS supervision uses a separate raw-feature head.
- **SSL resume and health, inspection:** configuration/row validation, sampler offsets, saved global RNG, and per-step view-generator seeds are consistent. Health evaluation centers encoder features, uses entropy of singular values and off-diagonal feature correlation, and does not update batch-normalization statistics. Screen PCA whitening and Procrustes multiplication have the intended orientation under its declared transductive protocol.

## Validation scope and handoff

Apptainer failed with “fuse: device not found.” Numerical checks instead used the existing /home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python for Torch and /scratch/bsccm_venv/bin/python for parquet/NumPy. Bytecode was disabled, CUDA hidden, and temporary storage restricted to /tmp/v181bb10/codex-N007rt/. CLI source functions were extracted with Python AST rather than importing top-level workloads. Real normalization and synthetic view/metric checks used CPU; numerical libraries were allocated 48 threads for those checks.

No GPU or dataset training run, full feature export, full screen/binding pipeline, checkpoint interruption/restart experiment, or full-pack scan was performed. Normalization equivalence is sampled CPU evidence, not a CUDA/exhaustive guarantee. Resume finding 3 follows directly from control flow; no experiment directory was mutated. Scratch was removed after checks. Only this report was written; notebook updates, fixes, commits, and integration remain with Claude under the report-only scope.

**Repair plan:** correct retrieval and add collapse controls first; repair the marker-set predicate and SigLIP sampling; validate supervised resume identity; define independent calibration/preprocessing populations; repair the screen map execution blocker. Rerun affected evaluations before accepting earlier scores. No source fixes were made.
