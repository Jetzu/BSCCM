Verdict: C3's arithmetic reproduces, and independent checks support flagging the batch-1 pooled assay; the original 12-target GOAL is not established. C1 overstates the physical diagnosis, and C2 does not establish a common biological autofluorescence truth.

Review scope: read-only source/data review on `bsccm-data-model`. All new artifacts are in this `codex/` directory. No live code, targets, jobs, notebook records, or commits were changed. Source hashes are in `source_manifest.txt`; `train_lf.diff` captures the reviewed diff. The existing N023 item is the record being reviewed; creating another notebook item or committing would violate this review's explicit write boundary.

Reproduction environment: the requested `apptainer exec /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 ...` failed before Python with `fuse: device not found` / `image driver mount failure`. Checks instead used the existing `/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python` (Python 3.11.15, numpy 1.26.4, pandas 3.0.3, sklearn 1.9.0, pyarrow 24.0.0). These were short CPU analyses, with one BLAS/OpenMP thread, no scheduler submission and no GPU. Array comparisons and correlations are small; LF analyses sample 43,010 rows and fit small classifiers. Serial execution of these short checks avoids job/startup overhead; no training was launched.

Run from this directory, using the scripts delivered beside this report:

```bash
mkdir -p /tmp/$USER/codex-N023
export TMPDIR=/tmp/$USER/codex-N023
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
audit_python=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
"$audit_python" audit_data.py > audit_data.txt 2>&1
"$audit_python" audit_independent_lf.py > audit_independent_lf.txt 2>&1
"$audit_python" audit_edge_cases.py > audit_edge_cases.txt 2>&1
"$audit_python" audit_recovery.py > audit_recovery.txt 2>&1
env PYTHONPATH=/home/hpc/v181bb/v181bb10/BSCCM "$audit_python" /home/hpc/v181bb/v181bb10/BSCCM/scripts/rescore_targets.py --runs n014_sslinit_b1to0 n014_sslinit_b0to1 n022_base_b10 --out rescore_original.json > rescore_original.txt 2>&1
"$audit_python" /home/hpc/v181bb/v181bb10/BSCCM/scripts/dye_spectra.py dye_original.npy > dye_original.txt 2>&1
rmdir /tmp/$USER/codex-N023
```

All Python analysis commands completed with exit 0; the container startup attempt failed as described above. Each finding below cites the relevant command from this block and its saved output. Repo-relative source paths below are relative to `/home/hpc/v181bb/v181bb10/BSCCM`.

1. **Blocker for the original GOAL: `.6344` measures 11 targets, and most of the increase is target removal.**

   Evidence: `audit_edge_cases.py`, output `audit_edge_cases.txt:10`:

   ```text
   v4_12              0.5741935109790313
   v4_common11        0.6286892000626177
   v5_common11        0.6343488939441637
   drop_target_gain   0.054495689083586485
   normalization_gain 0.0056596938815459685
   fraction_gain_from_dropping 0.9059154209885678
   ```

   The original rescorer independently gives b1to0 `.6289737833`, b0to1 `.6397240046`, 11 valid targets each (`rescore_original.txt:2`, `:4`; full precision in `rescore_original.json`). Thus 90.6% of the apparent `.0602` uplift comes from excluding A. The other `.00566` is a label-side normalization effect with fixed predictions, not training improvement. On b1to0, **every available test label and every per-target r stays unchanged**, including A at `-.06236`; only eligibility changes (`audit_data.txt:45`). On b0to1, A has no test labels and is correctly skipped rather than scored as zero (`audit_data.txt:47`).

   `.agent-research/notebook/GOAL.md:3` still specifies 12 v4 targets and `:5` specifies `.62`. `scripts/rescore_targets.py:20` even calls the available-target mean `r_mean_12` when only 11 exist. `scripts/train_lf.py:95-98` permits both a 12-available-target mean and a training-eligible mean. These can describe different tasks.

   The N023 item already says “NOT model progress” (`.agent-research/notebook/items/N023-label-qc-per-batch-dye-amplitude-single.md:23`); that part is correct. The blocker is interpreting this new number as satisfying the existing GOAL or carrying its `.62` success threshold forward unchanged. Zero batch-1 A training labels also means a v5 b1to0 regressor has no direct supervision for that output; removing it from a mean does not restore that capability.

   **Fix:** name and freeze a new common-11-target benchmark; report its exact target list, per-target coverage and all historical baselines. Keep the original 12-target GOAL unresolved/unavailable. Report A separately, and propagate the missing capability into downstream 12-target claims. Rename `r_mean_12` to an available-target name. Assess v5 training only against the re-scored fixed-model baseline on identical target sets; re-establish noise and any new success threshold on that benchmark.

2. **Major: the “3–5x dimmer / 4% / therefore residual” physical proof is not supported by the estimator.**

   Evidence: original `dye_spectra.py` invocation above reproduces `dye_original.txt:6`: batch-0 CD123's alleged dye vector is `[4.025, -33.529, -5.653, -19.262, -13.873, -7.817]`. Its norm is dominated by negative changes outside 426–446; the b0/b1 cosine is `-.736` (`:29`). CD56 similarly has cosine `-.149` and an alleged amplitude ratio `4.29` (`:34`). These population contrasts cannot simply be treated as the same fluorophore's brightness across batches.

   `scripts/dye_spectra.py:15-18` and `scripts/label_qc.py:34-40` select positives using the very unmixed estimate under audit, use an arbitrary top 10%, and retain negative spectral components in the norm. Marker-positive prevalence and autofluorescence composition affect the result. The strict `u < percentile(u,40)` also fails on floor ties: CD3 batch 1 has **zero negative rows**, producing NaNs. `audit_data.py` output `audit_data.txt:74` gives `strict bottom40 n 0`, and the original script prints the empty-slice warning (`dye_original.txt:1-2`).

   As a separate route, I used the supplied `selection_example_*_positive_cells_batch_*` spectrum-estimation gates, positive-minus-other-cell means, and nonnegative clipping. The resulting b1/b0 norm ratios are CD123 **.679**, HLA-DR **.291**, CD14 **.413**, CD56 **.710** (`audit_data.txt:64-81`), versus the claimed .28/.19/.36/4.29. This alternative is itself population-dependent, not a calibrated brightness truth; its purpose is to show that the claimed ratios are estimator-sensitive. The scripts do not estimate the claimed “4% of the pooled all-cell channel” using a mixture model, matched prevalence or recovery controls; the assertion appears in N023's item `:20`.

   I also tried simple NNLS spectral refits (`audit_recovery.py`). Batch-1 replicate 0 can show positive mono–lym contrast with alternative bases; replicate 1 changes sign with the chosen A spectrum. These are exploratory, ill-conditioned and not spatially detrended, so **they are not validated replacement labels**. They demonstrate why a vanished canonical contrast cannot establish chemical irrecoverability. Conversely, the released full-model A has substantial signal on CD16-only controls that contain no A antibody: batch-1 q90 **87.81**, versus **41.92** on all cells (`audit_recovery.txt`, `A NEGATIVE/POSITIVE CONTROL` rows). That supports crosstalk concern without proving every all-cell A measurement is residual.

   **Fix:** repair empty/floor-tie handling and distinguish population contrasts from dye spectra/brightness. Recompute using the documented spectrum gates and explicit autofluorescence/background controls; quantify separation, conditioning and negative-control recovery. Withdraw the numerical 4% attribution unless it is calculated. Describe masking as conservative exclusion of an unvalidated assay, not demonstrated absence of marker information. Do not choose a replacement spectral basis by whichever restores the expected contrast.

3. **Major methodological limitation: the original QC is supervised and outcome-informed; its “fluorescence-independent” wording is too strong.**

   Evidence: `scripts/label_qc.py:13,24-28` uses `n022_base_fov` encoder features and fluorescence-derived gate labels. `/hnvme/workspace/v181bb10-bsccm/runs/n022_base_fov/result.json:3-22` records FOV training on `targets_v4.npz`; `scripts/train_lf.py:125-139` trains marker regression and gate classification jointly. Therefore fluorescence labels, including A, shaped the representation even though fluorescence pixels are not inference inputs. `scripts/label_subset_check.py:11-14` repeats this dependence. `.agent-research/notebook/items/N023-label-qc-per-batch-dye-amplitude-single.md:7` explicitly starts from observed negative model performance. `scripts/label_qc.py:23,46-57` examines all splits, including evaluation labels, and presents model r in the same QC table. Fixing CANON before this table is not equivalent to selecting the target and intervention before observing held-out performance.

   **The independent attacks did not overturn the assay warning.** `audit_independent_lf.py` uses the pre-finetuning SSL encoder (`n007_ssl_aug/probe_fov/lf_feats_all.npy`) and fits PCA/logistic only on batch-0 split-0 gated cells. Output `audit_independent_lf.txt:1-7`:

   ```text
   batch-0 held-out-FOV accuracy .91384
   mono>lym d: batch 0 +1.48760; batch-1 rep1 -.14267; rep0 -.29507
   ```

   Separately, the three non-A fluorescence channels alone classify batch-0 gates at `.99725` accuracy and yield A contrasts `+1.21391 / -.44125 / -.60545` (`audit_data.py`, output `audit_data.txt:50-56`). A rank-calibrated variant has the same signs (`:57-63`). These routes exclude A from classifier inputs and remove A-regression supervision as a necessary explanation. They do not provide independently annotated batch-1 cell types, and gates share the unmixing system.

   **Fix:** replace the independence claim with precise provenance and include these independent controls. Record v5 as an exploratory, outcome-informed benchmark revision. Freeze the label-QC and metric rules before a genuinely untouched evaluation cohort; do not infer intent to game metrics merely from this adaptivity. The reviewed evidence supports caution about A, not a claim that the original QC was blind.

4. **Major: per-slide autofl z-scoring is not a validated correction to a common biological truth.**

   Evidence: `scripts/make_targets_v5.py:5-8,25-28` assumes same blood draw makes each slide's marginal median/IQR differences technical. Same origin does not establish equal cell mixtures, equal handling effects or antibody-independent unmixing errors. The transform removes real between-slide shifts/scales as readily as technical ones.

   With SSL-only LF classes, batch-1 lymphocyte/granulocyte autofl means are:

   ```text
   slide       v4 lym / gran       v5 lym / gran
   CD45         .67290 / .23783     .50682 / -.23613
   CD14       -1.20259 / -.92904   -.79848 /  .14573
   unstained1   .31061 / .36269    -.35458 / -.10714
   ```

   Command: `audit_independent_lf.py`; output `audit_independent_lf.txt:8,24,26`. The class relationship reverses between staining conditions and remains reversed after normalization. A common positive slide-scale/offset correction cannot repair such a sign reversal. This may reflect class-dependent contamination, handling/biology, or LF-domain classification errors; this check does not identify which. It rules out treating aligned marginal medians/IQRs as sufficient validation of a shared autofl target.

   All 375,657 autofl values change (`audit_data.txt:9`), including unstained slides. Importantly, each slide's transformation is positive affine: v4/v5 correlations are 1 (`target_diff.csv`, also `audit_data.txt:24-42`). **Within-slide ordering and Pearson r are preserved**; this audit does not show destruction of within-slide biology or artificial improvement of a fixed single-slide GS Pearson score. Cross-slide pooling, relative expression units and regression training weights do change.

   **Fix:** identify the estimand as within-slide standardized autofl unless a calibration control establishes comparability. Keep unstained autofl as a separately reported downstream truth; validate co-stained autofl against appropriate controls before pooling it. Report per-slide and within-class behavior and retain pre-normalization values/calibration parameters. Do not present marginal normalization alone as biological validation.

5. **Minor, conditional on downstream v5 adoption: split-0 calibration is disjoint for LF tests, but overlaps the GS evaluation cohort.**

   Evidence: `scripts/make_targets_v5.py:26-28` correctly uses split 0. LF batch-holdout tests use split 2 (`train_lf.py:23-24`), so I found no split-2 label leakage in this transformation. However, `train_gs_bind.py:20-23,157-158` evaluates **all** histology rows in the held-out batch, not only split 2. `audit_edge_cases.py`, output `audit_edge_cases.txt:11-12`:

   ```text
   batch 0: 10015 GS eval rows; 8007 participate in split-0 calibration
   batch 1:  5697 GS eval rows; 4495 participate in split-0 calibration
   ```

   The new GS-slide mappings are approximately `v5 = 5.86838*v4 - 3.65759` and `v5 = 4.75092*v4 - 1.83027`. This is use of evaluation-cohort labels to define numerical units, not evidence of model-input leakage or inflated single-slide Pearson r; affine invariance protects that r. Calibration overlap already exists in earlier target preparation, and current `train_gs_bind.py:18` still hardcodes v4. Thus this is a limitation of the blanket “leak-free” claim and a migration requirement, not a claim that current GS results secretly changed.

   **Fix:** document a separate calibration cohort for GS, exclude evaluated histology rows from calibration, or explicitly declare transductive target calibration. Define disjoint calibration when reporting MSE, cross-slide pooled scores or other scale-dependent quantities.

6. **Minor current impact, real utility bug: rescore protocol/training eligibility can disagree with training.**

   Evidence: `scripts/rescore_targets.py:10` guesses protocol from substrings, and `:14` counts all rows of a batch, or marks every FOV target trained. `scripts/train_lf.py:20-27,96` instead uses the actual filtered/subsampled training rows. The exact AST expression from the rescorer and a reproduction of the supported `--max-train` selection are exercised by `audit_edge_cases.py`; output `audit_edge_cases.txt:4-9`:

   ```text
   model_copy -> fov
   n023_v5_b01_note_b100 -> b1to0
   max_train=1000: actual eligible markers 0;
     rescorer says fov 12, b1to0 11, b0to1 12
   ```

   A renamed/copy run can silently get a different protocol; the added `b100` substring can even override a real `b01`. Counting batch totals ignores GS exclusion, split selection and `max_train`. **For the three supplied runs**, stored config, evaluation batches/splits and threshold eligibility all match the rescorer's guesses (`audit_data.txt:44,46,48`), so this does not explain the current `.6344`.

   **Fix:** read and validate `result.json` config and saved checkpoint training indices, or save those indices in evaluation artifacts. Apply the new mask to actual training indices and fail on missing/ambiguous provenance. Record an explicit fixed target set for comparable headline metrics.

The target-construction implementation itself passed the independent audit. `audit_data.py` reconstructed the entire v5 Y/mask from v4, using split-0 calibration and independent antibody-index selection, with **float32 exact equality, maximum error 0** (`audit_data.txt:43`). Row/global-index alignment is exact; all original non-Y/mask keys are unchanged; `Y_v4` is exact; all outputs are finite; all masked entries are zero; `qc_masked` exactly equals the 27,074 removals (`:1-5`). All eight single targets are bitwise unchanged; batch-0 grouped targets are unchanged; only the intended batch-1 grouped targets and the 19 autofl slides change. The complete per-target/per-slide audit is `target_diff.csv`. No `is_all`, indexing, accidental-target, masked-zero or split-statistics bug was found.

Integration plan: first fix benchmark naming/coverage and assay-QC claims; next correct spectra estimation and persist evaluation provenance; then validate autofl calibration and downstream migration separately. Real files implicated are `scripts/dye_spectra.py`, `scripts/label_qc.py`, `scripts/label_subset_check.py`, `scripts/rescore_targets.py`, `scripts/train_lf.py`, and, only after the calibration decision, `scripts/make_targets_v5.py` / `scripts/train_gs_bind.py`, together with N023/GOAL/knowledge records. This review supplies evidence and suggested fixes, not implementation changes.

Not checked: GPU training or the pending v5 retrains; checkpoint tensor/feature-cache hash provenance; raw-image re-extraction; independent batch-1 manual cell annotations; experimental proof of which residual components are biological; fresh-isolation generalization; exhaustive downstream GS/MK claims or every historical run. LF classifiers are sampled and were not tuned against model scores. The exploratory spectral refits are sensitivity controls, not a validated assay-recovery pipeline. The requested container environment could not be used. Scratch `/tmp/$USER/codex-N023` was deleted before handback.
