Verdict: **headline numbers reproduce, but “clinical parity / trade-off removed” and “GS truth is eval-only” are too strong. No blocker invalidating the within-slide correlations was found.**

Repository paths below are relative to /home/hpc/v181bb/v181bb10/BSCCM, branch bsccm-data-model. Evidence artifacts are in this codex directory. The parent updated the N025 item during review; reviewed_N025_item.md preserves the conclusions reviewed here. reviewed_train_lf.py preserves the HEAD source before unrelated live N026 edits. No live source, notebook, branch, or commit was changed by this review.

Exact reproduction commands, from this directory:

```bash
mkdir -p /tmp/$USER/codex-N025
export TMPDIR=/tmp/$USER/codex-N025
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python audit_N025.py > audit_N025.txt
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python audit_detrend_N025.py > audit_detrend_N025.txt
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python audit_supplement_N025.py > audit_supplement_N025.txt
```

All three commands completed successfully. They only read dataset/repository files and print evidence. Computation was light, single-core CPU, including four spatial-regression sensitivity fits; no GPU submission or neural retraining.

1. **Major — C3 fails upstream of the ridge: nonlinear spatial detrending uses GS autofl values to construct non-GS training labels.**

   Direct exclusions work: scripts/train_gs_bind.py:35,44 requires hist_row < 0; reviewed_train_lf.py:21 excludes GS rows. However, scripts/make_targets_v4.py:15-19 fits HGB on all observed slide/target split-0 rows, including GS. v5 inherits v4 (make_targets_v5.py:12); v6 inherits v5 (make_targets_v6.py:6). GS autofl therefore affects the function subtracted from non-GS labels, which train the LF encoder and autofl ridge. This is more than affine calibration.

   **Command/output:** the detrending command above first reconstructs saved v4 exactly, then refits the same HGB excluding GS. audit_detrend_N025.txt:1-12:

   ```text
   slide 2:  training all/nonGS/GS = 13095 / 5088 / 8007
   saved v4 reproduction max error = 0
   nonGS training-label change RMSE = .063222; original SD = .149539
   slide 19: training all/nonGS/GS = 11099 / 6604 / 4495
   saved v4 reproduction max error = 0
   nonGS training-label change RMSE = .269307; original SD = .210396
   ```

   Old/GS-excluded label correlations on non-GS training rows are .965942/.974668, so these changes are not just offsets/scales. In anchored FOV evaluation, split-2 GS values themselves were **not** used in the spatial fit: the demonstrated problem is indirect supervision from training-cohort GS truth, not demonstrated per-cell test-label leakage.

   The added v6 mean/SD also includes GS (make_targets_v6.py:10-11). audit_N025.txt:11-12 finds 8,007/13,095 b0 and 4,495/37,251 b1 calibration rows are GS. B0 mean/SD changes from .106632/.896903 with GS to .082399/.877553 without GS. This operation is affine per batch and cannot inflate fixed within-slide Pearson by itself, but changes training units/loss weights.

   Cross-slide evaluation includes *all* held-out-batch GS rows (train_gs_bind.py:23-24), including those split-0 rows used in normalization **and nonlinear detrending** (audit_supplement_N025.txt:3-9). That evaluation cohort is not untouched by target preprocessing.

   **Scope control:** fixed seed-0 predictions rescored against GS-excluded spatial truth still show a large gain: b0 v4 .381554 versus TT6 .575664; b1 .683412 versus .834996 (audit_detrend_N025.txt:5-6,11-12). This sensitivity control does not replace retraining.

   **Concrete fix:** fit all target calibration/spatial models on a declared non-GS calibration cohort, rebuild targets, and retrain teachers/ridges/students. Until then disclose GS-truth use in preprocessing. Real files: scripts/make_targets_v3.py, make_targets_v4.py, make_targets_v5.py, make_targets_v6.py. Upstream v3 also calibrates on split 0 without a GS exclusion (:15,33-41). Keep within-slide Pearson.

2. **Major — C1's clinical parity is unproven with three student seeds; the stated success criterion is not met literally.**

   reviewed_N025_item.md:15 requires autofl at least v5 and clinical performance at least v4 within seed spread; :33 declares parity/removal.

   **Command/output:** audit_N025.py independently recomputes all 12 fine AUROCs from saved MLL23 predictions, before JSON rounding. audit_N025.txt:109-118:

   | Student seed | v4 MLL23 mean | TT6 mean | TT6 minus v4 |
   |---|---:|---:|---:|
   | 0 | .613068 | .609088 | -.003980 |
   | 1 | .646384 | .622187 | -.024197 |
   | 2 | .606815 | .601631 | -.005183 |

   Means are .622089 versus .610969; difference -.011120. A descriptive paired 95% Student-t interval across these three seeds is **[-.039293, +.017052]**. This neither establishes equivalence nor proves a significant loss. “Within one SD” is not an equivalence test; no numerical noninferiority margin is registered. gs_autofl_within.py:25 reports population SD: TT6 sample SD is .010406 versus population .008497; v4 .021271 versus .017368. These seeds vary the student only, with one fixed LF teacher per version.

   The mean also hides losses: TT6/v4 band CD16 .577958/.620024, eos CD16 .736143/.774618, myelocyte CD16 .761684/.819925 (audit_N025.txt:124-133). These are descriptive differences, not separately powered significance findings.

   The autofl threshold fails on b1: TT6 .854665 versus v5 .867118, below v5 in all corresponding seeds (audit_N025.txt:30-39,60-69,79-81). TT6 improves b0 and greatly improves both slides versus v4; it does not dominate v5 on both slides.

   **Concrete fix:** report “substantially recovers the clinical mean while improving autofl relative to v4; clinical noninferiority unproven.” Prespecify a numerical margin, replicate enough independent seeds/teachers, and report paired uncertainty and individual contrasts. Fresh external validation is needed for finding 3. This is a reporting/validation change, not an aux arithmetic repair.

3. **Major — MLL23 is now an adaptively reused benchmark, although contrasts were fixed and images were not used in gradient training.**

   The chronology is explicit: N022 reports clinical results/teacher choices (.agent-research/notebook/items/N022-fine-grained-teacher-rare-population-tai.md:15-20); N024 observes the v5 clinical loss and proposes two teachers (N024-v5-label-r18-supervised-teacher-for-the.md:19,22,28); N025 aims to remove that observed loss (reviewed_N025_item.md:7). Prior MLL23 outcomes therefore informed the next architecture, recommended after testing on the same benchmark.

   **Command/output:** nl -ba on those item files, plus audit_N025.txt:89-104: every arm has the same class sequence and 14,533 cells. apply_gs_student.py:75-78 deterministically chooses seed-0 subsets capped at 1,500/class. Exactly 12 intended fine contrasts, with the stated signs, enter each mean; the three lineage references are excluded. Fixed endpoints do not restore an untouched model-selection test.

   **Concrete fix:** call MLL23 a reused development benchmark for N025; freeze the chain/endpoints before evaluating unseen patients or another source. More seeds on these cells measure optimization variation, not fresh clinical generalization. I found no evidence that MLL23 images/labels entered the gradient loop.

4. **Minor — “cleanest target” overstates a change in supervision population, scale, and representation; improvement is slide-specific.**

   make_targets_v6.py:8-11 removes stained-slide labels and rescales retained values; it does not independently quantify or repair their noise. Retained v4/v5/v6 labels are positive affine transforms per slide (audit_N025.txt:14-18), with identical within-slide ordering.

   **Command/output:** audit_N025.txt:5-10 and audit_supplement_N025.txt:11-17:

   ```text
   observed autofl: v5 375657; v6 63067; removed 312590
   actual autofl ridge fit: v5 60000 rows; v6 37844
   v6 ridge rows by batch: b0 5088; b1 32756
   total LF FOV training/eval rows unchanged: 317935 / 39435
   ```

   Fewer/different autofl-supervised cells are real; fewer *total LF training cells* are not the explanation. The representation is also retrained with a changed multitask loss. v6 improves GS b0 .579361 versus v5 .509999 but decreases b1 .846356 versus .867118 (audit_N025.txt:76-82). Identical evaluation rows and affine invariance exclude a scoring-scale trick, not training-scale/population effects.

   GOAL does reproduce: v6 .638976 versus v5 .640352, matched configs and bitwise-identical marker targets/masks (audit_N025.txt:4,148-161). “Flat at the recorded noise scale” is supported; exact invariance is not.

   **Concrete fix:** say “unstained-only supervision improves b0 prediction under this configuration.” To claim cleaner supervision causally, hold features/target scales fixed, use matched-count and batch-balanced ridge controls, and ablate LF retraining separately. No assay replacement is warranted from these scores alone.

5. **Minor, latent interface bug — auxiliary columns/rows use primary-file metadata without validation. Supplied files are aligned, so current N025 numbers survive.**

   train_gs_bind.py:40-45 loads auxiliary Y/mask, derives j from primary names, and uses primary split/hist_row/gidx. Reordered auxiliary columns silently select the wrong target; reordered rows can defeat exclusion relative to that auxiliary file. :42 also reuses primary stained_tr for auxiliary feature normalization without asserting row correspondence.

   **Command/output:** audit_N025.txt:163: rolling auxiliary columns makes requested autofl select grp:CD16 at primary index 12, while auxiliary autofl is index 0. For current files names, split, hist_row, slide, batch and FOV arrays are exactly equal (:2-3), so this is a future-use bug, not evidence that these aux fits include GS rows.

   **Concrete fix:** resolve a separate auxiliary target index from TA['names']; validate row identities, split, hist_row and feature length before fitting, or explicitly reject mismatches. Add a meaningful permutation/misalignment rejection check. Real file: scripts/train_gs_bind.py.

6. **Major for marker-estimate interpretation — four autofl underestimates do not establish that CIA marker estimates are conservative.**

   reviewed_N025_item.md:23 infers marker conservatism from four autofl examples. scripts/gs_mk_estimates.py:12-15 multiplies correlations; no bound fixes its bias direction when conditional independence fails. Autofl/marker residual covariance can differ; marker teacher accuracy also comes from a different population.

   **Command/output:** the exact orthogonal-vector counterexample in audit_supplement_N025.py gives (audit_supplement_N025.txt:21):

   ```text
   r(S,T)=.8; r(T,M)=.8; product estimate=.64; actual r(S,M)=.28
   ```

   Take unit-variance orthogonal X,E, T=X, S=.8X+.6E and M=.8X-.6E. The product can overestimate; the opposite autofl bias does not rule out this marker error structure. This does not refute measured autofl scores.

   **Concrete fix:** remove “conservative”; label marker estimates assumption-dependent with unknown bias direction. Report covariance sensitivity or valid bounds. Clinical class AUROC does not supply missing same-cell GS-marker accuracy. The immediate change is to the interpretation, not product arithmetic.

7. **Minor, pre-existing and outside headline correlations — FOV error-bound ingredients mix batches.**

   train_gs_bind.py:175-181 chooses tb from the first GS test row and estimates teacher error a on that batch, but computes student-teacher error b across *all* GS test rows. Anchored FOV runs contain both batches.

   **Command/output:** audit_supplement_N025.txt:19 gives tb=0 while TT6 has 1,014 b0 and 609 b1 GS cells. Those a/b populations do not match.

   **Concrete fix:** compute ingredients and target variance per batch/slide, or disable multi-batch FOV bounds. Current aux autofl is index 12, outside range(12), so the added AUX skip has no effect for the actual aux target. It is correctly positioned for hypothetical auxiliary marker columns. No effect on reported within-slide Pearson or MLL23.

Checks that survived:

- **Rows/metric:** all 15 anchored runs contain the expected split-2 GS rows, ordered by hist_row, on slides 2 and 19 (1,014/609 cells). CSV within-slide numbers reproduce to about 1e-15. Changing truth to v5/v6 affects fixed-prediction correlations only at float32 roundoff. Fixed v4 seed-0 pooled r instead changes .254959 to .476855 from v4 to v5 truth (audit_N025.txt:20-29,70). The within-slide metric is justified.
- **Aux mechanics:** primary marker teacher predictions are bitwise unchanged in TT5/TT6 (audit_N025.txt:86-87). Aux overwrites precede training, and distillation uses teach, not measured GS Y (train_gs_bind.py:45,211-212). Using primary non-GS stained_tr as the feature scaler is permissible for these aligned files; that choice itself does not leak GS truth.
- **Teacher/probe exclusion:** all r18 teachers record include_gs=0, including /hnvme/workspace/v181bb10-bsccm/runs/n025_r18_v6_fov/result.json. Probe train/test rows contain zero GS cells (audit_N025.txt:83-85). probe_lf.py:162,180-190 freezes the encoder, uses eval mode, and exports globally ordered features without probe labels feeding back into it. bsccm/gpu_norm.py:12-14,30-32 supplies LED/phase, not fluorescence pixels.
- **Fair comparisons:** all anchored configs match on 3,000 steps, batch size, LR, freeze/dim, augmentation, anchor weight .5, PBC half A, FOV holdout and DANN=0; final metrics are step 3,000, not selected intermediate checkpoints (audit_N025.txt:20-69). Historical v4 source hardcodes targets_v4, explaining omitted args. Cross-slide configs use 2,500 steps, no anchors, seed 0 (audit_supplement_N025.txt:2-9).
- **Aggregation:** all recomputed MLL23 AUROCs match published rounding within .0005 (audit_N025.txt:105); no below-chance fine contrasts are omitted. apply_gs_student.py is byte-identical to its N022 commit: the commands git show a7a8676:scripts/apply_gs_student.py | sha256sum and sha256sum scripts/apply_gs_student.py both return 063c2a3e20b24be6a3db6c2560538b6da2b87a71d272d24f079601fb49820009.

Handoff plan: keep the reproduced metric table; qualify parity, cleanliness and CIA language; declare benchmark reuse; exclude GS from upstream calibration/detrending and rerun the chain; add aux-schema validation and per-batch bound handling. Relevant real files are named per finding. Notebook integration/commits remain the integration agent's responsibility under this read-only brief.

Not checked: neural retraining/GPU inference; raw-image re-extraction; checkpoint/feature-cache hash provenance; individual clinical-image identities or patient-level independence (saved arrays lack patient IDs); original chat pre-registration timestamps; independent assay noise/contamination; new-isolation validity. The v4 b1-to-b0 reference n013_gsbind_v4_x0_s0 has no test_preds.npz, so .322523 was checked only against result.json, not recomputed (audit_supplement_N025.txt:2). Spatial rescoring is not a clean retrain. No fixes were landed. Scratch /tmp/$USER/codex-N025 was deleted before handback.
