# Codex WRITER brief: N039 stage 1 -- label v8 (per-cell background) + pre-registered QC
Repo /home/hpc/v181bb/v181bb10/BSCCM (branch bsccm-data-model; package bsccm/, scripts/). Data /hnvme/workspace/v181bb10-bsccm: READ-ONLY for you (never write there). Write ONLY into your out-dir given below; all scratch (venv, pip/uv caches, temp files) under /tmp/codex_<item>/ . apptainer does not work inside your sandbox (fuse): build a uv venv under /tmp matching the SIF's library versions (numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1, pyarrow 26.0.1; torch only if needed, CPU build). Do not edit repo files; deliver new/patched files in your out-dir; a Claude integrator lands them. Binding project rules: labels may be cleaned, LF INPUTS keep real artefacts (augment, never clean inputs); fluorescence is NEVER an LF-model input; Giemsa cells (hist_row >= 0) never enter any fit/calibration; all label calibration fitted on split-0 rows of the same draw; slide position carries no biology; the held-out draw's images may not be used in any training. Read .agent-research/notebook/GOAL.md, KNOWLEDGE.md (esp. K007, K019, K028) and results/N038-review-lf-mk-common-11-batch-held-out-go/diagnosis.md first.
Out-dir: .agent-research/results/N039-label-v8-re-derive-single-antibody-marke/codex/ . Item: .agent-research/notebook/items/N039-label-v8-re-derive-single-antibody-marke.md (Plan = the contract).
Background: targets come from scripts/make_targets_v3.py (asinh/q10/z calibration of the authors' *_single_antibody_model_unmixed
columns from /hnvme/workspace/v181bb10-bsccm/bsccm_sm.parquet), then make_targets_v4.py (position detrend; use BSCCM_DETREND=xfit,
BSCCM_CLEAN=1) and make_targets_v5.py (mask/per-slide z). Raw per-channel columns exist: Fluor_<ch>_total_raw, _background,
_background_lowess, _shading for ch in 426-446, 500-550, 550-570, 585-625, 627-673, 690-. The authors' unmixing uses the smoothed
background_lowess; in the dim dye channel shared by CD3/CD19/CD56 its error makes 34-49 % FOV-level label variance
(results/N038-*/agents/ab/e_bgmodel.py/.csv; d2_within_fov.csv).
Task:
1. Recover the authors' unmixing map: per marker x draw (single-antibody slides), test whether <marker>_single_antibody_model_unmixed
   is (near-)linear in XS = (total_raw - background_lowess)/shading over the 6 channels (+ the autofluor unmixed column if needed).
   Report fidelity (r, residual sd) on split-0 clean cells. If faithful (r >= .99), v8 raw value = same map applied to
   XL = (total_raw - background)/shading. If not faithful, implement a documented NNLS/least-squares unmixing with the dye spectrum
   from the authors' selection_example_<M>_positive_cells_batch_<b> gate and an autofluorescence basis from same-draw unstained
   split-0 cells (see agents/ab/d_ringlabel.py, e_bgmodel.py for a crude version that UNDERPERFORMED: ens r .24 vs .30 on CD19).
2. Write scripts/make_targets_v8_raw.py (deliver in out-dir) that produces a v3-compatible raw column set so the EXISTING pipeline
   (make_targets_v3 -> v4 xfit -> v5) can run with an env switch (e.g. BSCCM_RAW=v8) -> targets_v8c[x]. Provide the minimal patch
   to make_targets_v3.py as a unified diff in the out-dir (default path unchanged).
3. Pre-registered QC (compute on CPU in /tmp, write qc_v8.csv + qc_v8.md in out-dir), per single target x draw: (a) FOV-level excess
   share (vs 20-permutation null, FOVs >= 3 cells) <= .05; (b) CD19 positive mixture component 1-6 % with d' >= 3 in both draws;
   (c) r(v8, v5) >= .95 for CD16/CD45/CD14; (d) draw-1 lymphocyte stratum (use LF-independent definition: authors' gates or low
   CD16/CD14 label) r(label, own dye) >= 0 for HLA-DR/CD19/CD14. Report PASS/FAIL per criterion; do not tune criteria.
4. Do NOT score any model on v8 (no eval_preds use) -- the QC must be decided before scoring.
Deliver: code, diff, qc_v8.csv/md, and a short report.md (what you did, fidelity numbers, QC table, caveats).
