# Codex WRITER brief: N064 -- noise-ceiling bracket for the v10 GOAL (CPU analysis; decides whether .70 is reachable)

Repo /home/hpc/v181bb/v181bb10/BSCCM READ-ONLY; deliver into THIS folder: n064_ceiling.py, n064_cpu.sbatch (partition preempt_cpu,
48 cores, wrapped exactly like /home/hpc/v181bb/v181bb10/BSCCM/scripts/cpu_job.sbatch -- read it and ~/.claude/skills/work-log/bin/preempt-run.sh;
python3 -I -B inside /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif; outputs to /hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling/),
README.md (exact command, CPU-h estimate), report.md, a smoke log on <= 2,000 cells. Scratch /tmp/codex_n064/. No apptainer in your sandbox:
smoke with a uv venv (numpy, pandas, scipy, sklearn) on memmapped arrays.

Context: GOAL = mean Pearson r over 11 targets (8 single-antibody markers + 3 pooled groups, not grp:CD123/HLA-DR/CD14) between LF-image
predictions and labels Y in /hnvme/workspace/v181bb10-bsccm/targets_v5cx_v10.npz on the held-out draw's split-2 cells (both directions).
Best .6862, target .70. Labels = authors' unmixed fluorescence intensities curated by our pipeline (scripts/make_targets_v3.py ->
v4.py (detrend, BSCCM_DETREND=xfit) -> v5.py; BSCCM_COFACTOR map for v10 = CD123/CD14/CD19/CD3/CD56/HLA-DR 500, CD16/CD45 150 --
see .agent-research/results/N051-label-v10-per-marker-asinh-cofactor-pre/). Raw per-cell 6-channel fluorescence crops are in the pack
(see bsccm/data.py, scripts/pack.py, scripts/extract_backgrounds.py, and N039 .agent-research/results/N039-label-v8-re-derive-single-antibody-marke/
which re-derived per-cell intensities from pixels with the authors' spectra / unmixing; reuse that code path).
Task: for every labelled cell on the GOAL evaluation cells (and a matched sample of training-draw cells), compute TWO independent
estimates of the label from disjoint pixel halves (checkerboard partition of the foreground; background from disjoint halves of the
background region as in N039), push each half through the SAME unmixing and the v10 curation (cofactor, per-slide normalisation;
detrending may be skipped if it needs model fits -- state it) and report per target and direction: split-half r between the two halves,
Spearman-Brown full reliability rho = 2r/(1+r), r_max = sqrt(rho), and the GOAL-style mean of r_max. Also: (b) noise floor from
unstained cells (label spread on cells that carry no antibody, if the targets file provides them), (c) FOV-level label error share
(variance of FOV means of residual y - ensemble prediction, predictions in runs/n048_v9_base_<b10|b01>/eval_preds.npz +
n048_v9_zoom2 + n053_cytoland averaged; G = global row ids). Pre-registered reads (write them in report.md before computing):
optimistic mean r_max <= .72 -> .70 is near-ceiling; CD19/CD56 r_max >= .75 -> "label-limited" refuted. Note the bias directions:
pixel split-half captures photon/pixel noise only, not shared background/FOV/stain-lot error -> reliability is an UPPER bound.
Handback <= 1k tokens.
