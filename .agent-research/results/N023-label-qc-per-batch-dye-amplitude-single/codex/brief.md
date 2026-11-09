# Red-team brief (adversarial reviewer, read-only analysis; do not edit repo files)

You are the red-team for item N023 of the BSCCM project (repo /home/hpc/v181bb/v181bb10/BSCCM, branch bsccm-data-model).
Your cwd is the only place you may write: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N023-label-qc-per-batch-dye-amplitude-single/codex/
Scratch, pip caches, venvs, temp files: /tmp/$USER/codex-N023 ONLY (mkdir it). Never write to results/ outside codex/, never to /hnvme.
Python with numpy/pandas/sklearn: `apptainer exec /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 ...` (set PYTHONPATH=/home/hpc/v181bb/v181bb10/BSCCM for `bsccm.*` imports). Login node: keep each computation < 10 min and light (sample rows).

## The claim under attack
Context: BSCCM blood-cell dataset; label-free (LF) LED-array images -> surface-marker (MK) regression. GOAL = batch-held-out mean Pearson r over 12 markers (train batch 1 test batch 0 and vice versa). Notebook item: .agent-research/notebook/items/N023-label-qc-per-batch-dye-amplitude-single.md (read it first).
N023 claims:
 C1. On batch-1 'all' slides the pooled label grp:CD123/HLA-DR/CD14 does not measure its markers (unmixing residual), so masking it is a legitimate label fix, not metric gaming. Evidence: scripts/label_qc.py (LF-class contrast mono>lym d +1.21 batch 0 vs -0.08/-0.28 batch 1), scripts/dye_spectra.py (dye 3-5x dimmer in batch 1, shares the 426-446 channel with bright CD45/CD16).
 C2. Per-slide robust normalisation (scripts/make_targets_v5.py) of grp:* on the two batch-1 'all' slides and of autofl on all slides removes technical offsets (same blood draw per batch) and is leak-free (stats on split-0 rows).
 C3. Re-scoring v4-trained models on v5 labels (scripts/rescore_targets.py -> rescore_v4_models.json) gives GOAL .6344 vs .5742 on v4; this is a re-based yardstick, not progress. The 'valid' mean excludes targets with < 1000 training labels in the run's train batch (train_lf.py r_mean_markers_valid, see `git diff scripts/train_lf.py`).
Files: scripts/label_qc.py, scripts/dye_spectra.py, scripts/label_subset_check.py, scripts/make_targets_v5.py, scripts/rescore_targets.py, scripts/train_lf.py (diff), outputs in the N023 results dir (label_qc.csv, dye_spectra.txt, subset_check.txt, rescore_v4_models.json). Data: /hnvme/workspace/v181bb10-bsccm/{targets_v4.npz,targets_v5.npz,bsccm_sm.parquet,bsccm_index.parquet}; LF features runs/n022_base_fov/probe/lf_feats_all.npy (row = global index); eval preds runs/n014_sslinit_b1to0/eval_preds.npz (P, G).

## Your ONE task
Try to break C1-C3. Look for: (a) circularity - is the LF-class assignment contaminated by fluorescence (the LF teacher n022_base_fov was trained on v4 fluorescence labels incl. this target; does that bias the class means?), (b) a batch-1 LF-classifier failure (domain shift) that would fake the flipped contrast - check with an independent route, e.g. the grp:CD16/CD45/CD3-19-56 channels as gates, (c) whether the masking/normalisation was chosen by looking at model performance (metric gaming), (d) bugs in make_targets_v5.py (is_all, masks, z-scoring, split-0 stats, Y[~mask]=0, any target accidentally altered that should not be - diff v4 vs v5 per target per slide), (e) bugs in rescore_targets.py / r_mean_markers_valid (protocol inference from the run name, 'valid' rule, the b0to1 case where the masked target has no test labels), (f) whether per-slide normalisation of autofl destroys real biology or leaks (autofl is the ground truth for a downstream Giemsa student on unstained slides), (g) anything else that would make "GOAL .634 on v5" misleading.
Reproduce numbers yourself where possible; cite file:line or the exact command for every finding.

## Contract
Write findings to codex/redteam_N023.md: numbered findings, each with severity (blocker/major/minor), evidence (command + output), and a concrete fix. Final message <= 600 words summarising the findings. Do not modify any file outside codex/. Before finishing, delete /tmp/$USER/codex-N023.
