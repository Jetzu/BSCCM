# Codex DIAGNOSIS RED-TEAM brief (N059). Write only into this folder (diag_redteam.md); repo read-only; do not fix anything.
Claim (Claude, diagnosis.md): the remaining reachable gap to .70 is a draw-specific READOUT of HLA-DR and the pooled-panel groups
(grp:CD45, grp:CD16); the dim markers CD19/CD56 are label-limited; calibration is not the gap (oracle per-slide affine +.003).
Evidence: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N059-review-3-lf-mk-v10-goal-6862-target-70-a/{attempts.md,
diagnosis.md, n059_diag.py, n059_diag.csv}; GOAL.md, KNOWLEDGE.md (K019, K028-K040). Data: /hnvme/workspace/v181bb10-bsccm
(targets_v5cx_v10.npz, runs/<run>_<b10|b01>/eval_preds.npz, runs/n039_ceil_fov_v5_mt0/eval_preds.npz).
Question: what simpler explanation fits the attempts equally well? Where does the oracle mislead (e.g. the in-distribution model is
v5-trained, sees the held-out draw's labels AND images, has a different training-set composition; per-slide affine is the wrong
oracle for pooled targets with 1-2 slides; Spearman vs Pearson)? You may run small CPU checks with python3 + numpy if available in your
sandbox (no apptainer). Report findings ranked, each with evidence; "NOTHING BREAKS" if none. diag_redteam.md <= 1.5k tokens.
