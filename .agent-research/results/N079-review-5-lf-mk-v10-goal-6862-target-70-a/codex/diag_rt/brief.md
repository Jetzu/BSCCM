# Codex RED-TEAM brief (N079 diagnosis). Hat: red-team -- change nothing in the repo; write only diag_redteam.md here; scratch in /tmp.
Claim (/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N079-review-5-lf-mk-v10-goal-6862-target-70-a/diagnosis.md): "LF->marker decoding on BSCCM is limited by the information in its labelled set plus evaluation-label noise;
the model decodes a JOINTLY CONSISTENT per-cell profile (single-marker outputs composed linearly reproduce the measured pooled channels
as well as the pooled head: .778 vs .783); progress needs new information (external all-marker data S-BIAD452, richer supervision,
stronger representation)."
Evidence: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N079-review-5-lf-mk-v10-goal-6862-target-70-a/{attempts.md,diagnosis.md,n079_joint_consistency.py,n079_joint_consistency.csv}; K033/K043 in notebook/KNOWLEDGE.md;
predictions /hnvme/workspace/v181bb10-bsccm/runs/<run>/eval_preds.npz, labels targets_v5cx_v10.npz. Python: apptainer exec --bind
/hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I (or a /tmp venv if FUSE fails).
Do not fix anything. Find why the claim is wrong: (a) is "jointly consistent" vacuous -- would independent per-marker predictors (no
shared embedding) or a cell-type-only predictor give the same composed r? Build the control (e.g. composed r from a 3-class lymph/gran/mono
mean predictor, or from shuffled-within-class single predictions). (b) Is the data-limit reading supported, or could the plateau be
optimisation/representation (N078 pending)? (c) Does the DeepIFC comparison hold (different labels, magnification, metric)? (d) Is the
S-BIAD452 bet likely to transfer at all (modality gap)? Reproduce numbers; severity per finding; NOTHING BREAKS if none. <= 2k tokens.
