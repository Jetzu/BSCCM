# Codex IDEA-WRITER brief (N059 project review). Write only into this folder (ideas_codex.md); repo read-only.
Read: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/GOAL.md, KNOWLEDGE.md, and
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N059-review-3-lf-mk-v10-goal-6862-target-70-a/{attempts.md,declined.md,diagnosis.md,n059_diag.csv}.
Code: scripts/train_lf.py (GOAL harness), bsccm/encoders.py, scripts/make_targets_v*.py.
Task: give >= 5 ideas to raise the v10 GOAL from .6862 to .70, >= 2 of which CONTRADICT how the project works now (e.g. change the
unit of prediction, what the loss sees, the decomposition, the training signal). Respect declined.md and the constraints (inductive:
no held-out-draw labels or images in training; no new data beyond the public BSCCM release; re-basing labels is not progress).
Each idea: mechanism; why it moves the DIAGNOSED bottleneck (draw-specific readout of HLA-DR + pooled-panel groups; label-limited
CD19/CD56; FOV-level label error); expected gain (reasoned, cite numbers from the files); cheapest falsifier with GPU-h (1 x H100;
a GOAL run of the r50 recipe costs ~0.5 GPU-h per direction); build cost. Write ideas_codex.md (<= 2.5k tokens).
