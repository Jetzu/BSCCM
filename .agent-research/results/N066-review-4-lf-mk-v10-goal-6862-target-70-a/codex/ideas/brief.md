# Codex IDEA-WRITER brief (N066 project review 4). Hat: write -- but ONLY ideas_codex.md in this folder; repo read-only; scratch in /tmp.
Read: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/GOAL.md, notebook/KNOWLEDGE.md, and /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/{attempts.md,declined.md,diagnosis.md,
n066_perslide.csv,n066_fovresid.csv,n066_altlabels.csv}; previous review ideas: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N059-review-3-lf-mk-v10-goal-6862-target-70-a/ideas.md.
Code: /home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py (GOAL harness), bsccm/encoders.py, bsccm/heads.py, scripts/make_targets_v*.py,
N064 label reconstruction code in /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N064-noise-ceiling-bracket-for-the-v10-goal-p/codex/.
Task: give >= 5 ideas to raise the v10 GOAL (registered best .6862, descriptive 6-way .6929) to .70, >= 2 of which CONTRADICT how the
project works now (unit of prediction, what the loss sees, the training signal, the decomposition). Respect declined.md and the
constraints (inductive: no held-out-draw labels or images in training; no data beyond the public BSCCM release; evaluation labels
stay v10 -- re-basing labels is not progress, but TRAINING targets may be anything derived from training-draw data).
Each idea: mechanism; why it moves the DIAGNOSED bottleneck (background-model label error, FOV-clustered, partly independent between
background models; b01 pooled groups + HLA-DR; weak targets label-starved); expected gain (reasoned, cite numbers from the files);
cheapest falsifier with GPU-h (1 x H100; one GOAL run of the r50 recipe ~0.5 GPU-h per direction, Cytoland ~1.1 h); build cost.
Write ideas_codex.md (<= 2.5k tokens). Do not run training jobs.
