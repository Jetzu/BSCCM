# Codex IDEA-WRITER brief (N079 review 5, BSCCM). Hat: write -- ONLY ideas_codex.md in this folder; repo read-only; scratch in /tmp.
Read: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/GOAL.md, notebook/KNOWLEDGE.md (Headline + K043/K044), /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N079-review-5-lf-mk-v10-goal-6862-target-70-a/{attempts.md,
declined.md,diagnosis.md,n079_joint_consistency.csv}, /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N079-review-5-lf-mk-v10-goal-6862-target-70-a/../N077-external-data-scout-public-label-free-su/{data_scout.md,study.json},
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N079-review-5-lf-mk-v10-goal-6862-target-70-a/../N066-review-4-lf-mk-v10-goal-6862-target-70-a/ideas.md. Code: scripts/train_lf.py (GOAL harness; flags incl. --dense-aux,
--init-full, --train-panel, --aux-label), scripts/train_ssl_lf.py, bsccm/encoders.py.
Context: USER 2026-11-23: decoding ALL markers of a cell jointly is the priority; external datasets ALLOWED; other encoders may be
fully fine-tuned. External candidate verified: S-BIAD452 (ImageStream-X 60x, whole-blood WBC, 12 channels per cell: BF1, BF2, SSC +
9 marker fluorescence images CD15, Siglec8, CD14, CD19, CD3, CD45, CD4, CD56, CD8 -- all markers on EVERY cell, ~98k cells, 3 donors).
Task: >= 6 ideas to raise BSCCM LF->marker decoding (v10 GOAL .6862 -> .70, and joint multi-marker decoding), >= 3 using NEW
information (external data, richer supervision, stronger representation), >= 2 contradicting how the project works now. For S-BIAD452
specify concretely how to bridge the modality gap (pixel-size matching, channel mapping BF/SSC vs 24 LED channels, two stems vs shared
stem, per-dataset heads/affine adapters, which BSCCM targets map to which IFC markers, pooled-panel equivalents), how to validate joint
decoding on S-BIAD452 cells themselves (held-out donor), and the cheapest falsifier. Each idea: mechanism, why it moves the diagnosed
bottleneck, expected gain (reasoned, cite numbers), cheapest falsifier with GPU-h (1 x H100; r50 GOAL run ~.5 GPU-h/direction), build
cost. Respect declined.md (inductive; no lineage/exclusivity priors; eval labels v10). Write ideas_codex.md (<= 2.5k tokens).
