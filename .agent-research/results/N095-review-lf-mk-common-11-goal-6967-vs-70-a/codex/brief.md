ROLE: (1) idea-writer and (2) diagnosis red-team for a stalled ML project. Read-only analysis; write ONLY two files:
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N095-review-lf-mk-common-11-goal-6967-vs-70-a/codex/ideas.md and /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N095-review-lf-mk-common-11-goal-6967-vs-70-a/codex/diag_redteam.md (each <= 1500 words). Do not modify any other file.
Project: BSCCM - predict 11 surface-marker fluorescence targets per white blood cell from label-free LED-array microscopy images
(24 channels, 128x128). GOAL = mean Pearson r over common-11 targets, train on one blood draw, test on the other, mean of both
directions; inductive (no held-out-draw data in training). Target .70, record .6967, noise .0027, improved = > .7021.
Read: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/GOAL.md (if present), /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N095-review-lf-mk-common-11-goal-6967-vs-70-a/attempts.md, /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N095-review-lf-mk-common-11-goal-6967-vs-70-a/declined.md, /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N095-review-lf-mk-common-11-goal-6967-vs-70-a/diagnosis.md,
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/KNOWLEDGE.md (Headline/K028/K030/K041-K046), /home/hpc/v181bb/v181bb10/BSCCM/README.md (overview sections).
(1) IDEAS: give >= 5 ideas, >= 2 that contradict how we work now. Each: mechanism; why it moves the diagnosed bottleneck; expected
gain (reasoned); cheapest falsifier with GPU-h; build cost. Respect declined.md.
(2) DIAGNOSIS RED-TEAM: "Claim: the bottleneck is the test-draw-specific label/acquisition shift of draw 1. Evidence: attempts.md,
diagnosis.md. Do not fix. What simpler explanation fits the attempts equally well? Where does the oracle mislead? NOTHING BREAKS if
none." Cite file paths/numbers for every claim; you may run read-only python on saved CSVs/npz (apptainer exec -B /hnvme
/hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I; scratch only in /tmp/n095_codex).
