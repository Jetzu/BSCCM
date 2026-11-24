# Codex RED-TEAM brief round 2 (N086, BSCCM). Hat: red-team (attack, change nothing in the repo). Writable root = this folder ONLY; repo read-only; scratch in /tmp/n086_codex2. No SLURM, no GPU.
Round 1 findings: ../codex_round1/findings_round1.md (your previous review). Claude reproduced all of them and changed
/home/hpc/v181bb/v181bb10/BSCCM/figures/overview/make_data.py (hand-checked list C replaces the log-derived progress; per-target models
seed-matched) and make_figures.py (fig2 d seed-matched controls + all N053 arms + assertions; fig3 a seed-matched, d caption; fig4 now
2x3 with per-marker heatmap, all 4 marker pairs, pair-error panel; fig5 teacher-level values, tests paired by teacher / slide).
Figures regenerated: figures/overview/fig1..fig5 .png/.json.
## Claim
Every round-1 objection is now fixed, and the fixes introduced no new error: each value in list C equals the item's own measurement as
recorded in /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N0xx-*.md with the stated kind (single / composite / ensemble)
and label era; no measured common-11 GOAL attempt in N023-N084 is missing from C (other than the stated omissions); every .json note is true.
## Do not fix anything. Find why the claim is wrong: (a) a wrong value/kind/era in C or a missing attempt; (b) a fix that is itself wrong
(seed pairing, teacher pairing, heatmap rows/cols mislabelled, diverging colour scale misleading); (c) a caption sentence the data do not
support; (d) a remaining flattering selection. Cite file and line with the numbers compared. Say NOTHING BREAKS if you find nothing.
Deliver: findings.md (<= 300 words) in this folder.
