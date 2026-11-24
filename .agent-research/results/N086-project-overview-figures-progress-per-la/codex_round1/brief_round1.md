# Codex RED-TEAM brief (N086, BSCCM). Hat: red-team (attack, change nothing in the repo). Writable root = this folder ONLY; repo read-only; scratch in /tmp/n086_codex. No SLURM, no GPU.
## Claim
The five overview figures in /home/hpc/v181bb/v181bb10/BSCCM/figures/overview/ (fig1_progress .. fig5_giemsa, each .png + .json sidecar) faithfully show the
project's measured results: every plotted number equals the value recorded in its source file, labels/groups are attached to the right numbers,
the controls are the right seed-matched controls, and the caption notes in each .json (notes, stats) are true.
## Evidence
- Data builder: /home/hpc/v181bb/v181bb10/BSCCM/figures/overview/make_data.py (writes figures/overview/data/{progress,per_target,mm3}.csv; run in container
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif with `apptainer exec -B /hnvme <sif> python3 ...` -- do NOT re-run it, read its outputs)
- Plotter: /home/hpc/v181bb/v181bb10/BSCCM/figures/overview/make_figures.py (pfig library: ~/.claude/skills/paper-figures/pfig)
- Sources: the CSVs named in each .json "source" field under /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N0xx-*/ ; item values in
  /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/INDEX.md and notebook/items/N0xx-*.md (value: field; goal_effect).
- Facts to check against: registered best GOAL .6862 (N053); N084 pre-registered ensemble .6911; N083 MM3 seeds ConvNeXt-S .7735/.7688 vs ResNet-50 .7563/.7590;
  N080 macro r full .596 / single .526 / pooled .542 / type oracle .814; CD4-CD8 in T true -.81, full -.36, single -.07, pooled +.16; N082 50%/10ep .6460,
  50%/20ep .6575, 100%/10ep .6730; label-version eras: v4 = N007-N022, v5 = N023-N047 and N049-N050, v10 = N048 and N051+.
## Do not fix anything. Find why the claim is wrong:
(a) a plotted value that does not match its source (wrong column, wrong arm, wrong seed pairing, deltas against the wrong control, mean vs per-seed mixups);
(b) mislabelled groups/eras/directions (e.g. draw 0->1 vs 1->0 swapped in fig3 b; b10/b01 mixed), or a panel whose "control" is not seed-matched;
(c) a statistic in the .json that is computed on the wrong values (paired test on unmatched seeds, deltas tested instead of absolute values, Holm family wrong);
(d) a flattering selection: an attempt that exists in INDEX.md with a measured GOAL but is silently missing from fig1/fig2, or a best-of selection presented as typical.
Cite file and line per objection, with the exact numbers you compared. Say NOTHING BREAKS if you find nothing.
Deliver: findings.md (<= 400 words) in this folder.
