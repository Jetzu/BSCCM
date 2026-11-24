# Codex RED-TEAM brief round 4 (N086, BSCCM): NEW figures 6-9 + their captions. Hat: red-team (change nothing in the repo).
Writable root = this folder ONLY; scratch /tmp/n086_codex4. No SLURM/GPU.
Files: /home/hpc/v181bb/v181bb10/BSCCM/figures/overview/{make_architecture.py, make_alignment.py, fig6_architecture.*, fig7_architectures.*,
fig8_alignment_crossmodal.*, fig9_alignment_domain.*, CAPTIONS.md (sections Figure 6-9)}, data/alignment_metrics.csv (extracted by a Claude
subagent from the notebook; every row has a source file:line) and data/alignment_notes.md. Notebook items: .agent-research/notebook/items/N0xx-*.md,
result files .agent-research/results/N0xx-*/. Model code: scripts/train_lf.py, bsccm/encoders.py, bsccm/cytoland.py.
## Claim: every number plotted in fig6-9 and every number/sentence in CAPTIONS.md Figure 6-9 equals the source (item text or result file),
the groups are labelled correctly (e.g. held-out slide order b1/b0, teacher pairing, which arm is which method), the architecture description
(input shape, channels, params, heads, losses, training recipe, SSL details) matches the code and job commands, and no comparison is unfair
(different eval sets/protocols presented as comparable without saying so).
## Do not fix anything. Find why the claim is wrong. Check especially: (a) alignment_metrics.csv rows used by make_alignment.py against their
cited sources; (b) fig8 b mixes N045 (ridge readout, v5c truth) with N054 (student evaluation) values -- is that comparison valid as captioned?;
(c) fig9 b post-hoc alignment values vs per-teacher values; (d) fig6 member list / dims / params vs code and the N053/N084 job commands;
(e) overclaims in the 'Says:' lines. Cite file:line with numbers. Say NOTHING BREAKS if you find nothing. Deliver findings.md (<= 350 words).
