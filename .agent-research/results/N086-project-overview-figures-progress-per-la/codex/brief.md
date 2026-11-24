# Codex RED-TEAM brief round 3 (N086, BSCCM), NARROW. Hat: red-team (change nothing in the repo). Writable root = this folder ONLY; scratch /tmp/n086_codex3. No SLURM/GPU.
Round 2 findings: ../codex_round2/findings_round2.md. Claude reproduced all six and changed:
- /home/hpc/v181bb/v181bb10/BSCCM/figures/overview/make_data.py list C (per item: BEST single-network arm, seed mean where several seeds,
  + its own ensembles/composites; kind 'multi-network' now holds ensembles AND the N063 composite; N048 on frozen v5 .6522/.6456; N051 holds
  the v10 re-score .6804 + v10 retrain .6788 + base seed 0 .6730; added N040 .6529, N047 .6571, N065 .6929 (captioned post hoc), N026 .6422,
  N031 .6442, N037 .6445, N041 .6450, N046 .6510, N055 .6704, N063 .6766, N068 .6718 (best arm), N084 .6744).
- make_figures.py: fig1 caption; fig4 e = registered 36-pair metric from runs/n080_*_d*/metrics.json pairwise_correlation_bias (abs
  predicted_minus_true, mean over pairs, per donor); fig4/fig5 captions.
## Claim: the round-2 objections are fixed and the new entries/captions are correct. Check ONLY the changed entries and captions against
the notebook items (.agent-research/notebook/items/N0xx-*.md) and result CSVs, and the fig4 e computation. Cite file:line with numbers.
Say NOTHING BREAKS if you find nothing. Deliver findings.md (<= 200 words) in this folder.
