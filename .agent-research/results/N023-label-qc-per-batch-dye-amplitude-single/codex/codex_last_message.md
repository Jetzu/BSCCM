Verdict: **the assay warning survives independent checks, but `.6344` does not establish the original 12-target GOAL.**

Findings and reproducible evidence are in [redteam_N023.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N023-label-qc-per-batch-dye-amplitude-single/codex/redteam_N023.md).

- **Blocker for the original GOAL:** `.63435` averages 11 targets. About **90.6%** of the increase comes from dropping the pooled target. On common targets, v4 already scores `.62869`.
- **Major:** dye estimates depend strongly on population selection; CD3 batch 1 has an empty negative selection. The claimed “4% residual” attribution is unsupported.
- **Major:** QC was supervised and outcome-informed. However, SSL-only features and independent fluorescence gates also reproduce the problematic batch-1 contrast.
- **Major:** autofl normalization does not establish comparable biological truth across slides.
- **Minor:** downstream GS calibration overlaps evaluation rows; rescore protocol inference and eligibility counting are fragile.

The v5 target file **exactly matched my independent reconstruction**, with no unintended target changes found. Original rescoring numbers reproduced.

No live files changed. Training and fresh-isolation validation were not tested. The supplied container failed; a local analysis environment was used. Scratch was deleted.