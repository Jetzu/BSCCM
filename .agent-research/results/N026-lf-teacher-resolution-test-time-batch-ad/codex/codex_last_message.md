Verdict: **C1’s scores reproduce, but the scale-specific advantage is under-supported; dual-zoom collapse has a material BN confound.**

Report: [redteam_N026.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N026-lf-teacher-resolution-test-time-batch-ad/codex/redteam_N026.md).

- Omitted zoom+zoom control scores **.64910**, shrinking the mixed-scale advantage to **.00213–.00265**.
- Estimated single-run seed SD is **.00266**, based on only two seeds.
- Separate-view BN recalibration substantially rescues dual-zoom performance on diagnostic samples.
- SSL pretraining included every evaluation image, requiring a transductive qualification.

Row alignment, scoring, distinct seed checkpoints, evaluation zoom, and resume flag checks passed. The report includes commands, outputs, concrete fixes, and unchecked limits. All artifacts remain inside `codex/`; no scratch remains.