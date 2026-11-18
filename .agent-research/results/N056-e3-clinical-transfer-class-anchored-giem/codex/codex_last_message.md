Delivered the three requested scripts, Slurm wrapper, supporting modules, teacher manifest, and checksums in this folder.

- [README.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/README.md): exact GPU commands; estimated **4.1–5.8 H100 GPU-hours**.
- [report.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/report.md): fixed protocol, thresholds, reference fractions, and interpretation limits.
- [cpu_smoke.log](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/cpu_smoke.log): all checks passed, including actual source inference and bitwise-identical SIGTERM/resume.

All three existing T checkpoints were verified; their v6c autofl teacher is shared.

**RQ3 limitation:** expert BSCCM classes lack individual-marker observations. Literal calibration returns NA; a separately labeled proxy analysis is implemented.

No GPU jobs or clinical evaluation predictions were run.