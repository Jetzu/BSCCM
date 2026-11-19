Delivered [report.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N049-noise-aware-positive-negative-marker-lab/codex/report.md) and [implementation.patch](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N049-noise-aware-positive-negative-marker-lab/codex/implementation.patch), with soft targets and fit diagnostics for both draws.

Seven CPU tests pass, including bit-identical default training and resume checks. A tiny real-data CPU smoke also passed.

Eta/phi remain prior-dependent; several marker fits are degenerate or weak, explicitly flagged in the report. Repository/data unchanged; no GPU or jobs used.