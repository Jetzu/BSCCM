Delivered [analysis](./n064_ceiling.py), [Slurm job](./n064_cpu.sbatch), [README](./README.md), [report](./report.md), and [smoke log](./smoke.log).

The 2,000-cell smoke passed fidelity, curation replay, and preemption/resume checks. The complete aggregate ensemble rescore reproduces **0.686184**.

Smoke pixel-only mean `r_max` is **0.999865**, but local-background reconstruction agrees poorly with actual v10 CD19/CD56 labels. This therefore **does not establish that .70 is reachable**.

The full 69,576-cell pixel job is prepared but unsubmitted; estimated allocation is **1–8 CPU-h**. Exact submission command is in README.