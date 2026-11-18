Delivered all requested v2 scripts, contrasts, house-wrapper sbatch file, and documentation. V1 remains untouched.

- **CPU smoke passed for all three units:** 156.77 seconds.
- **SIGTERM/resume passed:** saved at step 10, exited in 0.75 seconds, resumed to bitwise-identical GS weights.
- Aggregation produced **10 primary, 24 full-grid, and 6 diagnostic rows**.
- Completion includes all controls and **24 per-marker C1 rows**.

Exact launch commands and unit order: [README.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v2/README.md). Changes and verification: [report.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v2/report.md).

GPU estimate targets **7.6 hours** under stated throughput assumptions. Actual CUDA and Slurm execution remain untested.