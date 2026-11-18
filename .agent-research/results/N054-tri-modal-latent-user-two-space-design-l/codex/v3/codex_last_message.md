Implemented in v3; v2 and production data remain untouched.

- Primary A1-GS/direct use split-1 latents, LF-pretrained initializers, fixed sigma 0 and 600-step final iterates. Added holdout curves and pre/post tables.
- A2 uses entropic GW; labelled Euclidean secondary rows and identity sanity results remain.
- Added source-validation stacking, plain-MLP controls, and gains against all six controls.
- Added LF overlap audit counts, `-I -B` throughout, and default prefix `n054v3`.

Full CPU smoke **passed in 173.75 s**, including SIGTERM/resume with bitwise-identical weights. GW recovered **100%** of synthetic rotation/permutation identities across four cases. No bytecode directories were created.

Exact sbatch commands: [README.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v3/README.md). Changes, evidence and limitations: [report.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v3/report.md).

Revised estimate: **8.6 GPU-hours total**, unmeasured. No GPU jobs submitted.