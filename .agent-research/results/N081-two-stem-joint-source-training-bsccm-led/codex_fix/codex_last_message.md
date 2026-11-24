Completed in `codex_fix`; repository unchanged.

- Added per-source BN buffers with shared affine weights, checkpoint serialization, and strict resume.
- Fixed donor exclusion for real early/late run names.
- Added masked training-row affine initialization, isolated IFC RNG streams, correct encoder LR grouping, vectorized D4 augmentation, and source gradient/clip logging.
- Added training-donor-only packer statistics and provenance.

**22/22 CPU tests pass**, including N074 byte-equality checks against baseline `4cf89b09…`, BN/RNG isolation, nonzero-IFC weight-replay checks, and strict resume rejection. Your v10 SD figures reproduced.

Delivered `train_lf.py`, `train_lf.diff`, `pack_sbiad452.py`, `pack_sbiad452.diff`, tests, logs, and provenance. See [README.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/codex_fix/README.md) and [final checks](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/codex_fix/tests/final_checks.log).

Existing packs need rebuilding with the revised statistics scope. GPU behavior and transfer accuracy remain untested.