Implemented in v2; v1 untouched.

- Configured six H runs, three H_matched runs, and all seven existing T seeds.
- Registered primary contrasts and enforced the student-seed noise floor.
- Added gradient-share matching, alignment controls, both fixed taus, separate RQ3 errors, single-file caches, and 12-worker loading.
- Reproduced and excluded 16 PBC duplicate filenames.

CPU smoke passed, including real SIGTERM resume with bitwise equality for H and H_matched. Artifact hashes also pass.

H_matched matches shares per update for 200 steps, then freezes the mean weight. GPU execution remains pending; the conditional estimate is 5.7 H100 GPU-hours.

Exact commands and limitations: [README.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/v2/README.md). Evidence: [validation.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/v2/validation.md).