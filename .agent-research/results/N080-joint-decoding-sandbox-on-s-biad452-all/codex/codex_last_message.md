Delivered `train_ifc_sandbox.py`, `tests/`, `README.md`, and `codex_last_message.md`.

All requested training arms, joint metrics, KNN baseline, and checkpoint/resume support are implemented. **31 CPU tests passed**, plus an independent CLI smoke run.

The supplied smoke pack lacks donor 2, so integration tests used synthetic fixtures. No GPU runs or SLURM jobs; repository and real data remain untouched.