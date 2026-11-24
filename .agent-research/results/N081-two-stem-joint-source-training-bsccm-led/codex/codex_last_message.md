Delivered [train_lf.py](train_lf.py), [train_lf.diff](train_lf.diff), [tests/](tests/), and [README.md](README.md) with exact b1to0/b0to1 commands.

All **13 CPU tests passed**, covering flag-off byte equality, two-step runs in both directions, donor exclusion, shared-trunk gradients, IFC evaluation artifacts, and strict checkpoint loading. BSCCM preparation, forwarding, and evaluation remain unchanged.

The repository stayed read-only. No GPU or SLURM jobs ran. README documents the existing packer’s all-donor normalization caveat; GOAL improvement remains untested.