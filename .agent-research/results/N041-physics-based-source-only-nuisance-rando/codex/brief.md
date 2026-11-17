# Codex WRITER brief: N041 -- physics-based source-only nuisance randomisation + consistency in scripts/train_lf.py
Repo /home/hpc/v181bb/v181bb10/BSCCM (branch bsccm-data-model; package bsccm/, scripts/). Data /hnvme/workspace/v181bb10-bsccm: READ-ONLY for you (never write there). Write ONLY into your out-dir given below; all scratch (venv, pip/uv caches, temp files) under /tmp/codex_<item>/ . apptainer does not work inside your sandbox (fuse): build a uv venv under /tmp matching the SIF's library versions (numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1, pyarrow 26.0.1; torch only if needed, CPU build). Do not edit repo files; deliver new/patched files in your out-dir; a Claude integrator lands them. Binding project rules: labels may be cleaned, LF INPUTS keep real artefacts (augment, never clean inputs); fluorescence is NEVER an LF-model input; Giemsa cells (hist_row >= 0) never enter any fit/calibration; all label calibration fitted on split-0 rows of the same draw; slide position carries no biology; the held-out draw's images may not be used in any training. Read .agent-research/notebook/GOAL.md, KNOWLEDGE.md (esp. K007, K019, K028) and results/N038-review-lf-mk-common-11-batch-held-out-go/diagnosis.md first.
Out-dir: .agent-research/results/N041-physics-based-source-only-nuisance-rando/codex/ . Item: .agent-research/notebook/items/N041-physics-based-source-only-nuisance-rando.md (Plan = contract).
Code to read: scripts/train_lf.py (prep(): raw int16 records -> bsccm/gpu_norm.py GPUNorm -> normalised LF; currently only +-4 px
translation), bsccm/gpu_norm.py (log((raw-30)/(background_crop-30)/slide_scale)/s_c), bsccm/data.py (channel identities, LED
families BF / DPC / DF-low / DF-high, phase; note directional DF channels have no rotated partner -> no rotations/flips that break
channel identity).
Task: add opt-in flags (defaults OFF and added to _NEW_DEFAULTS so existing checkpoints still resume):
  --phys-aug P (probability a training sample is perturbed; plan .7), --consist W (weight of prediction-consistency MSE between two
  independently perturbed views; plan .1). Perturbations in the RAW-COUNT domain before GPUNorm (so they act like the instrument):
  per-illumination-family multiplicative gain (log-normal sd .1, shared within a family), small additive offset, Poisson-matched shot
  noise (signal-dependent, using a camera gain estimate you derive and document from source-draw border pixels -- source draw only),
  and defocus approximated by angle-proportional sub-pixel shifts of each LED image (direction/magnitude from the LED's illumination
  angle; +-1 um equivalent; document the geometry you assume, from bsccm/data.py or the BSCCM paper's LED layout). Phase channel:
  leave unperturbed unless physically justified. Eval path unchanged (no augmentation at eval). Keep throughput: GPU ops, batched.
Deliver: patched train_lf.py (full file) + unified diff, any new module (e.g. bsccm/phys_aug.py), a CPU unit test (test_phys_aug.py)
that checks: identity at P=0; gain/shift statistics; no channel mixing; dtype/shape; deterministic under a seed. A short report.md.
Do not run GPU training. Do not change defaults or any other behaviour.
