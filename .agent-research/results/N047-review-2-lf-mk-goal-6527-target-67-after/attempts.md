# Attempts since review 1 (N038), all scored on the frozen GOAL (v5 labels, common-11, batch-held-out, mean of 2 directions)
best .6527 (N031), noise .0027 (flat zone +-.0054), target .67 (unconfirmed)

| item | scale | what changed | GOAL vs noise |
|------|-------|--------------|---------------|
| N039 | module | label v8 = authors' unmixing with PER-CELL background; pre-registered QC failed -> retrain cancelled; clean ceiling measured (v5-trained FOV model .678) | void |
| N042 | module | low-pass input probe (weak targets drop <= .011) -> weak-marker architecture killed | void |
| N041 | module | physics-based raw-count augmentation (+ consistency, +/- tail weighting) | .6531 ens / .6450 single (+.0004 / +.0025) |
| N040 | paradigm | noisy student: N031 ensemble pseudo labels on all training-draw cells (+ phys aug) | .6564 ens (+.0037, 16/22, p .001); single .6529 (+.0103) |
| N046 | knob | noisy student round 2 (teacher = N040 students) | .6546 (worse than round 1) |
(Not GOAL attempts, running: N044 residual flow image model; N045 SOTAlign Giemsa<->LF alignment.)

**All 5 attempts assumed the GOAL is moved by changing how the per-cell predictor is TRAINED (labels for training, augmentation,
architecture, extra pseudo-supervision) while the thing it is SCORED against -- pooled Pearson r on frozen v5 labels of one draw,
which N038 showed carries FOV-level background error and draw-specific label definitions -- stays fixed and is reachable.**
