# Codex WRITER brief: N049 noise-aware positive/negative marker labels (BSCCM)
Repo /home/hpc/v181bb/v181bb10/BSCCM (read-only for you; deliver new/patched files in .agent-research/results/N049-noise-aware-positive-negative-marker-lab/codex/ only). Data /hnvme/workspace/v181bb10-bsccm
read-only. Scratch ONLY /tmp/codex_n049/ (uv venv, torch CPU). No GPU; deliver a CPU unit test + smoke on tiny data. Do not submit jobs.
Contract: item .agent-research/notebook/items/N049-noise-aware-positive-negative-marker-lab.md (Plan). Read GOAL.md, KNOWLEDGE.md (K001, K007, K019, K028), the
trainer scripts/train_lf.py (masked multi-task MSE; options --pseudo/--phys-aug/--lowpass recently added; resume fingerprint _cfg/_NEW_DEFAULTS
must keep old checkpoints resumable), and the label files targets_v5.npz (keys Y, mask, split, batch, hist_row, names).
Build: (1) scripts/n049_noise_mixture.py: per single marker x draw, fit on split-0 clean cells of the TRAINING draw only a 2-component
mixture (Gaussian or skew-robust; justify) of the curated label with an explicit observation-noise layer (labelling efficiency eta,
false-positive rate phi, bounded priors) by EM; write per-cell soft targets P(true pos | y) for all masked cells of that draw plus
parameters JSON; report fit diagnostics (component separation d', eta, phi, BIC vs 1 component). (2) Opt-in train_lf.py flags (defaults
off, added to _NEW_DEFAULTS): --pospos FILE (soft-target npz) --pos-w W: an extra linear head with one logit per single marker,
BCE on soft targets where mask is true (never on missing), added to the loss; eval writes eval_preds with an extra array of
probabilities; GOAL predictions unchanged (intensity head). (3) CPU tests: default path bit-identical (fixed seed tiny run on CPU via
the same functions), masking, BCE on soft targets, resume fingerprint. Deliver report.md with design decisions and fit diagnostics
for the real data (CPU, minutes). Rules: fluorescence never an LF input; Giemsa cells excluded; no test-draw information in fitting.
