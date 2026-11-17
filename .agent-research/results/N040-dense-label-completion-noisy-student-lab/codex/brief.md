# Codex RED-TEAM brief: N040 noisy-student options in scripts/train_lf.py (Claude wrote them)
Repo /home/hpc/v181bb/v181bb10/BSCCM. Do NOT edit repo files; write ONLY .agent-research/results/N040-dense-label-completion-noisy-student-lab/codex/redteam_N040.md. Scratch under /tmp/codex_n040/.
Data /hnvme/workspace/v181bb10-bsccm read-only. apptainer fails in your sandbox: use a uv venv in /tmp if you need to execute.
The uncommitted diff of scripts/train_lf.py is in .agent-research/results/N040-dense-label-completion-noisy-student-lab/codex/train_lf_uncommitted.diff; it contains THREE features: --lowpass (N042,
already reviewed by me), N041 physics augmentation (reviewed by a Claude red-team, integrated with fixes -- you may comment), and the
NEW N040 options to review: --dump-train-preds (load a finished checkpoint, predict every training row, save train_preds.npz, exit),
--pseudo / --pseudo-w (teacher predictions as extra MSE targets only where a cell has no real label). Item contract:
.agent-research/notebook/items/N040-dense-label-completion-noisy-student-lab.md.
Check: (1) dump mode cannot modify the teacher run (ckpt, result.json, wandb) and predicts exactly the rows the student will train on;
(2) pseudo targets never come from the held-out draw, never from Giemsa cells, never overwrite real labels, masking/weighting correct;
(3) the dataloader tuple unpacking changes (*pz, *_) are correct in every loop (train, eval, DANN, AdaBN); (4) resume fingerprint
(_cfg/_NEW_DEFAULTS) behaves for old checkpoints and for new pseudo runs; (5) anything that crashes on GPU or silently changes the
default path. Reproduce with commands + output where possible; mark [unverified] otherwise. Verdict: LAND / LAND WITH FIXES / DO NOT LAND.
