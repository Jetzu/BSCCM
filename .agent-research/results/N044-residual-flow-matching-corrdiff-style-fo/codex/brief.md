# Codex RED-TEAM brief: N044 residual flow matching (Claude wrote it)
Repo /home/hpc/v181bb/v181bb10/BSCCM (read-only for you). Write ONLY /scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/n044_codex/redteam_N044.md. Scratch only under /tmp/codex_n044/.
Data /hnvme/workspace/v181bb10-bsccm read-only. apptainer does not work in your sandbox: use a uv venv in /tmp (torch CPU) if you execute.
Diff under review: /scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/n044_codex/n044.diff (scripts/fm_common.py, fm_train.py, fm_eval.py). Design: --mode rcfm: a frozen reg-mode CondUNet
(EMA weights of --mean-ckpt, same draw/fm_prep/targets) gives mu = E[y|LF]; the flow learns r = (y - mu)/sigma_c (sigma_c = per-channel
residual rms on 4,096 fixed training cells, stored in the checkpoint), conditioned on LF + mu (31 condition channels; CFG dropout zeroes
LF, flag and mu together); sampling returns mu + sigma_c * r_hat. fm_eval wraps both plain and residual models in fm_common.Generator;
the LF-shuffled control recomputes mu from the shuffled LF. Context: N034/N035 items in .agent-research/notebook/items/.
Check: (1) any train/eval inconsistency (scaling, conditioning, CFG null, amp/dtype, EMA vs raw weights of the mean net, res_scale on
resume); (2) leakage: mean net trained only on the same training draw split 0; no test-draw information enters training or sigma_c;
(3) plain cfm/reg paths unchanged (default behaviour + old checkpoints still evaluate identically); (4) the shuffled-LF control is
still a valid control for rcfm; (5) resume config: mean_ckpt_sha256 pinned before the check; anything that crashes on GPU.
Reproduce with commands + output where possible; [unverified] otherwise. Verdict: LAND / LAND WITH FIXES / DO NOT LAND.
