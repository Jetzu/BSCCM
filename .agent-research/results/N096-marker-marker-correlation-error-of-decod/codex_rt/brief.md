ROLE: red-team reviewer (read-only hat). Do NOT fix code; report. Write your report ONLY to /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-marker-marker-correlation-error-of-decod/codex_rt/redteam.md (<= 1500 words).
TASK: find correctness bugs, leakage and invalid comparisons in the N096 experiment code written by Claude, before its results are used.
Context: BSCCM project (label-free microscopy -> surface-marker regression). User asked to look at the correlation error between decoded
markers and try to fix it. Pre-registration: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N096-marker-marker-correlation-error-of-decod.md (read the Plan section).
Files:
 1. /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-marker-marker-correlation-error-of-decod/n096_bsccm.py - A: correlation error (r(pred_a,pred_b) - r(label_a,label_b) on co-measured test cells) of the N085 record
    ensemble and members; B: linear decoders on frozen n051 features with an MSE + lambda x correlation-matching loss, a post-hoc
    recolouring arm, lambda selection on an inner source-draw split, and a 'source-draw label corr' floor. Data:
    /hnvme/workspace/v181bb10-bsccm/targets_v5cx_v10.npz (Y, mask, names, cls3, slide, fov, batch), feature dumps
    /hnvme/workspace/v181bb10-bsccm/runs/n051_v10_base_{b10,b01}/features.npz (H, P, G, is_eval). The official scorer it must agree with
    for GOAL: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N085-stronger-members-convnext-s-with-120-ep/n085_score.py.
 2. /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-marker-marker-correlation-error-of-decod/n096_train_ifc_sandbox.py - sandbox trainer with a new --corr-match/--corr-source loss; diff vs the N087 version in
    /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-marker-marker-correlation-error-of-decod/codex_rt/sandbox_patch.diff. Job script /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-marker-marker-correlation-error-of-decod/n096_sandbox.sbatch. Eval: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex/n087_sandbox_eval.py.
Check especially: (a) any held-out (test draw / test donor) label reaching training, lambda selection or the recolouring fit;
(b) whether the corr-matching target and the reported correlation error use the same pair definitions/rows; (c) whether the
recolouring math does what the docstring says; (d) whether GOAL_dir equals the official per-direction GOAL definition (common-11,
masked rows); (e) resume/config issues in the sandbox patch (cfg fingerprint, checkpoint selection with the extra loss term);
(f) anything that would make "corr matching fixes/does not fix the error" a wrong conclusion. You may run read-only python with
apptainer exec -B /hnvme /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I (no writes outside /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N096-marker-marker-correlation-error-of-decod/codex_rt/; scratch to /tmp/n096_rt).
For each finding: severity (blocker/major/minor), file:line, concrete failure scenario, how you verified it. "NOTHING BREAKS" if none.
