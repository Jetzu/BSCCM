# Codex RED-TEAM brief (N085 GOAL claim). Hat: red-team (change nothing). Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N085-stronger-members-convnext-s-with-120-ep/codex_rt ONLY; scratch /tmp/n085_rt.
No SLURM/GPU; CPU in /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif (apptainer exec -B /hnvme ... python3 -I).
Claim: the pre-registered union ensemble E8 (definition: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N085-stronger-members-convnext-s-with-120-ep/n085_score.py docstring, written before the new members finished)
scores GOAL .6967 (b10 .7066, b01 .6869) > bar .6916 -> IMPROVED over the record .6862 (N053). Output: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N085-stronger-members-convnext-s-with-120-ep/n085_scores.csv.
Evidence: n085_score.py; member runs /hnvme/workspace/v181bb10-bsccm/runs/<run>/eval_preds.npz (P, G); targets_v5cx_v10.npz;
canonical scorer /home/hpc/v181bb/v181bb10/BSCCM/scripts/score_goal_arms.py; job ledgers /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N085-stronger-members-convnext-s-with-120-ep/jobs.txt and earlier items
(.agent-research/notebook/items/N053-*, N063-*, N065-*, N083-*, N084-*, N085-*).
Do not fix anything. Find why the claim is wrong:
(a) recompute E8 independently with the canonical scoring (common-11, eval rows, masks, both directions) -- same number?
(b) are all members' eval rows identical and aligned before averaging; any member with NaN/missing targets handled differently?
(c) leakage: did any member's encoder (SSL) or fine-tune see the held-out draw (check --protocol and --init-ssl / --exclude-batch of
    every member per direction: ex0 encoders for b10 = b1to0, ex1 for b01)?
(d) does the member set equal the pre-registered definition exactly (no swapped run, no extra seed)? Were any members selected after
    seeing their own test-draw scores (state which, if any; the pre-registration date vs job times)?
(e) is the bar right (.6862 + 2 x .0027) and is the gain robust to dropping any single member (report the 8 leave-one-out GOALs)?
Cite file:line with numbers; MAJOR/MINOR; NOTHING BREAKS if nothing. findings.md (<= 350 words).
