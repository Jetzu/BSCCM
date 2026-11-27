# Codex RED-TEAM brief (N087, Claude-written code). Hat: red-team (change nothing in the repo). Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex_rt ONLY;
# scratch /tmp/n087_codex_rt. No SLURM/GPU; CPU checks in /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif (apptainer exec -B /hnvme ... python3 -I).
Code under review:
1. /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex_rt/train_lf_claude.diff = Claude's uncommitted edits to /home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py: --dump-features
   (load the finished checkpoint, write pooled features H/P/G/is_eval for train then eval rows, exit) and --bottleneck k /
   --bottleneck-init projector (Linear(NF,k)+LayerNorm, or the pretrained SSL projector MLP 2048-2048-2048-256 loaded from
   <dirname(--init-ssl)>/ckpt.pt['projector'], between pooled features and all heads).
2. /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/n087_flow_decoder.py (job 1442504 running it now): conditional rectified-flow decoder vs MSE MLP on frozen features; sandbox part
   re-creates the N080 fold preprocessing (copied from scripts/train_ifc_sandbox.py main) and gating (exec of n080_eval.py source);
   BSCCM part scores GOAL on frozen n051_v10_base features.
Claims: (a) all defaults are byte-identical in behaviour (cfg written to checkpoints unchanged for old runs -> existing runs still resume;
evaluation unchanged); (b) the dump writes exactly the rows/order train_lf uses, eval rows flagged correctly; (c) the bottleneck wraps
the encoder correctly (heads, c3/c10, dual-zoom refusal, LR param groups, adabn, channels_last) and the projector init loads the right
weights; (d) n087_flow_decoder.py is inductive (decoders fitted on training-donor / training-draw rows only), its sandbox preprocessing
reproduces the N080 runs (it asserts this), its gating matches n080_eval.py, the flow (training loss, Heun sampler, K samples) is
correct, and its metrics mean what the column names say (per-marker r, 36-pair whole-donor correlation error, within-population pairs,
W1, sd ratio; BSCCM GOAL identical to scripts/score_goal_arms.py).
Do not fix anything. Find why a claim is wrong; cite file:line; MAJOR/MINOR; NOTHING BREAKS if nothing. findings.md (<= 400 words).
3. Claude's integration edits to the Codex decoder sweep: diff /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex/n087_decoders.codex_orig.py -> /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex/n087_decoders.py
   (per-target loss weighting like train_lf, wider ridge grid + logging, ranks 1,2,4,8,12,full, reference assert only for n051 dumps).
