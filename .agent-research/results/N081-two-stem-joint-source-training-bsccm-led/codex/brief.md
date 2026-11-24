# Codex WRITE brief (N081, BSCCM). Hat: write. Writable root = this folder (/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/codex) ONLY; repo read-only; scratch in /tmp/n081_codex;
# CPU smoke only, no SLURM.
## ONE task: optional second data source in the GOAL harness (two input stems, shared trunk, shared marker heads)
Read: /home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py (current working copy, all flags default-off must stay byte-identical: follow
the _NEW_DEFAULTS / _run_args pattern; existing equality test suite: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/../N074-dense-privileged-supervision-auxiliary-d/verify/test_dense_aux_integrated.py
+ /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/../N074-dense-privileged-supervision-auxiliary-d/codex/tests/), bsccm/encoders.py, scripts/pack_sbiad452.py (external pack
format: lf.npy (N,3,128,128) BF1/BF2/SSC, labels.npz Y (N,9) names CD15,Siglec8,CD14,CD19,CD3,CD45,CD4,CD56,CD8, sample, chan_scale),
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/../N079-review-5-lf-mk-v10-goal-6862-target-70-a/{diagnosis.md,agents/codex_ideas.md (idea 1 = this)}.
Spec: --ext-ifc DIR (default '' = off), --ext-w (loss weight, default 0.5), --ext-train-donors (default 'sample0,sample1,sample2' minus
--ext-test-donor), --ext-test-donor (default sample2 -> an extra eval on that donor's cells written to eval_ifc.npz/metrics in result.json).
Architecture: the BSCCM path is unchanged (SSL-init r50 with 48-ch stem incl. presence planes). Add a separate IFC stem = new conv1
(3 -> 64, 7x7 s2 like the r50 stem; initialise from the mean of the BSCCM stem's brightfield-like input filters, document) feeding the
SAME bn1/layer1-4 (shared trunk). Heads: BSCCM 'reg' head unchanged; the 5 overlapping markers (IFC CD14, CD19, CD3, CD45, CD56 <->
BSCCM single:CD14/CD19/CD3/CD45/CD56) are read by the SAME reg rows followed by a learned per-dataset affine (scale, offset) per marker for
IFC; an extra linear IFC-only head for CD15, Siglec8, CD4, CD8. Training: each BSCCM step adds one IFC batch (bs = --bs) with loss
ext_w * masked MSE; IFC inputs divided by chan_scale, +-4 px shift + 90-degree rotations/flips (IFC channels are rotation-invariant).
BSCCM evaluation, eval_preds.npz and GOAL scoring stay EXACTLY as now. Never use IFC data from the test donor in training.
Tests: flag off byte-identical (reuse the N074 suite against the current file); flag on: 2-step CPU smoke on a tiny IFC fixture + the
N074 BSCCM fixture; IFC donor split assertion; shared trunk receives gradients from both sources.
Deliver: train_lf.py (full patched copy), train_lf.diff, tests/, README.md (exact commands for b1to0/b0to1 runs), <= 300-word handback.
