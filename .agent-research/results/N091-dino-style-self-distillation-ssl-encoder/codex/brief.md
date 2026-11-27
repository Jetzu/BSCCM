# Codex WRITE brief (N091, BSCCM). Hat: write. Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N091-dino-style-self-distillation-ssl-encoder/codex ONLY; repo read-only (deliver patched copies + diffs);
# scratch /tmp/n091_codex. CPU smoke only, no SLURM/GPU. Test env: /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif (apptainer exec -B /hnvme).
## ONE task: a DINO-style self-distillation objective in the in-domain SSL trainer
Read: /home/hpc/v181bb/v181bb10/BSCCM/scripts/train_ssl_lf.py (SIGReg/JEPA-style 2-view trainer: illumination-preserving views from
bsccm/ssl.py make_views_batch, 48-channel stem = 24 data + 24 channel-present planes with random channel dropout, projector MLP
2048-2048-2048-proj_dim, resumable ckpt, --rows all --exclude-batch B = only unlabelled cells of the training draw), bsccm/ssl.py,
scripts/train_lf.py (how --init-ssl loads encoder_final.pt strictly: the exported encoder must stay a plain timm resnet50 state dict with
48 input channels), .agent-research/results/N088-literature-review-user-11-24-label-free/lit_A_labelfree_representations.md (proposal P3).
Spec: --objective {sigreg, dino} (default sigreg = BYTE-IDENTICAL behaviour, checkpoints/configs of old runs still resume; prove with an
equality test on a tiny fixture: same seed -> identical loss trajectory and identical encoder weights after N steps vs the current file).
dino: student = encoder + DINO head (3-layer MLP 2048 hidden, GELU, 256 bottleneck, L2-normalised, weight-normalised last layer to K
prototypes, default K=4096, last-layer norm frozen first epoch); teacher = EMA copy of encoder + head (momentum cosine .996 -> 1.0), no
gradient; loss = cross-entropy between teacher softmax((t - center)/tau_t) and student log-softmax(s/tau_s) over all view pairs with
different views; center = EMA (.9) of teacher batch means; tau_s .1, tau_t warmup .04 -> .07 over the first 30 % of epochs; views = the
EXISTING illumination-preserving make_views_batch (no rotations/flips; 2 global views; optional --local-crops n of 64 px centre-jittered
crops resized to 64 for the student only, default 0); weight decay cosine .04 -> .4; same optimizer/schedule/AMP/ckpt/resume/wandb
conventions as the file; checkpoint stores student, teacher, head(s), center, momentum state; encoder_final.pt = TEACHER encoder
(standard DINO practice) as a plain state dict loadable by train_lf.py --init-ssl. Log loss, teacher entropy and KL(teacher||uniform)
per epoch to detect collapse.
Tests: equality test above; dino 3-step CPU smoke on a tiny fixture (tiny model allowed via an existing or new test hook); resume after
interrupt gives identical next-step loss; encoder_final.pt loads with train_lf.py's strict loader; no held-out-draw row read
(assert on rows with --exclude-batch).
Deliver: train_ssl_lf.py (full patched copy), train_ssl_lf.diff, (bsccm/ssl.py patched copy + diff only if needed), tests/, README.md with the
exact commands for both draws (--exclude-batch 0 and 1, 40 epochs, bs 512) and the expected wall time on 1 H100, <= 300-word handback.
