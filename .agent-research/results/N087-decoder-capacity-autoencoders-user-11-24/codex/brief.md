# Codex WRITE brief (N087, BSCCM). Hat: write. Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex ONLY; the repo is read-only (deliver patched copies + diffs);
# scratch/venv/pip in /tmp/n087_codex only. CPU smoke tests only, no SLURM, no GPU. Python env for tests: the container
# /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif (apptainer exec -B /hnvme ... python3 -I ...; torch, timm, sklearn, numpy, pandas available).
## Context
Repo /home/hpc/v181bb/v181bb10/BSCCM. Task: label-free (LF) 24-channel cell images (23 LED illumination patterns + phase, 128x128) -> 13 per-cell
regression targets; GOAL = Pearson r over 11 targets (common-11), trained on one blood draw, tested on the other, mean of both directions
(protocols b1to0 / b0to1). Read: scripts/train_lf.py (training/eval, row selection lines ~95-125, prep()/GPUNorm, fwd(), evaluate(), the new
--dump-features path), scripts/score_goal_arms.py (the GOAL scorer), bsccm/gpu_norm.py, scripts/extract_subset_feats.py (frozen SSL encoder
with 48-ch stem = 24 data + 24 channel-present planes), scripts/train_ssl_lf.py (SSL row selection --rows all --exclude-batch),
scripts/train_ifc_sandbox.py + .agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/n080_eval.py (S-BIAD452 sandbox: 9 markers
on every cell, 3 donors, regimes full/single/single_pooled), .agent-research/notebook/items/N062-*.md, N080-*.md, N082-*.md.
Feature dumps being produced now: /hnvme/workspace/v181bb10-bsccm/runs/n051_v10_base_{b10,b01}/features.npz with H (N x 2048 fp16, pooled
features of the fine-tuned ResNet-50), P (model predictions), G (row ids into targets_v5cx_v10.npz), is_eval (0 train rows, 1 eval rows), names.
Recorded GOAL of these two models: .6730 (b10 .6867, b01 .6592).
## Deliver (all in /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex/)
1. n087_decoders.py FEATS... --targets targets_v5cx_v10.npz --out DIR: for each features file (one per direction) fit on train rows, score
   on eval rows: (a) the model's own P (reference; ASSERT it reproduces the recorded GOAL within 1e-4 using the SAME scorer code as
   score_goal_arms.py, import it or factor it, do not re-derive); (b) per-target ridge (masked rows, lambda by an inner 10 % split of train rows
   grouped by field of view); (c) reduced-rank linear decoder: shared d->k bottleneck + k->13 output, masked MSE, k in 1,2,4,8,16,32,64,128,
   full; (d) MLP 2x512 GELU dropout .1; (e) kNN regression (cosine, k=50) per target. Features standardised with train-row statistics only.
   Output decoders.csv (feature_set, decoder, k, direction, target, r) + goal.csv (feature_set, decoder, k, b10, b01, GOAL). CPU/GPU agnostic.
2. n087_extract.py --encoder {ssl,ae} --ckpt PATH --rows-from features.npz --out FILE: frozen features for exactly the rows/order of an existing
   features.npz (same normalisation as train_lf eval: GPUNorm, no shift; SSL = timm resnet50 48-ch stem with presence planes as in
   extract_subset_feats.py; AE = the encoder of item 3). Writes the same npz format (H, G, is_eval, names; no P).
3. n087_image_ae.py --protocol b1to0|b0to1 --dim k --out DIR: convolutional autoencoder on the normalised 24x128x128 LF input; encoder ->
   k-d linear bottleneck -> decoder; masked reconstruction MSE (+ optional --l1); trained ONLY on cells of the training draw (same rows as
   train_ssl_lf.py --rows all --exclude-batch <held-out batch>; assert no held-out-draw row). Preempt-safe like the repo's jobs: checkpoint
   <= 30 min and on SIGTERM/USR1, resume-latest from DIR/ckpt.pt; bf16 autocast; reasonable default epochs so one run is ~1 H100-hour at k=64.
   Log train/val reconstruction MSE per epoch to DIR/log.json. Final encoder weights DIR/encoder_final.pt.
4. n087_cluster_eval.py FEATS... --out DIR: clustering quality on draw-0 cells with cell-class labels (targets npz cls10 >= 0; these exist only
   in draw 0, so use features whose encoder did NOT train on draw 0, i.e. the b10 / b1to0 files): k-means k=10 (10 restarts) NMI and ARI vs
   cls10, kNN (k=15, cosine) balanced accuracy with 5-fold CV grouped by field of view, and the same after PCA to 16/64/256 dims (train-free:
   PCA fit on the evaluated cells is allowed here, it is unsupervised). Output cluster.csv.
5. Sandbox marker prior: patched copy of scripts/train_ifc_sandbox.py (+ diff) with --marker-prior {none,linear,ae} --prior-dim k: fit the
   prior on the FULL 9-marker labels of the sandbox's VALIDATION donor (never the test donor, never hidden labels of the training donor);
   linear = top-k PCA subspace of the standardised marker vectors; ae = MLP autoencoder 9->64->k->64->9 (fit before model training, frozen).
   The model head then outputs z (k) and predictions = frozen prior decoder(z); training loss unchanged (masked MSE on revealed markers).
   Default none = BYTE-IDENTICAL behaviour (re-run the existing sandbox test suite: .agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/
   codex/tests if present, else write equality tests). Plus n087_sandbox_eval.py that runs the N080 post-hoc evaluation (n080_eval.py logic:
   per-marker r, cell-type oracle, within-type r, pairs) on arbitrary run directories AND the pre-registered 36-pair metric from each run's
   metrics.json (model.pairwise_correlation_bias[*].predicted_minus_true, mean |.| per donor).
6. README.md: exact commands for every run (GPU placement is decided by me), expected wall time; tests/ with CPU smoke tests for 1-5
   (tiny fixtures; the AE and decoders must run 2 steps on CPU). <= 300-word handback in codex_last_message.
Constraints: inductive everywhere (nothing from the held-out draw/donor enters fitting except where stated for unsupervised PCA in item 4);
never write outside /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex; do not touch running jobs or /hnvme runs other than reading.
