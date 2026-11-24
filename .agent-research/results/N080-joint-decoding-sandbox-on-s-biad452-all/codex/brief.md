# Codex WRITE brief (N080, BSCCM). Hat: write. Writable root = this folder (/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/codex) ONLY; repo /home/hpc/v181bb/v181bb10/BSCCM read-only;
# scratch/venvs/caches in /tmp/n080_codex. No GPU here: CPU smoke tests only (tiny). Do not submit SLURM jobs.
## ONE task: train_ifc_sandbox.py -- a joint-decoding sandbox on the external S-BIAD452 imaging-flow dataset
Context: BSCCM (our main data) measures ONE marker per cell (single-antibody slides) + 4 pooled channels on 'all' slides; the user's top
priority is decoding ALL markers of each cell jointly, which BSCCM cannot test (review N079: pooled channels mostly measure cell type,
cell-type-only predictor .730 vs model .737). S-BIAD452 has 9 markers measured on EVERY cell -> we can simulate BSCCM's regime and
measure joint decoding against real joint truth. Read: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/../N079-review-5-lf-mk-v10-goal-6862-target-70-a/{diagnosis.md,agents/outsider.md,
agents/codex_ideas.md}, /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/../N077-external-data-scout-public-label-free-su/data_scout.md, /home/hpc/v181bb/v181bb10/BSCCM/scripts/pack_sbiad452.py
(packed format) and scripts/train_lf.py (house style: preempt-safe resume from <out>/ckpt.pt, SIGUSR1/SIGTERM checkpoint, result.json).
Data (packed by pack_sbiad452.py, will exist at /hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack/): lf.npy float16 (N,3,128,128) =
[BF1 log-ratio, BF2 log-ratio, SSC asinh] (divide by labels.npz chan_scale), labels.npz: Y (N,9) z-scored per sample, I_raw (N,9),
names [CD15,Siglec8,CD14,CD19,CD3,CD45,CD4,CD56,CD8], sample in {sample0, sample1_early, sample1_late, sample2_early, sample2_late}
(donor = sample0 / sample1 / sample2), part, object_number. A smoke pack with ~400 cells for tests: /scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/sbpack_smoke
(re-create a tiny fixture in /tmp if missing; never write into the real pack dir).
Spec:
- Model: timm resnet50 (pretrained=False, in_chans=3, num_classes=0) + linear head to 9 (+ optional rank-2 covariance head, see arms);
  AdamW lr 5e-4 OneCycle, bs 256, bf16 autocast on cuda, +-4 px translation + random 90-degree rotations/flips (all three IFC channels
  are rotation-invariant physics: brightfield + side scatter, no directional LEDs), --epochs default 15.
- Donor folds: --test-donor {0,1,2}; validation = one other donor fixed by rule; training = the rest. Report on the test donor only.
- Training regimes (--regime): full = all 9 labels per training cell; single = each training cell reveals exactly ONE marker chosen
  uniformly at random with a fixed per-cell seed (BSCCM-like); single_pooled = single for 90 % of cells, the other 10 % reveal ONLY two
  pooled values like BSCCM's 'all' slides: P1 = asinh((I_CD3 + I_CD19 + I_CD56) / c) and P2 = CD45 (linear intensities summed before asinh,
  per-donor standardised; c = training-donor robust scale), predicted from the model's own per-marker outputs via a learned monotone
  per-marker inverse map (or: a pooled head on the shared features -- implement both as options and document); masked MSE so only revealed
  values give loss. Optional --cov-rank 2: Gaussian NLL with diagonal + rank-2 covariance on the full regime (Codex N079 idea 4).
- Metrics on the test donor (write eval_preds.npz and metrics.json): per-marker Pearson r and macro mean; per-cell standardised vector
  RMSE and cosine; co-expression fidelity = Frobenius distance between predicted and true marker correlation matrices, overall AND within
  coarse populations defined from TRUE labels for evaluation only (lymphocytes = CD45 hi & CD15 lo etc.; document thresholds; no
  population labels enter training); key pairs CD3-CD19, CD3-CD56, CD4-CD8, CD14-CD15 predicted vs true r within lymphocytes/granulocytes;
  double-positive rate CD3+CD19+ (thresholds from training donors) predicted vs true.
- Literature checks (review N079 agents/lit_panel_merging.md: Infinity Flow / Mocking 2023 / HistoPlexer): also report (i) CELL-level vs
  CLUSTER-level correctness: r within each true population vs r of population means; (ii) over/under-correlation: predicted minus true
  pairwise r for all 36 marker pairs (HistoPlexer saw joint models over-correlate); (iii) gating F-score of standard gates applied to
  predicted vs true values (thresholds from training donors); (iv) ensemble-free uncertainty is optional.
- Baseline arm --knn-impute K (default 50): Infinity-Flow-style imputation in the model's own embedding space -- for each test cell, average
  each marker over the K nearest TRAINING cells that revealed that marker (single regime) or all K (full regime); score with the same metrics.
- Tests (CPU, tiny): masking correctness per regime (exactly one revealed per single cell; pooled cells reveal only pooled), donor split
  has no overlap, metrics on synthetic data with known answers, resume from ckpt.
Deliver in this folder: train_ifc_sandbox.py, tests/, README.md (commands for 3 regimes x 3 test donors, expected GPU-h on one H100),
codex_last_message <= 300 words. Do not edit anything outside this folder.
