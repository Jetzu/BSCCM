# Codex WRITER brief: N060-N062 (review N059 portfolio) -- three new options in the GOAL harness scripts/train_lf.py

Repo /home/hpc/v181bb/v181bb10/BSCCM READ-ONLY. Deliver into THIS folder: a modified copy `train_lf.py` (+ any new module, e.g.
`bsccm_heads.py`), `PATCH.diff` against scripts/train_lf.py, README.md (exact commands), report.md, CPU smoke log. Scratch /tmp/codex_n060/.
No GPU / apptainer in your sandbox: CPU smoke with tiny --max-train and a resnet18; Claude integrates and runs GPU jobs.
Read first: .agent-research/results/N059-review-3-lf-mk-v10-goal-6862-target-70-a/{diagnosis.md, ideas.md, agents/*.md},
KNOWLEDGE.md K028-K040, scripts/train_lf.py (GOAL recipe: --arch resnet50 --init-ssl <n031 encoder> --epochs 10 --bs 256 --lr 5e-4
--w-cls .5 --targets targets_v5cx_v10.npz, protocols b1to0/b0to1), bsccm/data.py, bsccm/gpu_norm.py, scripts/make_targets_v5.py.
Rules: the GOAL eval (evaluate(), eval_preds.npz P in label units, 13 targets) must stay byte-identical in meaning; every new option off
by default and omitted from _cfg() at default (_NEW_DEFAULTS) so old checkpoints still resume; training uses only the training draw;
no held-out-draw labels/images; no lineage/exclusivity priors (user ruling).

N060 --head {mse, hlgauss, twopart} (default mse):
- hlgauss: per target a histogram head (64 bins over [q0.001, q0.999] of that target's TRAINING labels, widened 10 %), HL-Gauss
  targets (Gaussian, sigma = .75 bin width, Imani & White 2024 / Farebrother et al. 2024), CE loss masked like the MSE, prediction =
  expectation over bin centres (label units); add --hl-mse w (default .1) MSE on the expectation. All 13 targets.
- twopart: for the dim single markers (single:CD123, single:HLA-DR, single:CD19, single:CD56) a gate logit + two amplitude outputs:
  pred = sigmoid(g)*mu_pos + (1-sigmoid(g))*mu_neg; loss = MSE(pred) + --gate-w (default .5) * class-balanced BCE(g, y > thr_t) where
  thr_t = the TRAINING-label quantile --pos-q (default .98) of that target (fixed before training, saved in ckpt); other targets MSE.
N061 (training-only nuisance modelling; prediction at test = shared head only):
- --fov-re w (default 0): mixed-effects "crowd layer" (Rodrigues & Pereira 2018): learnable per-FOV offset per target b[fov, t] and
  per (slide x predicted cls3) affine (a, c) applied ONLY inside the training loss: loss on a*p + c + b - y; L2 shrinkage w*(b^2) and
  (a-1)^2 + c^2 with --re-shrink (default 1.0); FOV/slide ids from the targets file; never used in evaluate().
- --pair-loss w (default 0): within-FOV pair-difference MSE [(p_i - p_j) - (y_i - y_j)]^2 for same-target same-FOV pairs inside a
  batch (sampler option --fov-batch k: batches built from k FOVs so pairs exist), added to the usual loss.
- --w-autofl x (default 1.0): weight of the autofl target in the regression loss (0 = drop co-stained autofl supervision).
N062 --panel-head (default off): latent molecular factors z (8, one per single marker) from the encoder features; single targets =
  per-target monotone affine link of z_m (positive slope); pooled targets = assay decoder of W z, W non-negative, initialised from the
  panel design (grp:CD3/CD19/CD56 = CD3+CD19+CD56; grp:CD45 = CD45; grp:CD16 = CD16; grp:CD123/HLA-DR/CD14 = CD123+HLA-DR+CD14) and
  learnable, plus per-TRAINING-slide affine adapters on pooled outputs used only in the training loss (test = shared decoder);
  autofl and class heads unchanged. Pooled cells thus supervise the single-marker factors.
Report: the exact arms you recommend (my plan: N060 hlgauss, twopart; N061 fov-re + w-autofl 0, pair-loss + w-autofl 0, w-autofl 0 alone;
N062 panel-head; each 2 seeds x 2 directions), GPU estimate (~.4 GPU-h b1to0, ~.2 b0to1 per run for r50), and anything in this brief that
is wrong about the data. Handback <= 1.2k tokens.
