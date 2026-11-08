You are the WRITER for one task in a research repo. Write code only into your cwd (this codex/ dir). Do not modify anything else.
Scratch, pip caches, venvs, temp files: ONLY under /tmp/v181bb10/codex-N007/ (create it), never in the repo or /hnvme. Delete /tmp/v181bb10/codex-N007/ when done.
No training runs, no GPU (this login node has none). Small CPU unit tests are fine (seconds).

## Context (read these files, read-only)
- /home/hpc/v181bb/v181bb10/BSCCM/bsccm/data.py — LFNorm (flat-field normalised label-free LED images, 23 channels, optional phase channel -> 24 channels, float32 (C,128,128)) and CellSet. Use LFNorm; do not reimplement normalisation.
- /home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py — the supervised baseline (style, checkpoint/resume pattern, wandb-offline pattern, timm usage). Match its conventions.
- Data: /hnvme/workspace/v181bb10-bsccm/{pack/*.npy, targets_v2.npz, cellmeta.npz, led_norm.npz, backgrounds_p50.npy} (read-only). N = 412,941 cells. targets_v2.npz keys: Y (N,13) z-scored marker targets, mask (N,13) bool, cls3 (N,) int16 (-1 = unlabeled; 0 lymph, 1 gran, 2 mono), split (N,) int8 (0 train/1 val/2 test by field of view), batch (N,), hist_row (N,) (>=0 for the 15,712 Giemsa-paired cells), names (13,).
- Python env: `apptainer exec --bind /hnvme/workspace/v181bb10-bsccm,/home/hpc/v181bb/v181bb10/BSCCM /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 ...` (torch 2.9.1, timm 1.0.31, numpy 2.4). If apptainer cannot run in your sandbox, skip tests and say so.

## Channel physics (important)
Channel order = bsccm.data.LED: 0 Brightfield, 1-17 darkfield (DF_50 .. DF_90, some "_Bottom"/"_Right" directional), 18-21 DPC_Bottom/Left/Right/Top, 22 LED119 (single off-axis LED), 23 = phase (if phase=True).
NO rotations/flips: directional channels would be silently relabelled. Allowed geometric aug: translation (+-6 px, reflect pad), small isotropic scale 0.9-1.1. Allowed photometric aug: per-channel gain jitter (x exp(N(0,0.05))), additive Gaussian noise (sd 0.05 in normalised units), mild Gaussian blur (sigma 0-0.8 px).

## TASK: self-supervised pretraining of the label-free (LF) encoder, two arms, one script
Files to write (in cwd):
1. `bsccm_ssl.py` (module): 
   - `make_views(x, arm, n_views, rng)`: x = (24,128,128) tensor. Returns list of (view (48,128,128), ) where view = concat[channels with dropped ones zeroed (24), presence mask planes (24, constant 0/1)].
     - arm "aug": every view keeps all channels except independent channel dropout p=0.1 per channel (never drop all), + the allowed augs (independently per view).
     - arm "xillum" (cross-illumination): partition the 24 channels randomly into disjoint subsets, one per view (each view >= 4 channels; keep phase channel in at most one view), + the same allowed augs. Two views of the same cell therefore share NO channel.
   - `sigreg(z, n_dirs=256, n_t=17, t_max=5.0)`: Epps-Pulley Gaussianity statistic. z (B,D) embeddings. Sample n_dirs random unit vectors a (fresh each call, on z.device); project p = z @ a (B,n_dirs); standardise nothing (target is N(0,1) per direction). For t in linspace(-t_max,t_max,n_t): empirical CF phi(t) = mean_b exp(i t p) (use cos/sin separately); target exp(-t^2/2); weight w(t)=exp(-t^2/2); statistic = trapezoid over t of |phi - target|^2 * w; return mean over directions, multiplied by B (as in the Epps-Pulley test) — make the B-scaling a flag, default on.
   - `ssl_loss(zs, lam)`: zs = list of V view embeddings (each (B,D), from the projector). centre = mean over views; pred = mean over views of mean_b ||z_v - centre||^2; reg = mean over views of sigreg(z_v); return (1-lam)*pred + lam*reg, plus both parts for logging. Default lam = 0.05.
   - Unit tests in `test_bsccm_ssl.py`: (a) sigreg of N(0,I) samples << sigreg of a collapsed/non-Gaussian batch (e.g. all-equal, or uniform*5); (b) make_views shapes, xillum views disjoint, masks consistent with zeroed channels, no flips used; (c) loss is finite and backprops.
2. `train_ssl_lf.py` (script; must be runnable as `python3 -I train_ssl_lf.py ...` from the repo's scripts/ dir after I copy it there — so insert the repo root into sys.path the same way train_lf.py does and import `from bsccm.data import LFNorm, W` and `from bsccm.ssl import ...` (I will copy bsccm_ssl.py to bsccm/ssl.py)):
   - args: --arm aug|xillum, --out DIR, --arch resnet50 (timm, in_chans=48, num_classes=0), --epochs 40, --bs 512, --views 2, --lr 2e-3 (AdamW, wd 0.05, cosine with 2-epoch warmup), --lam 0.05, --proj-dim 256 (projector MLP: feat -> 2048 -> 2048 -> proj_dim, BN+GELU), --workers 15, --rows all|train (default: all 412,941 cells — SSL may use every cell incl. test FOVs because no labels are used; also include a --exclude-batch option for later batch-holdout experiments), --max-steps (debug).
   - Dataset: returns the V views of one cell (use LFNorm(g, all 23 channels, phase=True)). Workers must open memmaps lazily (LFNorm already does).
   - bf16 autocast, channels_last, grad clip 3.0.
   - Preempt-safe: checkpoint (model, projector, opt, sched, epoch, step, rng) atomically (tmp + os.replace) every 20 min and at epoch end; `--resume auto` behaviour by default (load <out>/ckpt.pt if present); on SIGTERM set a flag, save a checkpoint at the next step and exit 0. Keep only latest ckpt + `encoder_final.pt` (encoder state_dict only) at the end.
   - wandb offline like train_lf.py (id = basename of out); also print one JSON line per epoch: {"epoch", "loss", "pred", "sigreg", "secs", "imgs_per_s"}.
   - Embedding health metrics each epoch on a fixed batch of 1024 cells (single full view, no aug): effective rank of encoder features (exp of entropy of normalised singular values) and mean |off-diagonal| of the feature correlation matrix. Print/log them (collapse detector).
3. `probe_lf.py` (script): frozen-encoder linear probe, used for BOTH SSL arms and the supervised baseline.
   - args: --ckpt PATH --kind ssl|sup (ssl: encoder_final.pt state_dict of a timm model with in_chans=48 and full-presence input [24 data + 24 ones]; sup: train_lf.py ckpt.pt, encoder = model['enc'] keys prefixed 'enc.', in_chans=24) --arch, --out DIR, --n-train 60000, --n-test 20000, --workers 15.
   - Rows: train rows from split==0, test rows from split==2 (both: cells with any marker mask or cls3>=0); sample with a fixed seed. Extract pooled features (no aug) in bf16 on GPU.
   - Per target j (13): ridge regression (closed form, standardised features, alpha chosen by 5-fold CV on train from [1e-2..1e4] log grid) fitted only on rows with mask[:,j]; report test Pearson r per target and mean over the 12 marker targets (exclude 'autofl' from the mean but report it).
   - cls3: multinomial logistic regression (sklearn, lbfgs, C by CV) on labelled rows; report accuracy and balanced accuracy.
   - Within-class r for the 4 'grp:' targets on cls3-labelled test rows (per class), and the class-oracle r (r between truth and the per-class mean of truth) — same definitions as train_lf.py's evaluate().
   - Also save the extracted features for the 15,712 Giemsa-paired cells (hist_row>=0) to <out>/lf_feats_hist.npy with the matching global indices, for the later Giemsa<->LF alignment screen.
   - Write <out>/probe.json and a CSV with one row per metric.
4. `README_codex.md`: what you wrote, how to run each script (exact commands), test results, anything you could not verify.

## Contract
- Reuse LFNorm; no hidden data-leak: never feed the 6 fluorescence channels (pack/fluor.npy) to any LF model.
- Code must be correct first, readable second; no dead code paths, no silent fallbacks (fail loudly).
- Handback (your final message, <= 400 words): files written, test results (exact), design choices you had to make, open risks.
