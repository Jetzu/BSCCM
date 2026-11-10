# N028 red-team (Claude) — virtual Wright/Giemsa pipeline by Codex

Reviewed: `codex/vs_common.py`, `vs_register.py`, `vs_train.py`, `vs_eval.py`, `vs_smoke.py`, `README.md` against `codex/brief.md`.
Environment for all checks: `apptainer exec /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I <script>` on helma2 (FUSE worked; sif has torch 2.9.1, timm 1.0.31, scipy 1.17.1, PIL 13.0, wandb 0.31.0). CPU only, 16 workers, <= 400 real cells per check, scratch in /tmp/$USER/rt-N028 (deleted). The check scripts import the Codex functions unchanged (`vs_register.phase_shift`, `vs_register.ncc`, `vs_common.registered_rgb`, ...).

**Verdict: NOT safe to launch.** One blocker: the registration step makes alignment worse than doing nothing. Fixing it is cheap (about 20 lines in `vs_register.py`) and does not touch the training or eval code.

---

## B1 [BLOCKER] Whitened phase correlation picks spurious shifts on cross-modal pairs; the registered GS is more misaligned than the unregistered crop

`vs_register.py:41-57` (`phase_shift`), especially line 50 `cross / np.maximum(np.abs(cross), 1e-8)` (full spectral whitening). Lines 54-56 then refine by NCC only within +-2 px of that peak. The same estimator drives the global scale fit (`fit_one`, line 64).

The sign convention is **correct**: `ncc()`, `phase_shift()` and `registered_rgb()` agree. The estimator, however, returns essentially random shifts for LF brightfield vs GS grey. Whitening equalises all frequencies, and the high-frequency content of the two modalities (GS granules/chromatin vs LF noise) does not correspond. The smoke "phase_sign" control (`vs_smoke.py:33-36`) only shifts an image against itself, where phase correlation is perfect, so it cannot catch this.

**Evidence.** Brute-force NCC over +-12 px served as reference, on 400 real cells (120 b0/120 b1 train, 60/60 test, 40 suspects) at scale 0.69. Core of the check:

```python
a, b = gaussian_filter(lf, 1), gaussian_filter(registered_rgb(grey, s), 1)
N = [[R.ncc(a, b, y, x) for x in range(-12, 13)] for y in range(-12, 13)]   # brute-force optimum
py, px, ppeak, _ = R.phase_shift(lf, registered_rgb(grey, s), 12)        # Codex estimator
# sign control: full-field NCC of LF vs registered_rgb(grey, s, +by, +bx) / (-by, -bx) / (0, 0)
```

Output:

```
phase_vs_brute_exact 0.115 | within 1 px 0.19 | median disagreement 5.4 px | p90 11.2 px
plain (unwhitened) xcorr vs brute within 1 px: 0.975   (= the method of claude_alignment_probe.py)
median |shift|: phase 5.5 px (p90 10.4)  vs  brute 2.2 px (p90 5.0)
median NCC: phase choice 0.818 | brute optimum 0.867 | zero shift 0.858
fraction of cells where ZERO shift beats the phase choice: 0.73
sign control at brute optimum: applied +shift 0.865 > zero 0.852 > inverted -shift 0.821
                               (+shift >= zero in 99.5 %, > inverted in 96 %)
```

**Consequence.** Training targets as produced would sit about 5.4 px median (p90 about 11 px, at 128 px) from the LF, against about 2.2 px with no registration at all. That exceeds the 5-px box blur, the +-4 px paired translation and the DINO patch size (about 8 LF px). The full GPU run would learn blurred or misplaced stain. Eval PSNR/SSIM would be scored against the same misregistered targets. The scale fit is degraded too: with the phase estimator, the per-batch optima on the same cells were 0.76 (b0) and 0.72 (b1); with brute-force NCC they are 0.70 and 0.68, with smooth curves.

**Fix (cheap).** Replace `phase_shift` with an unwhitened ZNCC search, used for both the scale fit and the per-pair shift. About 25 ms per pair, so roughly 10 s for all 15,712 pairs on 48 workers:

```python
def ncc_shift(a, b, max_shift=12):
    a, b = gaussian_filter(a.astype(np.float32), 1), gaussian_filter(b.astype(np.float32), 1)
    if a.std() < 1e-6 or b.std() < 1e-6: return 0, 0, -1., 0.
    peak, dy, dx = max((ncc(a, b, y, x), y, x) for y in range(-max_shift, max_shift+1)
                                              for x in range(-max_shift, max_shift+1))
    return dy, dx, peak, peak - ncc(a, b, 0, 0)          # 4th field: gain over zero shift
```

Alternatively keep the FFT path but drop the whitening (`ifft2(fa*conj(fb))` on z-scored images, as in the probe).

Also add a real-pair control to `vs_smoke.py`: on >= 50 training pairs, assert median full-field NCC(LF, `registered_rgb(..., dy, dx)`) > NCC at zero shift.

Re-verified with that estimator:
- Brute-force scale curve peaks at 0.68 (both batches; b0 0.70, b1 0.68; per-cell IQR 0.66-0.74), so batch-holdout scale transfer is fine.
- Optima at the +-12 search edge: 0.75 %.

---

## M1 [MAJOR] The quality gate (NCC >= 0.4) rejects nothing and cannot detect mismatched or mis-oriented pairs

`vs_register.py:91` (`--min-peak .4`) and line 167 (`good = peak >= min_peak & not boundary`).

The NCC between LF-BF and GS grey is dominated by the round cell blob.

**Evidence:**
- 0/400 sampled cells rejected in every group. Median peak: b0 train 0.84, b1 train 0.79, b0 test 0.85, b1 test 0.80, `gs_pair_suspect` cells 0.77.
- **Deliberately mismatched pairs** (LF of cell i vs GS of a random other cell j, 80 pairs, Codex estimator): median peak 0.77, **98.75 % pass the gate**. AUC true vs mismatched = 0.68.
- Wrong orientation of GS (flips, rot90, transpose) at the correct scale: median NCC 0.853-0.863 vs identity 0.872. Identity is the most frequent best orientation (54/120 vs <= 13 for each alternative), so orientation is fine at population level. Per pair, though, the median margin of identity over the best alternative is -0.0025.
- Even with brute-force NCC (150 true vs 150 random pairs): matched 0.873 vs mismatched 0.831, AUC 0.71. A null-calibrated threshold (p99 of mismatched = 0.94) would reject 98 % of true pairs. Gradient-magnitude NCC is worse (AUC 0.61).

So per-pair image similarity cannot serve as pair QC here. The report's `n_dropped` would read as "all pairs good".

**Fix:**
- Keep only the search-boundary rejection (registration failure).
- Exclude the existing `targets_v6['gs_pair_suspect']` cells (an area-outlier audit from N003; 567 GS cells, 450 in split 0) from **training pairs only**, never from test.
- Store a shuffled-pair null (median/p95 NCC of random pairs from training cells) in the registration report.
- Remove the "quality flag" claim from the README.

---

## M2 [MAJOR] wandb metrics will never sync, and the README's run location sits on a nearly full /home inside the committed notebook

`vs_train.py:148-150`: `wandb.init(..., dir=str(out), ...)`.

**Evidence (tested in the sif):** with `WANDB_DIR=/tmp/.../envdir` set (as `gpu_job.sbatch:12` does), `dir=out` wins. Offline runs land in `out/wandb/offline-run-*` and the env dir stays empty. The only login cron, `scripts/wandb_sync_cron.sh:9`, syncs `D=/hnvme/workspace/v181bb10-bsccm/wandb/wandb/offline-run-*`, so with the code as written no N028 metric reaches wandb. This violates the "no cron = no metrics" rule.

Each preempt segment also creates a separate offline dir with the same run id (tested: 2 segments gave 2 dirs). Whether a plain `wandb sync` appends the second one rather than overwriting is **[unverified]**.

README `RUN="$PWD/.agent-research/results/N028.../run-fov"` puts ckpt, wandb, `generated.npz` (README: up to about 2 GB at 256 px, 2x during export) and eval partials under $HOME. `quota -s`: /home is at **99,749 M of a 100 G soft quota** (memory: a full /home blocks git). `.agent-research/.gitignore` ignores `*.pt`/`*.npz` but not wandb run dirs, `*.wandb` or `*.npy`, so committing `.agent-research/` would pull in wandb files.

**Fix:**
- Drop `dir=str(out)` so the sbatch's `WANDB_DIR` applies and the existing cron picks the runs up.
- Set `RUN=/hnvme/workspace/v181bb10-bsccm/runs/N028-<holdout>` and copy only `metrics.json`/`metrics.jsonl`/`montage.png`/`per_cell.npz` into `results/N028.../`.
- Check that appending segments works in the cron (`wandb sync --append`, or an online resume).

---

## m1 [minor, brief deviation; becomes major once the gate is tightened] Generated images exist only for accepted held-out cells

`vs_common.py:113` filters `te` by `good`; `vs_eval.py:84`, 114 and 184 export only those cells. The brief asks for "generated images for **all** held-out cells". Generation needs only LF, so a GS-derived filter should not restrict the set the downstream autofl/student test runs on (the only same-cell molecular truth). Today about 0 cells are rejected (M1), so the practical effect is nil.

**Fix:** generate for every held-out row and store `good`. Compute PSNR/SSIM/DINO/retrieval on `good` only, and report both n.

## m2 [minor] Retrieval pools both slides

`vs_eval.py:158`, 178-179. For `--holdout fov` the 1000 candidates mix batch 0 (1014 test cells) and batch 1 (609). Slide colour alone lets a generated image rule out most other-slide candidates, which inflates top-1/top-5.

**Fix:** also report retrieval and colour statistics within each batch. PSNR/SSIM/cosine are already per batch.

## m3 [minor] README's "feed images/255" to the GS student has a 1-2 % intensity mismatch

`generated/255` approximates `H / rgb_scale`. `apply_gs_student.py:31` uses `H / percentile(H[:500:5], 99.9)` = 3922.0, while `rgb_scale` = 3992 (fov), 4007 (batch0) or 3892 (batch1), a ratio of 0.992-1.022.

**Fix:** feed `images/255 * rgb_scale / 3922` (stored `rgb_scale` is in the npz). Values were clipped at `rgb_scale`, so bright background saturates slightly differently than in the student's training.

## m4 [minor] The held-out test set is evaluated and logged every epoch; split 1 goes unused

`vs_train.py:172-185`, `--eval-every 1` on `te`. The code never selects a checkpoint, but per-epoch test curves invite human early-stopping. Split 1 (994 b0 + 593 b1 GS cells) is unused for `fov`.

**Fix:** monitor on split 1 during training (fov), or on a split-1 subset of the training slide for batch holdouts. Touch the test set only in `vs_eval.py`.

## m5 [minor] One non-finite bf16 batch kills the run without a resubmit

`vs_train.py:34` and 44: `clip_grad_norm_(..., error_if_nonfinite=True)` raises. rc != 0 without a signal means preempt-run exits and no successor is queued; up to 15 min of work is lost since the last save.

**Fix:** skip the step (zero the grads), count and log it, and abort only after k consecutive non-finite steps.

## m6 [minor] Training losses never reach wandb

`vs_train.py:181-182` logs only held-out eval to wandb. Train loss averages exist only per segment in `metrics.jsonl`.

**Fix:** `wb.log` the epoch-segment train losses (and step-level every N steps).

## m7 [minor] The smoke test cannot catch cross-modal registration failure

`vs_smoke.py:33-36` (same-content synthetic shift) and lines 60-65, which force `good=True` on all 8 cells. See the B1 fix.

---

## Checked and found correct (no action)

- **Shift sign and geometry.** `registered_rgb` output pixel `o` = source `centre + (o+0.5-size/2)*step - d*398*scale/128`, a translation of +d applied to GS. This is consistent with `ncc()` and with the phase-correlation convention. Empirically, applying the correct shift raises NCC over zero shift in 99.5 % of cells, and the inverted shift lowers it (B1 table).
- **128 vs 256 field.** The 2x2-pooled 256 crop differs from the 128 crop by 0.15 % (relative mean abs). NCC with LF is identical (0.8097 vs 0.8097).
- **No edge effects.** Maximum source extent is 0.8*398/2 + 12*398*0.8/128 = 189 px < 199.
- **Leakage.** Scale fit and `rgb_scale` use training rows only (`vs_register.py:122`, 143). Held-out GS images are used only to register held-out pairs for scoring. No checkpoint selection. `autofl`, `Y`, `mask` and `fluor` are referenced nowhere in the code. Registration and checkpoint embed holdout and targets/registration/normaliser/DINO sha256.
- **LF handling.** `GPUNorm` is used unchanged with `ch='all', phase=True, shift=0` (24 channels). Translation happens after normalisation, with no rotation or flip. `translate_pair` applies the same shift to x and y, doubled for 256 (synthetic ramp test: identical shifts).
- **Losses.**
  - `robust_l1` = 0.5 L1 + 0.5 L1 of box-blurred images (5/9 px, `count_include_pad=False`).
  - The DINO patch-token cosine runs with a no-grad target and a grad path through the frozen ViT; DINO params receive no grads.
  - DinoBloom loading is identical to `train_gs_bind.py:141-143` (strict, `backbone.` stripped, `mask_token` dropped).
  - ImageNet mean/std on [0,1] RGB match the GS student's `pipeline`.
  - PatchGAN is off by default.
- **bf16 autocast path, simulated on CPU** (`torch.autocast('cpu', bf16)` patched into `vs_train.amp`, real cells, width 32, DINO-S plus PatchGAN): no dtype errors, finite losses. UNet has 1.96 M params, so ckpt is about 24 MB.
- **Resume.**
  - Persistent workers honour `sampler.epoch`/`start` changes. A toy test reproduced the exact remaining batches of epoch 0 and the full epoch 1.
  - DataLoader iteration does not consume the global torch RNG, so translation RNG restore is exact.
  - Config mismatch is rejected.
  - Writes are atomic (tmp + fsync + `os.replace`).
  - The signal flag is serviced after the optimizer step. Exit code 0 lets preempt-run keep the successor.
  - Pending eval is replayed.
- **Eval.**
  - PSNR uses data range 1.
  - SSIM uses an 11-px Gaussian, sigma 1.5, valid conv, constants for L=1.
  - Retrieval ties are pessimistic (toy: duplicate candidate gives top-1 0.5).
  - `generated.npz` rows are ordered as `te` (ascending hist_row), with `gidx`/`hist_row` stored.
  - Metrics are on the quantised uint8 output.
- **Throughput.** `Pairs.__getitem__` takes 8.7 ms per sample (128) and 6.7 ms (256) in one process, so 12 workers are not the bottleneck. The sif provides wandb, scipy and PIL.
