# N049 red-team (Claude) of the Codex delivery

Date 2026-11-17. Scope: codex/ (train_lf.py replacement, bsccm/n049_pospos.py, scripts/n049_noise_mixture.py, tests). I did not edit any repo file and submitted no jobs. All runs used the bsccm.sif container (Python 3.11.16, torch 2.9.1+cu128, CPU, login node). Scratch was /tmp/n049_rt/, and the check scripts are kept in /tmp/n049_rt/chk/.

**Verdict: LAND WITH FIXES.** The trainer integration, loader and fitter are correct against the contract, and the default path is safe. However, the soft targets that would be used for the planned v9 run are degenerate for 3 to 4 of the 8 markers (MAJOR 1). That must be fixed, or those markers excluded, before the GPU run.

## Verified (reproduced first-hand)

1. **Baseline and diff.** The current repo `scripts/train_lf.py` equals HEAD, with SHA256 `7aeed6fd...`. That is the baseline Codex recorded. `implementation.patch` applies cleanly to a fresh copy and reproduces the delivered files byte for byte.
   - The diff contains only the N049 lines plus the `--device` plumbing (`.cuda()` → `.to(device)`, autocast `enabled=(device=='cuda')`, `pin_memory`, and worker kwargs only when workers>0). For `device='cuda'` these are call-for-call equivalent. This is checked by inspection only; GPU bitwise equality is [unverified].
   - No option was lost: `--lowpass`, `--phys-aug`, `--consist`, `--pseudo`, `--pseudo-w`, `--dump-train-preds`, `--dual-zoom` and `--adabn` are all present.
2. **Old checkpoints resume.** I parsed the stored cfg of 10 real checkpoints with the new parser and `_NEW_DEFAULTS` (n048_v9_base_b01, n048_v9_zoom2_b10, n046_student2_base_b10, n040_student_zoom2_b01, n041_phys_b10, n041_phystail_b01, n042_lp15_b10, n031_base_b01, n037_zoom2_b10, n039_ceil_fov_v5_mt0). The new `_cfg()` equals the old cfg for all 10, including zoom, phys_aug+consist, pseudo and lowpass runs. Check script: `chk/resume_cfg.py`.
3. **Delivered tests.** All 7 pass in the container in 6.2 s. This includes the CPU bit-identity of the default path against the original trainer and strict loading of an old checkpoint.
4. **Soft-target supervision, checked on the planned combination.** I loaded `v9fit/pospos_v9_draw{0,1}.npz` against `targets_v5cx_cf150.npz` with the trainer's real `tr`. Check script: `chk/check_v9.py`.
   - `ZM` equals the curated single-marker mask on clean source rows. It is 0 on the other draw, on eval rows and on Giemsa rows. Supervised entries: 111,413 for b0to1 and 201,177 for b1to0.
   - I recomputed the posterior from the JSON parameters on the trainer's global rows. It equals `ZY` exactly for all 8 markers in both draws, so alignment by global row and marker order is correct.
   - `pz` indexing is correct with and without `--pseudo`, following RawSet's dict insertion order.
   - BCE selects observed entries before computing, averages per marker and then across markers, and gives exactly zero for an all-missing batch.
5. **Mixture fitter.**
   - `fit_mask` covers split {0} and the source draw only. `fit_rows_sha256` matches. No Giemsa rows are in `G`.
   - Refitting with the delivered script on `targets_v5cx_cf150.npz` reproduces both v9fit archives bit for bit.
   - The archive schema matches the loader.
   - `v5cx_cf150` and `v5` have identical mask, split, batch, hist_row, cls3, cls10 and names; only Y differs.
   - Fail-closed checks work: a v9 archive is rejected against `targets_v5.npz` ("provenance/source labels differ"), and the wrong direction is rejected.
6. **GPU dtype.** The pos head and BCE run outside autocast on `h.float()`. Under forced bf16 autocast (CPU emulation, `--pospos` + `--pseudo`, 2 epochs), BCE received float32 logits and targets, and `P_pos` was finite (`chk/autocast_smoke.py`). `masked_soft_bce` already uses BCE-with-logits. `hashlib.file_digest` needs Python ≥3.11; the container has 3.11.16.
7. **Real-LF smoke of the exact planned setup.** Both directions ran with all 23 LED channels + phase, `--targets targets_v5cx_cf150.npz` and the v9 archives, 32 training cells and 24 eval cells. Loss was finite, `P_pos` had shape (24, 8), and resume with identical arguments succeeded (`chk/real_v9_smoke.py`).

## Findings

### MAJOR 1: rare-marker soft targets collapse to "negative" (reproduced, `chk/check_v9.py`, `chk/gmm_check.py`)

eta, phi and pi are not identified separately, so the ridge step sets phi = min(1/48, q). Whenever the observed-positive fraction q is below about 3-5%, phi absorbs most or all of the observed-positive component. The result is that clearly separated positive cells get low or zero targets. The table shows v9 training rows with y > mu1:

| Draw | Marker | d′ | ΔBIC | q | pi | Target on clearly positive cells |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | CD19 | 5.19 | +3286 | 0.0299 | 0.0098 | 0.31 max |
| 0 | CD56 | 5.75 | +3716 | 0.0220 | 0.0013 | 0.054 max |
| 0 | HLA-DR | 6.46 | +8944 | 0.0300 | 0.0099 | 0.31 max |
| 1 | CD19 | 8.77 | +2576 | 0.0092 | 0 | exactly 0 for every cell |
| 1 | HLA-DR | 5.57 | +4753 | 0.0115 | 0 | exactly 0 for every cell |

- **Consequence:** the BCE head is trained to call stained cells negative, which directly contradicts the intensity MSE through the shared encoder. AUROC against these labels is undefined (all-zero targets) or meaningless.
- **Root cause, partly:** the tied variance makes the "positive" component a far tail. A full-covariance GMM on the same cells gives a high-component weight of 25% versus 1.2% for draw-1 HLA-DR, and 8% versus 3% for draw-0 CD19.
- **No way to drop a marker:** the loader requires the archive mask to equal the curated mask, so degenerate markers cannot be excluded without a code change.
- **Fix before the GPU run (pick one):**
  - The fitter writes a per-marker `use` vector, false when pi<.01, ΔBIC≤0 or d′<1. `load_pospos` zeroes `ZM` for unused markers and pins `use` in `pospos_id`.
  - Bound the false-positive share of observed positives, e.g. phi(1-pi) ≤ 0.2 q.
  - Drop the noise layer and train on P(S=1|y) with fixed smoothing.

### MINOR

- **m1. CD123 local optimum (reproduced, `chk/cd123_ll.py`).**
  - The v9 CD123 fit has coincident means (d′=0.04, q≈0.45), which gives near-constant targets of about 0.47.
  - A tied 2-Gaussian solution exists with log-likelihood higher by 10,799 (draw 0) and 19,445 (draw 1). It describes a 0.4-0.6% outlier tail at y≈11-12, with 52 and 122 cells above 8.
  - So the warning "one component preferred by BIC" is wrong: the percentile multi-starts never seed an extreme tail. CD123 is degenerate either way, so exclude it (MAJOR 1). Also add a tail initialisation such as quantiles (50, 99.5) and/or robust emissions.
- **m2. The head's initialisation consumes the global torch RNG (reproduced, `chk/rng_check.py`).**
  - `--pospos X --pos-w 0` should be a no-op control, but 102 of 130 state tensors and P differ from the default run.
  - Enabled runs are therefore not seed-paired with the baseline (different data order and translations). That matters when the kill band is ±.0054.
  - Fix: create the head under `torch.random.fork_rng()` with a fixed sub-seed. Do this before launching.
- **m3. The tests only work from the delivery directory (reproduced).** Copied into the repo layout after the patch, 2 of 7 error:
  - the baseline shim misses `.to('cuda')` (the error is "no NVIDIA driver"), because `REPO/scripts/train_lf.py` is now the patched file;
  - the test writes to `ROOT/diagnostics/cpu_smoke.json`, which does not exist in the repo.
  Either keep the tests in `results/`, or adapt them: take the baseline from `git show <base>:scripts/train_lf.py` and remove the diagnostics write.
- **m4. Scoring is not implemented.** There is no AUROC or expected-mixture-intensity alternative, and `P_pos` is saved for eval rows only. AUROC against soft labels needs the other draw's archive (fitted on the eval draw's split 0, which is legitimate for scoring only). It is undefined for markers whose targets are all zero. Pre-register the scoring script.
- **m5. The noise layer is a fixed squash of the GMM posterior.** The EM updates of eta and phi are overwritten by the ridge projection every iteration. The noise layer therefore reduces to a fixed affine squash of the tied-GMM posterior, c1·P(S=1|y) + c0·P(S=0|y), with constants set by the prior modes. Codex discloses this. Phrase N049 results as prior-fixed label smoothing, not as measured labelling efficiency.
- **m6. Nondeterminism on GPU (nit) [unverified on GPU].** `scatter_add_` in `masked_soft_bce` is nondeterministic on CUDA. This is negligible next to cuDNN's existing nondeterminism.

## Landing guidance

- Land `scripts/train_lf.py`, `bsccm/n049_pospos.py` and `scripts/n049_noise_mixture.py` once the m2 fix (fork_rng) is in.
- Before the GPU run, implement the MAJOR 1 exclusion or repair, and refit v9.
- Do not copy `tests/test_n049.py` into repo `tests/` unchanged (m3).
