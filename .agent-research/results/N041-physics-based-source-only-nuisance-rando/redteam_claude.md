# N041 red-team (Claude) of the Codex delivery `codex/`

Date 2026-11-12. Reviewer did not edit repo files or submit jobs. Scratch: `/tmp/n041_rt/` (scripts in `/tmp/n041_rt/scripts/`, outputs `mag.out axis.out tail.out sat.out run_*.log`).
Container for every run: `apptainer exec --bind /hnvme/workspace/v181bb10-bsccm --bind /tmp/n041_rt /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I ...` (CPU, torch 2.9.1+cu128, numpy 2.4.6).

## Verdict: LAND WITH FIXES

There are no blockers. The code does what the brief asked: the default path is bit-identical to the current version, there is no leakage, and the enabled path runs end to end on CPU.

One major design/contract issue: the plan-given gain size is 4-24x the real source variation. Three minor fixes are listed at the end.

## Findings, ranked

### BLOCKER: none

### MAJOR

**M1. The family-gain size (sd 0.1, from the plan/brief) does not come from source border statistics, and is 4-24x too large in normalised units.**
The Plan says "perturbation ranges from training-draw border statistics". Codex calibrated only the camera gain (shot noise) and the offset from borders. The family gain is hard-coded at log-sd 0.1.

Because GPUNorm divides by a per-channel `s_c` that varies 10x (BF 0.038 ... DF50 0.395), a single log-sd hits BF about 10x harder than DF50.

Measured on 256 source cells (draw 0, split 0, non-Giemsa; `scripts/mag.py`): sd of the per-cell border DC shift that the augmentation causes, in normalised units, against the real source-draw sd of per-cell border DC (n=1500, `scripts/axis.py`):

| family | aug (gain only) | real, draw 0 / draw 1 | ratio |
|---|---|---|---|
| BF | 2.40 | 0.10 / 0.11 | ~24x |
| DF_90 | 1.98 | 0.13 / 0.11 | ~16x |
| DF_80 | 1.25 | 0.08 / 0.08 | ~15x |
| DPC (4) | 0.85-0.90 | 0.06-0.13 | ~7-14x |
| LED119 | 0.45 | 0.13 / 0.15 | ~3x |
| DF_50 | 0.28 | 0.07 / 0.06 | ~4x |

N005's `led_norm_check.json` gives the same picture. The residual FOV-ring CV after slide_scale is BF 0.004, DF 0.006-0.05, DPC 0.008-0.016, against the augmentation's 0.10. Even the full between-slide range that slide_scale already removes is only about ±5%. The mean border DC is +-0.01 in both draws, so between draws there is essentially no DC-brightness shift for the gain to make the model invariant to.

Side effect: BF saturates at 4095 in 2.5% of perturbed cells (0.6% of BF pixels; `sat.out`). The real BF maximum is 3410, so the data never saturates.

Codex followed the brief, so this is not a code defect. But at this size the arm mainly tests invariance to a 2.4-sigma BF DC jitter that does not exist, and a flat result could wrongly kill the physics hypothesis.

Fix: derive a per-family log-sd from source split-0 border statistics (e.g. the sd over cells of log(border-median ratio), about 0.004-0.03), or add a `--phys-gain-sd` flag. Either way, record the choice in the item.

### MINOR

**m1. Tail weights under N041 fit on split-0 only.** This is a gated but unrequested behaviour change; the brief said "do not change any other behaviour".
- Under the batch protocol every source row (splits 0, 1, 2) is a training row. The split-0 rule applies to label calibration, not loss weighting.
- The numerical effect is negligible (`tail.out`, targets_v5, tail 0.5): per-row correlation of old vs new weights is 0.9928-0.9999. The new/old ratio p1-p99 stays within 0.95-1.18. Top-1% tail mass is identical to 3 decimals in both directions.
- It does not affect arm (a) vs arm (b). It does break exact comparability with N022 `n022_tail05_b10` (tail 0.5, focal 2, all source rows).
- Fix: revert to `m_tr[tr] = True`, or document it in the item.

**m2. `changes.diff` lacks `new file mode` headers.** `git apply --check` fails with `error: dev/null: No such file or directory`. `patch -p1` applies cleanly, and the result is byte-identical to the delivered `scripts/train_lf.py`, `bsccm/phys_aug.py` and `test_phys_aug.py` (now under `tests/`). Integrate by copying, or with `patch -p1`.

**m3. Shot noise doubles the noise variance in every perturbed view.** High-frequency noise rises x1.41 in all 23 channels (`mag.out`, e.g. BF 0.365 -> 0.512, DF_90 0.687 -> 0.971). This is the documented "one exposure-equivalent", but it is strong.

The offset is effectively a no-op: DC sd 0.01-0.06 normalised units, because border levels sit 200-3100 ADU above the 30-count baseline and the data has no pixel at or below 30. No clamp-floor hits occur after augmentation.

**m4. Resume compares the recomputed calibration floats exactly.**
- Reproduced bit-exact: the container recomputation equals both delivered JSONs exactly, under OPENBLAS 1 or 16 threads and against Codex's separate uv venv, on an AMD EPYC 9535.
- A different CPU microarchitecture after preemption was not tested [unverified]. A last-bit difference there would raise "refusing to resume" and end the preempt chain.
- Optional hardening: on resume, reuse `s['cfg']['phys_calibration']` when `rows_sha256` matches, or compare with a tolerance.

**m5. The ±1 um defocus size is plan-given and the data cannot confirm it [unverified].**
- Centroid offsets of DF50_Right vs DF80_Right (each relative to its full ring) are uncorrelated or slightly negative: corr -0.12 (draw 0), -0.25 (draw 1). Pure shared defocus predicts positive correlation with slope 1.84.
- So natural defocus is not detectable over morphology-driven centroid scatter (sd about 3 px).
- The coefficients do check out by hand. Half-ring mean slope q/sqrt(1.33^2-q^2) x 2/pi / 0.166 um gives DF80 about 3.0 px/um (Codex 3.15), DPC about 1.38 (1.37), DF50 about 1.65 (1.71), LED119 1.896 (1.896).
- The pixel size 0.166 um and objective NA 0.5 come from the BSCCM zarr metadata.

**m6. Cost and side effects of the consistency loss.**
- One extra forward and backward pass per step.
- BN running stats are updated twice per step.
- The unmasked MSE also trains the head of the target with no source labels (grp:CD123/HLA-DR/CD14 in b1to0). This is harmless, since that target is excluded from `r_mean_markers_valid`.
- n031_base took 498 s (b0to1) + 1590 s (b1to0), about 0.58 GPU-h per arm. With consistency, expect about 1.6-2x: roughly 2-2.3 GPU-h for both arms, against 1.5 planned. That is far below the 25 GPU-h approval threshold.

**m7.** The phase channel is not re-registered to the shifted DPC raw images. Codex documented this, and it is compatible with the plan.

## Checks that passed (with reproductions)

**1. Default-path identity.** I reviewed the diff against the working copy (SHA f54525d8...) line by line. With PHYS False:
- the new `ap.error` checks pass;
- the tail mask reduces to `tr`;
- `prep` calls `.cuda()` in the same order;
- the only addition is `consistency = p.new_zeros(())`, which uses no RNG and no graph;
- the loss is unchanged;
- `_NEW_DEFAULTS` omits phys_aug/consist at 0.0, so `_cfg` is unchanged.

Empirical check: I made CPU-converted copies of both files (sed `.cuda`->cpu, `ev=ev[:256]`; the CPU conversion keeps the diff unchanged) and ran `--protocol b0to1 --targets targets_v5.npz --max-train 256 --bs 32 --epochs 1 --workers 2 --seed 0`.
- Eval predictions bit-equal: True.
- Model weights bit-equal: True.
- Checkpoint `cfg` equal: True.
- train_loss 2.42733496427536 in both.
- Only `secs` differs. `result.json` args gain two keys at 0.0.
- An old-code checkpoint resumed by the new code printed `resumed 1` and exited 0.

**2. Unit tests: 11/11 OK** in bsccm.sif, both from the delivery layout and from the integrated `tests/` layout.

**3. Leakage and constraints.**
- `cal_rows = tr ∩ split0 ∩ hist_row<0`; `calibrate_source_borders` validates one draw, split 0 and hist_row<0 before any read. The test confirms only the selected rows are read.
- The recomputed delivered calibrations contain draws {0} and {1}, splits {0}, max hist_row -1, and are bit-exact.
- N041 requires b1to0/b0to1 and refuses include_gs, DANN and AdaBN. All four guard rails fire:
  - `--protocol fov --phys-aug .7`
  - `--consist .1` without `--phys-aug`
  - `--phys-aug .7 --adabn 100`
  - `--phys-aug 1.5`
- Only `led_array` is read: no fluorescence, and dpc only as the unchanged phase.
- The eval path `prep(train=False)` skips augmentation; `evaluate` and `fwd` are unchanged.

**4. Physics form.**
- GPUNorm's background is the global p50 flat field (`backgrounds_p50.npy`, cropped at the cell's FOV position) times a fixed per-slide x channel `slide_scale`. Neither is estimated from the cell crop, and neither is perturbed.
- The gain multiplies (raw-30) of the whole crop, cell and local background together. So it does not cancel: it survives as a uniform additive log(g)/s_c over the whole normalised crop, and cell-vs-local-background contrast is unchanged.
- That is the correct form for an illumination or instrument intensity change. Its size is the problem (M1).
- Family indices match `CHSETS`: BF {0, 22}, DF-low 1-8 (DF_50..DF_65), DF-high 9-17 (DF_70..DF_90), DPC 18-21. LED119 (NA 0.3994, below objective NA 0.5) belongs in BF.
- Defocus frame, checked empirically on both draws (n=1500 each, centroid of above-background signal):
  - Right channels vary along x (sd 2.9-3.6 vs 0.8-1.8 px) and Bottom channels along y.
  - corr(ΔRight_x, ΔBottom_y) is -0.55 to -0.92 (Codex frame predicts <0).
  - DPC corr(ΔL-R_x, ΔB-T_y) is +0.82/+0.89 (predicts >0).
  - DPC_R vs DF80_R x is +0.33/+0.37 (>0).
  - LED119 vs DPC_Top y is +0.13/+0.21 (>0, weak).
  - There is no transposed or reflected axis.

**5. Tail weights:** see m1.

**6. Consistency loss.**
- The two views are independent: separate physics draws and separate ±4 px rolls.
- The MSE is taken on float32 regression outputs outside autocast, with gradients through both views (the test asserts p.grad = -p2.grad). There is no label masking, by design.
- End-to-end CPU runs of arm a (`--phys-aug .7 --consist .1`) and arm b (`+ --tail-weight .5 --focal-c10 2`) exited 0.
- An identical rerun printed `resumed 1` with the calibration pinned in `cfg`, and `phys_aug_calibration.json` was written (draw 0, 204 rows at max-train 256).

**7. GPU (static review only; the login node has no GPU) [unverified on GPU].**
- grid_sample input (B·23,1,H,W) matches grid (B·23,H,W,2), and the B-major reshape order is consistent.
- `torch.poisson` on CUDA float32 is supported.
- The device check compares cuda:0 with cuda:0.
- Transient memory for the augmentation is about 3 GB at bs 256.
- I found nothing that should crash. The integrator's GPU smoke test should confirm.

## Fixes before landing
1. M1: set the gain log-sd per family from source border statistics, or add `--phys-gain-sd` and decide. If 0.1 is kept, record it as a plan deviation.
2. m1: revert the split-0 tail-weight restriction, or document it in the item.
3. m2: integrate by copy or `patch -p1`; `git apply` fails on the diff.
4. Optional, m4: tolerance or reuse-on-resume for the pinned calibration.
