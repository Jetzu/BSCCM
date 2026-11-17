N041 delivery: opt-in raw-count augmentation and prediction consistency

Files to integrate: `scripts/train_lf.py` (full patched file), `bsccm/phys_aug.py`, and `test_phys_aug.py` (place the test under `tests/`). `changes.diff` includes all three at their integration paths. The patch is against the working copy of `train_lf.py`, SHA256 `f54525d87a73b79d6cf5b59b12002b8f9d6712dd60e4bc9755c6ebc5f3d06da9`, including the existing N042 low-pass option. Repository files and data were not edited.

Both new flags default to zero and are in `_NEW_DEFAULTS`. Off: no calibration, augmentation import, extra forward pass, or new random draws. Eval and model-forward function bodies are unchanged. Enabled: add `--phys-aug .7 --consist .1` to the existing N031 base command, retaining its direction-specific inductive SSL init and `--targets targets_v5.npz`. Arm (b) additionally uses the existing `--tail-weight .5 --focal-c10 2` (N022's focal exponent). The writer ran no training.

The first view keeps the original supervised losses. A second, independently sampled view supplies `consist * mean((prediction1 - prediction2)**2)` across all samples and all K regression outputs, with gradients through both predictions. Each view independently retains a 30% physics-unperturbed fraction; the existing ±4-pixel translation still applies. Raw/phase/row tensors transfer once when using two views. New perturbations use batched device operations, one `grid_sample`, and `torch.poisson`; no per-cell CPU augmentation loop. GPU throughput/memory have not been measured.

The enabled path requires a batch-held-out protocol and rejects Giemsa inclusion, DANN, and AdaBN. Its calibration uses at most 512 non-Giemsa split-0 rows from `tr`, selected with a private seed-0 RNG. It never indexes other image rows. The calibration is written to `phys_aug_calibration.json` and included in checkpoint config checks. With N041 enabled, tail-density weights also fit/normalize on source split-0 rows; the original tail-weight behavior is retained when N041 is off. Prebuilt labels, other losses, training-row selection, normalization, and evaluation are retained.

For each active sample:

- Gain: `exp(N(0,.1²))`, independently for BF, DPC, DF-low (channels 1–8), and DF-high (9–17); shared within each family. LED119 joins BF because its NA is 0.3994.
- Offset: independent Gaussian per family, shared across its images. Its SD is the source-border estimate of noise in a 128-pixel mean, bounded to [0.5,3] ADU. This is a small baseline-jitter convention, not a measured temporal offset-drift distribution.
- Shot noise: for gain-adjusted signal `s`, add `k*Poisson(max(s,0)/k)-max(s,0)`, where `k` is fitted ADU/electron. Added variance is `k*max(s,0)`. Recorded noise/artefacts remain; the augmentation adds one exposure-equivalent shot variance. Signal below the 30-count camera baseline is preserved, with zero additional shot variance there. Only active outputs are clipped to the 12-bit ADC range. Float32 avoids int16 rounding/overflow before GPUNorm.

Camera estimate: outer 8-pixel borders, non-overlapping 2×2 tiles. The Haar residual `(a-b-c+d)/2` cancels constant/linear shading and has the pixel-noise variance. In 12 signal quantiles, `median(residual²)/0.4549364` estimates variance. Fit `variance=k*(mean(raw)-30)+read_variance`, constrain the read intercept nonnegative, and take the median positive channel slope. This spatial effective gain is approximate: fixed pattern, texture, spatial correlation, and quantization can bias it; some channel fits are weak/nonpositive. Full fits and selected row IDs are supplied in the calibration JSONs. These are independent source-specific audits, never pooled or hard-coded into training.

| Training source | Effective ADU/electron | Offset SD: BF / DPC / DF-low / DF-high (ADU) |
|---|---:|---|
| Draw 0 (`b0to1`) | 0.59448 | 2.621 / 3.000 / 1.245 / 1.441 |
| Draw 1 (`b1to0`) | 0.52075 | 2.900 / 3.000 / 1.245 / 1.441 |

Defocus geometry: the [BSCCM paper](https://arxiv.org/html/2402.06191v1), [authors' LED-mask code](https://github.com/Waller-Lab/BSCCM/blob/dfd2fad7e882db98bbc5d08e3e81a062fb5a90e8/bsccm/led_array_calibration.py), and [quasi-dome layout](https://github.com/Waller-Lab/BSCCM/blob/dfd2fad7e882db98bbc5d08e3e81a062fb5a90e8/bsccm/quasi_dome_design.json) define illumination. Layout SHA256: `93fb92f6f5098c285388b72434a2a38bc103aedf07cd8119337b7398ec4b0e4a`. Use the authors' 8-mm z-offset and equal LED weights; assume PBS refractive index 1.33 and image axes x=NA_x, y=−NA_y. From each LED's directional NA vector `q`, compute ray slope `q/sqrt(1.33²-|q|²)`, then average over that channel's LEDs. Displacement is this mean slope times one shared `z ~ Uniform(-1,+1)` µm, divided by the metadata's 0.166 µm/pixel. Bilinear sampling uses border replication. The coefficients are embedded; training needs no network access.

This first-moment approximation gives zero shift for symmetric BF/full DF rings, ±1.3704 px/µm for directional DPC, 1.7066/2.0245/2.5050/3.1475 for directional DF, and −1.8964 vertically for LED119. “Sub-pixel” means continuous fractional-pixel resampling: ±1 µm physically permits >1-pixel shifts. Symmetric-pattern blur and wave-optical changes are outside this approximation. Phase is not perturbed or recomputed. No rotations, flips, channel relabeling, fluorescence input, or input cleaning were added.

Validation: 11 CPU unit tests passed (`test_results.txt`): exact P=0 identity/RNG preservation; gain/offset/probability/shift statistics; family sharing; impulse displacement/channel isolation; dtype/shape/determinism; unchanged inactive samples; signal-dependent shot variance; synthetic gain recovery and pre-read source/split/Giemsa rejection; real GPUNorm off/eval equivalence and phase preservation; legacy config omission; symmetric consistency gradients. Syntax checks passed, and AST comparison confirms unchanged `evaluate` and `fwd`. No GPU execution. The isolated `/tmp/codex_N041/venv` matches all five requested scientific-library versions and uses torch 2.9.1+cpu (`environment.json`).

Reproduce CPU checks from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/codex_N041 OMP_NUM_THREADS=2 \
  /tmp/codex_N041/venv/bin/python -m unittest -v test_phys_aug.py
```
