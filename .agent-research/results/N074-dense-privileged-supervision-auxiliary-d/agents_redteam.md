# N074 red-team of the Codex dense-aux patch (Claude, 2026-11-19)

Scope: codex/{train_lf.py, train_lf.diff, bsccm/dense_aux.py, pack_fluor_aux.py(.sbatch), tests/}. No files changed except this report.
CPU checks used /tmp/n074_codex/venv, scratch in /tmp/n074_rt (reg*.py, cap.py, shiftloss.py, cof.py).

## Verdict: no BLOCKER. One MAJOR (only with --diag-aug), five MINOR.

### MAJOR-1: fluorescence is not co-registered with the LF crop, and the anti-diagonal reflection turns that into contradictory targets
Measured on real pack rows: mean FFT cross-correlation peak of corner-subtracted fluorescence vs the DPC phase image, 600 cells per
draw x panel. Results are identical for draw 0 and draw 1 and for 'all' and 'unstained'. Offsets (dy,dx) in pixels at 128, in CHANNELS order:
426-446 (0,+2) | 500-550 (-1,+2) | 550-570 (-1,+4) | 585-625 (-1,+5) | 627-673 (-1,+3) | 690- (-1..-2,+6..+7).
Against the darkfield sum: +3..+8 px in x. Centre of mass: fluorescence x=66.5, phase x=64.1.
Code: dense_aux.py:61-62 together with train_lf.py:200-206. These lines reflect the target along with the input, so an instrument offset (dy,dx)
becomes (-dx,-dy) in the reflected samples. For 690-, the signal sits 7 px to the right in unreflected views and 7 px up in reflected views.
The N068 reflection is designed to be indistinguishable from a real view, so the decoder gets two conflicting geometries.
On real 'all' cells (shiftloss.py), the best mixture prediction leaves an irreducible MSE of 0.04-0.11 against a total standardised variance
of about 1.13, in 4 of 6 channels. The share of the predictable cell signal lost is much larger. The aux gradient also rewards the encoder
for detecting the reflection, which works against the purpose of N068.
The README claim "registered under ... anti-diagonal augmentation" holds only for test_d, because that test builds the input from the target.
Fix: (a) estimate per-channel offsets on the training draw's split-0 cells and shift-correct the fluorescence in the packer before pooling, or
(b) refuse --diag-aug when --dense-aux is set, or (c) skip the dense loss for reflected samples.
Without diag-aug the constant offset is learnable, so this does not matter there.

### MINOR
1. GS test cells feed the packer statistics. pack_fluor_aux.py:187 and :204 use all split-0 unstained cells.
   Of these, 8007/13095 (61%) in draw 0 and 4495/37251 (12%) in draw 1 are hist_row>=0 Giemsa cells. train_lf keeps exactly these cells
   out of training to protect the clean GS autofl test.
   Only 18 scalars per draw are affected and nothing reaches the GOAL eval draw, so the effect is negligible. Still, exclude hist_row>=0.
2. Gradient-clip coupling. clip_grad_norm_(model.parameters(), 2.0) now includes the decoder gradients. When those dominate the norm, the
   scalar heads and encoder get a lower effective learning rate, so the "omit dense flags" control is not LR-matched.
   train_loss also now includes the dense term. Log dense_loss and the gradient norms separately.
3. Synchronous I/O. DenseTargets.get (dense_aux.py:165) fancy-indexes 256 random 48 KB rows from a 20 GB memmap on /hnvme (Lustre).
   It does this in the main process every step, with no worker prefetch, plus a GPU sync from isfinite().all().
   Epoch 1 may be I/O-bound. Read the rows in the RawSet workers instead and keep the guard.
4. Import path. pack_fluor_aux.py:19 imports dense_aux from <script dir>/bsccm, and code_sha256 hashes that same path. Both break if the
   packer is integrated into scripts/.
5. Durability cost. Each of about 12.9k chunks pays a full 20 GB memmap flush, an fsync, a JSON fsync and a directory fsync on Lustre.
   This is probably fine in 4 h but has not been measured. Consider --chunk-size 256-512.

### Checked and holding
- (a) Leakage: DenseTargets only allows rows in tr from the single source draw, and raises (not assert) before reading pixels. All packer
  statistics are per draw. The flag refuses fov, include-gs, dann and adabn.
- (b) Fluorescence never enters x. evaluate() is AST-identical, and neither the decoder nor the reader runs at eval.
- (c) Capture (cap.py, real N031 checkpoint): the 48-channel ResNet-50 map is (B,2048,4,4), and avgpool(map)==h. The method is restored
  after an exception. Under bf16 autocast with channels_last the map is bf16 channels_last, the decoder runs in fp32 and the encoder grad
  is nonzero. The dual_zoom second view runs after the patch is restored. The harness has no torch.compile.
- (d) Mechanics: the roll matches GPUNorm (CPU randint (2,B), the same dims, RNG order preserved). The odd-shift area overlap is exact,
  the zoom crop offset is identical and the reflection has no channel swap.
- (e) dense_mse normalisation is correct: sum w*err^2 / (6 sum w). Decoder fp32 activations are about 1-2 GB at bs 256.
- (f) Channel order 5..0 matches N064, and the corner median uses r>=64. Real cofactors are 1.5-6.3 ADU, so the 1-ADU floor is not hit.
- (g) The flag-off tests exec the current repo working copy (sha 88f70936 = provenance). train_lf.diff reproduces exactly against it.
