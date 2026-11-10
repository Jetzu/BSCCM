N028 writer deliverable: paired LF → virtual Wright/Giemsa pipeline. Run these files in place, or copy `vs_{common,register,train,eval,smoke}.py` together into `scripts/`. All imports work with `python3 -I` from the repository root. No existing source file needs modification. Requires the supplied SIF packages plus SciPy and Pillow.

From the repository root, choose a **shared, persistent** output directory with space for checkpoints and generated images. These example commands keep outputs under the N028 results folder; the coordinator can change `RUN` to another shared location. This writer has not submitted jobs or run training.

```bash
S=.agent-research/results/N028-virtual-wright-giemsa-staining-from-labe/codex
RUN="$PWD/.agent-research/results/N028-virtual-wright-giemsa-staining-from-labe/run-fov"
mkdir -p "$RUN"

# Main agent runs this on an allocated CPU node: 48 cores, one native thread/worker.
# Independent per-cell FFT/resize work; no reason to allocate a GPU here.
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif \
  python3 -I "$S/vs_register.py" --workers 48 --holdout fov --out "$RUN/register.npz"

# Main agent submits, after recording N028 job/config/GPU-hours and wandb sync cron.
CMD="$S/vs_train.py --registration $RUN/register.npz --holdout fov --out $RUN/train --epochs 40 --bs 32" \
  sbatch scripts/gpu_job.sbatch

# Submit after training completes, with the same registration file.
CMD="$S/vs_eval.py --checkpoint $RUN/train/ckpt.pt --registration $RUN/register.npz --out $RUN/eval --bs 32" \
  sbatch scripts/gpu_job.sbatch
```

Repeat in separate directories with `--holdout batch0` or `batch1` in **both registration and training**. Batch holdout trains on all GS cells of the other batch; FOV holdout uses split 0 versus split 2, excluding split 1. Optional training flags: `--out-size 256`, `--perceptual B`, `--patchgan`; default is 128px, frozen DinoBloom-S, no GAN. Do not change training settings when resuming; use a fresh output directory. `workers`, evaluation cadence, checkpoint interval and logging mode may change. Paths containing spaces are unsuitable for the existing sbatch wrapper's unquoted `$CMD` expansion.

Registration fits a global centre-crop fraction on 256 training pairs, scanning 0.60–0.80 in 0.01 steps. RGB intensity scale is the training sample's 99.9th percentile. Each pair gets a Hann-windowed integer phase-correlation shift (±12 LF pixels), refined by overlap NCC within ±2 pixels. Positive NCC <0.4, flat images and search-boundary solutions are rejected. Output keys: `gidx`, `hist_row`, `scale`, `dy`, `dx`, `peak` (NCC), `phase_peak`, `good`, `rgb_scale`, fitting provenance and counts. Positive shifts translate GS toward increasing LF coordinates. Cropping samples the original GS at shifted coordinates without wraparound; 256px uses the same physical field. The initial scale-fit stage restarts if interrupted; the subsequent pair stage checkpoints every five minutes and on SIGTERM/SIGUSR1. Rerun the identical command. Check boundary-scale warnings and rejection counts before training; do not tune registration on test performance.

The model is a four-level GroupNorm UNet with 24 LF inputs (all 23 LEDs plus phase, using unchanged `GPUNorm`). Only paired ±4px translations are used, with replicate padding. The pixel loss averages ordinary L1 and box-blurred L1 (5px at 128, 9px at 256). This reduces penalties from residual small displacement while retaining a sharp pixel constraint; it avoids letting each pixel independently choose a convenient shifted target. Add 0.1× spatial DinoBloom token cosine loss. Checkpoint loading exactly strips `backbone.` from the `teacher` dictionary, excludes `mask_token`, and loads strictly. Optional conditional PatchGAN uses hinge loss, weight 0.01. It may increase invented texture.

Training uses bf16 CUDA autocast, channels_last, AdamW, fixed LR and a deterministic per-epoch shuffle. Atomic `train/ckpt.pt` saves model/optimizers, RNG, next-batch cursor, pending evaluation and config/data-calibration hashes every 15 minutes (maximum flag value 20), at epoch boundaries and on either signal after the current update. Pending evaluation replays on resume. Workers only read/crop deterministically; parent RNG controls translations. GPU kernels need not be bitwise deterministic. Evaluation JSON lines go to `train/metrics.jsonl`; held-out scores never select a checkpoint. Wandb defaults to offline; `sync_lineage.jsonl` records its stable run ID and each SLURM job ID. **The coordinator must install the login-node sync cron for this run's wandb directory before submission.** A lineage file alone does not sync metrics.

Evaluation also checkpoints every five minutes and on signals; rerun the identical command. Outputs:

- `metrics.json`: mean per-cell PSNR and 11px Gaussian SSIM; frozen DinoBloom-B embedding cosine; generated→real and reverse exact-cell top-1/top-5, using a fixed random set of 1000 accepted held-out cells as both queries and candidates. Ties within 1e-7 rank pessimistically. Smaller sets explicitly report their candidate count.
- `generated.npz`: `images` = uint8 NHWC, `gidx` = global LF rows, `hist_row` = original GS rows, `rgb_scale`, `out_size`, `meta`. All accepted held-out pairs, ordered by histology row. Rejected pairs are excluded and counted. Generation streams through a resumable memmap; final NPZ writing avoids a full RAM copy. Reserve roughly **2×** final image storage during export (1623×128px ≈80 MB final; 10k×256px ≈2 GB final).
- `per_cell.npz`: PSNR/SSIM/cosine, row identities, batch, retrieval and montage selection IDs. `montage.png`: 24 seeded random held-out rows, real | generated | contrast-scaled LF brightfield (fewer rows in the smoke test). Colour means/stds are population statistics over all RGB pixels in [0,1]; summary scores are also stratified by batch.

Metrics and embeddings use the exported quantized generated RGB, compared with registered float real GS. For the existing GS student, feed `images/255` through its RGB `pipeline`, **not** the uint16/4095 `prep`. Compare student outputs for generated versus identically registered/cropped real GS; also retain the original full-field real-GS baseline, since cropping/resolution changes can alter the student. Test same-cell **v6 autofl truth within each batch**, alongside student/teacher agreement. Surface-marker predictions have no same-cell GS molecular truth. Good PSNR, DINO similarity or downstream agreement alone cannot exclude hallucinated chromatin/granules, background shortcuts or teacher-shared bias. Registration may favor easy cells and cannot correct rotation, deformation or stain-induced morphology changes. 256px output does not recover new LF information.

Planning estimate, **not measured**: H100 80GB, width 32, batch 32, Dino-S, no GAN: roughly 8–16 GB GPU memory and 1–4 minutes per ~12.5k-pair epoch including FOV evaluation; 256px roughly 12–24 GB and 2–6 minutes. Dino-B/GAN increase cost. GPU tensor convolutions/attention are expected to exceed 2× a 48-core CPU run, but that comparison and timings remain unverified. Forty default epochs ≈0.7–2.7 GPU-h; approve any aggregate request above 25 GPU-h before submission. CPU registration is parallel; the GPU optimizer sequence cannot be parallelized across independent steps, while batches and data workers are parallel.

Bounded CPU verification (eight actual cells; two forward/backward transactions, plus synthetic signal controls; no training run):

```bash
mkdir -p /tmp/$USER/codex-N028
# Preferred, when FUSE works:
TMPDIR=/tmp/$USER/codex-N028 PYTHONDONTWRITEBYTECODE=1 \
  apptainer exec /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif \
  python3 "$S/vs_smoke.py" --tmp /tmp/$USER/codex-N028/smoke --threads 48
# This host: FUSE failed; fallback needed timm installed ONLY under the temporary directory.
TMPDIR=/tmp/$USER/codex-N028 \
  /home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -m pip install \
  --no-deps --cache-dir /tmp/$USER/codex-N028/pip-cache \
  --target /tmp/$USER/codex-N028/deps timm==1.0.31
TMPDIR=/tmp/$USER/codex-N028 PYTHONPATH=/tmp/$USER/codex-N028/deps \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=48 \
  /home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u \
  "$S/vs_smoke.py" --tmp /tmp/$USER/codex-N028/smoke --threads 48
rm -rf /tmp/$USER/codex-N028
```

See `smoke_result.json` for the retained test record. Full-dataset registration, CUDA/bf16 execution, real training, SLURM resubmission, offline wandb synchronization, 1000-cell retrieval and downstream student validity remain for the coordinator. N028 notebook updates, independent Claude red-team review and any commit belong to the integration agent; this writer touched only this folder.
