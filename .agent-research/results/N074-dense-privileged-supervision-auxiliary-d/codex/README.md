# N074: optional dense privileged supervision

This folder contains a full patched copy of the **current working-copy** `scripts/train_lf.py`, including N068/N073, and `train_lf.diff` against that file. Nothing was installed into the repository. No SLURM job was submitted and no production pack was written.

`--dense-aux ''` (default) preserves the legacy model, optimizer groups, scalar-loss arithmetic, RNG draws, checkpoint keys/config, and argument dumps. The new defaults are omitted using `_NEW_DEFAULTS`; dense supervision has its own guard and does not activate `NNEW`. An inactive dense weight/width is omitted too. Evaluation's function body is unchanged.

## Implementation

`--dense-aux /path/fluor_aux64.npy` adds a training-only decoder; `--dense-w` defaults to **0.5**, `--dense-width` to **64**. The usual `enc(x)` call still produces the pooled features for every scalar head. A temporary capture of timm's `forward_features` obtains its NCHW pre-pool map during that same call, including the existing SSL presence planes or LFBackbone LED gate/resize. It restores the original method immediately. There is no second encoder pass or alternate pooling operation.

The decoder projects to its width with a 1x1 convolution, applies FiLM from a learned 16-dimensional antibody-condition embedding, then uses four bilinear-upsampling/3x3-convolution/GELU blocks and a six-channel output convolution. Conditions are `unstained`, `all`, and each of the eight single antibodies. Its output is `(B,6,64,64)` in `CHANNELS` order. The weighted pixel MSE divides by the sum of pixel weights times six: disk `r<32` has weight 1; corners weight .25. The mask follows the target's augmentation. Only the primary input view contributes dense loss when dual zoom or consistency is enabled.

ResNet-family backbones and **Cytoland** are supported. Token-only `dinov3_s`, `dinov3_b`, and `dinobloom_b` are rejected before loading their weights. Other architecture names are conservatively refused. Dense supervision requires `b1to0` or `b0to1` and `include-gs=0`, `dann=0`, `adabn=0`. It supports the existing diagonal augmentation/TTA, LED dropout, censor loss, zoom and SSL initialization. It leaves scalar targets and all evaluation labels untouched. The decoder and fluorescence reader are unused at evaluation; decoder weights remain in training checkpoints for resume. There is no image-to-marker inference/readout (K027).

`DenseTargets.get()` raises an explicit `AssertionError` before reading pixels unless every row belongs to the exact selected training-row set **and** its one source draw. It also validates global row identity, split/draw/panel digest, channel order, float16 shape and completion metadata. Resume config pins the transform metadata and the packed file's path/size/mtime.

## Fluorescence packing and choices

The production output is exactly **one** float16 array `(412941,6,64,64)`, row = global index, plus `fluor_aux64.json`. The array payload is 20,296,876,032 bytes (about 18.9 GiB). No chunk files are made. JSON's atomic temporary file is replaced in place.

For each cell/channel: reverse native channels `5..0` to `['426-446','500-550','550-570','585-625','627-673','690-']`; subtract the median of its own 128-pixel crop outside the disk centred at `(64,64)`, `r<64`; mean-pool 2x2; apply `asinh(x/s_c)`; subtract that draw/channel's split-0 mean and divide by its population standard deviation. Negative background-subtracted pixels are retained. No shading correction, clipping, unmixing, scalar fluorescence labels or held-out-draw statistics enter another draw's transform.

The unspecified robust-scale estimator is fixed to **1.4826 x MAD** of pooled foreground pixel values on that draw's split-0 unstained cells. For bounded memory/read cost, use up to **4096 evenly spaced eligible global rows**, and **64 fixed evenly spaced foreground positions** per selected cell. The same positions apply to every channel. This is a robust sampled estimate, not an exact full-pixel median. The **1 ADU cofactor floor** avoids amplifying a zero/near-zero noise scale. Standardization uses **all pixels of all split-0 cells** in that draw, including unstained cells, streamed with float64 sums/squared sums; SD floor `1e-6`. Constants, sample positions/counts/row hashes and estimator choices are recorded. These choices resolve unspecified details; the requested transform order and draw isolation are unchanged.

`GPUNorm` ordinarily draws CPU integer shifts and rolls the normalized 128-pixel LF image. With a dense target present, `prep()` requests unshifted normalization and reproduces the same CPU draw/roll explicitly. The target is nearest-expanded 64->128, receives the identical cyclic integer shift, zoom and selected anti-diagonal reflection, then is mean-pooled back to 64. An odd LF shift is a half target pixel: this computes its exact area overlap for the piecewise-constant 64-pixel representation. Original subpixel fluorescence detail is already unavailable after packing; shifting transformed/prepooled targets cannot reconstruct it. This introduces the expected interpolation smoothing for odd shifts, with no displacement error on that representation. Fluorescence channels are **never swapped**. LF low-pass/physical perturbations change only LF intensities, not registration.

The packer bounds pending worker results to twice the worker count. It checkpoints source-only standardization accumulators after each chunk and each finished cofactor fit. An interrupted unfinished cofactor fit (at most 4096 cells/draw) is recomputed. For output, it flushes/fsyncs each written chunk **before** committing the next-row offset in atomically replaced/fsynced JSON. Uncommitted chunks are overwritten on restart. SIGTERM/SIGUSR1 stop at a durable boundary with exit 143. A completed identical run is a no-op. A directory flock prevents concurrent packers there; changed inputs/code/chunk size/calibration options require a new output. The sbatch uses 48 cores on `preempt_cpu` and the same `preempt-run.sh` chain as N064. Worker count can change on resume.

## Exact commands

All commands below are run from this folder. Scratch, environments and caches stay in `/tmp/n074_codex`.

```bash
export N074_STAGE=/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N074-dense-privileged-supervision-auxiliary-d/codex
cd "$N074_STAGE"
mkdir -p /tmp/n074_codex
export TMPDIR=/tmp/n074_codex
export XDG_CACHE_HOME=/tmp/n074_codex/cache
export PIP_CACHE_DIR=/tmp/n074_codex/pip-cache
export PYTHONPYCACHEPREFIX=/tmp/n074_codex/pycache
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
```

The requested container probe failed with `squashfuse_ll ... fuse: device not found`. The fallback used for the tests:

```bash
python3 -m venv /tmp/n074_codex/venv
/tmp/n074_codex/venv/bin/pip install torch==2.15.0+cpu torchvision==0.29.2+cpu \
  --index-url https://download.pytorch.org/whl/cpu
/tmp/n074_codex/venv/bin/pip install timm==1.0.31 numpy==2.5.4 pandas==3.1.1 pyarrow==26.0.1
/tmp/n074_codex/venv/bin/python -I -B tests/test_dense_aux.py > tests/cpu_tests.log 2>&1
bash -n pack_fluor_aux.sbatch
/tmp/n074_codex/venv/bin/python -I -B tests/run_staged.py --help
```

Tests create and pack a 64-row fixture under `/tmp/n074_codex/test`; its global source row IDs are in `tests/provenance.json`. Tests rerun from scratch and deliberately remove **only their named fixture outputs** there. For a separate restart of the finished fixture pack:

```bash
/tmp/n074_codex/venv/bin/python -I -B pack_fluor_aux.py \
  --source /tmp/n074_codex/test/fixture --out /tmp/n074_codex/test/fluor_aux64.npy \
  --smoke --workers 2 --chunk-size 7 --cofactor-cells 12
```

Production packing command, **provided but not executed** (also encoded in `pack_fluor_aux.sbatch`):

```bash
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$N074_STAGE/pack_fluor_aux.py" \
  --source /hnvme/workspace/v181bb10-bsccm \
  --out /hnvme/workspace/v181bb10-bsccm/fluor_aux64.npy --workers 48
```

`tests/run_staged.py` supplies the read-only repository imports and this folder's auxiliary module, so the staged full copy is runnable without copying files into the repo. After normal integration as `scripts/train_lf.py` plus `bsccm/dense_aux.py`, the ordinary harness command works directly.

Example future b1to0 GPU run, **not executed**:

```bash
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$N074_STAGE/tests/run_staged.py" \
  --out /hnvme/workspace/v181bb10-bsccm/runs/n074_dense_s0_b10 \
  --protocol b1to0 --channels all --phase 1 --arch resnet50 \
  --init-ssl /hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/encoder_final.pt \
  --targets targets_v5cx_v10.npz --epochs 10 --bs 256 --lr 0.0005 \
  --workers 15 --seed 0 --include-gs 0 --w-cls 0.5 \
  --dense-aux /hnvme/workspace/v181bb10-bsccm/fluor_aux64.npy --dense-w 0.5 --dense-width 64
```

For b0to1 use `--protocol b0to1`, `n031_ssl_aug_ex1/encoder_final.pt`, and a distinct output directory. For a matched control omit the three dense options. Existing N068/N073 flags can be appended unchanged. No training command above is a CPU smoke command.

## Verification and remaining limits

See `tests/cpu_tests.log` and `tests/provenance.json` for the actual run and source hashes.

- Flag off: exact tensor equality for forward outputs, training loss and gradients against the **unpatched working copy**, with/without SSL and with/without N068/N073 enabled; equal config, model keys, argument dump and serialized checkpoint bytes. A complete four-step CPU run also produces identical final model tensors and evaluation predictions.
- Flag on: ResNet-50 loads the **actual N031 source-only SSL checkpoint**, uses all 48 channels including correct absent-LED presence planes, emits `(2,6,64,64)`, has finite combined gradients, and receives nonzero encoder gradients from **auxiliary loss alone**. A complete small run writes and strictly resumes its dense checkpoint.
- Evaluation: exact AST equality of `evaluate()`; decoder and dense-label reader replaced by failing stubs during eval, which completes successfully. Held-out/non-training/out-of-range requests are rejected.
- Packer: all **64 actual fluorescence crops** exactly reproduce an independent hand calculation including channel order, corner median, pooling, sampled robust cofactor, source-only standardization and float16 bytes. Corrupting the uncommitted suffix and restarting reproduces the original file. A real SIGTERM exits 143 with partial progress and CLI restart reproduces every byte. Completed restart is a no-op; no chunk files remain.
- Registration: actual `prep()` tested on asymmetric six-channel synthetic images for all 81 shifts in `[-4,4]^2`, mixed reflected/unreflected samples and zoom 1/2. Targets equal the identically transformed input after removing **only the LF directional permutation**.
- Actual locally pretrained Cytoland's gate/resize/pooled output and spatial capture are checked; unsupported token architectures fail clearly before weight loading.

Only 64 unique real crop rows were read, CPU batches were two, and no GPU was used. Scalar labels and LF calibration in the fixture are synthetic, so these checks establish mechanics, not GOAL accuracy. The unchanged evaluator emits NaN correlations on the four-cell evaluation fixture because its per-target minimum is 50 cells; these are expected and are not NaN training losses. Existing result-file ResourceWarnings are left unchanged.

The full 412941-row pack, 48-worker throughput/filesystem durability at scale, cluster container runtime, GPU AMP/memory/throughput and any effect on the v10 GOAL remain unverified. The decoder is float32 outside the existing encoder autocast context. The sampled cofactor and odd-shift interpolation are explicit modeling choices, not measured improvements. File size/mtime and metadata are pinned for resume; the 20 GB pack is not rehashed on every training startup. Rebuild under a new filename if content changes.
