# N007 writer delivery

Verdict: implementation ready for integration and runtime review; runtime tests are blocked by this sandbox's container startup failure. No training or GPU work was run.

## Files and integration plan

Copy these real code files after review:

| Delivery file | Repository destination | Purpose |
| --- | --- | --- |
| `bsccm_ssl.py` | `bsccm/ssl.py` | Illumination-safe views, Epps–Pulley SIGReg, multiview loss |
| `train_ssl_lf.py` | `scripts/train_ssl_lf.py` | Both SSL arms, resume/checkpoint handling, offline wandb, encoder health |
| `probe_lf.py` | `scripts/probe_lf.py` | One frozen probe implementation for SSL and supervised checkpoints |
| `test_bsccm_ssl.py` | Keep beside this delivery's `bsccm_ssl.py` | Eight CPU unittest cases |

`bsccm/data.py` and `scripts/train_lf.py` were read and are reused without edits. LFNorm supplies every model input: LED channels 0–22 and phase, with no fluorescence inputs. Scripts add the repository root to `sys.path` before package imports, matching `train_lf.py`, and have main guards for worker imports. They are intended to run under `python3 -I` after copying to `scripts/`.

The existing N007 notebook item is the parent record. The integrator must attach this delivery, verification evidence and subsequent review to that item, and commit only after control/red-team verification. No notebook, live-tree, Git or other outside-cwd files were modified by this writer; the explicit cwd-only task restriction takes precedence over general notebook-writing instructions.

## Exact commands

The following commands are instructions, not runs performed in this task. Install the three code files at the destinations above first. Dependencies are torch, timm, numpy, wandb, scikit-learn and threadpoolctl in the SIF environment.

Set up the environment in Bash. All temporary/cache files go under the designated temporary directory. `-B` also prevents Python bytecode writes when `-I` ignores environment settings.

```bash
export BSCCM_REPO=/home/hpc/v181bb/v181bb10/BSCCM
export BSCCM_DATA=/hnvme/workspace/v181bb10-bsccm
export BSCCM_N007=$BSCCM_REPO/.agent-research/results/N007-ssl-pretraining-of-the-lf-encoder-cross
export BSCCM_TMP=/tmp/v181bb10/codex-N007
mkdir -p "$BSCCM_TMP"
export TMPDIR=$BSCCM_TMP
export APPTAINER_TMPDIR=$BSCCM_TMP
export APPTAINER_CACHEDIR=$BSCCM_TMP/apptainer-cache
export XDG_CACHE_HOME=$BSCCM_TMP/cache
export TORCH_HOME=$BSCCM_TMP/torch
export PYTHONDONTWRITEBYTECODE=1
run_bsccm() {
    apptainer exec --bind "$BSCCM_TMP,$BSCCM_DATA,$BSCCM_REPO" \
        "$BSCCM_DATA/env/bsccm.sif" "$@"
}
```

CPU unit tests (seconds; no data reads):

```bash
cd "$BSCCM_N007/codex"
run_bsccm python3 -B -m unittest -v test_bsccm_ssl
```

After integration, run both arms from `scripts/` on a bf16-capable GPU allocation:

```bash
cd "$BSCCM_REPO/scripts"
run_bsccm python3 -I -B train_ssl_lf.py \
    --arm aug --out "$BSCCM_N007/aug" --arch resnet50 \
    --epochs 40 --bs 512 --views 2 --lr 2e-3 --lam 0.05 \
    --proj-dim 256 --workers 15 --rows all
run_bsccm python3 -I -B train_ssl_lf.py \
    --arm xillum --out "$BSCCM_N007/xillum" --arch resnet50 \
    --epochs 40 --bs 512 --views 2 --lr 2e-3 --lam 0.05 \
    --proj-dim 256 --workers 15 --rows all
```

Repeat the same command to auto-resume. `--resume none` rejects an existing checkpoint/final export; `--resume /absolute/path/ckpt.pt` imports a compatible checkpoint. Training configuration and eligible rows must match; output directory, worker count and debug step limit may change. `--rows train` restricts SSL to split 0. `--exclude-batch 0` excludes batch 0; multiple batch IDs may follow the flag.

A two-step debug command writes a resumable checkpoint and exits successfully without a final encoder export:

```bash
run_bsccm python3 -I -B train_ssl_lf.py \
    --arm xillum --out "$BSCCM_N007/debug-xillum" \
    --arch resnet18 --bs 8 --workers 0 --max-steps 2
```

`--max-steps` is a global committed-step limit. Raise/remove it on resume. Keep other arguments identical. It does not change the scheduler's planned 40-epoch horizon.

Both SSL probes (frozen features extracted on GPU, ridge/logistic fitting on CPUs):

```bash
run_bsccm python3 -I -B probe_lf.py \
    --ckpt "$BSCCM_N007/aug/encoder_final.pt" --kind ssl --arch resnet50 \
    --out "$BSCCM_N007/probe-aug" --n-train 60000 --n-test 20000 --workers 15 --cpus 48
run_bsccm python3 -I -B probe_lf.py \
    --ckpt "$BSCCM_N007/xillum/encoder_final.pt" --kind ssl --arch resnet50 \
    --out "$BSCCM_N007/probe-xillum" --n-train 60000 --n-test 20000 --workers 15 --cpus 48
```

Supervised probe, using the same sampling and evaluation (replace the checkpoint variable with the actual baseline run path, and match its architecture):

```bash
export BSCCM_SUP_CKPT=/absolute/path/to/supervised/run/ckpt.pt
run_bsccm python3 -I -B probe_lf.py \
    --ckpt "$BSCCM_SUP_CKPT" --kind sup --arch resnet18 \
    --out "$BSCCM_N007/probe-sup" --n-train 60000 --n-test 20000 --workers 15 --cpus 48
```

For the supervised checkpoint format actually read in `train_lf.py`, weights come from checkpoint `m` entries prefixed `enc.`. Supervised checkpoints must have 24 input channels (all LEDs plus phase). SSL probes require the encoder-only final export, with 48 channels. State loading is strict. All probes use seed 0 by default and save the sampled indices, so comparisons can verify identical rows.

The supplied commands submit no scheduler jobs. Placement, >=2x GPU speed justification, GPU-hour budget approval where applicable, run registration and login-node wandb sync cron are the submitting agent's responsibility. Dense CV solves use 48 BLAS threads by default. Folds run sequentially because each solve uses that CPU fraction and simultaneous folds would multiply peak RAM and oversubscribe BLAS.

After commands finish and no process uses the temporary directory:

```bash
rm -rf -- /tmp/v181bb10/codex-N007
```

## Implementation decisions

- `make_views` returns a list of tensors, each `(48,128,128)`. Its RNG is a device-matched `torch.Generator`. Augmentation uses positive isotropic affine sampling with reflection at the boundary, translation <=6 px, scale 0.9–1.1, channel gain jitter, noise and separable Gaussian blur. Dropped channels are zeroed after all augmentations; mask planes stay binary and spatially constant. There are no channel permutations, rotations, flips or circular shifts.
- Cross-illumination randomly partitions all 24 channels as evenly as possible. It supports 2–6 views with >=4 channels each; each channel, including phase, appears in exactly one view. Augmentation-only views use independent 10% channel dropout, restoring one random channel if all were dropped.
- SIGReg samples fresh unit directions every call, integrates CF error with the stated Gaussian weight and trapezoidal quadrature, and defaults to multiplying by batch size. Disable only that scaling with `scale_by_batch=False`. It does not standardize embeddings. The prediction loss uses squared L2 norm (sum over embedding coordinates); its multiview centre remains differentiable. SIGReg runs outside autocast in float32 (float64 inputs retain float64).
- AdamW uses wd=0.05, gradient norm clip 3, bf16 forwards and channels-last encoder inputs. The projector is Linear–BN–GELU–Linear–BN–GELU–Linear, with no output BN. Learning rate increases linearly for two epochs, then follows cosine decay. Debug runs shorter than two epochs only traverse the planned warmup.
- Shuffle and augmentation are deterministic by seed/epoch/cell. Like the supervised baseline, training drops the final incomplete minibatch each epoch; this keeps BatchNorm valid and the SIGReg batch-size factor constant. Checkpoints store encoder, projector, optimizer, scheduler, next epoch/batch, global step, Python/NumPy/CPU/CUDA RNG, eligible rows, configuration and partial-epoch sums. This avoids replaying committed batches and avoids dependence on persistent-worker prefetch. It promises the same sampled data on resume, not bitwise CUDA determinism.
- Checkpoints use fsynced temporary files and `os.replace`, every 20 minutes and at epoch end. SIGTERM sets a flag, then the next completed step checkpoints and exits 0. An interrupted/debug run exports no partial final encoder. Completing training retains `ckpt.pt` plus encoder-only `encoder_final.pt`; offline wandb history remains in its normal directory. A resumed completed checkpoint regenerates the final export if needed.
- Health uses a fixed, seed-selected 1024-cell subset of the eligible SSL rows, full presence and no augmentation. Effective rank uses centred encoder features and entropy of normalized singular values, not squared singular values; exactly constant features get rank 0. Correlation is also centred; zero-variance columns contribute zero correlation. `secs` and `imgs_per_s` count training time and unique cell presentations, excluding health/checkpoint time and the view multiplier. Partial-epoch timing is preserved on resume.
- `--rows all` is explicitly transductive to test-FOV images, as requested. It reads only split/batch metadata for inclusion; no target values or marker masks enter SSL. Use `--rows train` for an inductive SSL comparison. The health set follows exclusions, including batch exclusions.
- Ridge uses float64 closed-form primal/dual eigensolves, intercepts and fold-local StandardScaler. A seven-point alpha grid is `10**arange(-2,5)`, selected by five-fold mean validation MSE on each target's observed train rows. One eigensolve evaluates all alphas in a fold. No test labels select hyperparameters or scaling.
- Logistic regression uses multinomial lbfgs for the three classes, max_iter=2000, seven C values from 1e-3 through 1e3, and stratified five-fold balanced accuracy for selection. StandardScaler is inside the CV pipeline. Convergence warnings become errors. The script fails if training classes lack five examples, counts exceed available rows, or features/observed targets are nonfinite.
- Pearson definitions and group reporting thresholds match the baseline: >20 test observations per target, >=50 gated rows per group, >30 rows per within-class metric. Insufficient group metrics are explicit JSON nulls (empty CSV values). Missing marker metrics make the 12-marker mean fail explicitly. The group class oracle uses test truth's per-class mean, exactly as the baseline; it is descriptive and never enters a learned head.

Probe outputs are `probe.json`, `probe.csv`, sampled `probe_rows.npz`, `lf_feats_hist.npy`, `lf_feats_hist_indices.npy` (global rows), and `lf_feats_hist_hist_row.npy` (histology pack rows). Feature rows and both index vectors have identical ordering; all `hist_row>=0` cells are included, regardless of train/test sample membership. Those three arrays support later LF–Giemsa matching without ambiguity.

## Verification evidence and limitations

On 2026-11-08:

1. Environment startup command (exit **255**):
   ```bash
   TMPDIR=/tmp/v181bb10/codex-N007 APPTAINER_TMPDIR=/tmp/v181bb10/codex-N007 APPTAINER_CACHEDIR=/tmp/v181bb10/codex-N007/cache PYTHONDONTWRITEBYTECODE=1 apptainer exec --bind /tmp/v181bb10/codex-N007,/hnvme/workspace/v181bb10-bsccm,/home/hpc/v181bb/v181bb10/BSCCM /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -B -c 'import torch, timm, sklearn; print(torch.__version__, timm.__version__, sklearn.__version__)'
   ```
   Error: `container creation failed: image driver mount failure ... squashfuse_ll ... fuse: device not found, try 'modprobe fuse' first`.
2. Host fallback import checks (both exit **1**): `python3 -B -c 'import torch, numpy; ...'` fails with `ModuleNotFoundError: No module named 'torch'`; a separate `import numpy` fails similarly. No packages were installed. Runtime unit tests were skipped, as explicitly allowed when Apptainer cannot run. **0/8 runtime tests executed; no runtime pass is claimed.**
3. All four Python sources passed `ast.parse` and in-memory `compile(..., 'exec')`, exit **0**, without writing bytecode. The exact repeatable source-check command is:
   ```bash
   python3 -B - <<'PY'
   import ast
   from pathlib import Path
   for p in sorted(Path('.').glob('*.py')):
       source = p.read_text()
       ast.parse(source, filename=str(p))
       compile(source, str(p), 'exec')
       print('syntax OK:', p)
   PY
   ```
4. A read-only stdlib `zipfile`/`struct` inspection of `targets_v2.npz` confirmed the Unicode `(13,)` names array, exactly one `autofl`, the four `grp:` names, N=412,941, and exactly 15,712 `hist_row>=0` cells (exit **0**). This was a metadata check, not a normalization or data-loader execution. Additional AST source checks confirmed that both repository-root insertion expressions match the baseline, that the SSL module makes no flip/rotation/circular-roll calls, and that eight CPU tests are defined (exit **0**).

The eight unexecuted CPU tests cover Gaussian versus collapse/mean shift, the scalar CF integral and batch scaling, partitions/masks across 2–6 views, augmentation dropout/reproducibility, positive isotropic transforms/reflection/no flips, all-dropped recovery, invalid arguments, and finite differentiable loss.

Not verified: runtime imports/API compatibility inside the SIF; CPU tensor tests; actual LFNorm image loading; GPU memory at bs=512, numerical throughput or >=2x CPU speedup; CUDA/bf16 forwards/backwards; optimizer/scheduler resume equivalence; SIGTERM delivery and round-trip checkpoints; wandb persistence/sync; numerical ridge equivalence to sklearn; full feature extraction/probe fitting; or end-to-end quality/collapse behaviour. None of those was run. Memory use and logistic convergence require checking on the intended allocation; lower batch size if necessary and keep it fixed on resume.

The temporary directory created for this writer session was deleted at handback. Control and independent Claude red-team review remain for the integrator before landing or committing.
