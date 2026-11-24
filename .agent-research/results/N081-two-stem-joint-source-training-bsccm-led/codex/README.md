# N081: optional IFC second source

`train_lf.py` is the complete patched current harness; `train_lf.diff` applies to `scripts/train_lf.py`. Use `tests/run_staged.py` for the staged copy's read-only repository imports. Integration needs only the harness change. The repository was not modified. Only CPU fixtures were run, with no SLURM submission.

## Behavior

`--ext-ifc ''` disables all additions. `_run_args()` removes all four IFC options when disabled, including nondefault dependent values; `_NEW_DEFAULTS` preserves checkpoint compatibility. BSCCM `prep()`, `fwd()`, and `evaluate()` are unchanged. `eval_preds.npz` and downstream GOAL scoring retain their meaning.

Enabled mode requires SSL-initialized `resnet50`, `--channels all --phase 1`, and the ordinary linear MSE regression head. BSCCM's convolution still takes 48 channels, including presence planes. IFC adds a bias-free 3→64 convolution, 7×7, stride 2, padding 3. Both sources use the **same** bn1, activation, maxpool, layers 1–4, and global pool. BatchNorm parameters and running statistics are shared; each training step processes BSCCM followed by IFC.

The IFC stem starts from the mean of the BSCCM convolution's **Brightfield data-channel** filters, copied to all three IFC channels without rescaling. The current schema has one Brightfield channel, index 0, so this is a singleton mean. Phase, directional DPC, darkfield, LED119, and the presence filters are excluded. SSC receives this initialization only; its filter trains independently afterward.

IFC CD14/CD19/CD3/CD45/CD56 select the original `reg.weight`/`reg.bias` rows for the corresponding `single:` markers, followed by IFC scale/offset parameters initialized to 1/0. BSCCM uses the identity affine. A four-output linear head predicts IFC CD15/Siglec8/CD4/CD8. Names determine column routing, including reordered packs.

Every BSCCM optimizer step adds one IFC batch of `--bs` cells, cycling a shuffled loader as needed. One backward pass uses `BSCCM_loss + ext_w * IFC_masked_MSE`. MSE averages observed cells within each marker, then present markers. Nonfinite targets and optional `labels.npz['mask']` entries are excluded. Inputs are divided by `chan_scale`, then independently rotated by multiples of 90°, optionally reflected, and rolled by integer shifts in `[-4,4]²`. Evaluation has no augmentation.

Requested donors default to `sample0,sample1,sample2`; the test donor defaults to `sample2` and is always removed from training, including explicit lists. Unknown donors, no remaining training labels, or fewer than `--bs` eligible cells fail. Training reads assert the donor boundary before accessing each input/label row. IFC loaders use separate RNG generators; training shuffle is seeded per epoch.

Epoch metrics appear under `log[i]['ifc']`; final metrics are `result.json['final']['ifc']`: per-marker Pearson/count/MSE, macro Pearson, five-overlap Pearson, and masked MSE. Undefined correlations are omitted; undefined aggregate metrics are null. `ifc_train_loss` is unweighted. Final `eval_ifc.npz` contains `P,Y,M,G,sample,names`; `G` indexes the IFC pack and contains only test-donor cells.

Checkpoint config pins effective splits, row hashes, label-file SHA256, scales, and LF absolute path/size/mtime/schema. Full LF hashing is avoided; keep packs immutable. `--init-full` remains strict and requires matching IFC modules when IFC is enabled.

## CPU verification (executed)

From this folder:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/tmp/n074_codex/venv/bin/python -I -B tests/test_n074_regression.py > tests/n074_regression.log 2>&1
/tmp/n074_codex/venv/bin/python -I -B tests/test_ifc.py > tests/ifc_cpu.log 2>&1
```

Existing environment: PyTorch `2.15.0+cpu`, timm `1.0.31`; no installation needed. All scratch artifacts are under `/tmp/n081_codex`. Tests reconstruct the N074 64-cell fixture from real LF/phase/fluorescence crops with synthetic scalar targets/calibration. IFC is a synthetic 24-cell, three-donor pack. SSL fixture weights are synthetic except the shared-gradient test and reused dense-backward test, which load the actual source-only N031 checkpoint.

`tests/n074_support.py` reuses all seven N074 integrated checks inherited from its original `codex/tests`. Adaptations redirect paths, use today's baseline/packer, and compare the baseline's `_run_args()` instead of `vars(a)`, because today's baseline already removes inactive dense/aux options. No numerical assertion is relaxed. `tests/preempt_probe.py` redirects the existing SIGTERM check to today's packer.

N074 checks exact flag-off outputs/losses/gradients/config/argument dumps/model keys/serialized checkpoint bytes, unchanged evaluator AST, and full-loop final weights/eval predictions. It also checks dense supervision, packing/restart, augmentation registration, and architecture restrictions. IFC checks donor assertions, poisoned test-donor inputs/labels, normalization, D4/shift transforms, masking, independent shared-trunk gradients, exactly five shared regression rows, disabled dependent flags, reordered markers, dual zoom, and **two-step full runs in both directions**, including artifacts and strict resume/split-change rejection. See logs and `tests/provenance.json`.

## b1to0 / b0to1 commands (provided, not executed)

These retain the existing v10 20-epoch censored-loss recipe and add IFC at weight 0.5. Output stays inside this folder. The packer's default output directory below was absent during this task: supply a completed `lf.npy`/`labels.npz` pack there, or set `IFC_DIR` to an existing pack. No packing or GPU run was performed.

```bash
export N081_STAGE=/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/codex
export BSCCM_REPO=/home/hpc/v181bb/v181bb10/BSCCM
export BSCCM_DATA=/hnvme/workspace/v181bb10-bsccm
export IFC_DIR="$BSCCM_DATA/external/sbiad452_pack"
export WANDB_MODE=disabled

apptainer exec --nv --bind "$BSCCM_DATA" --bind "$BSCCM_REPO" \
  "$BSCCM_DATA/env/bsccm.sif" python3 -I -B "$N081_STAGE/tests/run_staged.py" \
  --out "$N081_STAGE/runs/n081_ifc_s0_b10" \
  --protocol b1to0 --channels all --phase 1 --arch resnet50 \
  --init-ssl "$BSCCM_DATA/runs/n031_ssl_aug_ex0/encoder_final.pt" \
  --targets targets_v5cx_v10.npz --epochs 20 --bs 256 --lr 0.0005 \
  --workers 15 --seed 0 --include-gs 0 --w-cls 0.5 --censor-loss 1 \
  --ext-ifc "$IFC_DIR" --ext-w 0.5 \
  --ext-train-donors sample0,sample1,sample2 --ext-test-donor sample2

apptainer exec --nv --bind "$BSCCM_DATA" --bind "$BSCCM_REPO" \
  "$BSCCM_DATA/env/bsccm.sif" python3 -I -B "$N081_STAGE/tests/run_staged.py" \
  --out "$N081_STAGE/runs/n081_ifc_s0_b01" \
  --protocol b0to1 --channels all --phase 1 --arch resnet50 \
  --init-ssl "$BSCCM_DATA/runs/n031_ssl_aug_ex1/encoder_final.pt" \
  --targets targets_v5cx_v10.npz --epochs 20 --bs 256 --lr 0.0005 \
  --workers 15 --seed 0 --include-gs 0 --w-cls 0.5 --censor-loss 1 \
  --ext-ifc "$IFC_DIR" --ext-w 0.5 \
  --ext-train-donors sample0,sample1,sample2 --ext-test-donor sample2
```

For matched controls, use distinct output directories and omit all four IFC options. For seed 1, change `--seed` and output names in both arms. IFC test labels are reported only; do not select settings using that donor.

## Limits

`pack_sbiad452.py` calculates fluorescence cofactors and `chan_scale` across all donors, then standardizes labels per donor. This harness consumes `Y` and `chan_scale` as specified and never trains on test-donor cells. Cell-level exclusion is enforced, but that packer is **not fully inductive for preprocessing statistics**. A strict donor-held-out study needs training-donor-only fitted preprocessing, outside this harness task.

These fixtures establish mechanics, not transfer accuracy. The unchanged BSCCM evaluator emits NaN aggregate correlations on four eval cells because no target meets its count threshold; training losses and IFC predictions are finite. GPU AMP, full-pack throughput, container execution, and GOAL improvement remain untested. Completed-run resume checks checkpoint loading; inherited training RNG state is not fully restored for bitwise interrupted continuation.
