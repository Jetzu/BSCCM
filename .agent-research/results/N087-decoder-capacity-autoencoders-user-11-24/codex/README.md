# N087 decoder capacity, image autoencoders and marker priors

The two N051 feature dumps reproduce the recorded common-11 baseline. The reference gate measured **0.68674590** for b10, **0.65918953** for b01 and **0.67296771** for their mean. Per-target and aggregate results are in [reference_check/decoders.csv](reference_check/decoders.csv) and [reference_check/goal.csv](reference_check/goal.csv). The original and factored scorer CLIs also produced byte-identical CSVs on the archived eval predictions (GOAL 0.67296784).

## Integration

| Files | Destination and purpose |
|---|---|
| `score_goal_arms.py` + `score_goal_arms.diff` | Patch `scripts/score_goal_arms.py` to expose the existing common-11 selection, per-target Pearson calculation and grouped means. Its CLI still produces the three historical CSVs. |
| `n087_decoders.py`, `n087_cluster_eval.py` | Add to `scripts/` for the paired regression and draw-0 classification analyses. |
| `n087_image_ae.py`, `n087_extract.py` | Add both to `scripts/`. The extractor imports the training encoder and normalisation function. |
| `train_ifc_sandbox.py` + `train_ifc_sandbox.diff` | Patch `scripts/train_ifc_sandbox.py` to fit and freeze the validation-donor prior. |
| `n087_sandbox_eval.py` | Add to `scripts/` for post-hoc evaluation of new and archived runs. |
| `tests/` | Writer verification suite. Run from this staged directory. |

Both diffs apply to their live source files and reproduce the delivered copies byte-for-byte. This package uses the existing `train_lf.py --dump-features 1`, `bsccm/gpu_norm.py`, data packs and N080 test fixtures.

## Run environment

The examples below use Bash arrays. Run this setup once per shell. Set `DEVICE=cpu` or `DEVICE=cuda` for the placement you select. `O` defaults to a new directory within the staged package and may be overridden for production. Set `S=$R/scripts` to use the integrated scripts.

```bash
set -euo pipefail
R=/home/hpc/v181bb/v181bb10/BSCCM
C="$R/.agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex"
S="$C"
W=/hnvme/workspace/v181bb10-bsccm
SIF="$W/env/bsccm.sif"
O=${N087_OUT:-"$C/runs"}
T="$W/targets_v5cx_v10.npz"
PACK="$W/external/sbiad452_pack"
DEVICE=${DEVICE:-cpu}
mkdir -p "$O"
FLAGS=(-B /hnvme -B "$R:$R")
if [[ "$DEVICE" == cuda ]]; then FLAGS+=(--nv); fi
PY=(apptainer exec "${FLAGS[@]}" "$SIF" python3 -IB)
declare -A protocol=([b10]=b1to0 [b01]=b0to1)
declare -A excluded=([b10]=0 [b01]=1)
declare -A ft ssl
for d in b10 b01; do
  ft[$d]="$W/runs/n051_v10_base_$d/features.npz"
  ssl[$d]="$W/runs/n031_ssl_aug_ex${excluded[$d]}/encoder_final.pt"
done
```

The input feature files already exist. Their original, fully specified dump commands are recorded in [the N087 jobs file](../jobs.txt). All production commands below read these inputs. The AE logs to offline W&B and records the launcher job ID when supplied. `WANDB_DIR` selects the offline spool. Register production runs and their job IDs with the site's notebook and login-node sync lineage at submission.

## Regression and frozen features

**Reference check.** This reads predictions and row IDs before any decoder fitting. The assertion applies separately to each direction and to their mean, with absolute tolerance `1e-4`.

```bash
"${PY[@]}" "$S/n087_decoders.py" "${ft[b10]}" "${ft[b01]}" \
  --targets "$T" --reference-only --out "$O/reference"
```

**Extraction and AE training.** Each protocol/dimension combination is independent. Inside each combination, extract after training finishes. Resume an interrupted AE by running the identical training command with the same output directory. For a fresh experiment, use a new output path.

```bash
for d in b10 b01; do
  "${PY[@]}" "$S/n087_extract.py" --encoder ssl --ckpt "${ssl[$d]}" \
    --rows-from "${ft[$d]}" --device "$DEVICE" --out "$O/ssl_$d/features.npz"
  for k in 16 64 256; do
    run="$O/ae${k}_$d"
    "${PY[@]}" "$S/n087_image_ae.py" --protocol "${protocol[$d]}" --dim "$k" \
      --device "$DEVICE" --out "$run"
    "${PY[@]}" "$S/n087_extract.py" --encoder ae --ckpt "$run/encoder_final.pt" \
      --rows-from "${ft[$d]}" --device "$DEVICE" --out "$run/features.npz"
  done
done
```

The AE takes the complete SSL-eligible source draw and reserves 1% of its (slide, field of view) groups for validation. At seed 0 this gives 272,272 training and 2,708 validation cells for b10, and 136,525 training and 1,436 validation cells for b01. With bf16 convolution and a linear spatial bottleneck, the default objective reconstructs randomly withheld 16×16 patches shared across the 24 channels (25% of patches). Only finite withheld pixels contribute. `--mask-fraction 0` trains on all finite pixels. `--l1 W` adds weighted reconstruction MAE. Validation uses fixed masks, and `log.json` reports pixel-weighted MSE for each completed epoch. The atomic `ckpt.pt` records the next batch, phase, optimizer and partial-epoch totals at epoch boundaries, every 20 minutes during batch processing, and on SIGTERM/USR1.

Extraction uses complete images, the repository’s fixed flat-field and slide calibration through `GPUNorm(..., 'all', True, shift=0)`, and a float16 output matrix. SSL appends 24 all-one presence planes. The extractor compares exported SSL weights with the sibling training checkpoint and verifies the recorded source rows. For an independently archived bare SSL export, `--ssl-excluded-batch` accepts a documented exclusion supplied by the caller. The AE export contains its protocol. In both cases the output keeps the reference's exact `G`, `is_eval` and `names`.

**All decoder arms.** The labels before `=` pair the two directions of each representation. The script infers the source and evaluation draws from row metadata. For one standalone pair, pass both plain paths and optionally `--feature-set NAME`.

```bash
feats=()
for d in b10 b01; do
  feats+=("finetuned=${ft[$d]}" "ssl=$O/ssl_$d/features.npz")
  for k in 16 64 256; do feats+=("ae$k=$O/ae${k}_$d/features.npz"); done
done
"${PY[@]}" "$S/n087_decoders.py" "${feats[@]}" --targets "$T" --device "$DEVICE" \
  --seed 0 --epochs 10 --bs 512 --lr .001 --out "$O/decoders"
```

Features use the source-row mean and population SD. Ridge selects each target's penalty by masked validation MSE on a 10% group holdout, scales on the inner fitting fold during selection, and refits on all labelled source rows for that target. The penalty grid is `1e-6, 1e-4, .01, 1, 100` for mean squared error plus λ‖w‖². Reduced-rank and MLP heads use the same fixed ten-epoch budget. The factorized linear widths are 1, 2, 4, 8, 16, 32, 64 and 128. Their effective output rank cannot exceed 13. `full` is a direct affine head. The MLP uses two 512-unit GELU hidden layers and 0.1 dropout. Exact cosine kNN averages 50 observed training labels per target, or all available neighbors in a smaller bank.

`decoders.csv` has one row per representation, head, width, direction and target, including the two non-GOAL targets. Scores without support or variance are blank. Ridge also reports a missing score for targets without support on both inner folds. `goal.csv` requires 11 finite Pearson scores per direction and both directions for GOAL. Only dumps containing `P` have a reference arm.

## Draw-0 clustering

```bash
clusters=("finetuned=${ft[b10]}" "ssl=$O/ssl_b10/features.npz")
for k in 16 64 256; do clusters+=("ae$k=$O/ae${k}_b10/features.npz"); done
"${PY[@]}" "$S/n087_cluster_eval.py" "${clusters[@]}" --targets "$T" \
  --out "$O/clusters" --seed 0
```

The current dump contains 1,181 labelled draw-0 evaluation cells. The N051 run and the extractor establish training provenance for this workflow. The five-field NPZ schema carries only row partitions for independently supplied dumps. The evaluator checks source/evaluation draw IDs, selects `cls10 >= 0` and aligns comparisons by global row ID. For each native representation and each feasible PCA projection (16/64/256), it fits k-means with ten clusters and ten initializations and scores NMI and ARI. Native features retain their original units. PCA mean-centers the evaluated cells without whitening. Exact cosine 15-NN uses five disjoint (slide, field of view) folds. `knn_balanced_accuracy` scores pooled out-of-fold predictions and gives each represented class equal weight. `knn_mean_fold_balanced_accuracy` gives each fold equal weight. `cluster.csv` records unachievable dimensions explicitly, e.g. PCA-256 on an AE-16 latent.

## S-BIAD452 marker priors

**18 new single-marker runs.** Each (prior, dimension, donor) triple is independent. The validation donor supplies its full nine-dimensional marker vectors, standardized in the trainer's target coordinates. PCA uses the leading directions. The AE fits 9→64→k→64→9 with GELU, Adam and 30 passes. Both freeze the decoder, center and scale before fitting the image head. Training uses the existing revealed-entry loss. The validation donor therefore supplies extra joint supervision as well as checkpoint selection. The third donor is the held-out performance test.

```bash
for prior in linear ae; do
  for k in 2 4 6; do
    for d in 0 1 2; do
      "${PY[@]}" "$S/train_ifc_sandbox.py" --pack "$PACK" --regime single \
        --marker-prior "$prior" --prior-dim "$k" --test-donor "$d" \
        --workers 12 --threads 8 --device "$DEVICE" --out "$O/n087_${prior}${k}_d$d"
    done
  done
done
```

**Post-hoc evaluation.** These commands use the nine archived N080 controls. Set `REMAKE_CONTROLS=1` to reproduce the controls in fresh N087 run directories and evaluate those predictions. The default control ignores prior dimension.

```bash
arms=()
for d in 0 1 2; do
  for regime in full single single_pooled; do
    control="$W/runs/n080_${regime}_d$d"
    if [[ ${REMAKE_CONTROLS:-0} == 1 ]]; then
      control="$O/control_${regime}_d$d"
      "${PY[@]}" "$S/train_ifc_sandbox.py" --pack "$PACK" --regime "$regime" \
        --marker-prior none --test-donor "$d" --workers 12 --threads 8 \
        --device "$DEVICE" --out "$control"
    fi
    arms+=("$regime=$control")
  done
  for prior in linear ae; do
    for k in 2 4 6; do arms+=("${prior}$k=$O/n087_${prior}${k}_d$d"); done
  done
done
"${PY[@]}" "$S/n087_sandbox_eval.py" "${arms[@]}" --pack "$PACK" --out "$O/sandbox_eval"
```

The `n087_eval_*.csv` files report per-marker Pearson r under both N080 target conventions, the training-donor true-type oracle, within-type r, within-group pairs, population fractions and the summary. Each extra metric is tied to its run and test donor. `n087_eval_36pairs.csv` gives the mean absolute predicted-minus-true correlation from all 36 entries in the model's `metrics.json`, plus a defined-pair count. Undefined pairs leave that donor's score blank. The separate N080 `pair_corr_abs_err` column uses its four within-group comparisons. As a reader check, averaging the stored 36-pair errors over the three archived donors gives 0.18908 (full), 0.22949 (single), and 0.23681 (single_pooled).

## Wall-time budget

| Work unit | Planning estimate |
|---|---|
| Reference gate, both dumps | A few seconds on CPU. Reads metadata and predictions. |
| One SSL or AE extraction | 5–20 min with H100-class compute and local pack bandwidth, depending on draw. |
| One image AE, k=64 | About 1 H100-hour **if throughput is 1,500 cells/s**. The default targets 5.4M training presentations, giving 20 b10 or 40 b01 epochs. Validation and I/O add overhead. Six runs budget about 6 H100-hours at that throughput. |
| All decoders, one pair of 2,048-wide matrices | Approximately 0.5–3 h on a 48-core CPU allocation or 5–30 min on H100-class hardware. Exact cosine search and ridge dominate wide inputs. |
| Clustering, all five representations | Budget a few minutes on CPU for the current 1,181-cell cohort. |
| One sandbox fold | Budget 5–10 min on H100-class hardware. Archived N080 single folds took 302, 299 and 480 seconds. The 18 new folds budget about 1.5–3 h in aggregate. |
| Sandbox post-hoc across all runs | Budget several CPU minutes. Donor gates are reused across matching cohorts. |

The new production runtimes and CPU/GPU speed ratios are estimates. The writer ran CPU smoke tests only. Placement and scheduling remain with the submitter.

## CPU verification

All 18 N087 tests and the 31 archived N080 tests passed in the supplied SIF’s Python environment. The suite trains each neural decoder and the image AE for two optimizer steps on tiny fixtures. It also exercises real signals, resume equivalence, frozen weights, withheld-label perturbations, byte-for-byte control outputs and score agreement with N080. An additional four-row extraction on each actual N031 checkpoint produced finite 4×2048 float16 matrices with identical global IDs, split flags and names.

Run the following on a machine with Apptainer. The SIF provides the scientific packages. Pytest is installed into a temporary system-site-packages venv.

```bash
cd "$C"
TMPBASE=/tmp/n087_codex
mkdir -p "$TMPBASE"/{home,cache,root}
export APPTAINER_CACHEDIR="$TMPBASE/cache" APPTAINER_TMPDIR="$TMPBASE/root"
export TMPDIR="$TMPBASE/root" APPTAINERENV_TMPDIR="$TMPBASE/root"
export APPTAINERENV_XDG_CACHE_HOME="$TMPBASE/cache" APPTAINERENV_TORCH_HOME="$TMPBASE/cache/torch"
export APPTAINERENV_CUDA_VISIBLE_DEVICES='' APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_OPENBLAS_NUM_THREADS=2 APPTAINERENV_OMP_NUM_THREADS=2 APPTAINERENV_MKL_NUM_THREADS=2
TEST_IMAGE="$SIF"
# On the writer host /dev/fuse is unavailable. The same SIF can run as a directory.
if [[ ! -e /dev/fuse && ! -d "$TMPBASE/sif-root" ]]; then
  apptainer sif dump 4 "$SIF" > "$TMPBASE/image.squash"
  unsquashfs -no-xattrs -no-progress -processors 12 -d "$TMPBASE/sif-root" "$TMPBASE/image.squash"
fi
if [[ -d "$TMPBASE/sif-root" ]]; then TEST_IMAGE="$TMPBASE/sif-root"; fi
BOX=(apptainer exec --home "$TMPBASE/home" -B /hnvme -B "$R:$R" "$TEST_IMAGE")
"${BOX[@]}" python3 -IB -m venv --system-site-packages "$TMPBASE/venv"
CHECK=("${BOX[@]}" "$TMPBASE/venv/bin/python" -IB)
"${CHECK[@]}" -m pip install --disable-pip-version-check --cache-dir "$TMPBASE/cache" pytest==9.2.0
"${CHECK[@]}" -m pytest -q -p no:cacheprovider --basetemp="$TMPBASE/new-suite" "$C/tests"
BSCCM_REPO="$R" bash "$C/tests/run_n080_suite.sh" "${CHECK[@]}"
```

The local fixtures exercise reconstruction and decoding, both real SIF SSL exports were probed on CPU, and the real GOAL gate passed. Full-scale fits, CUDA numerical behavior, scheduler-driven preemption and W&B synchronization were outside the CPU smoke run.
