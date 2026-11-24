# N080: S-BIAD452 joint-decoding sandbox

`train_ifc_sandbox.py` predicts all nine measured markers of each held-out cell from BF1, BF2 and SSC. The three training regimes isolate the cost of observing one marker per cell and the value of adding synthetic pooled slides. Co-expression is tested against measured joint truth, including variation within populations; good population means alone do not establish cell-level decoding.

The implementation is standalone and does not import or modify the BSCCM repository. No jobs were submitted and no GPU training was performed during development.

## Data and fixed folds

Pass the read-only output of `scripts/pack_sbiad452.py` through `--pack` (default `/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack`). Required files are:

- `lf.npy`: float16 `(N,3,128,128)`, BF1 log-ratio / BF2 log-ratio / SSC asinh. Each channel is divided by the supplied `chan_scale`.
- `labels.npz`: finite `Y` and `I_raw`, each `(N,9)`, canonical `names`, `sample`, `chan_scale`. Optional `part` and `object_number` are carried into predictions. Marker order is CD15, Siglec8, CD14, CD19, CD3, CD45, CD4, CD56, CD8.

`sample1_early` and `sample1_late` always stay together, as do the two sample2 acquisitions. Validation is `(test_donor + 1) % 3`; training is the remaining donor.

| Test | Validation | Training |
| --- | --- | --- |
| 0 | 1 | 2 |
| 1 | 2 | 0 |
| 2 | 0 | 1 |

All three donors must exist. The supplied ~400-cell smoke pack currently has only sample0 and sample1_early, so it cannot support this protocol. The test suite creates an entirely synthetic three-donor pack in `/tmp/n080_codex` instead. The real pack path was unavailable on this CPU host.

**Calibration scope:** the requested packed `Y` uses a global fluorescence cofactor and per-sample z scores computed by the packer using all cells of that sample; its `chan_scale` also uses cells across samples. This is a retrospective, assay-standardised target, with pre-existing transductive calibration. The sandbox honours those supplied arrays. It does not claim a prospective, completely training-calibrated assay. A stricter study needs a newly packed training-calibrated target and LF scale; do not silently compare it to this estimand. All additional pooled statistics and model fitting here use training observations only. Test gates use frozen training thresholds, and no test statistics are used to recalibrate predictions.

## Training arms

The production backbone is `timm.create_model('resnet50', pretrained=False, in_chans=3, num_classes=0)` and a linear nine-output mean head. Defaults: 15 epochs, batch size 256, AdamW, peak LR `5e-4`, weight decay `.05`, OneCycle with 10% warmup, CUDA bf16 autocast, gradient norm clipping at 10. Very short smoke schedules use at least two warmup steps when possible to avoid OneCycle's zero-length warmup corner case. Images receive independent random 90-degree rotations/reflections and integer translations in `[-4,4]` pixels, shared across channels, with zero background padding. No directional-LED restriction applies to these IFC channels.

- `--regime full`: masked MSE with all nine entries visible.
- `--regime single`: exactly one uniformly selected marker per cell. A stateless hash of packed row ID and `--seed` fixes the revealed marker for the entire run, including resume. The same row has the same single marker in both sparse arms.
- `--regime single_pooled`: independently assign each cell to pooled-only supervision with probability .10 (approximately 10%, deterministic by row/seed). Other cells reveal exactly one marker. Pooled cells reveal no individual marker, including CD45. They reveal exactly two pooled values.

The synthetic raw pooled measurements are `P1 = asinh((I_CD3 + I_CD19 + I_CD56)/c)` and `P2 = I_CD45` (linear CD45). Signed compensated intensities are retained. `c = max(1.4826 * MAD(raw summed intensity), 1)` uses only revealed pooled **training** cells. Both pooled measurements are standardised by their means and standard deviations on those cells. Each fold has one training donor, so this is training-donor standardisation; its statistics are frozen for validation/test. No hidden single labels supply pooled calibration. At least two pooled training cells and two revealed singles per marker are required for inverse-map initialisation; use a sufficiently large smoke fixture.

Both pooled implementations are available:

| Option | Pooled prediction | Consequence |
| --- | --- | --- |
| `--pooled-mode inverse` (default) | Each predicted marker `y_m` maps to raw intensity `c*sinh(softplus(a_m)*y_m+b_m)`; sum CD3/CD19/CD56 before asinh and standardise. CD45 uses the same inverse map. | Pooled gradients reach the marker outputs. |
| `--pooled-mode head` | Separate linear two-output head on shared image features. | Pooled supervision shapes features; it does not force individual means to add to the pool. |

Inverse-map slopes/biases are initialised by positive affine fits from the revealed training single stains only, then learned jointly. An auxiliary loss (default `--map-weight .1`) fits the same observed single measurements in `asinh(I_raw/c)` coordinates. It adds no observations. This affine-in-asinh inverse is an approximation, especially across early/late sample normalisations. The sinh argument is clamped to `[-20,20]` for numerical stability, making the map nondecreasing at its extreme tails. There are no lineage constraints or forbidden marker combinations.

Sparse loss is the mean squared error over revealed single entries plus `--pool-weight` (default 1) times mean squared error over revealed pooled entries, plus the optional inverse calibration loss. Empty masks contribute zero. Hidden entries are zeroed before constructing supervision tensors. Validation uses the same fixed regime/masks as training; the epoch with minimum validation loss is selected. Validation is never added to the neighbour bank or refitted into the training model. The test donor is scored after training completes.

`--cov-rank 2` is available only with `--regime full`: a linear head predicts nine positive diagonal variances and a `9 x 2` factor, giving `Sigma = diag(softplus(d)+1e-4) + U U^T`. Training uses joint Gaussian NLL with Cholesky solves in float32. The usual metrics still score posterior means; `gaussian_nll` and saved covariance parameters assess the distribution. A covariance head need not improve the correlation structure of the means.

## Runs: three regimes × three test donors

Use an environment with PyTorch, torchvision, NumPy and timm. Run from **this deliverable folder**, keeping outputs underneath it. The loops below are direct invocations, with no scheduler submission.

```bash
export XDG_CACHE_HOME=/tmp/n080_codex/cache
export TORCH_HOME=/tmp/n080_codex/torch
export PYTHONPYCACHEPREFIX=/tmp/n080_codex/pycache
PACK=/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack
for regime in full single single_pooled; do
  for donor in 0 1 2; do
    python train_ifc_sandbox.py --pack "$PACK" \
      --out "runs/${regime}_d${donor}" --regime "$regime" --test-donor "$donor"
  done
done
```

The pooled-head comparison and full-label covariance arm each require separate outputs:

```bash
for donor in 0 1 2; do
  python train_ifc_sandbox.py --pack "$PACK" --regime single_pooled \
    --pooled-mode head --test-donor "$donor" --out "runs/single_pooled_head_d${donor}"
  python train_ifc_sandbox.py --pack "$PACK" --regime full --cov-rank 2 \
    --test-donor "$donor" --out "runs/full_cov2_d${donor}"
done
```

Planning estimate on **one H100**, ~98k total cells, 128 px, 15 epochs: **0.2–0.6 GPU-hours per run**, or **1.8–5.4 GPU-hours for the nine-run core grid**. Each additional three-fold arm adds roughly 0.6–1.8 GPU-hours. These are unmeasured estimates, including validation/embedding passes and a tiled GPU neighbour search; storage speed and donor sizes can dominate. Time an initial run before reserving the full grid. CPU smoke timing is not an H100 benchmark.

## Neighbour baseline

`--knn-impute K` defaults to 50; `0` disables it. The selected model embeds training and test cells in evaluation mode. For each marker, an exact Euclidean search averages that marker's values from the K nearest **training cells that revealed it**. Full-label banks share one search; sparse banks contain only the corresponding observed singles. Pooled cells supply no single-marker neighbours. K is reduced to bank size when necessary and the effective K/count are recorded; an empty marker bank raises an error.

The search uses the model's unnormalised embedding, with no test-fitted transform. Both query/reference axes are tiled (`--knn-query-chunk 256`, `--knn-reference-chunk 4096`), and distances run on the selected device. This is a neighbour-imputation comparator in the spirit of panel merging, not a reproduction of Infinity Flow's original per-marker regressors. Every prediction is scored with the same metric function as the neural mean head. K is prespecified, not chosen on the test donor.

## Evaluation definitions and outputs

All thresholds use the complete **training** truth for evaluation only, after model fitting; hidden training labels used to define evaluation gates never enter a model loss or neighbour value. Thresholds are stored in `metrics.json`. `hi`/positive means greater than the training 75th percentile for that marker. The CD45 inclusion threshold is its training 25th percentile; CD15-low is at or below its training median.

Coarse populations are disjoint and defined from **true test labels**:

- Lymphocytes: CD45 above q25, CD15 at/below q50, CD14 at/below q75.
- Granulocytes: CD45 above q25 and CD15 above q75.
- Monocytes: CD45 above q25, CD15 at/below q75, CD14 above q75.
- Other: the remainder.

Gate F1 applies the same thresholds/logic independently to predicted and true vectors: the three named populations, T (lymphocyte/CD3+), B (lymphocyte/CD3−/CD19+), NK (lymphocyte/CD3−/CD19−/CD56+), CD4 T and CD8 T (lymphocyte/CD3+/subset marker+), eosinophils (granulocyte/Siglec8+), neutrophils (granulocyte/Siglec8−), and CD3+CD19+ across all cells. These are reproducible quantile-based coarse gates, not curator-supplied or clinically validated identities. Small/empty populations should be inspected using their reported counts.

`metrics.json` contains `model` scores, with the identical baseline scores under `model.knn`:

- Each marker's Pearson r and macro mean over defined markers.
- Per-cell vector RMSE in the supplied standardised Y coordinates (mean/median) and mean vector cosine. This does not standardise each cell across its own markers.
- True/predicted `9 x 9` marker Pearson matrices and their full-matrix Frobenius distance overall and within each coarse true population.
- Within-population, per-marker cell-level r versus per-marker r of the **four population means**, with equal population weights for the latter. Four means provide a very small cluster-level sample; always read the cell-level results beside it.
- All 36 marker pairs' predicted r minus true r, exposing over- and under-correlation. CD3–CD19, CD3–CD56, CD4–CD8 and CD14–CD15 are additionally reported within true lymphocytes and granulocytes.
- Predicted/true CD3+CD19+ rates and gate F1 with TP/FP/FN/counts. Rate agreement alone does not establish that the same cells are double positive.

A correlation needs at least three cells and nonzero variance; otherwise it is JSON `null`. Macro means omit undefined markers. Frobenius distances use only entries defined in both matrices, with `valid_entries` and `total_entries` explicitly reported; partial distances are not comparable to complete distances. A completely constant predicted vector has undefined cosine; those cells are excluded from the mean and counted. F1 is `null` when both positive sets are empty. No missing correlation is silently filled with zero.

Artifacts in each output directory:

| File | Contents |
| --- | --- |
| `setup.json` | Recipe/fingerprint, donor counts, observed-label counts, pooled calibration, architecture. |
| `ckpt.pt` | Last training state, best validation model, optimizer, scheduler, epoch/batch cursor, running loss, row IDs, Python/NumPy/Torch/CUDA RNG. |
| `result.json` | Complete/interrupted status; completed runs include arguments, history, selected epoch and metrics. |
| `metrics.json` | Test-only metrics and frozen gate thresholds. |
| `eval_preds.npz` | Test row IDs, names, sample/part/object IDs, truth `Y`, predictions `P`, per-cell `vector_rmse`/`vector_cosine`, optional `P_knn`, `cov_diag` (positive diagonal variance), `cov_factor` (rank-2 factor). Undefined per-cell cosine is NaN in this array. |

Report each donor separately. This is three biological donors; thousands of cells do not create thousands of independent donor replicates. The main comparisons concern joint decoding on IFC, not a promised gain on BSCCM. Metrics follow the requested N079 literature checks: cell versus cluster correctness (Mocking/panel merging), per-cell joint truth beyond marginal imputation (Infinity Flow), and correlation inflation (HistoPlexer). The supplied [N079 literature review](../../N079-review-5-lf-mk-v10-goal-6862-target-70-a/agents/lit_panel_merging.md) contains the source discussion.

## Resume and preemption

Rerun the identical command with the same `--out` to resume `<out>/ckpt.pt`. Checkpoints are atomically replaced every 200 optimizer steps by default, every epoch, and at the next completed batch after SIGUSR1/SIGTERM. A handled signal writes `result.json` with `status: interrupted` and exits successfully; rerun to finish. A hard kill recovers from the last periodic checkpoint. Signals during prediction/neighbour search stop at a batch/tile boundary; the completed training checkpoint is retained and scoring restarts on resume.

Resume validates the recipe, complete labels-file SHA256, LF file size/mtime/path, and train/validation/test rows. Changing epochs, donor, regime, seed, or model recipe requires a fresh output directory. Worker/thread counts, checkpoint interval and KNN evaluation settings can change. LF content is not fully hashed; pack files must remain immutable. Checkpoints are trusted local files loaded with `weights_only=False`. Exact resumed CPU trajectories are tested; CUDA kernel determinism across hardware/software versions is not guaranteed. The checkpoint remains the latest training model even when the selected best epoch is used for final scoring.

## Tiny CPU verification

All installation caches, fixtures and temporary results can stay under `/tmp/n080_codex`. A CPU-only environment was used here:

```bash
mkdir -p /tmp/n080_codex
export UV_CACHE_DIR=/tmp/n080_codex/uv-cache
uv venv /tmp/n080_codex/venv
uv pip install --python /tmp/n080_codex/venv/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/n080_codex/venv/bin/python numpy timm pytest
export PYTHONPYCACHEPREFIX=/tmp/n080_codex/pycache
export XDG_CACHE_HOME=/tmp/n080_codex/cache
export TORCH_HOME=/tmp/n080_codex/torch
/tmp/n080_codex/venv/bin/python -m pytest tests -q -p no:cacheprovider \
  --basetemp=/tmp/n080_codex/pytest
```

An independent CLI smoke run (the fixture destination must be new):

```bash
/tmp/n080_codex/venv/bin/python tests/make_fixture.py --out /tmp/n080_codex/fixture
/tmp/n080_codex/venv/bin/python train_ifc_sandbox.py \
  --pack /tmp/n080_codex/fixture --out /tmp/n080_codex/cli_smoke \
  --test-donor 0 --regime single_pooled --epochs 2 --bs 32 \
  --device cpu --workers 0 --threads 1 --smoke-tiny-model --knn-impute 3
```

`--smoke-tiny-model` explicitly replaces ResNet-50 with an eight-feature toy encoder. Its scores are solely execution checks. The tests also run a real timm ResNet-50 forward/backward on two 128px CPU images. `--max-train/--max-val/--max-test` optionally select fixed donor-local subsets for development; defaults use all cells. `--stop-after-steps N` exercises preemption after N newly completed steps and is not part of the resume recipe.

Verification on this CPU host: **31 tests passed in 5.60 s**, plus the independent synthetic CLI run above completed with both `P` and `P_knn` for all 100 held-out cells. Tests include complete-model invariance after changing hidden single labels and test truth, exact batch-resume trajectory, epoch-boundary resume, SIGUSR1/SIGTERM during training, signal interruption during evaluation, short OneCycle schedules, and known-answer metric cases. Environment: Python 3.13.13, torch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.31, NumPy 2.5.4, pytest 9.2.0. No scientific performance conclusion is drawn from synthetic scores.
