# N045 — SOTAlign for frozen BSCCM Giemsa ↔ label-free embeddings

Delivered entirely under `/tmp/codex_n045/`; repository, data and upstream SOTAlign remain unchanged. No image extraction, fluorescence input, apptainer or GPU execution was used here.

`report.md` explains the experiment and limitations. `smoke/` contains the completed real-data CPU smoke: 768 distinct paired cells, 1,664 distinct cells total, six optimizer steps per learned arm, all four protocols and all seven methods. These are execution checks, **not trained performance estimates**.

## CPU setup and exact reproduction

The supplied `.venv` is ready on this host. To recreate it in this same delivery directory:

```bash
cd /tmp/codex_n045
export UV_CACHE_DIR=/tmp/codex_n045/uv-cache
uv venv /tmp/codex_n045/.venv --python 3.11
uv pip install --python /tmp/codex_n045/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/codex_n045/.venv/bin/python -r requirements.txt
```

`requirements-cpu.lock` records exact installed versions, including CPU torch. Use that lock with the PyTorch CPU index if exact recreation is needed; it is **not** a CUDA environment lock.

The delivered smoke was generated with:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/codex_n045/.venv/bin/python \
  /tmp/codex_n045/scripts/n045_sotalign.py \
  --out /tmp/codex_n045/smoke --smoke --threads 4 \
  > /tmp/codex_n045/smoke.log 2>&1
```

To verify/re-enter the completed run, add `--resume` to that exact command. Use another `--out` to regenerate rather than overwrite results. `--smoke` fixes the small sample/step budget and seed 0; it overrides numeric training-budget options. It preserves requested arms/protocols and loss choices.

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/codex_n045/.venv/bin/python \
  /tmp/codex_n045/tests/check_contracts.py
bash -n /tmp/codex_n045/scripts/run_n045_h100.sh
```

All five contracts pass: known retrieval answers including collapse, CCA versus upstream injected alignment-layer affinities, KLOT finite-difference gradients, actual split/pool exclusion checks, and bitwise-identical model weights after interrupted/resumed CPU training. Logs: `contracts.log`, `resume.log`; `launcher_contracts.log` also verifies actual signal forwarding and success/failure exit propagation. A second small run, `smoke_pairs_only/`, exercises `--sail-negatives pairs` for the four requested arms in one cross-slide direction.

## GPU execution

Use a CUDA-capable PyTorch environment on one H100. Install `requirements.txt` into that environment **after** CUDA torch; do not install the CPU lock. There are no timm, transformers, image, POT or upstream training-stack dependencies: only the embedding alignment classes are loaded.

For a direct run using your activated CUDA environment:

```bash
cd /tmp/codex_n045
python scripts/n045_sotalign.py \
  --out /tmp/codex_n045/gpu_direct --device cuda \
  --protocols cross_b0to1 cross_b1to0 fov_b0 fov_b1 \
  --seeds 0 1 2 --threads 8 --resume
```

That is 12 protocol/seed combinations, each with seven methods; it may exceed 30 minutes in total. For preemptible Slurm jobs, use the launcher, one protocol and alignment seed per job. Its checkpoints are atomic, configuration-checked, and written every 50 steps plus every validation. SIGUSR1/SIGTERM requests a checkpoint and exit 75; on Slurm the launcher then requests requeue. Completed arms are skipped; interrupted head evaluation is rerun. Checkpointed CPU continuation is verified; CUDA continuation is implemented but untested here.

Choose storage that survives node replacement and sits outside the read-only repo/data/source trees. The following example uses a sibling run directory; change it to your cluster's writable persistent scratch if needed. The writer has **not** created that directory. `N045_PYTHON` must point to your CUDA environment interpreter; `python` below is resolved from the submitting shell.

```bash
cd /tmp/codex_n045
export N045_RUN_ROOT=/hnvme/workspace/v181bb10-n045-runs
export N045_PYTHON=$(command -v python)
mkdir -p "$N045_RUN_ROOT/logs"
for protocol in cross_b0to1 cross_b1to0 fov_b0 fov_b1; do
  for seed in 0 1 2; do
    N045_PROTOCOL=$protocol N045_SEED=$seed \
      sbatch --partition=preempt \
      --output="$N045_RUN_ROOT/logs/%x_%j.out" \
      --export=ALL scripts/run_n045_h100.sh
  done
done
```

Add the local H100 resource constraint/account flags to `sbatch`; the launcher requests exactly one GPU and checks that the assigned device is an H100. Do not place concurrent writers in the same output directory. Each job uses `N045_RUN_ROOT/PROTOCOL_seedSEED`. The launcher requests 29-minute allocations with a signal 90 seconds before expiry, so a longer experiment can continue across allocations. An identical manual resubmission also resumes. Changing source, versions, inputs or options requires a new output directory; the fingerprint rejects accidental mixtures. Output/storage durability and Slurm requeue permission remain cluster responsibilities.

Combine jobs, preserving teacher identities:

```bash
python /tmp/codex_n045/scripts/merge_n045.py \
  --root "$N045_RUN_ROOT" --out "$N045_RUN_ROOT/combined"
```

`summary_by_teacher.csv` averages alignment seeds inside each LF teacher; `summary_across_teachers.csv` then averages teacher means and flags fewer than three LF teachers as exploratory. The supplied experiment has **one LF teacher**, regardless of alignment seed count. Do not interpret three alignment seeds as three teacher replications (K025/N032).

## Inputs and protocols

Defaults:

- Data root: `/hnvme/workspace/v181bb10-bsccm`.
- Repo: `/home/hpc/v181bb/v181bb10/BSCCM` (historical context only).
- SOTAlign: `external/SOTAlign` beneath data root, read-only.
- Giemsa: `emb/giemsa_dinobloom_B.npy`, 15,712 × 768, `hist_row` order.
- LF: `runs/n029_r18_v4c_fov/probe/lf_feats_all.npy`, 412,941 × 512, global-index order.
- Targets: `targets_v5c.npz`; measured detrended autofluorescence is the sole same-cell molecular truth on Giemsa cells.
- PBC: `emb/pbc_dinobloom_B.npy`; even rows/half A for unlabelled adaptation, odd rows/half B for evaluation. Labels enter evaluation only, including the fixed 5-NN sanity classifier.

The inverse mapping is checked against `pack/hist_cells.npy`, `targets.hist_row`, and both LF hist index sidecars. The separate LF hist extraction pass is compared to the canonical global LF array; all training and inference use the global array consistently.

| Protocol | Paired training | Model selection | Final retrieval/truth test |
|---|---|---|---|
| `cross_b0to1` | draw 0 Giemsa, split 0 | draw 0 Giemsa, split 1 | all draw 1 Giemsa cells |
| `cross_b1to0` | draw 1 Giemsa, split 0 | draw 1 Giemsa, split 1 | all draw 0 Giemsa cells |
| `fov_b0` | draw 0 Giemsa, split 0 | draw 0 Giemsa, split 1 | draw 0 Giemsa, split 2 |
| `fov_b1` | draw 1 Giemsa, split 0 | draw 1 Giemsa, split 1 | draw 1 Giemsa, split 2 |

FOVs cannot cross these splits. The training slide's split-2 pairs are unused in cross-slide fitting/selection. LF unpaired and ridge-head training rows are non-Giemsa, training draw, split 0; ridge validation uses that draw's non-Giemsa split 1. No test-slide Giemsa or LF features enter fitting or unpaired pools. The supplied frozen LF encoder did see **non-Giemsa cells from both draws** during its earlier FOV training; the cross-slide test is conditional on that encoder, not an encoder-inductive study.

## Methods and defaults

| Arm | Definition |
|---|---|
| `cca` | Upstream regularized CCA teacher, fitted on training pairs only |
| `siglip_pairs` | Original linear `AlignmentLayer` plus upstream pair SigLIP loss; no unpaired data |
| `sotalign` | Same student architecture/initialization, upstream semi-supervised SAIL + KLOT; X=PBC-A, Y=source-draw non-Giemsa LF |
| `sotalign_lf_only` | No extra X data; training Giemsa anchors provide X support against extra LF Y for KLOT; only LF supplies extra SAIL negatives |
| `raw_ridge` | Standardized raw Giemsa → raw LF ridge bridge; LF marker head stays on normalized raw LF features |
| `procrustes` | Training-pair-only PCA whitening then orthogonal rotation, a leakage-corrected N009-style rotation control |
| `n009_siglip` | Training-pair-only PCA whitening plus linear SigLIP maps without LayerNorm; N009-style architecture with the common N045 budget |

`raw_giemsa` additionally reports PBC raw-feature 5-NN in the **same metrics table**. Direct unaligned cross-modal cosine retrieval is undefined for 768-dimensional Giemsa versus 512-dimensional LF; `raw_ridge` supplies an explicit supervised bridge instead. Historical N009, N021 and N032 rows are included in `comparison.csv`, labelled `comparable=false`.

Defaults retain upstream `scripts/sotalign.sh`: Lion, lr 1e-4, betas .9/.99, weight decay 1e-5, cosine schedule, initial logit scale 20/bias −10, CCA penalties .1/.1, eig epsilon 1e-6, KLOT weight .001, student/teacher OT epsilon .05/.01, 100 iterations, no convergence early stop. The source's batch 32,768 and dimension 1,024 are reduced to 256 paired + 256 unpaired per side and dimension 128 (also capped by paired rank and input dimension). Pools contain up to 100k eligible LF cells and all 8,546 PBC-A cells. Training is 1,500 updates, with a source-FOV R@10 checkpoint choice every 100 steps. All learned arms use the same pair minibatches, seeds, initialization where architectures match, optimizer and update count.

Default semi-supervised SAIL includes unpaired negatives, matching upstream. Thus the SOTAlign arms change both the negative pool and KLOT relative to pairs-only. To isolate KLOT with an identical paired loss, run a separate, fully predeclared experiment:

```bash
N045_RUN_ROOT=/hnvme/workspace/v181bb10-n045-pairloss \
N045_PROTOCOL=cross_b0to1 N045_SEED=0 \
N045_PYTHON=$(command -v python) \
  bash /tmp/codex_n045/scripts/run_n045_h100.sh --sail-negatives pairs
```

Repeat protocols/seeds as above if using that variant. Do not select the variant from PBC or test-slide results. No loss weights, dimensions, learning rates or OT settings were tuned in this delivery; ridge alpha (1, 10, 100, 1000) is selected separately for each target on training-draw non-Giemsa split-1 labels. Raw ridge bridge alpha uses source Giemsa validation retrieval. No refit on validation/test follows selection.

Alternatives use explicit paths, a new `--out`, and the same frozen protocol:

```bash
python /tmp/codex_n045/scripts/n045_sotalign.py --device cuda \
  --out /tmp/codex_n045/gpu_scalematched \
  --giemsa /hnvme/workspace/v181bb10-bsccm/emb/giemsa_dinobloom_B_scalematched.npy

python /tmp/codex_n045/scripts/n045_sotalign.py --device cuda \
  --out /tmp/codex_n045/gpu_ssl \
  --lf-dir /hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/probe_fov \
  --allow-unverified-lf-provenance
```

The SSL provenance flag is an explicit attestation because its directory lacks the supervised `include_gs=0` manifest; it does not disable row-mapping checks. The primary supervised input has verified `include_gs=0`, `channels=all`, `phase=1`, 24 LF input channels. Neither alternative was trained/evaluated here. Scale metadata says crop side 398 px, so the scalematched name alone does not establish a meaningful scale intervention.

## Output contract

Each run writes:

- `metrics.csv` / `metrics.json`: current methods, including raw PBC features; null unavailable metrics are retained.
- `comparison.csv`: current metrics plus explicitly non-comparable historical results; no historical rows enter current means.
- `summary.csv`: means/SD over alignment seeds, separated by protocol, arm, target, truth and metric.
- `manifest.json`: exact configuration, package versions, source/target hashes and embedding file stat identities.
- `mapping_checks.json`, `preregistered_contrasts.json`, `completion.json`.
- Per protocol/seed: `partitions.npz` with **hist-row indices** for paired pools and **global indices** for head/LF pools; `partition_audit.json`; CCA teacher tensors.
- Per learned arm: resumable `last.pt`, validation-selected `best.pt`, `training.json` with SAIL/KLOT and Sinkhorn residuals; selector metadata. Load trusted checkpoints with `weights_only=False`.
- Per arm: ridge weights/scaler/target availability, per-cell predictions with row identities, geometry diagnostics and metrics. PCA/control maps are saved too.

Test Giemsa→LF R@1/R@10 use the entire protocol-specific test gallery, with chance 1/n and min(10/n,1). Ties are pessimistic. Autofluorescence Pearson r is reported separately for transferred Giemsa predictions and the same-cell LF head, using masked `targets_v5c.Y[:, autofl]`. A secondary **evaluation-only** correlation uses `gs_autofl_detrended.npy`, the N012 region-cross-fitted legacy residual; it is not interchangeable with the clean v5c truth. Predicted surface-marker agreement between Giemsa and LF is explicitly **not marker truth**.

PBC reports fixed 5-NN accuracy/balanced accuracy and all predeclared N021 contrasts plus the requested CD14 monocyte>all-other and CD3 lymphocyte>neutrophil contrasts. PBC-B morphology/labels never select a checkpoint or ridge/loss setting. IG/erythroblast labels are held out of supervision, but those classes' half-A images may enter unlabelled adaptation; they are not morphology-held-out classes here. Missing heads/classes produce null AUROC with a status/count, never a flipped sign or dropped row. MLL23 is omitted because there are no supplied frozen features.
