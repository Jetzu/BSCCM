# N054 tri-modal experiments

This folder is a standalone delivery. The BSCCM repository and data are read-only inputs. Training and scientific comparisons require your GPU runs; the delivered CPU smoke uses real data and a random, two-block ViT-tiny stand-in.

**Data corrections:** `targets_tt6c.npz` has **zero** single-marker cells with observed autofluorescence, so C1 returns an explicit unavailable result. It also retains 27,074 draw-2 pooled CD123/HLA-DR/CD14 observations that K019 rejects. Every new model and scorer masks those entries, retaining the original target values otherwise. The baseline's original LF teachers are reconstructed with their original masks; the common reference teacher uses the corrected mask. No labels are manufactured.

## Files and execution

- `n054_trimodal.py`: `lf_spaces | gs_encoder | flow | eval | completion`.
- `n054_clinical.py`, `n054_contrasts.py`: frozen inference and the verbatim registered PBC/MLL23 contrasts.
- `n054_select.py`: freezes the hyperparameter choice using training-slide split-1 autofl r only.
- `n054_gpu.sbatch`: one GPU, partition `preempt`, 16 CPUs, container launch, signal handling/requeue.
- `n054_submit.sh`: exact complete command grid and Slurm dependencies.
- `n054_smoke.py`, `smoke.log`, `smoke_results.json`: executable integration check and results.

Python dependencies are in `requirements.txt`; `timm==1.0.31` is required. There are no imports from repository scripts. The clinical entry point loads its two sibling modules explicitly so `python3 -I` works without modifying `sys.path`.

## Exact GPU commands, in dependency order

From this folder on the cluster:

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -c 'import torch, timm, numpy, scipy, sklearn, PIL; assert timm.__version__ == "1.0.31"; print(torch.__version__, timm.__version__)'
bash n054_submit.sh
```

`n054_submit.sh` is the executable command list; it prints every exact submitted command and job ID. It submits, in this order:

1. A1 / sigma=0 / batch1 `lf_spaces`, which creates the shared fixed ridge at `runs/n054_reference/ridge.npz`.
2. For **each** holdout `batch1`, `batch0` and **each** configuration `A1:0:0 A1:0.1:0 A2:0:0 A2:0.1:0 A3:0:0 A3:0:0.5` (arm:sigma:w_sig): `lf_spaces`, then `gs_encoder`, then `flow`, then `eval --validation-only`. Every other LF job waits for the reference job; every stage waits for its own predecessor. A3 `flow` is a recorded no-op.
3. `n054_select.py --runs <all 12 run directories> --out runs/n054_selection.json`, after all 12 validation jobs. This file is immutable on rerun unless the identical choices are reproduced.
4. `eval` for all 12 configurations and both A0 checkpoints, after selection is frozen. Reporting the full predeclared grid is intentional; use the frozen choices for the primary A1/A2/A3 comparisons.
5. `completion` once, in the A3/w_sig=0/batch1 directory, after that directory's evaluation. Completion independently fits all three LF-only protocols and does not depend on the GS holdout.
6. `clinical` for all 14 configurations, after each internal evaluation, with TTA=1, PBC all, and the original deterministic MLL23 cap of 1,500 per class. No clinical result affects selection.

For example, the exact commands for a single configuration, submitted serially, are:

```bash
R=/hnvme/workspace/v181bb10-bsccm/runs
FLAGS=(--arm A1 --holdout batch1 --sigma-bridge 0 --w-sig 0 --out "$R/n054_A1_s0_w0_batch1" --ridge-cache "$R/n054_reference/ridge.npz")
sbatch --wait n054_gpu.sbatch lf_spaces "${FLAGS[@]}"
sbatch --wait n054_gpu.sbatch gs_encoder "${FLAGS[@]}"
sbatch --wait n054_gpu.sbatch flow "${FLAGS[@]}"
sbatch --wait n054_gpu.sbatch eval --validation-only "${FLAGS[@]}"
# In the complete experiment, freeze selection across all candidates here.
sbatch --wait n054_gpu.sbatch eval "${FLAGS[@]}"
sbatch --wait n054_gpu.sbatch clinical --arm A1 --holdout batch1 --run "$R/n054_A1_s0_w0_batch1" --out "$R/n054_A1_s0_w0_batch1/clinical"
```

Use the complete submission script for the comparison; the single-configuration example is useful for a first GPU runtime check. Do not run two writing stages concurrently in one `--out`. Separate output directories are used for every arm/configuration/direction; clinical has its own checkpoint directory.

Outputs are `runs/n054_{arm}_s{sigma}_w{w_sig}_{holdout}` and `runs/n054_A0_{holdout}`. All defaults in the command list are intentional: seed=0, dim=256; GS freeze=8, batch=128, LR=1e-4, 2,500 updates; LF/flow/completion 2,500 updates, batch=256, LR=1e-3; tau=.25; flow Euler=32. The marker encoder has two 512-unit hidden layers and input `[y*mask, mask]`. All latents are L2-normalized and decoder inputs are scaled by sqrt(dim).

**Planning estimate:** approximately **12–30 GPU-hours** for the complete grid including all 14 clinical evaluations on an A100/H100-class device; roughly 0.3–0.9 GPU-hours per GS fit, with I/O/TTA clinical inference contributing much of the remainder. This is an estimate, not a measured GPU benchmark. The scheduler template requests three hours per stage. Start with the first configuration to measure local throughput and storage load. Preemptions add elapsed time. LF and identical GS fits are deliberately independent across output directories; sharing immutable fits could reduce cost later.

## Splits and objectives

Frozen LF inputs are only the two specified 512-dimensional image feature files. The hist feature/index/hist_row triplets explicitly locate the same GS cell's LF row. Feature standardization uses non-GS, labelled split-0 LF rows only. LF-side spaces and flows train on split 0 across both draws, validate on split 1, and report split 2. The ridge uses alpha=10, an unpenalized intercept, and a fixed target-specific seed to select at most 60,000 observed training rows per target. Its cache is checked against the training rows and normalization; all arms share it.

GS training uses only source-slide split 0, model selection uses source-slide split 1, and evaluation uses **every** FOV on the opposite slide. Source-slide split 2 is unused for GS fitting. Full counts (train/validation/test): batch1 holdout 8,007 / 994 / 5,697; batch0 holdout 4,495 / 593 / 10,015. This reserves validation FOVs that the original TT6c training did not reserve. The baseline is evaluated without retraining.

A1/A2 train space 1 with LF↔marker SigLIP plus masked LF regression and exact leave-one-entry-out marker decoding. Near-duplicate surface readouts are excluded as negatives using the N009 criterion, excluding autofl from the duplicate check. A3 uses LF regression plus that same SigLIP term; its GS objective is normalized latent MSE plus w_sig × SigLIP. A1/A2 jointly train P2 and DinoBloom on source GS/LF pairs with SigLIP. Those models and space 1 are frozen before flow training.

A1 flow endpoints are `(P2(f), P1(f))` of the same non-GS training LF cell. A2 uses independent source/marker-row batches and exact minibatch Hungarian OT; same global-cell matches are explicitly forbidden. Its target is `E_M(row)`, not `P1(f)`. The stochastic interpolant is `x_t=(1-t)z2+t*z1+sigma*t*(1-t)*eps` with analytic velocity `z1-z2+sigma*(1-2t)*eps`. This is a bounded-noise bridge with zero endpoint noise; deterministic inference integrates 32 Euler steps and normalizes the endpoint.

The stain/optics pipeline is copied from `train_gs_bind.py` with Macenko disabled, as in TT6c. New arms fit the 99.9th-percentile image scale on training-slide split-0 images only. A0 retains the original `H[:500:5]` scale for compatibility. The smoke-only backbone/resolution changes are explicit in metadata.

## Outputs and interpretation

Each evaluation writes `eval.json`, `eval.csv`, `eval_gs_preds.npz`, `lf_test_preds.npz`, and, for A1/A2, `lf_geometry.npz`. Validation has separate `validation.*` files and only source-slide split-1 GS cells. NPZs include global rows, hist rows where applicable, predictions, masks/truth, target names and embeddings; LF output includes batch IDs.

- M1: directly measured detrended GS autofl r.
- M2: each target's GS agreement with both the common 1,024-D ridge and its own LF path, paired with teacher-truth r on split-2 non-GS cells of the held-out draw. The product is **an assumption-dependent estimate**, not GS marker truth. The analogous autofl product and measured score are both recorded.
- M3: exact-cell cosine retrieval in the native matching space; pessimistic tie ranks and exact chance rates.
- M4: linear CKA, directed kNN overlap, and mutual-kNN graph overlap, k=10. Mutual overlap is twice the common mutual edge count divided by the sum of both graph edge counts. A1/A2 additionally report mapped-GS↔LF-space-1 geometry and LF-space-1↔LF-space-2 geometry.
- M5: LF-path and ridge r for each target, both draws combined and separately. A0's own LF path is its original two ridge teachers, because TT6c's output head was trained on GS latents.
- M6: per-cell PBC/MLL23 predictions, source filenames, embeddings, class means, and all 10 PBC + 15 MLL23 AUROCs. Contrasts and clinical RGB preprocessing follow the existing script; background is kept.

`completion/completion.{csv,json}` contains C1/C2; `coherence.csv` contains C3. Each protocol trains P_L, E_M, both heads and normalization **from scratch inside its own training draw(s)**. The fused model sees random 50% observed-entry masking during training and predicts all observed entries, as requested. Evaluation always masks the tested target. C2 inputs contain only other observed pooled groups. The failed draw-2 group has no test truth, and in the reverse direction it has no source training truth; both cases are explicitly marked unavailable. C3 conditions on the top 5% observed single-marker values and compares every other decoded marker with the same single-marker evaluation population's decoded mean. C3 supplies no validation of unmeasured marker coexpression.

## Resume and CPU smoke

Reissue the same command and output directory. `<out>/ckpt.pt` contains stage/job cursors, current and best model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG states. Periodic saves occur at validation boundaries; SIGTERM/USR1 triggers an atomic save at the next optimizer boundary and exit 99. The Slurm wrapper requeues that exit. Evaluation saves GS inference cursors in the same checkpoint; clinical inference checkpoints each batch in its separate output. Feature loading/closed-form ridge computation can be recomputed after interruption. Completed training stages export their validation-best model. Changed training configurations are rejected on resume. Never alter the input files in place between stages.

Reproduce the CPU check (all scratch under the required directory):

```bash
mkdir -p /tmp/codex_n054
UV_CACHE_DIR=/tmp/codex_n054/uv-cache uv venv /tmp/codex_n054/venv
UV_CACHE_DIR=/tmp/codex_n054/uv-cache uv pip install --python /tmp/codex_n054/venv/bin/python -r requirements.txt --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/codex_n054 /tmp/codex_n054/venv/bin/python -I n054_smoke.py > smoke.log 2>&1
```

The harness uses the same ≤2,000 LF / 300 GS-cell universe for all configurations, 20 optimizer steps, CPU threads=2, batch=4 for GS, and a random depth-2 ViT-tiny at 32 pixels. A0 loads the actual DinoBloom/TT6c checkpoints for its tiny held-out subset. Clinical smoke uses two images per class. It checks source/test isolation, leave-out masks, finite predictions, all registered contrast outputs, SIGUSR1 exit/resume, completed-stage resume, and bitwise equality of interrupted versus uninterrupted CPU GS training. Smoke scores are not experiment results.
