# N054 v2

This is a standalone replacement delivery. All files in the parent directory, repository code and input data remain untouched. CPU smoke artifacts are under `/tmp/codex_n054/`; production outputs use `runs/n054v2_*`. No GPU job was submitted by the writer.

## Launch exactly three units

Run from the cluster login shell. Parameters live in exported environment variables, so the house wrapper can replay the batch script after `PreemptMode=CANCEL`.

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v2
export N054_CODEDIR="$PWD"
export N054_RUNROOT=/hnvme/workspace/v181bb10-bsccm/runs
export N054_PREFIX=n054v2

apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -c \
  'import torch, timm, numpy, scipy, sklearn, PIL; assert timm.__version__ == "1.0.31"; print(torch.__version__, timm.__version__)'

LF_JOB=$(N054_UNIT=lf sbatch --parsable n054_unit.sbatch)
# Wait for the ENTIRE resubmission chain, not only its first Slurm job ID.
"$HOME/.claude/skills/work-log/bin/wait-job.sh" "$LF_JOB"
# Continue only when wait-job reports rc=0.
B1_JOB=$(N054_UNIT=batch1 sbatch --parsable n054_unit.sbatch)
B0_JOB=$(N054_UNIT=batch0 sbatch --parsable n054_unit.sbatch)
"$HOME/.claude/skills/work-log/bin/wait-job.sh" "$B1_JOB" "$B0_JOB"
# Continue only when both chains completed successfully.

apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I \
  "$N054_CODEDIR/n054_aggregate.py" \
  --units "$N054_RUNROOT/${N054_PREFIX}_batch1" "$N054_RUNROOT/${N054_PREFIX}_batch0" \
  --lf-unit "$N054_RUNROOT/${N054_PREFIX}_lf" \
  --out "$N054_RUNROOT/${N054_PREFIX}_tables"
```

Units batch1 and batch0 may run concurrently **after the LF unit's entire chain finishes**. The user can implement that chaining externally. A plain `afterok:<first-LF-job>` dependency is insufficient if LF is preempted: its successor has another ID. The holdout driver also checks all three LF stage manifests before starting. Avoid concurrent writers to the same unit directory. Use a new `N054_PREFIX=n054v2_r2` after changing code/configuration.

`n054_unit.sbatch` requests partition `preempt`, one GPU, 16 cores, 64 GB, eight hours and `--signal=B:USR1@900`. It executes the existing `~/.claude/skills/work-log/bin/preempt-run.sh -- apptainer exec ... python3 -I ...`, without `srun`. The house wrapper resubmits the batch script, forwards one SIGTERM to the command and accepts Python's checkpointed exit **143**. Cancel a chain using the house `preempt-cancel.sh <root-job-id>`.

## Unit contents and shared artifacts

1. `n054v2_lf`: fit `space1/lf_spaces.pt` (P1/E_M/D_M), `hub/lf_spaces.pt` (P_L/H/E_M), and the common fixed ridge **once**, using non-GS LF split 0 from both draws and split 1 for validation. Fit completion's FOV, draw0→1 and draw1→0 protocols once, independently of GS holdout. Completion uses **targets_v5cx_v10.npz**; GS-space models use **targets_tt6c.npz**. Metadata/feature row alignment is identical across these files. TT6c's 27,074 failed draw-2 group-8 observations are masked; v10 already masks them and is asserted to do so.
2. `n054v2_batch1`: hold out GS batch 1. Train one `space2/gs_encoder.pt` (P2/E_G SigLIP), then one hub GS encoder for each `w_sig ∈ {0, .5}`. Cache only source split-0/1 GS latents. Train nine small maps: four bridge families × two sigmas plus direct regression. Select on source split-1 autofl truth r, freeze `selection.json`, evaluate the entire grid and A0, and run clinical inference on **A0 + the four selected primary configurations only**.
3. `n054v2_batch0`: identical workflow with the GS slides reversed, reusing all LF-unit outputs.

Every two-space map references the exact same frozen space-1 artifact and the holdout's exact same space-2 artifact. The script does not copy or retrain them per bridge. Hub GS fits reference the single LF hub. GS training uses source split 0, source split 1 selects configurations, and the entire opposite slide is test-only. Non-GS LF fitting uses split 0 across draws, as specified; some non-GS unstained cells share slides/FOVs with GS cells, so this is GS-cell exclusion, not whole-slide exclusion for the LF side. Source split 2 is unused for GS fitting.

### Arms

- **A1-LF:** same-cell `(P2(f), P1(f))` LF bridge. Retained as the literal LF alignment design. Test evaluation additionally starts this map from `P2(f_own)` for an explicitly labelled **oracle diagnostic**.
- **A1-GS:** start from that sigma's A1-LF weights, then train on source split-0 GS pairs `(E_G(img_i), P1(f_i))`. Source split 1 supplies early-stop loss and sigma selection. GS latents are cached with deterministic evaluation preprocessing; the encoder remains frozen.
- **direct:** same GS pairs and exactly the same parameter count as the velocity net; fixed zero time input and direct normalized latent regression, without integration. This is a diagnostic and has no selected clinical run.
- **A2:** independently sample source-slide split-0 **GS** cells and different random non-GS LF split-0 cells. Minibatch Hungarian OT matches `z_G` to **P1(f)**, without biological correspondences. Equal global-cell IDs are forbidden before assignment and assignment collisions are measured/asserted. Source-validation GS and LF split-1 marginals supply early stopping.
- **A2-EM:** same source GS marginal, with full E_M(marker row) target marginal. Secondary only. D_M is trained on P1 inputs, exact leave-one-entry-out E_M inputs, **and full E_M inputs**.
- **A3:** GS latent regression is `(zg-zl).square().sum(-1).mean()` on unit vectors (`2−2cos`), plus `w_sig * SigLIP`. Both weights are retained.
- **A0:** load the original TT6c checkpoint and its original preprocessing/teachers, without retraining.

Bridge sigmas are `{0, .1}`, using the bounded stochastic interpolant and 32 deterministic Euler steps. Defaults remain seed 0, dimension 256, two 512-unit hidden layers, 2,500 updates, GS batch 128/LR 1e-4/freeze first eight DinoBloom-B blocks, LF/flow batch 256/LR 1e-3. SigLIP excludes nearby equal surface readouts and now also nearby autofl-only readouts (`|Δautofl| < .25`). Production augmentation and clinical preprocessing/25 contrasts are copied from v1's repository-derived implementations.

### Completion and claims

Completion is freshly fit within each protocol, including normalization and marker encoders. FOV scoring uses split 2; held-out-draw scoring trains/validates only the source draw and tests opposite-draw split 2. Supplied LF features still come from teachers trained across both draws; this limits source-only encoder claims.

Models train with random observed-entry masking. LF, fused and labels-only neural prediction losses are scored **only on observed entries removed from the input**. Validation/early stopping contains only those prediction errors, with no SigLIP component or visible-entry reconstruction loss. The labels-only model has its own marker encoder and the exact same output MLP architecture/parameter count as the fused model; its LF half is zero. Target-specific ridge controls fit source split-0 rows with the same evaluation input restrictions, alpha 10, unpenalized intercept and deterministic cap 60,000 rows. C1 fits on single-marker cells, C2 on pooled cells.

Each C1/C2 row compares (a) LF only, (b) fused, (c) labels-only MLP **and** labels-only linear ridge, (d) ridge on `[f, observed values×mask, mask]`. C1 excludes autofl from input; C2 permits only other observed pooled groups. The hidden-information flag requires **b > max(a, both c controls, d)** with all correlations available. It is descriptive; there is no significance test. C1 gives separate gains versus each control for **each observed single marker**, plus an aggregate row. `abs(labels-only r) >= .2` flags a marker for autofl predictability/crosstalk review; it cannot establish the cause. Failed or source-unobserved C2 targets are explicitly unavailable. C3 remains descriptive coexpression/coherence with no other-marker truth.

## Resume, signatures, and outputs

Each stage appends to its own `.log`. A `.done.json` binds the SHA256 of `n054_trimodal.py`, exact stage arguments and upstream artifact fingerprints to exported outputs. The driver skips only when the signature and existing output fingerprints match. Signature drift fails with a request for a fresh versioned directory. Training `ckpt.pt` also checks code and training arguments, retains latest/best weights, optimizer/scheduler and CPU/CUDA/Python/NumPy RNG. Interrupted validation has a pending-validation cursor. Source cache, internal GS inference, and clinical archive inference resume from batch cursors. Periodic checkpoints occur every 250 optimizer steps or 20 inference batches; SIGTERM/USR1 saves at the next boundary and exits 143. Stats/ridge/geometry loops check STOP between bounded chunks; closed-form stats/ridge work recomputes after interruption. Repeated signals only set STOP.

The aggregator **consumes** the frozen selected run paths; it never chooses using test or clinical scores. Its primary table has five rows per direction (A0, selected A1-GS, A1-LF, A2, A3). `secondary_full_grid.csv` has all twelve rows per direction, including A2-EM and direct. `primary_metrics.csv` / `secondary_metrics.csv` retain per-target M2/M5 and registered clinical AUROCs. `diagnostics.csv` reports both oracle sigmas and direct regression. Selection file hashes and source artifact identities are recorded in `aggregation.json`. Completion tables are included alongside these tables.

Internal evaluation retains M1 measured autofl r, M2 common-ridge/own-LF agreement and assumption-dependent r products, M3 same-cell retrieval, M4 CKA and mutual-kNN overlap, and M5 LF test r. Saved NPZs contain global rows, truth masks, predictions and latents. Gap diagnostics report mean/SD cosine and distances for native `z_L=P2(f_own)` plus P1-space cosine and within-P2 nearest-other distances; hub rows use `z_L=P_L(f_own)`. M2 products are not measured GS marker truth. No cell has both GS and surface-marker truth; C3 cannot validate unobserved coexpression. Clinical labels never affect fitting or selection.

## GPU budget

Target **7.6 GPU-hours total**, conditional on one fast A100/H100-class GPU and warm local storage. No GPU throughput was measured here.

| Work | Assumption | GPU-h |
|---|---|---:|
| Shared space 1 + LF hub | Two × 2,500 small-MLP updates, validation/ridge included | 0.25 |
| Completion | Three × 2,500 updates, all neural/ridge controls | 0.30 |
| GS encoders | Six fits × 0.85 h; ~1 s/update plus ~560 s validation/I/O per fit | 5.10 |
| Bridges + direct | Eighteen maps × ~30 s plus source caching/OT | 0.25 |
| Internal grid/A0 inference | Batched frozen inference/geometry; full-slide grids | 0.30 |
| Clinical | Ten selected/baseline runs × ~8.4 min; eight-view TTA | 1.40 |
| **Total** | Preempted computation and queue time excluded | **7.60** |

Clinical image volume, CPU memmap latency and GPU class can exceed these assumptions; measure the first unit before treating this as a quota. Sharing removes the repeated GS fits and full-grid clinical work that drove v1's estimate. No training count, grid, or production TTA was reduced to manufacture this estimate.

## CPU reproduction

Dependencies: see `requirements.txt` (including `timm==1.0.31`). With the existing CPU environment:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/codex_n054 \
  /tmp/codex_n054/venv/bin/python -I n054_smoke.py \
  --scratch /tmp/codex_n054/n054v2_smoke_new > smoke_reproduction.log 2>&1
```

The smoke runs all three `--stage all` units, 20 updates per fit, at most 2,000 non-GS LF and 300 GS cells, a random two-block ViT-tiny at 32 px, and real data. A0 uses actual checkpoints/backbone. It sends **SIGTERM mid-GS-stage to the driver**, checks exit 143 and the saved cursor, resumes the whole unit, and compares GS exports bitwise to uninterrupted training. It also verifies whole-unit skips preserve artifact mtimes, changed-code resume rejection, all flow endpoint populations, shared weights/capacity, masked-only completion validation, per-marker controls, selected-only clinical outputs, and primary/full-grid aggregation. Clinical smoke uses 16 PBC/24 MLL23 images; A0 uses one CPU view, new primary arms eight. Smoke scores are integration checks, not scientific comparisons.
