# N054 v3

This is a standalone replacement delivery. All files in the parent directory, repository code and input data remain untouched. CPU smoke artifacts are under `/tmp/codex_n054/`; production outputs use `runs/n054v3_*`. No GPU job was submitted by the writer.

## Launch exactly three units

Run from the cluster login shell. Parameters live in exported environment variables, so the house wrapper can replay the batch script after `PreemptMode=CANCEL`.

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v3
export N054_CODEDIR="$PWD"
export N054_RUNROOT=/hnvme/workspace/v181bb10-bsccm/runs
export N054_PREFIX=n054v3

apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B -c \
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
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$N054_CODEDIR/n054_aggregate.py" \
  --units "$N054_RUNROOT/${N054_PREFIX}_batch1" "$N054_RUNROOT/${N054_PREFIX}_batch0" \
  --lf-unit "$N054_RUNROOT/${N054_PREFIX}_lf" \
  --out "$N054_RUNROOT/${N054_PREFIX}_tables"
```

Units batch1 and batch0 may run concurrently **after the LF unit's entire chain finishes**. The user can implement that chaining externally. A plain `afterok:<first-LF-job>` dependency is insufficient if LF is preempted: its successor has another ID. The holdout driver also checks all three LF stage manifests before starting. Avoid concurrent writers to the same unit directory. Use a new `N054_PREFIX=n054v3_r2` after changing code/configuration.

`n054_unit.sbatch` requests partition `preempt`, one GPU, 16 cores, eight hours and `--signal=B:USR1@900`. It executes the existing `~/.claude/skills/work-log/bin/preempt-run.sh -- apptainer exec ... python3 -I -B ...`, without `srun`. The house wrapper resubmits the batch script, forwards one SIGTERM to the command and accepts Python's checkpointed exit **143**. Cancel a chain using the house `preempt-cancel.sh <root-job-id>`.

## Unit contents and shared artifacts

1. `n054v3_lf`: fit `space1/lf_spaces.pt` (P1/E_M/D_M), `hub/lf_spaces.pt` (P_L/H/E_M), and the common fixed ridge **once**, using non-GS LF split 0 from both draws and split 1 for validation. Fit completion's FOV, draw0→1 and draw1→0 protocols once, independently of GS holdout. Completion uses **targets_v5cx_v10.npz**; GS-space models use **targets_tt6c.npz**. Metadata/feature row alignment is identical across these files. TT6c's 27,074 failed draw-2 group-8 observations are masked; v10 already masks them and is asserted to do so.
2. `n054v3_batch1`: hold out GS batch 1. Train one `space2/gs_encoder.pt` (P2/E_G SigLIP), then one hub GS encoder for each `w_sig ∈ {0, .5}`. Cache only source split-0/1 GS latents. Train 14 small maps: the retained LF, in-sample GS, GW A2, Euclidean A2 and A2-EM two-sigma grids, direct LF pretraining, in-sample direct, and the two fixed out-of-sample primary maps. Freeze `selection.json`, evaluate the entire grid and A0, and run clinical inference on **A0 + five primary configurations only**. A1-GS/direct are fixed in advance; other primary families retain source-validation selection.
3. `n054v3_batch0`: identical workflow with the GS slides reversed, reusing all LF-unit outputs.

Every two-space map references the exact same frozen space-1 artifact and the holdout's exact same space-2 artifact. The script does not copy or retrain them per bridge. Hub GS fits reference the single LF hub. GS training uses source split 0, source split 1 supplies the disjoint map-training/holdout partition for primary A1-GS/direct; it selects configurations only in the other families, and the entire opposite slide is test-only. Non-GS LF fitting uses split 0 across draws, as specified; LF split-0 training includes the GS-held-out draw’s non-GS unstained cells: **6,604 slide-19 rows (2.08%) / 5,088 slide-2 rows (1.60%)** out of 317,935 non-GS training rows. These per-holdout counts are written to `data_audit.json`; some non-GS unstained cells share slides/FOVs with GS cells, so this is GS-cell exclusion, not whole-slide exclusion for the LF side. Source split 2 is unused for GS fitting.

### Arms

- **A1-LF:** same-cell `(P2(f), P1(f))` LF bridge. Retained as the literal LF alignment design. Test evaluation additionally starts this map from `P2(f_own)` for an explicitly labelled **oracle diagnostic**.
- **A1-GS primary:** `A1-GS_oos_s0`, initialized from `A1-LF_s0`. Train on out-of-sample source GS **split 1**, excluding a deterministic 20% of slide/FOV groups (rounded up). E_G optimizer training remains source split 0. Sigma is fixed at 0 and the budget at **600 updates**. Export the final iterate without checkpoint/sigma selection. `finetune_curve.json` contains step 0 and validation checkpoints (250, 500, 600), with loss, mapped cosine and measured autofl r on the entire grouped holdout, for information only. The encoder retains v2's split-1 validation; these are out-of-optimizer-sample latents, not an untouched model-selection set.
- **A1-GS secondary:** `A1-GS_insample_s{0,0.1}`, initialized from the matching LF map, 600 updates on source split 0, labelled in-sample.
- **direct primary:** identical `Velocity` capacity, fixed zero time, normalized direct regression. First train `direct_pre` on LF same-cell pairs `P2(f) -> P1(f)` for 2,500 updates, then fine-tune `direct` on exactly the same split-1 training/holdout partition, sigma 0 and 600-update final-iterate policy as A1-GS. `direct_insample` retains the split-0 secondary. Both maps' pre/post test numbers are exported in `pre_post.json` and aggregate `pre_post.csv`; A1-GS's pre row is `A1-LF_s0`.
- **A2 primary:** source-slide split-0 GS vs independently sampled different non-GS LF split-0 cells, using intra-space cosine-distance **entropic Gromov–Wasserstein** coupling. A small torch solver implements the squared-loss fixed-point/Sinkhorn method of [Peyré, Cuturi & Solomon (2016)](https://proceedings.mlr.press/v48/peyre16.pdf). No cross-space coordinate costs enter. Distance-profile initialization breaks uniform symmetry; epsilon .005, at most 40 outer iterations, 100 log-Sinkhorn iterations, uniform marginals with balancing round-off correction. Sample joint endpoint pairs from the coupling for flow matching. Both sigmas remain; source-validation selects between them. The solver is nonconvex and does not guarantee biological correspondences between independently trained spaces.
- **A2 Euclidean secondary:** `A2_euclidean_s{0,0.1}` retains the original Hungarian coordinate cost, explicitly labelled **frame-dependent**. Each GW run also writes `gw_identity_sanity.json`: same LF cells encoded as P2(f) and P1(f), reporting argmax identity rate, diagonal coupling mass and chance. The synthetic rotation/permutation unit test is separate from this empirical check; cross-model geometry may differ.
- **A2-EM:** same source GS marginal, with full E_M(marker row) target marginal. Secondary only. D_M is trained on P1 inputs, exact leave-one-entry-out E_M inputs, **and full E_M inputs**.
- **A3:** GS latent regression is `(zg-zl).square().sum(-1).mean()` on unit vectors (`2−2cos`), plus `w_sig * SigLIP`. Both weights are retained.
- **A0:** load the original TT6c checkpoint and its original preprocessing/teachers, without retraining.

Bridge sigmas are `{0, .1}`, using the bounded stochastic interpolant and 32 deterministic Euler steps. Defaults remain seed 0, dimension 256, two 512-unit hidden layers, 2,500 updates except the 600-update GS-map fine tunes, GS batch 128/LR 1e-4/freeze first eight DinoBloom-B blocks, LF/flow batch 256/LR 1e-3. SigLIP excludes nearby equal surface readouts and now also nearby autofl-only readouts (`|Δautofl| < .25`). Production augmentation and clinical preprocessing/25 contrasts are copied from v1's repository-derived implementations.

### Completion and claims

Completion is freshly fit within each protocol, including normalization and marker encoders. FOV scoring uses split 2; held-out-draw scoring trains/validates only the source draw and tests opposite-draw split 2. Supplied LF features still come from teachers trained across both draws; this limits source-only encoder claims.

Models train with random observed-entry masking. LF, fused, labels-only and plain-MLP neural prediction losses are scored **only on observed entries removed from the input**. Validation/early stopping contains only those prediction errors, with no SigLIP component or visible-entry reconstruction loss. The labels-only model has its own marker encoder and the exact same output MLP architecture/parameter count as the fused model; its LF half is zero. Target-specific ridge controls fit source split-0 rows with the same evaluation input restrictions, alpha 10, unpenalized intercept and deterministic cap 60,000 rows. C1 fits on single-marker cells, C2 on pooled cells.

Each C1/C2 row compares (a) LF only, (b) fused, (c_mlp/c_linear) labels-only MLP/linear ridge, (d) ridge on `[f, observed values×mask, mask]`, (e) linear stacking of `[pred_LF, pred_labels_MLP]` with intercept, and (f) a plain MLP directly on `[f, observed values×mask, mask]`. The plain MLP has the fused head's two 512-wide hidden layers, with no marker encoder or shared latent space, and trains under the same masking regime. Stacking uses only source split-1 rows and task-specific masks, with separate fits for each C1 observed-marker row and the aggregate; coefficients and global fit-row IDs are saved. Test rows never enter a fit. C1 excludes autofl; C2 permits only other observed pooled groups. The flag requires **fused > max(a, c_mlp, c_linear, d, e, f)** with all scores available. Each row reports gains against every control. C1 includes each observed single marker plus an aggregate; C3 remains descriptive.

## Resume, signatures, and outputs

Each stage appends to its own `.log`. A `.done.json` binds the SHA256 of `n054_trimodal.py`, exact stage arguments and upstream artifact fingerprints to exported outputs. The driver skips only when the signature and existing output fingerprints match. Signature drift fails with a request for a fresh versioned directory. Training `ckpt.pt` also checks code and training arguments, retains latest/best weights, optimizer/scheduler and CPU/CUDA/Python/NumPy RNG. Interrupted validation has a pending-validation cursor. Source cache, internal GS inference, and clinical archive inference resume from batch cursors. Periodic checkpoints occur every 250 optimizer steps or 20 inference batches; SIGTERM/USR1 saves at the next boundary and exits 143. Stats/ridge/geometry loops check STOP between bounded chunks; closed-form stats/ridge work recomputes after interruption. Repeated signals only set STOP.

The aggregator **consumes** the frozen selected run paths; it never chooses using test or clinical scores. Its primary table has six rows per direction (A0, fixed A1-GS, selected A1-LF/A2/A3, fixed direct). `secondary_full_grid.csv` has all seventeen rows per direction, including in-sample maps, LF pretraining, frame-dependent A2 and A2-EM. `primary_metrics.csv` / `secondary_metrics.csv` retain per-target M2/M5 and registered clinical AUROCs. `diagnostics.csv` reports both oracle sigmas, all three direct-regression phases, and the GW same-LF-cell identity sanity rows. `pre_post.csv` retains both maps’ before/after metrics, and `gw_identity_sanity.csv/json` compares cross-model coupling identity to chance. Selection file hashes and source artifact identities are recorded in `aggregation.json`. Completion tables are included alongside these tables.

Internal evaluation retains M1 measured autofl r, M2 common-ridge/own-LF agreement and assumption-dependent r products, M3 same-cell retrieval, M4 CKA and mutual-kNN overlap, and M5 LF test r. Saved NPZs contain global rows, truth masks, predictions and latents. Gap diagnostics report mean/SD cosine and distances for native `z_L=P2(f_own)` plus P1-space cosine and within-P2 nearest-other distances; hub rows use `z_L=P_L(f_own)`. M2 products are not measured GS marker truth. No cell has both GS and surface-marker truth; C3 cannot validate unobserved coexpression. Clinical labels never affect fitting or selection.

## GPU budget

Planning estimate **8.6 GPU-hours total**, conditional on one fast A100/H100-class GPU and warm storage; no GPU throughput was measured. Per-unit eight-hour requests remain sufficient under these assumptions because LF and the two holdouts are separate units.

| Work | Assumption | GPU-h |
|---|---|---:|
| Shared space 1 + LF hub | Two × 2,500 updates, validation/ridge | 0.25 |
| Completion | Three × 2,500 updates, extra plain MLP + stack fits | 0.40 |
| GS encoders | Six fits × 0.85 h, unchanged | 5.10 |
| Maps, caching and GW | 28 maps, 4 GW fits; allowance for iterative coupling | 0.75 |
| Internal full-grid/A0 inference | 34 configurations, pre/post diagnostics | 0.40 |
| Clinical | 12 primary/baseline runs × ~8.5 min, eight-view TTA | 1.70 |
| **Total** | Queue and repeated preempted computation excluded | **8.60** |

The torch GW solver uses float64 matrix products; GPU class, double-precision throughput and real latent convergence can materially change its cost. A 256×256 random coupling took about 0.21 seconds on two CPU threads; this is not a GPU benchmark. Measure production throughput before treating the estimate as a quota. No GPU jobs were submitted here.

## CPU reproduction

Dependencies: see `requirements.txt` (including `timm==1.0.31`). With the existing CPU environment:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/tmp/codex_n054 \
  /tmp/codex_n054/venv/bin/python -I -B n054_smoke.py \
  --scratch /tmp/codex_n054/n054v3_smoke_new > smoke_reproduction.log 2>&1
```

The smoke runs all three `--stage all` units, 20 updates per fit, at most 2,000 non-GS LF and 300 GS cells, a random two-block ViT-tiny at 32 px, and real data. A0 uses actual checkpoints/backbone. It sends **SIGTERM mid-GS-stage to the driver**, checks exit 143 and the saved cursor, resumes the whole unit, and compares GS exports bitwise to uninterrupted training. It also verifies whole-unit skips preserve artifact mtimes, changed-code resume rejection, all flow endpoint populations, out-of-sample FOV isolation, LF direct pretraining, both added completion controls, and GW rotation/permutation invariance, shared weights/capacity, masked-only completion validation, per-marker controls, selected-only clinical outputs, and primary/full-grid aggregation. Clinical smoke uses 16 PBC/24 MLL23 images; A0 uses one CPU view, new primary arms eight. Smoke scores are integration checks, not scientific comparisons.

Run the GW unit test alone with:

```bash
/tmp/codex_n054/venv/bin/python -I -B n054_gw_test.py
```

It writes `gw_test_results.json`, including three 48-point/16-dimensional cases and one 64-point/256-dimensional case. The full smoke includes it. `production_partition_audit.json` records the full-data split-1 FOV partition without fitting a model. `report.md` separates smoke evidence from production claims.
