# N092 field-of-view shortcut audit: how to run

`n092_audit.py` imports the unmodified N045 runner
(`../N045-sotalign-semi-supervised-ot-alignment-gi/codex/scripts/n045_sotalign.py`) by path.

Reused unchanged from N045:
- `Data` and `partition`. The script asserts that train/val/test equal the N045 v2 `partitions.npz`.
- `fit_cca`, `train_model` (SigLIP pairs), `LearnedMap`, `FrozenMap` and `retrieval`.

The N045 raw-ridge branch is written inline in N045's `run()`, so it cannot be called; `fit_raw_ridge` copies its logic.

Defaults are the N045 v2 settings (`n045_cpu.sbatch`): seed 0, 4,500 steps, 256 pairs per batch, dim 128, Lion lr 1e-4,
wd 1e-5, evaluation and checkpoint every 100 steps, and ridge alphas 1/10/100/1000.

## Arms

| arm | training positives | checkpoint selection |
|---|---|---|
| `raw_ridge`, `cca` | true pairs (as N045) | N045 (ridge alpha on val R@10) |
| `siglip_pairs` | true pairs (as N045) | best.pt = max true-pair val R@10 (N045); also last.pt |
| `siglip_pairs_multifov` | true pairs, single-cell fields dropped (the same cells as the control) | best.pt on true val of multi-cell fields; also last.pt |
| `siglip_samefov_shuffled` (CONTROL) | (GS of cell i, LF of a DIFFERENT cell j of i's field); one random cycle per field (`--control-seed 9200`); single-cell fields dropped | best.pt on val pairs shuffled the same way, so true pairs are never seen (`--control-selection true` switches to true val); also last.pt |

All arms are evaluated on the normal TRUE-pair test set.

## Metrics

Each metric is computed from the dumps only. Queries are GS and the gallery is the LF test cells, as in N045. Ties are
pessimistic (rank = #(sim >= sim_true - 1e-6)).

| column | meaning | chance column |
|---|---|---|
| `R1`, `R5`, `R10`, `median_rank` | (a) N045 retrieval over all test cells. `n045_retrieval_R10` is N045's own `retrieval()` on the same embeddings, and `n045_v2_R10` is the value in `v2_summary.csv`. | `chance_R10` |
| `fovhit1/5/10` | (b) Share of queries with ANY own-field LF cell in the top k | `chance_fovhit*`: exact hypergeometric per query, then averaged |
| `withinfield_acc` | (c) Queries whose field has >= 2 test cells: is the true partner top-ranked among that field's LF cells only? | `chance_withinfield_acc` = mean 1/n_field |
| `R10_excl_samefield` (and R1, R5) | (d) Rank after removing the query's OTHER same-field cells from the gallery | `chance_R10_excl_samefield` |
| `fieldmean_R10` (and R1, R5) | (e) Every embedding replaced by its field mean (per modality), giving rank >= n_field. This is the R@10 that field identity alone yields in this embedding. It is NOT shortcut evidence by itself, because averaging correct cell matches also matches the field means. | (none) |
| `R10_singlecell_field_queries`, `R10_multicell_field_queries` | R@10 split by whether the query's field has one cell or several | (none) |
| `R10_over_siglip_pairs` | Ratio of this arm's R@10 to `siglip_pairs` at the same checkpoint rule. This is the headline number for the control. | (none) |

## Exact CPU command lines (one per protocol)

Run these inside a 48-core `preempt_cpu` allocation. The 64-core login node also works but is not the intended place.

```bash
W=/hnvme/workspace/v181bb10-bsccm
R=/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N092-field-of-view-shortcut-audit-of-lf-giems
apptainer exec --bind $W $W/env/bsccm.sif env PYTHONPATH=$W/env/pylib_n045 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 python3 $R/n092_audit.py --protocols fov_b0 --threads 48 --out /home/hpc/v181bb/v181bb10/bsccm_runs/n092/fov_b0 --results-dir $R --resume
apptainer exec --bind $W $W/env/bsccm.sif env PYTHONPATH=$W/env/pylib_n045 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 python3 $R/n092_audit.py --protocols fov_b1 --threads 48 --out /home/hpc/v181bb/v181bb10/bsccm_runs/n092/fov_b1 --results-dir $R --resume
```

To submit a line, use
`sbatch -p preempt_cpu -c 48 --mem=64G -t 00:30:00 -J n092_fov_b0 -o /home/hpc/v181bb/v181bb10/BSCCM/logs/%x_%j.out --wrap "<line>"`
after setting `W` and `R` inside the wrap.

The run lasts well under the 2-hour preempt exemption, so preempt-run is not needed. The script is still preempt-safe:
- SIGTERM or SIGUSR1 checkpoints the run and exits 143 (N045 `train_model`).
- `--resume` continues from `last.pt` and skips finished arms.
- To use preempt-run anyway, it needs an sbatch FILE (not `--wrap`).

`--resume` requires the identical command and the identical script, because the fingerprint includes the sha256 of both
scripts.

### Expected runtime

About 2-4 min per protocol on 48 cores:
- N045 v2 measured `siglip_pairs` at 4,500 steps = 25 s on 48 cores (fov_b0 `training.json`).
- N092 trains 3 SigLIP arms plus ridge and CCA in seconds.
- Data loading takes under 1 s once cached.
- The full 7-arm N045 jobs took 10-11 min. Most of that was SOTAlign (251 s) and the marker heads and PBC evaluation,
  which N092 does not run.

## Outputs (`--out`, per protocol)

- `config.json` (config + fingerprint), `completion.json`.
- `<protocol>/seed0/partitions.npz`: train/val/test hist rows and field ids.
- `<protocol>/seed0/<arm>/`:
  - `best.pt`, `last.pt`, `training.json` (N045 `train_model` format);
  - `audit_selection.json`;
  - `control_pairs.npz` for the control (GS row → LF partner row, train and val);
  - `map.joblib` for raw_ridge.
- `<protocol>/seed0/dumps/<arm>__<best|last>.npz`:
  - `gs_emb` and `lf_emb`: TEST embeddings, L2-normalised float32, row i = the same cell;
  - `test_hist_row`, `test_global_index`, `slide`, `fov`, `field`, `batch`;
  - `n045_R1`, `n045_R10`;
  - `info` (JSON: arm, ckpt, selection).
- `<protocol>/seed0/metrics.csv|json`: one wide row per arm × checkpoint, with all metrics above. The same file is
  copied to `--results-dir` as `metrics_<protocol>_seed0.csv|json`.
- To recompute metrics only: `... n092_audit.py --from-dumps --protocols fov_b0 --out <same out>`.
- To audit the EXISTING N045 v2 models without training: `--reuse-n045 --arms raw_ridge cca siglip_pairs`. This loads
  `best.pt`, `last.pt`, `map.joblib` and `cca_teacher.pt` from `bsccm_runs/n045_v2/original_<protocol>`.

Training stdout lines print `"arm": "siglip_pairs"` for all three SigLIP arms, because N045's `train_model` is called
with that arm name. The output folder identifies the arm.

## Smoke and verification evidence (2026-11-27, login node, container `bsccm.sif`, 4-8 threads)

- `smoke_selftest.log` (`--selftest`):
  - analytic 33-cell toy: R1 12/33, R10 21/33, within-field acc 10/31 (chance 7/31), FOV-hit@10 1, field-mean R10
    21/33, median rank 3;
  - 148-cell random toy: vectorised metrics = brute-force loops for every metric, and R@1/R@10 = N045 `retrieval()`;
  - derangement checks: no fixed points, same field, a permutation, singletons dropped.
- `smoke_train.log`: end-to-end run of all 5 arms × fov_b0/fov_b1 on 40 random fields per block (782 cells), 20 steps,
  dim 32. It took 6.5 s. These numbers are execution checks only.
  - An independent check against `targets_v5c` fov ids confirms that every control training/val pair joins two
    different cells of the same field. The LF partners are a permutation of the GS rows.
  - SIGTERM at step 2062, then `--resume`, continued from step 2062 and completed. `--from-dumps` reproduces the table.
- `smoke_reuse_n045.log`, `metrics_fov_b{0,1}_seed0_reuse045.csv|json`: `--reuse-n045` on the real N045 v2 checkpoints.
  - Metric (a) equals `v2_summary.csv` to 4 decimals: siglip .3994/.3842, raw_ridge .1479/.1445, cca .1785/.2036.
  - Partitions equal N045 v2.

Preview from the existing N045 models (no control yet; seed 0):

| protocol, arm | R10 | FOV-hit@10 (chance) | within-field acc (chance) | R10 excl. same field |
|---|---|---|---|---|
| fov_b0 siglip_pairs best | .399 | .435 (.048) | .840 (.233) | .403 |
| fov_b1 siglip_pairs best | .384 | .432 (.064) | .799 (.282) | .386 |
| fov_b0 / fov_b1 cca | .179 / .204 | .224 / .255 | .672 / .676 | .180 / .204 |

Reading of the preview:
- Inside the right field, the true partner is picked at 3-4x chance.
- Removing same-field neighbours barely changes R@10.
- Finding the field (FOV-hit@10 ≈ .43) is what limits R@10.

This argues against a PURE field shortcut, pending the control.

## Fairness caveats of the control

1. The control learns every signal shared at field level. That includes technical nuisance (focus, background,
   illumination) and also regional biology (the local cell-type mix, smear thickness). So a high control score shows
   that field-level information suffices. It does not show that this information is technical.
2. The derangement is static: each GS cell keeps one fixed wrong LF partner for all 4,500 steps, and 2-cell fields are
   swaps. The head can partly memorise these noise pairs, so the control is a conservative field-only learner. A
   control far below the real arm is weaker evidence than a control close to it.
3. The control drops cells in single-cell fields (4.6% of training cells on b0, 8.9% on b1). Compare it with
   `siglip_pairs_multifov`, which is matched on exactly those cells, and at both `best` and `last` (no selection effect).
4. Neither the control nor metric (c) can exclude a WITHIN-field position shortcut. A cell's two images share its
   position in the field (vignetting, local focus), so high within-field accuracy may partly be position matching.
5. The audit uses one seed and one LF teacher (n029_r18_v4c_fov probe), as N045 v2.
