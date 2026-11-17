# N045 red-team (Claude) — Codex SOTAlign runner

Reviewed: `codex/scripts/n045_sotalign.py`, `codex/scripts/merge_n045.py`, `codex/tests/`, `codex/README.md`, `codex/report.md`, `codex/smoke/*`, plus the integrator's `n045_gpu.sbatch` / `jobs.txt`. Upstream: `/hnvme/workspace/v181bb10-bsccm/external/SOTAlign` (read-only). Scratch: `/tmp/n045_rt/` (login node). No repo files edited, no jobs submitted.

Env for all reproductions: `apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif env PYTHONPATH=/hnvme/workspace/v181bb10-bsccm/env/pylib_n045 python3 ...` (CPU).

## BLOCKER

**B1. All four GPU jobs died at argument parsing; nothing is running.** [verified]
`logs/bsccm_n045_137489{2,3,4,5}.out` each end in
`ValueError: output must be outside all source/data trees` (rc=1, 7-15 s). `n045_gpu.sbatch` passes `--out $W/runs/n045/...` with `W=/hnvme/workspace/v181bb10-bsccm`. That is inside the default `--data-root`, and `parse_args` (l. 823-826) refuses it. `/hnvme/workspace/v181bb10-bsccm/runs/n045/` does not exist. preempt-run does not resubmit a command that exits by itself, so the chain is dead.
Fix: write somewhere outside the data root (e.g. `$HOME/n045_runs/...`; outputs are small), or relax the guard so it allows `data_root/runs/`.

## MAJOR

**M1. The preemption chain breaks on the first preemption, and a preemption changes which checkpoint is selected.** [verified]
- On SIGTERM/USR1 the runner exits with code 75. Reproduced through apptainer: `timeout --preserve-status -s TERM 14 ... n045_sotalign.py ...` gives `rc=75`. `~/.claude/skills/work-log/bin/preempt-run.sh` l. 60-64 accepts only rc 0/143 after a signal; any other code makes it `scancel` the successor. The first preemption therefore ends the run silently. The Codex launcher used `scontrol requeue`, which does not fit Helma (PreemptMode=CANCEL). The integrator switched to preempt-run but kept the code 75.
  Fix: exit 0 or 143 on STOP.
- On STOP, l. 456 runs a validation at an off-schedule step, and that result can become `best.pt`. Reproduced: the interrupted sotalign history starts `[86, 100, 200, ...]`. Selection then depends on when the preemption happened. Fix: save `last.pt` on STOP but do not evaluate or select.
- Resume itself works: siglip_pairs resumed vs uninterrupted differs by at most 5e-5 (8 vs 16 threads).

**M2. A GPU is not needed (CLAUDE.md compute rule).** [verified]
The full default config (1,500 steps, batches 256/256, dim 128, 100k unpaired, 100k/10k heads) ran all 7 arms for `cross_b0to1` in **2 min 55 s** wall on 16 login-node CPU threads (`/tmp/n045_rt/full_cpu_x01.log`). One learned arm trains in seconds.
Run all 4 protocols on `preempt_cpu`. Also, `cross_b0to1`/`fov_b0` (and `cross_b1to0`/`fov_b1`) use the same train pairs, validation pairs and seed. They train bit-identical models, so half of the training is duplicated, and the PBC rows of those two protocols are the same model (not replication).

**M3. Scientific: the CCA teacher nearly collapses on PBC.** [verified]
This confounds `sotalign` vs `sotalign_lf_only`. The teacher is fitted and centred on BSCCM Giemsa features. PBC DinoBloom features carry a large domain offset. On a 256x256 KLOT batch, PBC-A vs unpaired LF compared with BSCCM-train Giemsa vs LF (`/tmp/n045_rt/teach.py`):

| X batch | mean pairwise cos of projected X | var(K) across X rows | teacher-plan entropy (uniform 11.09, permutation 5.55) | Sinkhorn err |
|---|---:|---:|---:|---:|
| BSCCM Giemsa | .012 | .070 | 6.73 | 5e-5 |
| PBC-A | **.600** | **.0035** | **8.12** | 4e-9 |
| PBC-A re-centred on its own (label-free) mean | .020 | .0082 | 7.36 | — |

On PBC batches the KLOT target is therefore mostly domain-offset structure. In addition, upstream expanded SAIL pushes BSCCM-LF pair embeddings away from PBC as negatives. This is not a code bug: upstream assumes the unpaired data come from the same domain.
Recommendation: before reading the PBC arm, (a) log these teacher diagnostics, and (b) add a preregistered variant that re-centres PBC on the PBC-A mean (label-free; upstream `CCA.fit` already accepts `x_mean`).

## MINOR

- **m1. Teacher CCA rank (faithfulness).** The runner uses `n_components = min(dim=128, ...)`. Upstream `sotalign.sh` never sets `--cca_topk_x`, so the teacher there uses all components (512 here). Measured effect is small: teacher val R@10 .162 (128) vs .165 (full). The README calls it the "upstream regularized CCA teacher"; either document the 128 or use `None` for the KLOT teacher.
- **m2. Undocumented training-loop deviations, equal across arms.** Gradient clipping at 2.0 (upstream default None). Cosine schedule without the 10 % warmup. fp32 instead of amp. 1,500 instead of 2,000 target steps. Selection on val R@10 instead of upstream mean_R@1. These should be listed in the README.
- **m3. Training budget is limiting.** Best steps are 1200-1400 of 1500 for every learned arm. For siglip_pairs, val R@10 is still .318 → .321 at 1300 → 1400. Lion 1e-4 from Xavier init underfits; a 3-5x steps check costs minutes on CPU.
- **m4. Selection is tie-prone.** Val R@10 moves in steps of 1/994 (or 1/593) cells. Running with 8 instead of 16 threads changed the sotalign best step from 1200 to 1300 (val .2877 vs .2887). That moved PBC AUROCs by up to .0076 and teacher-agreement r by .010. Add a continuous tie-breaker (val mean rank or MRR).
- **m5. `n009_siglip` is not N009.** `scripts/screen_gs_lf.py` used AdamW lr 1e-3 / wd 1e-2, batch 512, logit scale 10, transductive all-row PCA, the final step, all 10,015 b0 pairs and LF teacher n006v3. N045 uses Lion 1e-4 / wd 1e-5, batch 256, scale 20, PCA on train pairs, validation selection and 8,007 pairs. The README says so; never present it as comparable. Add an exact-recipe arm if a bridge to N009 is needed (cheap).
- **m6. `comparison.csv` uses the least comparable autofl references.** The rows are K011 legacy-truth numbers (.648/.658/.32); legacy vs v5c autofl correlate only .92/.94 on Giemsa cells. Directly comparable clean numbers exist in N029 `n029_metrics.csv`:
  - cross-slide TT6c b0→b1 .715, b1→b0 .601; v4c .593/.214;
  - within-slide TT6c .564/.848.
  These are on the same test cells (all cells of the held-out slide, or split 2). Their truth is affine-identical to v5c: I measured r(v4c, v5c autofl) = 1.0000 in every draw and split, with identical masks. Add these rows.

## Checked and OK

- **Leakage.**
  - CCA, PCA, ridge bridge and scalers are fitted on train pairs (source draw, split 0) only.
  - Checkpoint and alpha selection use source split 1.
  - Test sets: cross = all Giemsa cells of the other draw; FOV = split 2.
  - Unpaired LF, head-train and head-val pools are non-Giemsa cells of the source draw (splits 0 and 1). This is asserted in code and by `check_contracts`.
  - Giemsa autofl truth is read only in `evaluate()`.
  - PBC-B is used for evaluation only. PBC-A labels are used only for the evaluation kNN.
  - The K001 assertion holds.
  - Remaining exposure is pre-existing and documented: the LF encoder saw non-Giemsa split-0 cells of both draws, and v5c calibration uses both draws.
- **Upstream code really runs.** `MatchingModel`, `CCA`, `KLOT` and `sinkhorn` resolve to the upstream files. `AlignmentLayer.forward.__code__.co_filename` is upstream `sail_model.py` (line 221), compiled from the unchanged AST; the inherited `__future__.annotations` flag is harmless.
  - The KLOT config matches `sotalign.sh`: ε .05/.01, 100 iterations, symmetric, no stop criterion; supervised SAIL 0, semi-supervised SAIL 1, KLOT .001.
  - The teacher sees the raw unpaired inputs, as upstream does.
  - The `lf_only` custom path calls upstream methods.
- **Metrics.**
  - Retrieval rank = count(sim ≥ diag − 1e-6), i.e. N009's definition with pessimistic ties. Chance is 1/n and 10/n over the whole test gallery.
  - The autofl Pearson uses the v5c mask; all Giemsa cells are labelled.
  - The 10 N021/N032 PBC contrasts are identical to `scripts/apply_gs_student.py` (names, markers, positive/negative sets, AUROC direction).
  - PBC A/B is the same sorted-filename even/odd split, and the labels were re-verified against the archive.
- **Smoke reproduced.** All 1,037/1,037 metric rows agree within 1.3e-5.
- **Fair baselines.** Learned arms share pair minibatches, initialisation, optimiser, step count and selection rule.
- **GPU [unverified].** I could not run on a GPU. By code reading, the device moves are correct, `CUBLAS_WORKSPACE_CONFIG` is set before torch is imported, TF32 is off, and no deterministic-mode-only ops are used.

## Preview: red-team CPU run, `cross_b0to1`, seed 0, default config

Not the official run; files in `/tmp/n045_rt/full_cpu_x01/`.

| arm | test R@10 (chance .0018) | autofl transfer r | same-cell LF head r | PBC 5-NN | CD16 neut>lym | CD3 lym>neut |
|---|---:|---:|---:|---:|---:|---:|
| raw_ridge | .0169 | **.657** | .755 | .853 | .517 | .390 |
| procrustes | .0195 | .526 | .753 | .956 | .676 | .757 |
| cca | .0209 | .257 | .755 | .965 | .152 | .270 |
| siglip_pairs | **.0370** | .213 | .704 | .966 | .460 | .666 |
| sotalign | .0295 | .384 | .732 | .960 | .419 | .582 |
| sotalign_lf_only | .0335 | .288 | .705 | .963 | .578 | .913 |
| n009_siglip | .0251 | .082 | .753 | .951 | .631 | .442 |

## Verdict

**LAND WITH FIXES:**
1. B1: move `--out` outside the data root, or relax the guard. Resubmit.
2. M1: exit 0/143 on STOP, and do not evaluate or select on STOP.
3. M2: run on `preempt_cpu` (about 3 min per protocol) and de-duplicate the identical cross/FOV training.
4. m1, m2, m5: document the deviations from upstream and from N009.
5. m6: add the N029 clean references.

Strongly recommended before interpreting the PBC arm: M3 diagnostics plus the PBC re-centred variant, the m3 longer-budget check, and the m4 tie-breaker.
