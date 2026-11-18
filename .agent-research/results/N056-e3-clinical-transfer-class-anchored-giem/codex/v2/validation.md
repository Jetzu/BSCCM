# N056 v2 validation — 2026-11-18

Final CPU invocation (exit0):

```bash
/tmp/codex_n054/venv/bin/python -I -B n056_smoke.py \
  --out /tmp/codex_n056v2/smoke_final > cpu_smoke.log 2>&1
bash -n n056_unit.sbatch
```

Environment: existing read-only CPU venv, torch2.15.0+cpu, no CUDA device.
No GPU jobs submitted; no new clinical predictions, contrast scores or fractions
were generated. All synthetic tables live in /tmp and are labelled fixtures.
Source repo/data and v1 were read only. No packages installed.

| Finding | Change and observed evidence |
|---|---|
| P1-1 | Seven saved T checkpoints/configs verified; synthetic full 16-evaluation grid confirms equal teacher weighting despite unequal T seed counts, strict noise-floor decisions and primary16. Real tiny H seeds0/1 have identical hubs and different students. |
| P1-2 | Literal T SigLIP/ridge output/anchor recipe replayed on tiny real source and permitted anchor-A batches. T/H measured shares agree within1e-12 in each of four CPU calibration updates. Production is200 updates; CPU does not establish clinical effectiveness or full200-step GPU calibration. |
| P1-3 | All three controls serialize and produce finite unit embeddings; both fixed tau parameterizations checked. Existing OT/CCOT identity, permutation, serialization, inductive extension and finite-extreme checks pass. |
| P1-4 | Synthetic fractions confirm transfer_drift and bsccm_calibration separately, aligned-row interpretation labels, and unsupported measured-class fractions remaining NA. |
| P2-1 | Production arguments lock lr1e-3/batch256/2500 hub updates; code applies K019 batch1/group8 exclusion while retaining raw masks for T replay. |
| P2-2 | Report states both LF feature inputs and bans across-teacher SD comparisons; aggregation exports no cross-teacher SD ranking. |
| P2-3 | Cache interrupted after first transaction, resumed remaining batches without recomputing completed batch; exactly one cache file, no .parts. Default run paths and Slurm logs use /hnvme/runs/n056v2_*. |
| P2-4 | All PBC thumbnails audited. Seven CRC+size exact pairs and eight thumbnail pairs total (seven overlap exact groups),16 excluded filenames. Both halves retain8538 images and exclude all16 names. MLL23 fit/eval IDs disjoint (7268/7265). |
| P2-6 | Full real T source inference confirms cached normalized eight-view mean; alignment application accepts2-D mean vectors and identity mapping equals decoding those means. |
| P2-7 | Both real-source smoke training arms use12-worker DataLoaders. Real SIGTERM exits143 during student updates; resumed H and H_matched parameters are bitwise identical to uninterrupted runs. Matched reference optimizer/RNG/calibration trace is also identical. Completed runs exit immediately. |

Verbatim AST checks pass for legacy augmentation/anchor dataset and all25
contrast definitions. Actual full saved t0 T checkpoint loads and performs8-view
inference on two SOURCE TRAINING cells. Bash syntax and Python AST checks pass.

`duplicate_audit.log` records the independent hygiene run. `pbc_excluded.json`
contains every excluded archive filename, exact group CRC/size and thumbnail-pair
cosines with both files' CRC/size. Historical T training exposure to removed
anchors is disclosed; available checkpoints are reused as requested.

`MANIFEST.sha256` binds handoff code, preregistration, audit and final evidence.
GPU cost remains unmeasured. README gives a conditional~5.7 H100 GPU-hour target
and the slower-runtime~11–16 h scenario; no seed or clinical-case reductions are
made to manufacture a budget claim. H_matched explicitly uses adaptive weights
in calibration, freezing their mean after update200.
