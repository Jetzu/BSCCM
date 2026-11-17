# N040 red-team review

**Verdict: LAND WITH FIXES.** Fix F1–F4 before treating stage-2 results as the specified experiment. The ordinary finished-teacher path, source-row checks, real-label masking, and tuple changes passed the checks below. No repository implementation files were changed.

Reviewed 2026-11-12 against the supplied `train_lf_uncommitted.diff` and N040 item contract. The live `scripts/train_lf.py` matched the diff's new blob: `c492b6dad0dcff239531b85f896035d8dcf803fa`. Line numbers below refer to that version. N042 was outside this review; N041 interaction is noted below.

## Required fixes

### F1 · P1 · Dump mode can train and overwrite an unfinished teacher before rejecting it

**Location:** `scripts/train_lf.py:178–230`, especially training at 188 and the dump precondition at 226.

`--dump-train-preds` does not branch before training. With a missing checkpoint, `ep0=0`, so it runs the entire training loop, writes `ckpt.pt` and final `eval_preds.npz`, and only then raises the finished-checkpoint assertion. With a valid partial checkpoint, it resumes training and replaces the teacher checkpoint before raising, because `ep0` still contains the original incomplete epoch. This violates the dump operation's read-only teacher contract even though `result.json` is spared and W&B is disabled under normal Python execution.

**Reproduced**, executing the unchanged script source in an isolated CPU fixture (substitutions described below):

```text
RUN dump_missing dump True status AssertionError: dump-train-preds needs a finished checkpoint in --out wandb calls 0
missing dump wrote checkpoint True eval True result False
RUN teacher_run dump True status AssertionError: dump-train-preds needs a finished checkpoint in --out wandb calls 0
# The second case resumed a real epoch-1 checkpoint of a 2-epoch fixture.
incomplete dump changed existing files ['ckpt.pt']
```

**Fix:** Validate checkpoint existence, config/rows, and completed epoch before any training or run-owned write; execute dump in a dedicated branch that never enters training. Use explicit control flow rather than an assertion to disable W&B. Keep the ordinary finished-checkpoint behavior verified below.

N041 interaction: even a completed dump with `PHYS=True` reaches the unconditional calibration-file write at 185–186 before dumping. Skip that write in dump mode too. **[unverified execution]** This observation is from source inspection; physics augmentation was not enabled in the harness.

### F2 · P1 · Pseudo loss includes an untrained teacher head and exceeds the 11-target contract

**Location:** `scripts/train_lf.py:34–35,61,214`.

The item specifies dense pseudo labels for the common 11 targets. The target file actually has 13 columns, and `PM[_G0] = True; PM &= ~Mk` enables every missing column. This adds both `autofl` and `grp:CD123/HLA-DR/CD14` to pseudo supervision. In source batch 1, the latter has **zero real training labels** under v5 QC. The teacher's masked regression loss supplies no supervised gradient to that output row, yet the student is trained to match it on **269,283 cells**. This is distillation of an unsupported output, with gradients also affecting the shared encoder. Averaging two such teacher outputs does not establish a learned target.

**Reproduced on the real read-only v5 arrays:**

```text
target shape (412941, 13)
# Counts ordered [batch 0, batch 1], non-Giemsa source cells:
grp:CD123/HLA-DR/CD14 real counts [10210, 0] pseudo counts [117736, 269283]
autofl real counts [117736, 242209] pseudo counts [10210, 27074]
```

The existing `evaluate()` comment at 154 already recognizes the unsupported batch-1 target; the new pseudo mask does not respect it.

**Fix:** Select pseudo columns explicitly by the common-11 target names, preserving all real-label losses. Also require sufficient source supervision for any teacher output allowed into pseudo loss. Save target names and support metadata in dumps so column meaning and support can be checked. Any decision to pseudo-supervise additional outputs should be an explicit experiment change.

### F3 · P2 · Resume fingerprints teacher paths, not teacher supervision

**Location:** `scripts/train_lf.py:58,173–183,224`.

`_cfg()` includes the comma-separated teacher path and nondefault `pseudo_w`, but no teacher content digest or equivalent identity. Replacing `P` at the same path leaves config, training rows, and eval rows unchanged, so resume accepts different supervision midway through a run. `train_preds.npz` itself is overwriteable by another dump, making this plausible during normal iteration.

**Reproduced using the actual `_cfg()` definition:**

```text
pseudo path included /tmp/codex_n040/teacher.npz pseudo_w omitted at .3 True
overwriting teacher P preserves fingerprint True
changing weight changes fingerprint True
```

**Fix:** Include a stable digest of each teacher's actual predictions, row IDs, target schema, and relevant provenance in the saved fingerprint; verify it on resume before training or W&B initialization. Recording file paths alone does not pin the objective. Include default `pseudo_w` semantically, while retaining compatibility with checkpoints written before pseudo training existed.

### F4 · P2 · Missing artifact/weight validation permits silent target corruption and NaN training

**Location:** `scripts/train_lf.py:21,58–61,213–214`.

There is no check that each `P` is finite and exactly `(len(G), K)`, that `G` is a unique one-dimensional array of valid global indices, or that target columns match the student's target schema. NumPy assignment accepts `(N,1)` predictions by broadcasting one output across all targets. Duplicate row IDs pass and last assignment wins. No target names are exported, so same-shaped column permutations cannot be detected. `--pseudo-w nan`, `inf`, and negative values are accepted.

A nonfinite teacher prediction even at a **real-labeled, pseudo-masked position** poisons the pseudo loss: subtraction/squaring happen before multiplication by zero, and `NaN * 0` remains NaN. Thus the mask is logically correct for finite predictions but does not protect training against bad artifacts.

**Reproduced with the actual loader/mask statements and loss expression:**

```text
broadcast_one_column ACCEPTED overlap False PY [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
duplicate_rows ACCEPTED ... PY [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]
NaN_on_real_label ACCEPTED overlap False
NaN masked loss nan finite gradient False
parser accepted pseudo_w nan
parser accepted pseudo_w inf
parser accepted pseudo_w -1.0
```

**Fix:** Fail before run initialization on nonfinite or incorrectly shaped `P`, nonunique/out-of-range/noninteger `G`, schema mismatch, or nonfinite/negative pseudo weight. Keep the existing coverage, draw, Giemsa, and ensemble row-order checks. Use explicit exceptions for data-integrity checks rather than assertions that disappear with `python -O`.

## Checks that passed

| Requested check | Evidence and limits |
| --- | --- |
| Finished teacher dump preserves the run | CPU fixture: existing checkpoint, result, and eval-prediction hashes unchanged; only `train_preds.npz` added; zero W&B init calls; exit code 0. Model switches to eval and inference runs under `no_grad`. F1 covers invalid/incomplete teachers. |
| Exactly the student row population | Dump uses `mk(tr, False)`: ordered, no shuffle, no dropped remainder. Fixture dumped all 32 rows with batch size 5, including the last 2. Real v5 `has` filtering excludes **zero** additional non-Giemsa cells: `tr=127946` for batch 0 and `269283` for batch 1. At student batch size 256, 202/227 shuffled rows are dropped per epoch by the pre-existing `drop_last=True`; the dump correctly covers the entire eligible population rather than a fixed dropped subset. |
| No held-out-draw or Giemsa pseudo rows | Actual guard rejected a covered source set plus a held-out row, plus a Giemsa row, and an incomplete source set. For normal integer global IDs, coverage + one draw + matching training draw + `hist_row<0` enforce source-only non-Giemsa inputs. Predictions of *source-draw* split-2 cells are legal under these batch-held-out protocols; they are not held-out-draw cells. |
| Real labels remain authoritative | `Y` and `Mk` are unchanged; `PM & Mk` is empty. Finite-input gradient test gave exactly zero pseudo gradient at real-label positions. |
| Ensemble alignment | All teacher `G` arrays must be exactly equal before averaging. This prevents row-order mixups among accepted files. Column-schema checking is absent: F4. |
| Pseudo weighting | Per-target pseudo error is normalized by that target's number of missing entries, then averaged across targets with entries, then multiplied by `pseudo_w`. Unit errors produced `0.30000001192092896` with weight .3. Empty target denominators are clamped and targets without entries are excluded; the wholly empty pseudo mask has a differentiable zero fallback. `wt` affects real regression only. This implements an additive, separately normalized MSE, not a combined real/pseudo denominator. |
| Train tuple | `RawSet` iterates insertion-ordered `LAB.values()`: first `Y,M,c3,c10,W`, then appended `PY,PM`. Actual `RawSet`/DataLoader returned arity 8 without pseudo and 10 with pseudo; `*pz` was length 0/2 with correct extra-array order. |
| Eval, DANN, AdaBN tuples | Eval consumes eight fixed values and ignores extras; DANN and AdaBN consume only the initial raw/phase/global-row triple. Full-script CPU fixture completed pseudo training/eval, pseudo+DANN, and pseudo+AdaBN. DANN and AdaBN fixture target rows included split 0 so their loops actually executed. The stub encoder has no BN layers, so this checks AdaBN iteration/unpacking, not recalibration numerics. |
| Old checkpoint compatibility | Actual saved `cfg`, `tr`, and `ev` matched the new `_cfg()` and reconstructed rows for **all four** N031 base/zoom2 × b01/b10 checkpoints, each at epoch 10. Default N040/N041/N042 options are omitted and dump flag is excluded. |
| New pseudo checkpoint compatibility | CPU fixture trained and resumed a new pseudo run successfully. Nondefault pseudo weight changes the fingerprint. Same-path content replacement is the gap in F3. |
| Default path | With new options at defaults, the CPU fixture produced model weights **bitwise equal** to the pre-diff HEAD script after training. No N040 tensor/device operation runs when pseudo/dump are disabled, apart from compatible starred unpacking. Actual CUDA/default numerical equivalence remains **[unverified]**. |

These checks validate row IDs and current teacher checkpoint configurations. A `P,G`-only artifact cannot establish that an arbitrary teacher was itself trained inductively; storing its provenance would improve that audit trail. Existing N031 checkpoint configs reference the expected draw-specific SSL initialization paths.

## Reproduction and execution boundary

No Apptainer or GPU job was launched; no writes were made under `/hnvme/workspace/v181bb10-bsccm`. `nvidia-smi` was unavailable and the isolated environment reports `torch.cuda.is_available() == False`. **Real CUDA execution, BF16/autocast behavior, GPU memory consumption, full-image teacher inference, and numerical N041 interactions are [unverified].** No additional N040-specific CUDA type/device fault was evident from inspection: `py` and `pm` are transferred to CUDA, float prediction math is explicit, and boolean reductions/indexing are valid PyTorch operations. Scalar CUDA mask tests introduce synchronization but are not a crash by themselves.

Scratch artifacts are in `/tmp/codex_n040/`: the exact reviewed script, pre-diff HEAD script, `audit.py`, `audit.out`, `data_checks.py`, and `data_checks.out`. The audit uses the unchanged reviewed source, actual `RawSet`, real PyTorch loss/optimizer/scheduler code, and isolated synthetic `.npz`/raw packs. It substitutes a tiny encoder, normalization, CUDA transfers/autocast, and W&B; consequently its control-flow and file-write reproductions do not claim a real GPU/model smoke test. The data checker reads the real target arrays and checkpoints on CPU.

Commands used (repository reads and all execution outputs remain in scratch):

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM
mkdir -p /tmp/codex_n040
cp scripts/train_lf.py /tmp/codex_n040/train_lf_review.py
git show HEAD:scripts/train_lf.py > /tmp/codex_n040/train_lf_base.py
git hash-object scripts/train_lf.py
# c492b6dad0dcff239531b85f896035d8dcf803fa
UV_CACHE_DIR=/tmp/codex_n040/uv-cache uv venv /tmp/codex_n040/venv
UV_CACHE_DIR=/tmp/codex_n040/uv-cache uv pip install \
  --python /tmp/codex_n040/venv/bin/python numpy torch \
  --index-url https://download.pytorch.org/whl/cpu
# numpy 2.5.4; torch 2.15.0+cpu
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  /tmp/codex_n040/venv/bin/python -u /tmp/codex_n040/audit.py \
  > /tmp/codex_n040/audit.out 2>&1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  /tmp/codex_n040/venv/bin/python -u /tmp/codex_n040/data_checks.py \
  > /tmp/codex_n040/data_checks.out 2>&1
```

Additional selected output:

```text
finished dump changed existing files [] new files ['train_preds.npz']
dump row order exact True P shape (32, 13)
RUN student_run dump False status OK ... events [..., 'resumed 2']
RUN student_dann dump False status OK
RUN student_adabn dump False status OK
RawSet pseudo False arity 8 pz 0 ... extra order valid True
RawSet pseudo True arity 10 pz 2 ... extra order valid True
default model weights bitwise equal to HEAD baseline True
n031_base_b01 ep 10 cfg_equal True tr_equal True ev_equal True
n031_base_b10 ep 10 cfg_equal True tr_equal True ev_equal True
n031_zoom2_b01 ep 10 cfg_equal True tr_equal True ev_equal True
n031_zoom2_b10 ep 10 cfg_equal True tr_equal True ev_equal True
```
