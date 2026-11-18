**Verdict: LAND WITH FIXES.** The residual objective and sampling implementation are consistent in the checks below. Fix the mean-checkpoint lineage validation and the legacy CFM/reg resume regression before landing. I found no evidence of new test-draw leakage in the actual checkpoints inspected. CUDA execution remains **[unverified]**.

Reviewed the supplied `n044.diff`, SHA256 `2976b98eda9d98621ec06bde067340101953bcb85fbef8791d3b2e5dbf82a5a2`, against its original blobs (`fm_common.py`: `0afb3dc`; `fm_train.py`: `6c83294`; `fm_eval.py`: `e9fd2ab`). Applying it under `/tmp/codex_n044/after/` reproduced all three repository working files byte for byte. Line references below refer to those reviewed files. Read N034, N035, `fm_prep.py`, `vs_common.py`, and the normalization/data access implementation. No repository or data files were modified.

1. **Required fix: validate the mean checkpoint's content lineage before computing residuals.** `scripts/fm_train.py:48–52` checks mode, blood draw, prep hash, and the **targets pathname**, but ignores the mean checkpoint's `targets_sha256`, `lf_norm_sha256`, and `stain_names`.

   The targets hash matters because this file defines `batch`, `split`, and `hist_row`. Replacing it at the same pathname can move formerly trained cells into the current test split, while the existing mean gate still accepts the checkpoint. I reproduced that case using a synthetic three-cell targets file and the actual gate expression: one old training row became a current test row, the hashes differed, and the gate accepted it. This demonstrates a missing safeguard, **not observed leakage in the real N034/N035 artifacts**.

   Ignoring LF normalization hashes also creates a train/eval inconsistency: training can compute both mu and sigma with a mean checkpoint from another normalization, whereas the recursive mean load in `fm_eval.py:36` rejects it. Ignoring stain-name ordering can silently use the wrong stain embedding during training; evaluation checks the names later.

   **Fix:** require `_c['targets_sha256'] == cfg['targets_sha256']`, the full required LF normalization manifest to match, and `_m['stain_names'] == prep.stain_names`, before residual-scale estimation or training. Keep the existing mode/draw/prep checks. Fail explicitly on missing required provenance. Prefer a shared validation routine so training and evaluation enforce the same mean-checkpoint contract. The outer mean-checkpoint SHA pins its bytes; it does not establish compatibility with the current data.

2. **Required fix: preserve old CFM/reg resume compatibility.** `scripts/fm_train.py:18,37–43` adds `mean_ckpt=''` to every new config, including plain CFM and regression. Pre-N044 checkpoints omit this key. The strict dictionary comparison therefore rejects an otherwise identical resume. Executing the actual new parser/config/check code against both real N035 manifests produced:

   ```text
   ACTUAL_OLD_RESUME reg resume config mismatch: ['mean_ckpt']
   ACTUAL_OLD_RESUME cfm resume config mismatch: ['mean_ckpt']
   ```

   **Fix:** normalize the absent legacy field to `''` for plain CFM/reg before comparison, or omit irrelevant mean fields consistently from both compared configs. Preserve strict RCFM mean-path/hash checks and rejection of all substantive configuration changes. Old checkpoint **evaluation** still works; this defect affects resuming training.

The requested checks gave the following results:

| Check | Result and evidence |
| --- | --- |
| Residual target and decode | PASS. Training uses `(y - mu) / sigma`; generation uses `mu + sigma * r_hat`. The actual RMS block matched a direct per-channel, per-pixel reduction on synthetic batches, including a smaller last batch. No extra `YS` or linear-count conversion is introduced. |
| Condition and CFG null | PASS. RCFM uses 24 LF channels + presence flag + 6 mu channels. Mixed dropout and probability-one dropout zero LF, flag, and mu together. Sampling's null condition has all 31 channels zero, retaining the stain token as training does. |
| Frozen mean and EMA | PASS. Training loads `_m['ema']` and freezes/evaluates the mean net. Evaluation recursively loads its EMA. On the real smoke artifact, the loaded mean state equaled the recorded EMA exactly, while raw and EMA weights differed. Backward probes left all mean gradients absent. |
| Dtype and AMP | PASS on CPU probes; CUDA **[unverified]**. Actual CondUNet residual loss/backward and guided sampling passed with patches 1/2 in float32 and CPU bfloat16 autocast. Mean outputs, residual arithmetic, integration, and decoded outputs were float32 and finite. CPU autocast is not a CUDA kernel test. |
| Sigma selection | PASS for direct row selection. Local RNG seed 3 selects at most 4,096 rows from `tr`, which is clean, same-draw split 0. No validation/test loader feeds the RMS accumulator; scale estimation uses no translation. The smaller smoke cohort correctly uses all 2,000 training rows. |
| Sigma resume | PASS. The actual resume branch restores the saved six float32 scales bitwise and does not invoke the scale loader. Scale is saved as Python floats and moved to the input/training device for arithmetic. |
| Mean hash on resume | PASS. `mean_ckpt_sha256` is computed before comparing configs. Executing that code accepted an unchanged mean file and rejected replacement at the same pathname with `resume config mismatch: ['mean_ckpt_sha256']`. Evaluation also checks the recorded hash. |
| Plain CFM/reg numerics | PASS in tested cases. Old/new condition construction and training/eval losses were bitwise equal with fixed RNG. Plain sampling was bitwise equal at CFG 0/1/2. Real N035 regression and CFM checkpoints produced bitwise-identical before/after outputs on two held-out 128-pixel cells; CFM used two Heun steps and CFG 2. Full aggregate evaluation was **[unverified]**. |
| Shuffled-LF control | PASS. The evaluator deranges normalized LF within each stain/cohort, and the wrapper recomputes mu from that shuffled LF before both residual conditioning and affine decoding. A recorded-call probe verified this; the original cell's mu is not retained. |

The shuffled control remains a valid test of dependence on the correct cell's LF. It preserves the generator's stain/cohort input distribution while breaking the LF–target pairing, including the mean contribution. It is still one sample per shuffled cell, so it does not provide an equal-K shuffled CRPS comparison. Also, `cfg=0` removes conditional residual velocity guidance but decoding still adds the LF-dependent mu; it should not be described as a fully unconditional image baseline.

Actual checkpoint/data provenance checks found:

```text
N035 reg_b0: draw=0, epoch=30, step=23940
n_train=102097, recorded=102097
sigma candidate sample: n=4096, splits=[0], draws=[0], hist_row max=-1

N034 reg_v2: draw=1, epoch=30, step=50610
n_train=215838, recorded=215838
sigma candidate sample: n=4096, splits=[0], draws=[1], hist_row max=-1

Both mean checkpoints:
REAL_MANIFEST_MATCH {"prep_sha256": true, "targets_sha256": true, "lf_norm_sha256": true}

N044 smoke: draw=1, max_train=2000, epoch=1, step=16
mean_ckpt=/hnvme/workspace/v181bb10-bsccm/runs/n034/reg_v2/ckpt.pt
SMOKE_MEAN_HASH_MATCH True
res_scale=[0.2447445542, 0.2500145137, 0.2665624619,
           0.2485634387, 0.2360798120, 0.2635071874]
```

Thus the available means declare the expected training draws, their recorded training counts match the current clean split-0 cohorts, and their recorded input manifests match current files. Together with the reviewed trainer's row-selection code, this supports the intended lineage. Historical training reads cannot be proven from checkpoint metadata alone **[unverified]**. I did not recompute the entire real 2,000/4,096-cell RMS on CPU **[unverified]**; the existing smoke scales are finite and positive, and the scale algorithm was executed on synthetic batches.

There is an **inherited preprocessing qualification** to “no test-draw information enters training.” The current `fm_prep.npz` reports `a_c_batch=1`. For draw-0 training, that shared target transform therefore uses a noise constant estimated from draw-1 split-0 fluorescence. `fm_prep.py` also fits per-slide gains on split 0 and selects them using split-1 fluorescence; N035 explicitly discloses its draw-1 split-1 primary cohort as fluorescence-assisted. Per-slide test-draw gain fitting affects that draw's targets, and the shared `a_c` enters draw-0 training targets and hence sigma. N044 adds no direct test-draw residual-scale fitting, but a literal claim that the entire pipeline uses zero test-draw fluorescence would be false. Keep N034/N035's disclosure; changing this would require a separately specified preprocessing protocol.

Reproduction used a uv environment under the permitted scratch directory, with no apptainer:

```bash
mkdir -p /tmp/codex_n044
UV_CACHE_DIR=/tmp/codex_n044/uv-cache uv venv --python 3.11 /tmp/codex_n044/venv
UV_CACHE_DIR=/tmp/codex_n044/uv-cache uv pip install \
  --python /tmp/codex_n044/venv/bin/python torch numpy \
  --index-url https://download.pytorch.org/whl/cpu
UV_CACHE_DIR=/tmp/codex_n044/uv-cache uv pip install \
  --python /tmp/codex_n044/venv/bin/python scipy --index-url https://pypi.org/simple

PYTHONDONTWRITEBYTECODE=1 /tmp/codex_n044/venv/bin/python -u /tmp/codex_n044/probe.py
PYTHONDONTWRITEBYTECODE=1 /tmp/codex_n044/venv/bin/python -u /tmp/codex_n044/real_eval.py
PYTHONDONTWRITEBYTECODE=1 /tmp/codex_n044/venv/bin/python -u /tmp/codex_n044/lineage_resume.py
```

The retained harnesses and complete outputs are `/tmp/codex_n044/{probe,real_eval,lineage_resume}.{py,out}`. They execute the before/after code snapshots and AST-extracted production functions/branches with CPU globals, without invoking the CUDA-only training CLI. The real-evaluation harness uses the actual checkpoints and normalized LF for global rows 52946 and 52947 (draw 0, split 2). The lineage/resume fixtures only write under `/tmp/codex_n044/`. Selected output:

```text
ENV {"torch": "2.15.0+cpu", "numpy": "2.4.6", "cuda": false, "threads": 4}
COND plain before/after bitwise equal; residual channels=31; LF/flag/mu jointly drop (mixed and all-null)
SAMPLE plain before/after bitwise equal at cfg=0,1,2 (3 Heun steps)
LOSS plain cfm/reg before/after bitwise equal in training/eval mode
RESIDUAL wrapper exact affine decode; cfg null has all 31 zero channels; shuffled LF recomputes mean
SCALE actual block: training-only rows; per-channel RMS matches direct reduction; resume restores float32 bitwise without recompute
UNET patch=1 cpu_bf16=False finite residual loss/backward/cfg sampling; mean frozen; output=torch.float32
UNET patch=1 cpu_bf16=True finite residual loss/backward/cfg sampling; mean frozen; output=torch.float32
UNET patch=2 cpu_bf16=False finite residual loss/backward/cfg sampling; mean frozen; output=torch.float32
UNET patch=2 cpu_bf16=True finite residual loss/backward/cfg sampling; mean frozen; output=torch.float32
MEAN_GATE_ACCEPTS_INCOMPATIBLE targets_sha256
MEAN_GATE_ACCEPTS_INCOMPATIBLE lf_norm_sha256
ALL_PROBES_PASSED

REAL_OLD_CHECKPOINT reg before_after_bitwise_equal=True max_abs_diff= 0.0
REAL_OLD_CHECKPOINT cfm before_after_bitwise_equal=True max_abs_diff= 0.0
REAL_RCFM_CHECKPOINT finite=True dtype= torch.float32 shape= (2, 6, 128, 128) affine_decode_exact=True
REAL_MEAN weights_equal_ema=True raw_differs_from_ema=True
MEAN_PIN unchanged_resume_accepted=True
MEAN_PIN replaced_same_path_rejected= resume config mismatch: ['mean_ckpt_sha256']
ALL_REAL_CHECKS_PASSED

ACTUAL_OLD_RESUME reg resume config mismatch: ['mean_ckpt']
ACTUAL_OLD_RESUME cfm resume config mismatch: ['mean_ckpt']
SYNTHETIC_LINEAGE same_targets_path=True hashes_differ=True mean_gate_rejects=False old_train_overlaps_new_test= [0]
LINEAGE_RESUME_CHECKS_PASSED
```

No new device/dtype mismatch was apparent in the reviewed CUDA paths: mean inference shares the existing CUDA bfloat16 autocast helper, returns float32, and both training/evaluation explicitly move sigma to the working device. Actual CUDA forward/backward, optimizer resume, memory capacity at the configured batch sizes, full sampler settings, and CLI-level preemption/resume are **[unverified]**. The existing smoke checkpoint records 16 steps but is not a GPU run performed by this review.
