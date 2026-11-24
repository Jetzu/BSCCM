# N081 fix round

Deliverables are staged here only. `train_lf.py` starts from the previously delivered `../codex/train_lf.py`; `train_lf.diff` applies to the live repository's `scripts/train_lf.py` (SHA256 `4cf89b093fc7dd44aa13ce6a2a82f743eeec2bea03c0b536b298c755362ce8c8`). `pack_sbiad452.py` is a patched copy of the repository packer, with its own `pack_sbiad452.diff`. The repository was read only. All execution was CPU; scratch is `/tmp/n081_fix`. No training job was submitted.

## Fixes and root causes

The original IFC forward contaminated all 53 shared ResNet BN layers' running statistics. Each BN now registers `ifc_running_mean`, `ifc_running_var`, and `ifc_num_batches_tracked`, initially cloned from its SSL-loaded BSCCM buffers. The ordinary `running_*` buffers always belong to BSCCM outside an IFC forward. IFC training and evaluation swap buffer references in a `try/finally` context, preserving autograd's saved tensors and restoring BSCCM on exceptions. BN affine parameters remain shared. Both domains serialize in `model.state_dict()`; strict resume rejects missing IFC buffers and the changed setup version rejects old mixed-BN checkpoints. `--init-full` requires the complete matching model state.

Donors are extracted from run names with `^sample\d+`. Thus `sample1_early` and `sample1_late` both belong to `sample1`. The donor flags accept names such as `sample2`; run names are rejected. The test donor is removed from the requested training donors, and dataset reads assert donor membership before touching each input/label row. Effective train/test run names are recorded in checkpoint configuration.

For each shared marker (CD14/CD19/CD3/CD45/CD56), IFC affine initialization uses the actual masked, finite training rows:

```
scale0  = std(IFC training-donor labels) / std(BSCCM selected training-row labels)
offset0 = mean(IFC training-donor labels) - scale0 * mean(BSCCM selected training-row labels)
```

SD uses population normalization (`ddof=0`). Row selection includes the chosen draw, training-panel filter and `--max-train`. Missing/constant shared-marker training support fails explicitly. All means, SDs, counts and initial affine values are printed and pinned in checkpoint configuration. The baseline AdamW decays BN and bias parameters; this policy is retained. The new IFC affine alone has zero weight decay. The IFC stem uses the encoder's learning-rate multiplier.

Separate torch generators handle IFC initialization, augmentation, training shuffle and evaluation loading. Constructors run inside an isolated RNG context. D4 transforms operate on eight groups, followed by batched gather-based circular translations; channels remain together. `--adabn` with IFC fails early.

Gradients traverse each source graph once and accumulate before one joint norm-2 clip. Epoch logs contain `grad_norm_bsccm`, `grad_norm_ifc_weighted` (includes `ext_w`), `grad_norm_joint`, `clip_count`, and `clip_fraction`, alongside IFC loss/evaluation metrics. These are preclip mean norms. Joint clipping can scale BSCCM gradients when active. Two source activation graphs remain live until backward; GPU memory, throughput and GPU AMP were not measured. Budget roughly twice the source-only compute pending measurement. IFC loaders add two worker pools; lower `--workers` when needed.

## Packer and use

The copied packer adds `--stat-donors sample0,sample1` (default). Both fluorescence cofactors and channel-scale fits use only rows from these donors, including both early/late runs. Channel-scale sampling selects at most 5,000 eligible cells with a deterministic NumPy generator. `pack.json` records donor names, run names, eligible count and channel-fit count. Per-run label z-scoring remains the existing label definition; held-out label statistics define only held-out evaluation labels.

Rebuild old all-donor packs with this copy. Set `--stat-donors` to the effective training donors whenever the held-out donor changes. The harness accepts existing packs and does not infer or enforce their preprocessing provenance; `pack.json` supplies that audit record.

Use `tests/run_staged.py` to run the staged harness with imports from the read-only repository. Keep the existing v10 recipe and append:

```
--ext-ifc /path/to/rebuilt/pack --ext-w 0.5
--ext-train-donors sample0,sample1,sample2 --ext-test-donor sample2
```

Enabled IFC requires SSL-initialized `resnet50`, all channels, phase, and the ordinary linear MSE head. Omit IFC flags for the matched control. The BSCCM `prep`, `fwd`, and `evaluate` functions remain AST-identical to the live baseline. Flag-off configuration, outputs, gradients, checkpoint bytes and full-loop predictions retain exact equality.

## CPU verification

From this folder, using the existing CPU environment:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/tmp/n074_codex/venv/bin/python -I -B tests/test_n074_regression.py > tests/n074_regression.log 2>&1
/tmp/n074_codex/venv/bin/python -I -B tests/test_ifc.py > tests/ifc_cpu.log 2>&1
/tmp/n074_codex/venv/bin/python -I -B tests/test_fix.py > tests/fix_cpu.log 2>&1
/tmp/n074_codex/venv/bin/python -I -B tests/check_artifacts.py > tests/final_checks.log 2>&1
```

All 22 tests pass: seven reused N074 integration tests, six inherited IFC tests, nine fix-round tests. N074 path adaptations are in `tests/n074_support.py`; no numerical assertion was relaxed. Fixtures use 64 real BSCCM LF/phase/fluorescence cells with synthetic scalar labels/calibration, a 24-cell synthetic IFC pack, and synthetic SSL weights; inherited gradient checks also load the real N031 SSL checkpoint. New packer tests execute its full `main()` with tiny mocked Zarr/FCS sources, real fitting and output writes, and poison held-out inputs/labels to verify fit isolation.

BN isolation is tested across three optimizer steps with IFC active at `ext_w=0`, inactive clipping and RNG-identical controls: all 53 buffer triples, BSCCM sampled rows, augmented inputs and evaluation predictions match exactly. A second three-step test uses nonzero IFC gradients at weight 0.5 and dual zoom, replaying the joint model's weight trajectory into an IFC-skipped control: every BSCCM buffer still matches exactly. Weight replay isolates BN contamination because joint training intentionally changes shared weights. IFC evaluation, exception restoration, buffer serialization, missing-buffer resume rejection, real early/late donor names, training-only masked affine fits, encoder LR, source gradient norms and active clip logging are also covered. Both training directions complete two-step full-loop runs and strict resumes.

The real complete v10 training draws were loaded separately to verify affine-statistic selection. Reproduced BSCCM label SDs:

| Marker | Draw 1 | Draw 0 |
| --- | ---: | ---: |
| CD14 | 0.97931 | 1.00360 |
| CD19 | 0.36208 | 0.52086 |
| CD3 | 1.00487 | 1.01469 |
| CD45 | 0.98097 | 1.00083 |
| CD56 | 0.91698 | 0.83757 |

Full initialization records are in `tests/fix_cpu.log`; its IFC numerator comes from the synthetic fixture, not a production pack. `tests/provenance.json` records versions, source/artifact hashes and test results. Small BSCCM evaluation fixtures trigger inherited NaN aggregate correlations because they lack sufficient per-marker counts; training losses and IFC predictions remain finite. Transfer accuracy and GPU behavior remain untested. The baseline does not restore its complete RNG state for bitwise interrupted continuation; strict checkpoint/state/configuration loading is verified.
