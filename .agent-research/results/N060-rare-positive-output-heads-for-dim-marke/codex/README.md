Deliverable for integration into the read-only BSCCM repository. Place `train_lf.py` and `bsccm_heads.py` together in `scripts/`; `PATCH.diff` includes both files. No repository files were modified here. Run GPU commands in the existing BSCCM GPU environment; CPU dependencies below are only for verification.

```bash
# Reviewer/integrator only; this handback did not apply the patch.
cd /home/hpc/v181bb/v181bb10/BSCCM
patch --dry-run -p1 < .agent-research/results/N060-rare-positive-output-heads-for-dim-marke/codex/PATCH.diff
patch -p1 < .agent-research/results/N060-rare-positive-output-heads-for-dim-marke/codex/PATCH.diff
```

The 24 recommended GPU runs (6 arms × 2 seeds × 2 directions):

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM
bash .agent-research/results/N060-rare-positive-output-heads-for-dim-marke/codex/run_arms.sh \
  /hnvme/workspace/v181bb10-bsccm/runs
```

Alternatively run the delivered copy directly, without integration:

```bash
export PYTHONPATH=/home/hpc/v181bb/v181bb10/BSCCM${PYTHONPATH:+:$PYTHONPATH}
export TRAINER=/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N060-rare-positive-output-heads-for-dim-marke/codex/train_lf.py
bash "$(dirname "$TRAINER")/run_arms.sh" /hnvme/workspace/v181bb10-bsccm/runs
```

Exact baseline command; change `b1to0/ex0` to `b0to1/ex1` together for the reverse direction. Use seed 0 and 1. Add one arm's flags below to this command.

```bash
python scripts/train_lf.py --out /hnvme/workspace/v181bb10-bsccm/runs/n060_control_b1to0_s0 \
  --protocol b1to0 --arch resnet50 --channels all --phase 1 \
  --init-ssl /hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/encoder_final.pt \
  --epochs 10 --bs 256 --lr 5e-4 --w-cls .5 --targets targets_v5cx_v10.npz --seed 0
```

| Arm | Flags |
|---|---|
| N060 histogram | `--head hlgauss --hl-mse .1` |
| N060 gate/amplitudes | `--head twopart --gate-w .5 --pos-q .98` |
| N061 mixed effects | `--fov-re .01 --re-shrink 1 --w-autofl 0` |
| N061 differences | `--pair-loss .5 --fov-batch 64 --w-autofl 0` |
| N061 auxiliary ablation | `--w-autofl 0` |
| N062 factors | `--panel-head --re-shrink 1` |

CPU verification, from this delivery directory (all generated data/checkpoints stay under `/tmp/codex_n060`). The suite refuses to overwrite configuration-mismatched checkpoints; use a fresh `/tmp/codex_n060/runs` if changing code or test arguments.

```bash
export UV_CACHE_DIR=/tmp/codex_n060/uvcache
uv venv --python 3.11 /tmp/codex_n060/venv
uv pip install --python /tmp/codex_n060/venv/bin/python \
  torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /tmp/codex_n060/venv/bin/python timm==1.0.22 scikit-learn
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 WANDB_MODE=disabled
/tmp/codex_n060/venv/bin/python smoke_cpu.py --prepare /tmp/codex_n060/fixture
/tmp/codex_n060/venv/bin/python test_heads.py > cpu_smoke.log 2>&1
/tmp/codex_n060/venv/bin/python smoke_suite.py >> cpu_smoke.log 2>&1
```

The smoke fixture copies 528 real LF/phase cells and their unmodified labels/metadata; original global IDs are recorded in `original_rows.npy`. It selects source split-0 and evaluation split-2 FOVs on both draws. Each run uses `--arch resnet18 --device cpu --epochs 1 --bs 8 --max-train 64 --workers 0`, the ordinary training loop, and the unchanged `evaluate()`. These tiny subsets give NaN marker summary scores because existing minimum-count rules exclude them; finite training losses and all 13 predictions are checked. They are execution tests, not GOAL measurements.

Implementation contracts:

- Defaults retain the original MSE branch, parameter names/order, and configuration. Every new default is in `_NEW_DEFAULTS`, including CUDA placement. `evaluate()` is byte-identical. `P` always has 13 columns in the existing transformed label units. Adapters are called only in the training loss.
- Statistics use masked `Y[tr]` after `--max-train`. Histogram edges, gate thresholds/class weights, and source adapter IDs are checkpoint buffers; `head_setup.json` records calibration and a fingerprint of source rows/labels/masks/IDs. Resume checks this configuration. Unsupported histogram targets have placeholder support and zero loss; gates with only one observed class have no BCE.
- HL-Gauss uses 64 equal bins, q.001/q.999 support plus 10% of the span on each side, Gaussian bin integrals with sigma .75 bin widths, and normalized truncated mass. Out-of-support labels are clipped for CE only; expectation MSE uses the original labels. Fixed support bounds predictions, a particular risk for CD123's extreme tail.
- Two-part BCE weights each observed class to half its target's training count; the gate label is strictly `y > threshold`. It is an operational upper-tail class, not biological positivity or a calibrated prevalence estimate. BCE is added per target with the same present-target denominator as MSE. No extra amplitude supervision or prior correction is added.
- Mixed effects: `a=1+delta`, loss readout `a*p+c+b`, shrinkage `fov_re * (mean(b²) + re_shrink*(mean(delta²)+mean(c²)))`. Means are over source tables then weighted targets. Classes are detached predictions of the existing cls3 head. No true classes or external teachers choose strata. `--fov-re` is the shrinkage coefficient and enables the tables.
- Panel: single slopes are softplus-positive; pooled W starts exactly at the panel incidence matrix and learns positive weights on its stated edges. Off-panel edges stay zero. Each pooled assay has its own monotone affine+tanh decoder. Training-slide pooled affine adapters have `re_shrink*(mean(delta²)+mean(c²))` penalty. No cell-lineage or exclusivity constraints.
- Autofl weighting applies to the whole autofl regression target, including unstained source cells, and renormalizes the remaining target weights. Zero removes its supervised gradient from MSE/CE, pair loss, and adapter penalties.
- Pair loss uses every unordered, distinct-cell, same-target/same-FOV batch pair, per-target averaging, then target weighting. It acts on shared predictions. `--fov-batch k` takes `bs/k` cells from each of k distinct source FOVs. FOV draws are count-proportional; small FOVs use replacement, while pair loss excludes duplicate rows. One epoch has `floor(N/bs)` batches. This changes cell sampling and merits a sampler-matched control if the pair arm wins. Sampler epochs are deterministic on resume; other RNG behavior follows the legacy harness.
- Active N060--N062 options require b1to0/b0to1, no Giemsa, DANN, or AdaBN. Panel requires `--head mse`; HL-Gauss plus mixed effects is rejected since the categorical readout has no defined affine-adapter objective here. HL-Gauss can combine with pair loss/autofl weighting; two-part can combine with mixed effects. The prescribed six arms do not use unsupported combinations.
