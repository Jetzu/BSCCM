N039 stage 1 delivers a rejected v8 candidate: QC FAIL. Do not proceed to training/scoring.

Files:

- `scripts/make_targets_v8_raw.py`: new raw-label derivation, to land under repo `scripts/`.
- `make_targets_v3.diff`: minimal v3 overlay/output-directory patch; default raw/path unchanged.
- `pipeline.diff`: complete patch for v3/v4/v5, including separate output directory and final `targets_v8c[x].npz` naming. Apply this combined diff once; the individual diffs duplicate its hunks.
- `scripts/qc_targets_v8.py`: fixed QC, imports the raw script from the same directory.
- `scripts/run_stage1_v8.py`: executes patched copies in scratch; does not edit the repo.
- `qc_protocol.md`, `qc_v8.csv`, `qc_v8.md`, `report.md`: protocol, all results, decision and caveats.
- `unmixing_fidelity.csv`, `unmixing_maps.json`, `unmixing_validation.csv`: split-0 map fidelity, coefficients/floors and independent split-2 map validation.
- `validation.json`, `build_manifest.json`, build logs and position-detrend diagnostics: integration/provenance evidence.

Reproduce from this bundle (all commands write only to scratch and the chosen out-dir):

```bash
mkdir -p /tmp/codex_N039/tmp /tmp/codex_N039/cache
export TMPDIR=/tmp/codex_N039/tmp
export UV_CACHE_DIR=/tmp/codex_N039/cache
export PYTHONPYCACHEPREFIX=/tmp/codex_N039/pycache
uv venv /tmp/codex_N039/venv
uv pip install --python /tmp/codex_N039/venv/bin/python -r requirements.txt
/tmp/codex_N039/venv/bin/python scripts/run_stage1_v8.py --out "$PWD"
```

No Apptainer, GPU, torch, LF features/images or model prediction files are needed. The existing metadata/target NPZ and parquet files are only read.
The runner leaves the rejected raw data and `targets_v8cx.npz` under `/tmp/codex_N039/targets/`, writes reports/audit files to `--out`, and never advances to stage 2. A QC FAIL is a completed measurement, so the runner prints FAIL but exits successfully.

After integration, the equivalent pipeline invocation (in a separately writable target directory) is:

```bash
export BSCCM_CLEAN=1 BSCCM_DETREND=xfit BSCCM_RAW=v8
export BSCCM_TARGET_DIR=/tmp/codex_N039/targets
python scripts/make_targets_v8_raw.py --out "$BSCCM_TARGET_DIR"
python scripts/make_targets_v3.py
python scripts/make_targets_v4.py "$BSCCM_TARGET_DIR"
python scripts/make_targets_v5.py
```

This recipe produces a candidate for QC, not approval to use the failed candidate. `BSCCM_V8_RAW` optionally overrides the raw parquet location; the default is `$BSCCM_TARGET_DIR/raw_v8.parquet`. Group/autofl columns retain their released raw inputs and the existing clean xfit processing. No separate histology regression is fitted by the delivered v8 pipeline; it reuses the frozen `targets_v5cx.npz` pair-suspect annotation.

QC(d) cannot be assessed in a verified lymphocyte stratum with these data: the authors' gates cover only pooled slides. The reported fluorescence-only full-model proxy is diagnostic. The `--proxy-as-primary` option exists solely for an explicit user decision; it was not used for this measurement, and would not repair the failures of (a), (b) or (c).
