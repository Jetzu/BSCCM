# N056 handoff

Code and CPU validation only; Claude must run the GPU jobs. Read the sealed-design
[report.md](report.md) before inference. No clinical result is asserted here.
All default outputs stay in this delivery folder; source repo and data are read-only.

`teachers.json` records the exact saved N032/N029 configurations. All three T
checkpoints exist. Their marker teachers differ; their v6c autofl teacher is shared.
H learns a separate A3 hub for each corresponding pair. The N054 A3 hub objective
and architecture are copied into `n056_hub_core.py`; the anchor image dataset and
augmentation are verbatim in `n056_legacy.py`; `n056_contrasts.py` contains the exact
10 PBC and 15 MLL23 definitions. Preserve these sibling files when copying code.
`source_hashes.json` records the original script hashes. No GPU submission was made.

## Exact GPU commands (Claude, outside the writer sandbox)

The house `preempt-run.sh` wrapper requires its existing installation in `~/.claude`.
The sbatch file uses the existing BSCCM Apptainer image and `python3 -I -B`, no
`--mem`, no `srun`. It saves optimizer, scheduler, RNG and stage cursor on SIGTERM
and exits 143. Inference saves each batch atomically. Resubmissions resume without
restarting completed stages. Run parameters are environment variables for replay.

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex
export N056_CODEDIR="$PWD"
export N056_RUNROOT="$PWD/gpu_runs"
mkdir -p "$N056_RUNROOT"
for teacher in 0 1 2; do
  N056_UNIT=train N056_TEACHER="$teacher" sbatch n056_unit.sbatch
done
```

Wait for all three **preempt chains**, rather than just their initial job IDs:
check `gpu_runs/t{0,1,2}/H_train/done.json` (the house `wait-job.sh` can follow chains).
Do not use `afterok` on the initial H jobs: a preempted initial link can exit
successfully before its replacement finishes. Then:

```bash
for teacher in 0 1 2; do
  for arm in T H; do
    N056_UNIT=eval N056_TEACHER="$teacher" N056_ARM="$arm" sbatch n056_unit.sbatch
  done
done
```

Each evaluation first predicts source split-0 training cells, fixes literal and
proxy RQ3 thresholds, and writes `seal.json` containing the preregistration,
threshold, source prediction, model and code hashes. Only then does it predict
clinical fit cells, fit OT, and predict held-out clinical cells. Existing N032
clinical predictions are never consumed. Keep `report.md` and all Python files
unchanged after starting. Put any numerical report in `gpu_runs/results.md`.

After six `gpu_runs/t*/{T,H}/eval.done.json` files exist:

```bash
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$PWD/n056_eval.py" --stage aggregate --out "$N056_RUNROOT" --device cpu
```

To inspect/source-calibrate a single arm before allowing clinical prediction:

```bash
apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$PWD/n056_eval.py" --stage prepare --teacher 0 --arm T \
  --out "$N056_RUNROOT/t0/T"
```

Use `--stage fit` for clinical fit inference/OT alone; `--stage evaluate` requires
an existing seal. All production hyperparameters are locked in the report. The
training flags `--tiny --device cpu --steps 4 --hub-steps 4 --bs 2 --hub-bs 8`
select engineering-only behavior: randomly initialized tiny ViT, at most 96 LF
hub rows / 8 GS training cells and a 96-row normalization subset. Such artifacts
are explicitly refused by the clinical predictor. Production H uses the actual
DinoBloom ViT-B and original source normalization; student batches are stateless,
seeded samples from the same source/anchor image universes, enabling exact resume.
This changes batch ordering relative to the original legacy DataLoader.

## Outputs and interpretation

Per arm: `source.npz`, `thresholds.json`, `calibration_rows.npz`, proxy classifier,
`seal.json`, clinical fit/eval caches with archive-qualified image IDs and eight
view embeddings, frozen OT states, `contrasts.csv`, `fractions.csv`, `sanity.csv`.
OT fitting is capped at 2048 cells per distribution (three such fits for PBC CCOT).
Unconditional maps see no labels. CCOT consumes only PBC A anchor labels; frozen
student C routes all evaluation cells. No MLL23 class-conditional arm is fitted.
The OT decoder uses the fixed fitted column potential; it never solves a coupling
on the evaluation batch. Eight-view decoded predictions are averaged; fit/source
OT supports use normalized eight-view average embeddings. This extension is fixed
in advance, not selected by clinical metrics.

Aggregate: `contrast_means.csv`, `macro_means.csv`, `paired_contrasts.csv`,
`paired_macros.csv`, `fraction_means.csv`, `sanity_all.csv`, `sanity_means.csv`, `results.json`, plus
unaggregated tables. Each mean/SD uses precisely three teacher observations.
Conditional OT gets PBC summaries only. Missing contrasts invalidate the complete
macro score. `results.json` identifies the largest all-25 mean; inspect fine and
robust/failing strata before interpreting it. Within-slide autofl uses source
GS split-2 truth from v6c; it is never a fitting or model-selection signal.

RQ3 **cannot have the literal requested measured-class calibration**: expert
BSCCM gates cover pooled-marker cells, with zero observed individual-marker labels
on those cells. Primary fraction/MAE fields therefore stay NA. The explicitly
named proxy analysis uses a source-only LF class predictor, N008 GS pseudo
classes, source-only fluorescence mixture positivity, and source quantile matching.
It is an exploratory morphology/assay compatibility result, not a measured
single-cell clinical marker-positivity validation. Only four fixed reference
class-marker pairs receive absolute errors. Full definitions/sources are in the
preregistration. No expectation is used to fit source positive rates.

## CPU checks and resource budget

The existing CPU environment was read-only reused, without package installation:

```bash
/tmp/codex_n054/venv/bin/python -I -B n056_smoke.py \
  --out /tmp/codex_n056/smoke_fresh > cpu_smoke.log 2>&1
bash -n n056_unit.sbatch
```

Use a new scratch subdirectory per invocation. Checks cover AST equivalence,
real archive fit/eval disjointness (filenames only), OT analytic geometry and
unseen-point extension, conditional routing, missing RQ3 support and endpoints,
all 25 synthetic AUROCs, three-teacher pairing, tiny real source/anchor-A training,
real SIGTERM exit 143 and bitwise resume, and actual t0 T checkpoint eight-view
inference on two BSCCM source training images. Clinical test pixels/predictions
are not used in smoke. `cpu_smoke.log` contains the run evidence.

Planning estimate on one H100: three H student jobs ~0.8–1.0 h each including
hub fitting; six shared-inference evaluations ~0.25–0.4 h each; post-hoc OT/head
application ~0.2–0.4 h total: **4.1–5.8 GPU-h**, target <=6. This is an estimate,
not a measured timing. Eight-view inference dominates; actual archive counts
appear in the CPU log. Caches share image inference across raw/OT/CCOT and resume
by image batch. Expect roughly 5–10 GB GPU memory at default training batches,
CPU RAM ~4–8 GB and several GB of embedding caches. If slower hardware exceeds
the estimate, retain the protocol and report actual GPU-hours; do not trim or
select clinical cells based on results. No replacement T training is needed:
all three required checkpoints were verified present. If one disappears, restore
it or replay its exact saved `teachers.json` args with the original
`scripts/train_gs_bind.py` (that original script lacks checkpoint/resume support).
