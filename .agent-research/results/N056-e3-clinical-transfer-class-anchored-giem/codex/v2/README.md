# N056 v2 handoff

Code + CPU validation; GPU execution is pending and belongs to Claude under the
original writer brief. v1 is untouched. Read [report.md](report.md) before runs;
keep it, Python files and [pbc_excluded.json](pbc_excluded.json) immutable after
launch. Write clinical results separately under the GPU run root.

The grid is six H students (two seeds × three teachers), three H_matched students
(seed0 × three teachers), and seven existing T checkpoints. H_matched uses a
200-update instrumented T-recipe replay on the same augmented batches. The
anchor gradient share matches at each calibration update; freeze the mean chosen
weight afterward. Gradient logs contain raw term norms, weights and both shares.
This is adaptive early calibration, rather than a claim of equal anchor loss scales.
The sensitivity arm has one seed and cannot establish a superiority claim.

Primary macros cover four PBC held-out-biology contrasts and 12 MLL23 fine
contrasts. Every H-vs-T claim uses the preregistered student-seed noise floor;
all25 is secondary. No across-teacher SD ranking is produced. Alignment exports
mean shift, CORAL, kNN10, OT tau .91, OT tau .30, and PBC CCOT, all fitted and
applied to normalized eight-view means from the SAME inference cache. Raw scores
keep legacy eight-view prediction averaging. MLL23 fit sampling shares the eval
class composition and is labelled transductive.

## Exact GPU commands (outside writer sandbox)

The house preempt wrapper and the existing BSCCM container are required. No new
packages are needed: SQLite is Python standard library. Sbatch has no `--mem` or
`srun`; parameters are exported for successor replay. Logs and artifacts default
to `/hnvme/workspace/v181bb10-bsccm/runs/n056v2_*`, not /home.

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/v2
export N056_CODEDIR="$PWD"
export N056_RUNROOT=/hnvme/workspace/v181bb10-bsccm/runs/n056v2_main
mkdir -p "$N056_RUNROOT"
for teacher in 0 1 2; do
  for seed in 0 1; do
    N056_UNIT=train N056_TEACHER="$teacher" N056_SEED="$seed" N056_ARM=H sbatch n056_unit.sbatch
  done
  N056_UNIT=train N056_TEACHER="$teacher" N056_SEED=0 N056_ARM=H_matched sbatch n056_unit.sbatch
done
```

Wait for all **nine completed preempt chains**: each must have
`t{teacher}/{H_s0,H_s1,H_matched_s0}_train/done.json`. Do not use `afterok` against
only the first chain job. A completed successor validates its contract and exits
immediately. Then launch all 16 evaluations:

```bash
for teacher in 0 1 2; do
  for seed in 0 1; do
    N056_UNIT=eval N056_TEACHER="$teacher" N056_SEED="$seed" N056_ARM=T sbatch n056_unit.sbatch
    N056_UNIT=eval N056_TEACHER="$teacher" N056_SEED="$seed" N056_ARM=H sbatch n056_unit.sbatch
  done
  N056_UNIT=eval N056_TEACHER="$teacher" N056_SEED=0 N056_ARM=H_matched sbatch n056_unit.sbatch
done
N056_UNIT=eval N056_TEACHER=0 N056_SEED=2 N056_ARM=T sbatch n056_unit.sbatch
```

After all sixteen `t*/{T,H,H_matched}_s*/eval.done.json` markers exist:

```bash
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$N056_CODEDIR/n056_eval.py" --stage aggregate --device cpu --out "$N056_RUNROOT"
```

Source calibration can be run alone with `--stage prepare`; clinical fit inference
and alignments with `--stage fit`; scoring with `--stage evaluate`. Example:

```bash
apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  "$N056_CODEDIR/n056_eval.py" --stage prepare --teacher 0 --seed 0 --arm T \
  --out "$N056_RUNROOT/t0/T_s0"
```

Production is locked: hub lr1e-3/batch256/2500 updates/K019 mask, student
lr1e-4/batch128/3000 updates, DataLoader12 workers. The `--tiny --device cpu`
path is engineering-only and cannot generate clinical results.

## Files, exclusions and resume

Each inference cache is a SINGLE `.cache.sqlite` file with transactional batch
rows and model/ID/role contracts. No `.parts` directory or second concatenated
copy is written. Completed batches are reused after SIGTERM. Model checkpoints
atomically save stage/optimizer/scheduler/RNG plus gradient calibration state;
Python exits143 on SIGTERM. A crash can leave one temporary checkpoint or SQLite
rollback journal; the latter is recovered automatically by SQLite.

The duplicate audit reproduces seven exact pairs plus an additional near pair
(MO_415531/MO_231316 cosine .995795), excluding 16 filenames total. PBC retains
8538 images in each original half. The complete CRC32, byte size, thumbnail cosine
and excluded filenames are in `pbc_excluded.json`; both evaluation halves and H
anchor loading filter these names AFTER the original alternating split.
Historical T anchor exposure cannot be undone; removed images never enter v2
scoring. To reproduce this source-hygiene audit before launching anything:

```bash
/tmp/codex_n054/venv/bin/python -I -B n056_duplicates.py \
  --out /tmp/codex_n056v2/pbc_excluded_reproduced.json
```

Per seeded evaluation: source/sanity/clinical caches, calibration rows, frozen
thresholds and seal, alignment states, predictions, contrasts/fractions/sanity
CSVs. The seal binds source predictions, report, code, exclusions, model and
teacher pair before clinical prediction. Aggregate exports seed observations,
seed-averaged teacher means, paired deltas, maximum within-teacher seed ranges,
and registered decisions. Missing support stays NA, missing runs fail closed.
RQ3 reports `transfer_drift` and `bsccm_calibration` separately. Aligned fractions
say `not interpretable (shrinkage toward source)`. The literal measured-class
analysis remains unavailable; proxy fractions are exploratory.

## CPU checks and GPU budget

```bash
/tmp/codex_n054/venv/bin/python -I -B n056_smoke.py \
  --out /tmp/codex_n056v2/smoke_final > cpu_smoke.log 2>&1
bash -n n056_unit.sbatch
```

Use a fresh scratch directory. [cpu_smoke.log](cpu_smoke.log) and
[validation.md](validation.md) contain engineering evidence. Checks include real
source/anchor training, 12-worker SIGTERM resume with bitwise equality, T-recipe
matched gradient shares, one-file cache interruption/resume, unequal seed counts
and noise-floor decisions, both taus/three controls, and actual full T inference
on two SOURCE training images. No clinical performance is measured by smoke.

Target: **<=6 H100 GPU-hours**. A conditional planning budget is 3.15 h for nine
student+hub runs at 21 min each, 2.24 h for sixteen eight-view evaluations at
8.4 min each, plus ~.3 h for calibration replay/postprocessing: **~5.7 GPU-h**.
This requires roughly <=.42 s/student update and >=600 view-images/s during
inference, including I/O overhead; it is NOT a measured timing. The v1 budget
cannot be reused unchanged for the larger seed grid. If v1's .8–1 h per student
and .25–.4 h per evaluation actually hold, the full grid would cost ~11–16 h.
Measure the first job's logged elapsed time before launching the rest if the
6 h ceiling is strict; do not drop seeds, steps or clinical cases to meet it.
Source/clinical inference is shared by all alignments. Expect <24 GB GPU memory
for H, more for the brief's first200-step dual-model H_matched replay; an H100
80GB is the planning device. Allow ~8–16 GB CPU RAM and tens of GB on /hnvme for
all sixteen sets of cached eight-view features and checkpoints.
