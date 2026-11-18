N064 estimates conditional photon/pixel reliability using disjoint checkerboard foreground **and** corner-background pixels, N039 single-marker unmixing, reconstructed authors' pooled unmixing, and the frozen v10 curation. `report.md` contains the pre-registration, smoke results, complete aggregate FOV analysis, and interpretation limits. `smoke/` contains the numerical evidence and row IDs.

The full pixel run is prepared, not submitted. Run this exact command outside the sandbox on the cluster:

```bash
mkdir -p /hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling
sbatch /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N064-noise-ceiling-bracket-for-the-v10-goal-p/codex/n064_cpu.sbatch
```

The batch file uses partition `preempt_cpu`, 48 cores, 120G, two hours, the `cpu_job.sbatch` working-directory/container/bind pattern, and `~/.claude/skills/work-log/bin/preempt-run.sh`. Its exact wrapped analysis command is:

```bash
cd "$HOME/BSCCM"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
  apptainer exec --bind /hnvme/workspace/v181bb10-bsccm \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B \
  /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N064-noise-ceiling-bracket-for-the-v10-goal-p/codex/n064_ceiling.py \
  --source /hnvme/workspace/v181bb10-bsccm \
  --out /hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling --workers 48
```

Expected allocation is **1–8 CPU-h** (1.25–10 minutes × 48 cores), dominated by storage/cache conditions. The full plan has 69,576 unique pixel cells, about 13.7 GB of fluorescence records. The two-hour Slurm limit corresponds to a 96 CPU-h maximum per chain link. The warm-storage 2,000-cell smoke took 1.34 seconds total, 0.29 seconds in pixel extraction, with four workers; this does not measure cold-storage throughput or 48-worker scaling.

No Apptainer is available in the sandbox. The real-data smoke used a uv virtual environment and read-only `np.load(..., mmap_mode='r')` fluorescence arrays. Reproduce from this deliverable directory (scratch stays under the specified path; use a fresh output directory if code/inputs/seed change):

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N064-noise-ceiling-bracket-for-the-v10-goal-p/codex
mkdir -p /tmp/codex_n064
UV_CACHE_DIR=/tmp/codex_n064/uv-cache uv venv /tmp/codex_n064/venv
UV_CACHE_DIR=/tmp/codex_n064/uv-cache uv pip install \
  --python /tmp/codex_n064/venv/bin/python \
  numpy==2.5.5 pandas==3.1.1 scipy==1.18.1 scikit-learn==1.9.1 pyarrow==26.0.1
/tmp/codex_n064/venv/bin/python -I -B n064_ceiling.py \
  --source /hnvme/workspace/v181bb10-bsccm \
  --out /tmp/codex_n064/reproduce_smoke --workers 4 --max-cells 2000 \
  > /tmp/codex_n064/reproduce_smoke.log 2>&1
```

Python 3.13.13 was used for the uv smoke; the batch job uses the existing image's Python/packages without installing anything. NumPy, pandas, SciPy and pyarrow are the runtime dependencies; sklearn is present in the requested smoke environment. No training, GPU, torch, zarr decoding or source-repo changes are required. The N039 map JSON and source parquet/target/prediction files are read in place; all paths are recorded in `manifest.json`.

The pixel cap applies to unique fluorescence crops. Calibration checks use all clean split-0 aggregates, frozen spectral gates use the authors' original selections, and FOV/unstained diagnostics use all GOAL/control aggregate rows. Both directions' evaluation pixels and matched training pixels share the global 2,000-cell smoke budget. Full-run matched samples use clean split-0 rows in the opposite draw, without replacement, matched by antibody condition and count. Reported half correlations retain any zero-variance/negative-reliability failure as undefined; the GOAL mean is withheld unless all 11 targets exist in each direction.

The full run writes `report.md`, `reliability.csv` (target/direction/cohort r, rho, r_max), `summary.csv`, `fov_error_share.csv`, `unstained_spread.csv`, `unstained_raw_pooled.csv`, `reconstruction_fidelity.csv`, `pixel_fidelity.csv`, `unmixing_fidelity.csv`, `curation_replay.csv`, `half_labels.npz` (global G, A, B, full, names), `cell_plan.csv`, fitted pooled maps, manifest, runtime and decision JSONs, and atomic pixel checkpoints under `/hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling/`. The final `COMPLETE.json` makes a successor exit immediately. SIGTERM/USR1 save completed chunks and exit 143; resubmissions reuse those chunks. Changed code/input timestamps/seed/chunk selection refuse stale checkpoints. To stop a submitted chain, use the wrapper's `~/.claude/skills/work-log/bin/preempt-cancel.sh JOB_ID` (plain scancel triggers resubmission).

The smoke passed deterministic signal/resume checks in `validation.log` and the batch file passes `bash -n`. `smoke_initial_channel_audit.log` preserves the first failed channel-order audit; the final mapping follows the released global metadata, not an inferred correlation ordering.

High local-background split-half agreement does **not** demonstrate high reliability of actual v10 labels: the reconstructed-label fidelity table exposes substantial differences, especially CD19/CD56. The photon-only estimate omits shared background/FOV/stain-lot error. FOV residual share also contains model/composition error and finite-FOV-sample variance; its CSV includes weighted/unweighted quantities, singleton counts, a null share and ANOVA sensitivity. Marker labels on unstained cells are masked out in the target file; do not interpret their zero fills as measured zero noise.
