# N029 · Clean calibration cohort: no Giemsa (hist_row>=0) cell in ANY target fit (v3 calibration, v4 spatial HGB, v5/v6 normalisation) -> targets_v3c..v6c; rebuild the TT6 Giemsa chain on clean labels (red-team N025 F1)

type: fix · date: 2026-11-10 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal
Red-team N025 F1 (reproduced): v3 calibration (CAL = split==0) and the v4 spatial HGB (tr = m[sp[m]==0]) were fitted on rows INCLUDING Giemsa cells, so the only same-cell Giemsa truth (autofl) shaped training labels. Rebuild every target fit without Giemsa cells and re-run the Giemsa chain on clean labels.

## Plan
BSCCM_CLEAN=1 mode in make_targets_v3..v6 (all fits/calibrations exclude hist_row>=0; transforms still applied to all rows) -> targets_v3c..v6c. r18 FOV teachers on v4c and v6c -> probes -> TT6c (v4c markers + v6c autofl) and v4c single-teacher students, 3 seeds anchored + cross-slide b0/b1; MLL23/PBC; within-slide autofl truth on clean (v4c) truth. Compare with N025 (TT6 .578/.855 | .611; v4 .383/.689 | .622).
Incident: first submission (v4c job 1294906) stalled >77 min (original v4 run: 6 s): sklearn initialised OpenMP with all node cores BEFORE the script's OMP_NUM_THREADS setdefault -> 16 workers x 64 threads, load ~800 on 48 cores. Fixed (setdefault before any import, OMP_NUM_THREADS=3 exported); chain cancelled and resubmitted (.agent-research/results/N029-clean-calibration-cohort-no-giemsa-hist/jobs.txt; apply outputs now written to /hnvme runs dirs, not /home).

## Changes
- scripts/make_targets_v3-v6.py: BSCCM_CLEAN mode (default off); make_targets_v4.py: OMP thread cap before imports.

## Evidence
- Clean targets (BSCCM_CLEAN=1; v3c job 1294905, v4c job 1295472 [5 s; only the summary-CSV write failed: missing argv], v5c/v6c built on the login node): single-marker labels bit-identical to v4/v6; pooled grp labels max |dY| .23 (v3 intensity floor previously taken over the whole batch incl. GS cells); autofl corr(old, clean) .90 on GS cells, .997 non-GS. GOAL arms re-scored on v5c move <= +.0005 (ensemble .6517/.6522).
- Clean chain (jobs .agent-research/results/N029-clean-calibration-cohort-no-giemsa-hist/jobs.txt; n029_metrics.csv, transfer_multiseed.csv; plots .agent-research/plots/n029-gs-autofl-within-b0_2026-11-10_1129.png, .agent-research/plots/n029-mll23-fine-mean_2026-11-10_1129.png), all scored on CLEAN truth (targets_v4c), 3 student seeds: within-slide autofl b0 / b1: TT6c .564 / .848 vs TT6 (old labels) .570 / .845; v4c .395 / .654 vs v4 .373 / .684. Cross-slide (1 seed): TT6c .715 / .601 vs TT6 .745 / .623. MLL23 fine mean: TT6c .551 vs TT6 .611; v4c .546 vs v4 .622.

## Outcome
Same-cell autofl gain CONFIRMED on clean labels (TT6c .564/.848 vs single-teacher .395/.654) -> the preprocessing leak did not create it. Clinical MLL23 drops by ~.07 for BOTH retrained clean teachers (two-teacher and single) although marker labels barely changed -> most likely teacher-RUN variance (all earlier clinical comparisons used a single teacher run per version; red-team N025 F2) -> N032 measures it. Not a GOAL attempt.
<!-- on close: bin/effect.sh N029 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N029-slug/ + external absolute paths -->

## Next
N032 teacher seeds; report clinical transfer as mean over teacher x student seeds from now on.
