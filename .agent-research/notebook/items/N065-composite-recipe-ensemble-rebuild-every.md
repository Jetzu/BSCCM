# N065 · Composite-recipe ensemble: rebuild every ensemble member on the N063 recipe (panel head for pooled groups + twopart/fov-re for singles) -- composite base, composite zoom2, composite Cytoland; pre-registered 3-member ensemble vs the .6916 bar

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: module · value: 0.6899

## Goal
N063 made the composite (panel head for pooled groups + twopart / per-FOV-offset model for singles) the best single-model recipe
(+.0106 over fresh controls), but as one extra member it adds only +.004 to the N053 ensemble. Rebuild ALL ensemble members on it.

## Plan
New runs (both directions): composite zoom2 = tpfov + panel with --zoom 2, r50 SSL init, seed 2; composite Cytoland = tpfov + panel
with --arch cytoland --enc-lr-mult .1, seed 0. Composite member = singles from tpfov, pooled groups from panel (as N063).
PRE-REGISTERED primary: E3c = mean(composite base [N063 seeds 2+3 averaged], composite zoom2, composite Cytoland); success if
GOAL(E3c) > .6862 + .0054 = .6916. Secondary (descriptive): E3c + the three N053 members (6-way). Expected .690-.696.
Placement GPU 1 x H100; 8 runs, ~4.7 GPU-h (Cytoland ~1.1 h b10 / .6 h b01, r50 ~.4 / .2 h).

## Changes
No code change (flags only). Scorers: results/N065-.../n065_score.py (pre-registered), n065_pertarget_vs_n053.py (descriptive).

## Evidence
Jobs 1383672-79 (gpu_job.sbatch, 1 x H100, all exit 0; r50 11-30 min, Cytoland 70 min). Runs /hnvme/workspace/v181bb10-bsccm/runs/
n065_{tpfov,panel}_{z2,cyto}_{b10,b01}. tpfov = --head twopart --pos-q .98 --gate-w .5 --fov-re 1.0 --w-autofl 0; panel = --panel-head
--re-shrink 1; z2 = r50 SSL init --zoom 2 seed 2; cyto = --arch cytoland --enc-lr-mult .1 seed 0.

| set | b10 | b01 | GOAL |
|---|---|---|---|
| N053 ensemble (GOAL best) | | | .6862 |
| comp_base (N063 s2+s3) | .6901 | .6759 | .6830 |
| comp_z2 | .6786 | .6681 | .6734 |
| comp_cyto | .6756 | .6531 | .6644 |
| **E3c (PRIMARY)** | .6975 | .6823 | **.6899** |
| E3c + N053 (6-way, secondary) | .7036 | .6822 | .6929 |

PRIMARY E3c .6899 < bar .6916 -> FLAT (+.0037 over the GOAL best = 1.4 x noise).
Per target vs N053 ensemble (n065_vs_n053_per_target.csv): E3c gains on pooled groups (grp:CD45 +.025, CD3/19/56 +.008) and
CD123 +.014 / CD56 +.013, loses on singles CD19 -.010, CD45 -.005, CD3 -.004 -> composite helps the pooled/dim targets, the
old members are better on bright singles. The 6-way is >= N053 on 10/11 targets (CD45 -.001), GOAL .6929 (+.0067 = 2.5 x noise)
but it was the descriptive secondary, so it does NOT count; needs a pre-registered fresh-seed confirmation.
Plot: plots/goal-r-n065_2026-11-18_1831.png

## Outcome
<!-- on close: bin/effect.sh N065 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
results/N065-composite-recipe-ensemble-rebuild-every/: n065_score.py, n065_scores.csv, n065_per_target.csv,
n065_pertarget_vs_n053.py, n065_vs_n053_per_target.csv, plot_n065_goal.csv, jobs.txt; plots/goal-r-n065_2026-11-18_1831.*

## Next
Flat -> STALL 5/5 -> project review. Candidate for the review portfolio: pre-registered fresh-seed confirmation of the 6-way
(composite members seed 1 + N053 members) -- union-of-recipes ensemble; expected ~.692, i.e. right at the bar.
