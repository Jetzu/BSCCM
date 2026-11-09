# N015 · GS chain with the SSL-init r50 teacher (fov-trained, both batches): feature export + GS student cross-slide both directions, detrended autofl truth

type: exp · date: 2026-11-09 · status: done
goal_effect: flat · scale: module · value: 0.5742

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- SSL(aug)-init r50 teacher trained fov on v4 (job 1283352, 24 min): probe FOV r_mean .650 (r18 v4 .634), cls3 .945.
- GS student (jobs 1283354/5): b0->b1 autofl-truth .641 (teacher on b1 GS .400; student-vs-teacher markers .594); b1->b0 .303 (teacher .079; .504). With the r18 v4 teacher: .65/.66 and .32 (teacher .56/.31; agreement .68/.53).
- Plot: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/plots/gs-student-autofl-by-teacher_2026-11-09_1416.png
- => better MARKER teacher is worse on GS autofl and lowers GS-student agreement; autofl cannot rank teachers for the marker task. GS student's autofl (~.64/.30) comes mostly from Wright morphology.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N015 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N015-slug/ + external absolute paths -->

## Next
