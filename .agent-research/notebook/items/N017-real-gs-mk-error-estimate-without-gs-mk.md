# N017 · Real GS->MK error estimate without GS-MK pairs: per-marker bound + CIA estimate from a (teacher, stained held-out) and b (GS student vs teacher, held-out slide)

type: analysis · date: 2026-11-09 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Jobs 1285049 (train GS b0 slide -> test b1 slide) / 1285050 (b1 -> b0); teacher = ridge on frozen v4 r18 fov LF features; .agent-research/results/N017-real-gs-mk-error-estimate-without-gs-mk/bound_b0-b1.json, bound_b1-b0.json, gs_mk_estimates.csv.
- Correlation estimate under CIA with a calibrated teacher (T ~ E[M|L]): r(GS,M) ~= r(S,T) * r(T,M). b0->b1: CD16 .80, CD45 .78, CD14 .73, grp CD3/19/56 .70, grp CD45 .69, grp CD16 .66, CD3 .48, HLA-DR .43, CD56 .21, CD19 .21, CD123 .17, grp CD123/HLA-DR/CD14 .01. b1->b0 (student trained on 5.7k cells): CD16 .82, CD14 .67, HLA-DR .49, CD45 .44, grp CD3/19/56 .44, CD3 .40, CD56 .21, CD19/CD123 ~.03-.05.
- RMSE-based bounds (validated in N016): CIA R^2 CD16 .66/.67, CD45 .62, CD14 .54/.33, others low/negative; RMSE-based numbers are pessimistic where the student's scale differs from the teacher's (b includes calibration offsets) -> correlation estimates used for ranking.
- Autofl truth (only real GS truth): student .649 / .319 vs teacher .559 / .305.
- => estimated Giemsa-visible markers ranking: CD16 ~ CD45 ~ CD14 (lineage: neutrophil/leukocyte/monocyte) > lymphoid group CD3/19/56 > CD3 ~ HLA-DR > CD19/CD56/CD123 (~ not visible); matches the hematology expectation (N002 D).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N017 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N017-slug/ + external absolute paths -->

## Next
