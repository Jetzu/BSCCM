# N018 · LF information map: supervised r18 per LED-pattern group (BF, DPC4, phase, DF low/high NA, LED119, morphology BF+DPC+phase, all DF) -> per-marker r, v4 labels

type: exp · date: 2026-11-09 · status: done
goal_effect: n/a · scale: paradigm · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- 8 supervised r18 runs (FOV split, v4 labels; jobs 1285248-55; .agent-research/results/N018-lf-information-map-supervised-r18-per-le/info_map_fov_v4.csv, info_map_evals.json). Mean r (12 markers): ALL .635 | DF_all .619 | DPC4 .618 | morph (BF+DPC4+phase) .614 | LED119 .612 | DF_low .607 | DF_high .600 | BF .585 | phase .576.
- Every single group carries most of the information (BF alone = 92 % of ALL). CD123 is darkfield-borne (DF_all .384 ~ ALL .390 vs BF/phase/morph .24; DF_high .33 > DF_low .27) -> granule scatter (basophils/pDC). Lymphoid group CD3/19/56 and grp CD45/CD16 best from phase-contrast morphology (DPC4/morph .83-.84 vs DF .77-.80). LED119 (single coherent LED) strong for grp targets (.81-.83).
- Caveat: FOV split (within batch); batch-held-out map not run.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N018 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N018-slug/ + external absolute paths -->

## Next
