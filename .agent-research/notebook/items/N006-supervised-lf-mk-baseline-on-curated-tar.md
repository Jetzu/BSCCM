# N006 · Supervised LF->MK baseline on curated targets (resnet18, 23 LED + phase; FOV split + batch hold-out)

type: exp · date: 2026-11-08 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Official runs (targets_v3, GPU loader, GS cells excluded from training), resnet18 on 23 LED + phase, 8 epochs, seed 0: jobs 1275685 (fov), 1275686 (b1to0), 1275687 (b0to1); dirs /hnvme/workspace/v181bb10-bsccm/runs/n006v3_lf_sup_*; per-epoch logs + summary in results/N006-supervised-lf-mk-baseline-on-curated-tar/ (n006v3_*_log.json, n006v3_summary.csv, n006v3_per_marker_long.csv); plot .agent-research/plots/lf-to-mk-pearson-per-marker_2026-11-08_1945.png.
- Mean r over 12 markers: FOV 0.707 | batch 1->0 0.494 | batch 0->1 0.471. cls3 acc (FOV) 0.945.
- Per marker (fov / b1to0 / b0to1): CD16 .90/.89/.89, CD14 .85/.84/.79, CD45 .82/.71/.82, HLA-DR .63/.52/.40, CD3 .62/.46/.49, CD19 .64/-.05/-.07, CD56 .59/.20/.18, CD123 .57/.33/.35; grp CD3/19/56 .85/.66/.61, grp CD123/HLA-DR/CD14 .36/-.08/-.07; autofl .52/.30/.38.
- KEY: class-oracle (truth class means, 3 classes) on gated 'all' cells beats or matches the model: CD16 .954 vs .880, CD45 .858 vs .851, CD3/19/56 .823 vs .793 -> LF->MK signal is mostly cell-type identity. Within-class r 0.08-0.45 (best: CD3/19/56 within mono .45, within lymph .28; CD45 within gran .41).
- Batch hold-out costs ~0.22 mean r; CD19/CD56/CD123-group collapse across batches (batch-1 'all' CD19 at 35 % antibody; donor/isolation shift).
- Superseded (v2 targets, old loader, GS cells in training): results/N006-supervised-lf-mk-baseline-on-curated-tar/superseded_v2targets_job*.json.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N006 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N006-slug/ + external absolute paths -->

## Next
