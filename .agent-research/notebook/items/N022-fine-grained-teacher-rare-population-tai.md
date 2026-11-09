# N022 · Fine-grained teacher: rare-population tail weighting + class-balanced focal 10-gate head (SSL-init r50); fine metrics: within-class r, cls10 balanced acc, eosinophil-gate AUROC

type: exp · date: 2026-11-09 · status: open
goal_effect: pending · scale: module

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- MLL23 (Zenodo 14277609, CC BY 4.0; 12 class zips kept packed; up to 1,500 cells/class; NEVER used in training/alignment): pre-registered fine contrasts (chat 2026-11-09) for the reference students (baseline n019 / PBC-anchored n021 s0 / s1; .agent-research/results/N022-fine-grained-teacher-rare-population-tai/mll23_auroc_reference.csv):
  eos<segneut CD16 .34/.85/.75; myelocyte<segneut CD16 .62/.81/.81; band<segneut CD16 .67/.65/.61; hairy>lym HLA-DR .52/.86/.85; hairy>lym CD123 .31/.61/.72; hairy>lym CD19 .53/.43/.63; hairy<lym CD3 .42/.48/.63; plasma<lym CD45 .53/.58/.72; normoblast<lym CD45 .64/.50/.68; LGL>lym CD56 .56/.56/.44, CD16 .50/.52/.56; baso CD123 .35/.51/.37; lineage refs segneut>lym CD16 .86/.98/.98, mono CD14 .62/.93/.87, lym CD3 .64/.82/.86.
- => on an independent clinical set, PBC-anchored alignment transfers several fine facts (eosinophil + immature granulocyte CD16-low, hairy-cell HLA-DR+ / CD123+) in both seeds; fails for NK-like LGL (CD56/CD16) and basophil CD123 -- exactly the teacher's weakest markers.
- Fine-grained teacher runs (jobs 1287902-06) + chained probe -> anchored GS student -> PBC/MLL23 eval (base_fov: 1288250-52, tail05_fov: 1288253-55).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N022 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N022-slug/ + external absolute paths -->

## Next
