# N022 · Fine-grained teacher: rare-population tail weighting + class-balanced focal 10-gate head (SSL-init r50); fine metrics: within-class r, cls10 balanced acc, eosinophil-gate AUROC

type: exp · date: 2026-11-09 · status: done
goal_effect: flat · scale: module · value: 0.6306

## Goal
Make the LF teacher (and through it the Giemsa student) more fine-grained: rare / within-lineage populations (NK-LGL CD56/CD16, basophil CD123, eosinophil CD16-low, immature granulocytes), measured on the batch-held-out GOAL, fine metrics, and pre-registered MLL23 clinical contrasts.

## Plan
Arms on the N014 best config (SSL(aug)-init r50, 10 ep, lr 5e-4, targets_v4): base | tail05 (density^-0.5 tail weighting + class-balanced focal 10-gate head, gamma 2) | tail10 (alpha 1.0). FOV + b1to0. Chain FOV teachers -> anchored GS student (PBC half A anchors) -> PBC half B + MLL23 (never used) contrasts, 3 student seeds per teacher; reference = N021 anchored students (r18 sup teacher, 2 seeds). Kill: no GOAL gain > 2x noise and no MLL23 fine-contrast gain.

## Changes

## Evidence
- MLL23 (Zenodo 14277609, CC BY 4.0; 12 class zips kept packed; up to 1,500 cells/class; NEVER used in training/alignment): pre-registered fine contrasts (chat 2026-11-09) for the reference students (baseline n019 / PBC-anchored n021 s0 / s1; .agent-research/results/N022-fine-grained-teacher-rare-population-tai/mll23_auroc_reference.csv):
  eos<segneut CD16 .34/.85/.75; myelocyte<segneut CD16 .62/.81/.81; band<segneut CD16 .67/.65/.61; hairy>lym HLA-DR .52/.86/.85; hairy>lym CD123 .31/.61/.72; hairy>lym CD19 .53/.43/.63; hairy<lym CD3 .42/.48/.63; plasma<lym CD45 .53/.58/.72; normoblast<lym CD45 .64/.50/.68; LGL>lym CD56 .56/.56/.44, CD16 .50/.52/.56; baso CD123 .35/.51/.37; lineage refs segneut>lym CD16 .86/.98/.98, mono CD14 .62/.93/.87, lym CD3 .64/.82/.86.
- => on an independent clinical set, PBC-anchored alignment transfers several fine facts (eosinophil + immature granulocyte CD16-low, hairy-cell HLA-DR+ / CD123+) in both seeds; fails for NK-like LGL (CD56/CD16) and basophil CD123 -- exactly the teacher's weakest markers.
- Fine-grained teacher runs (jobs 1287902-06) + chained probe -> anchored GS student -> PBC/MLL23 eval (base_fov: 1288250-52, tail05_fov: 1288253-55).
- Teacher results (/hnvme/workspace/v181bb10-bsccm/runs/n022_{base,tail05,tail10}_{fov,b10} result.json; .agent-research/results/N022-fine-grained-teacher-rare-population-tai/n022_metrics.csv): FOV mean r base .654 | tail05 .674 (CD123 .50 -> .67); within-class r (pooled groups) .205 | .215; cls10 balanced acc .445 | .476; eos-gate AUROC .971 | .958. Batch-held-out b1to0: legacy 12-target .5749 | .5722 | tail10 .5724; common-11 on v5 labels .6343 | .6306 | .6320 (plot .agent-research/plots/lf-mk-common11-b1to0_2026-11-09_2042.png); b1to0 within-class .148 | .131 | .130. cls10 metrics are meaningless on b1to0 (gate labels exist only on the batch-0 'all' slide -> head untrained, chance .09-.14).
- GS students (jobs 1288250-55, 1288774-81; runs/n022_gs_anc_{base,tail05}_fov[_s1,_s2]; transfer_multiseed.csv): MLL23 mean AUROC over the 12 pre-registered fine contrasts: ref N021 (r18 teacher) .630 +- .024 (n=2) | base_fov .505 +- .017 (n=3) | tail05_fov .494 +- .050 (n=3) (plot .agent-research/plots/mll23-fine-contrast-mean-auroc_2026-11-09_2042.png). Per contrast ref vs base vs tail05: eos<segneut CD16 .80/.57/.51, hairy HLA-DR .86/.69/.60, plasma CD45 .65/.30/.34, myelocyte CD16 .81/.71/.60; LGL CD56 .50/.48/.49 and baso CD123 .44/.41/.38 fail in every arm. PBC-B: HLA-DR mono>neut .87 | .41 | .53.
- Student seed spread is large (PBC eos CD16 sd .25-.42), so single-seed teacher comparisons are uninformative.

## Outcome
FLAT / negative. Tail weighting + focal gates help in-distribution (FOV +.02, CD123 +.17) but not on held-out batches (common-11 b1to0 .6306 vs .6343) and not for clinical transfer. The r50 SSL-init teachers distil WORSE into the Giemsa student than the r18 supervised teacher (MLL23 fine .50 vs .63), repeating K011: the best marker teacher is not the best Giemsa teacher. NK/LGL CD56 and basophil CD123 fail everywhere -> label-limited (N023: CD56 dim, CD123 rare) rather than teacher-limited.
<!-- on close: bin/effect.sh N022 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N022-slug/ + external absolute paths -->

## Next
GS chain stays on the r18 supervised teacher; test a v5-label r18 teacher for the chain; reliability-weighted labels (IDEAS); NK/basophil need a different signal (darkfield-only heads, cell-level calibration), not loss re-weighting.
