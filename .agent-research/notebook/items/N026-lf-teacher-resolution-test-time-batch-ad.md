# N026 · LF teacher resolution + test-time batch adaptation: zoom2 centre crop, no-maxpool stem, both; AdaBN on unlabelled new-batch images (v5 labels, SSL-init r50)

type: exp · date: 2026-11-10 · status: done
goal_effect: improved · scale: module · value: 0.6515

## Goal
Lift the batch-held-out GOAL (common-11 v5, best .6404) where it is weakest: lymphoid subsets (CD19 .28-.37, CD56 .36, CD3 .56-.59) need within-lymphocyte morphology, but a cell occupies a ~41 px detection circle in the 128 px crop and the ResNet stem divides by 4 before layer1 (~10 px per cell; ~1 px at the last stage). Second lever: new-isolation shift via unsupervised test-time BN recalibration (AdaBN) on unlabelled images of the new batch (available in deployment; no labels).

## Plan
SSL(aug)-init r50, 10 ep, lr 5e-4, targets_v5, seed 0, b1to0 + b0to1 (1-seed screen). Arms: zoom2 (centre 64 px -> bilinear 128), nomaxpool stem (stride 2 before layer1), zoom2+nomaxpool. AdaBN (--adabn 20000: reset BN running stats, cumulative average over 20k target-batch images from TRAINING FOVs, split 0; labels never read) on every new arm and on the existing N023 v5 checkpoints (seeds 0/1, resumed at epoch 10, no further training). Baseline: N023 v5 seed 0 .6423 / 2-seed mean .6404; seed-ensemble reference .6439. Kill: < .6404 + .004.
Jobs (.agent-research/results/N026-lf-teacher-resolution-test-time-batch-ad/jobs.txt): 1294865-74. Placement 1 H100 each (preempt); nomaxpool ~2-3x compute; ~7 GPU-h total.

## Changes
- scripts/train_lf.py: --zoom, --stem nomaxpool, --adabn N (writes result_adabn json + eval_preds_adabn npz); resume cfg omits new options at default so older checkpoints still resume.

## Evidence
- AdaBN on the N023 v5 checkpoints (jobs 1294865/66/70/71; /hnvme/workspace/v181bb10-bsccm/runs/n023_v5_b10/result_adabn.json and the 3 sibling runs; 20k unlabelled target-batch split-0 images): common-11 b1to0 .6437 -> .6346 (s0), .6420 -> .6312 (s1); b0to1 .6409 -> .6357, .6348 -> .6294 => AdaBN HURTS (-.005 to -.011) on every run; the per-slide flat-field input normalisation already removes the batch shift and re-estimated BN statistics only move away from the trained ones. Dead.
- Resolution arms (seed 0, jobs 1294867-74; runs/n026_*): GOAL zoom2 .6404 (b10 .6456 / b01 .6351), nomaxpool .6319, zoom2+nomaxpool .6325; zoom2 seed 1 (jobs 1295183/85) .6440. Zoom helps lymphoid markers (CD19 b01 .42 vs ~.36, CD56 b10 .40 vs .36) but costs CD123 (-.07..-.11): cropping removes the darkfield context CD123 needs.
- Ensembles (prediction average, same eval rows; goal_arms.csv, ensemble_scoring.py; plot .agent-research/plots/goal-common11-multiscale_2026-11-10_0953.png): base+base (2 seeds) .6438; zoom2+zoom2 (2 seeds) .6491; base+zoom2 .6512 (seed-0 pair) / .6518 (seed-1 pair, independent models). On the N029 clean labels (v5c) every number moves <= +.0005 (.6517 / .6522).
- Shared-encoder dual-scale (--dual-zoom 2; jobs 1295184/86): COLLAPSED .507 -- red-team: separate-view BN recalibration largely rescues it on diagnostic samples -> BatchNorm running statistics mixed over two very different views (confound, not a dead idea).
- Red-team (Codex, codex/redteam_N026.md): scores, row alignment, distinct seeds, eval-time zoom and resume-cfg checks pass. Reproduced: zoom+zoom control .6491 -> the MIXED-SCALE advantage is only +.002-.003; seed SD of a single run ~.0027 (2 seeds). Reproduced: the SSL(aug) encoder (n007, rows='all', exclude_batch=[]) was pretrained on ALL cells incl. the held-out batch's unlabelled images -> every SSL-init GOAL number (N014 .5742, N023 .6404, this .6515) is TRANSDUCTIVE w.r.t. unlabelled test-batch images; inductive control = N031.

## Outcome
IMPROVED (same transductive-SSL protocol as the recorded best): 2-model ensemble of a full-view and a 2x-zoom SSL-init r50 = GOAL .6515 (two independent replicates .6512 / .6518) vs best single .6404 and same-scale ensemble .6438. Most of the gain is that zoom models ensemble well (zoom+zoom .6491); scale complementarity adds only +.002-.003. AdaBN, nomaxpool and the BN-confounded shared dual-scale model: no. Caveat: SSL-init numbers are transductive (N031 measures the inductive version).
<!-- on close: bin/effect.sh N026 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N026-slug/ + external absolute paths -->

## Next
N031 inductive SSL; dual-scale with view-specific BN (or two encoders trained jointly); zoom ensembles of more seeds.
