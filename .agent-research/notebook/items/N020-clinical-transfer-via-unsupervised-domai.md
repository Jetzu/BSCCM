# N020 · Clinical transfer via unsupervised domain adaptation: DANN against unlabelled PBC smears (half A), pre-registered lineage AUROCs on held-out PBC half B

type: exp · date: 2026-11-09 · status: done
goal_effect: flat · scale: module · value: 0.5742

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Held-out PBC half B (8,546 images; half A = unlabelled DANN target only). 8 pre-registered lineage AUROCs (order: CD16 neut>lym, CD16 neut>eos, CD14 mono, grp CD3/19/56 lym, CD3 lym, CD45 lym>neut, CD123 baso, HLA-DR mono>neut):
  baseline (no DA)      .49 .11 .70 .31 .36 .33 .48 .47  mean .41
  DANN .1 (job 1285661) .22 .27 .38 .37 .43 .52 .71 .80  mean .46
  DANN .3 (job 1285663) .82 .36 .26 .47 .54 .70 .62 .68  mean .56
  bg-whitening (test-time, baseline) .51 .18 .64 .21 .35 .47 .23 .90  mean .44
  (.agent-research/results/N020-clinical-transfer-via-unsupervised-domai/pbc_transfer_auroc.csv; pbc_bg_whitening_check.png). Domain-classifier loss ~.69 (chance) -> marginals aligned; in-domain retrieval R@10 .27-.28, student-vs-teacher .48-.49 (vs .54 without DANN).
- Plot: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/plots/pbc-lineage-auroc_2026-11-09_1543.png
- => no reliable clinical transfer: contrasts swing between configs (CD14 .70 -> .26); unconditional alignment mismatches classes (label-shift + extra PBC classes: platelets, erythroblasts, IG). Next: class-anchored alignment (anchors = neutrophil/lymphocyte/monocyte only), evaluation on classes never used for alignment (eosinophil CD16-neg, basophil CD123+).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N020 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N020-slug/ + external absolute paths -->

## Next
