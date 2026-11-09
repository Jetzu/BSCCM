# N021 · Class-anchored alignment to PBC (anchors neutrophil/lymphocyte/monocyte only; eosinophil + basophil never used) +/- DANN; held-out-class contrasts on PBC half B

type: exp · date: 2026-11-09 · status: done
goal_effect: flat · scale: module · value: 0.5742

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Anchored CE (anchors neutrophil/lymphocyte/monocyte; BSCCM side = LF-model lineage of each Wright cell; PBC half A anchor images only; eosinophil/basophil/IG/erythroblast/platelet NEVER used): jobs 1285843 (s0), 1286891 (s1); +DANN .3 job 1285845. Eval on PBC half B (.agent-research/results/N021-class-anchored-alignment-to-pbc-anchors/pbc_anchor_auroc.csv; */lineage_consistency.json).
- Anchored contrasts (CD16 neut>lym, CD14 mono, grp CD3/19/56, CD3, CD45 lym>neut, HLA-DR): baseline .31-.70 -> anchored s0 .79-1.00, s1 .94-1.00 (partly circular: class labels shape the alignment).
- HELD-OUT classes: CD16 neut>eos .11 -> .95 (s0) but .36 (s1) = NOT robust; CD16 neut>IG (pre-registered before results) .30 -> .92/.83; CD45 lym>erythroblast (pre-registered) .35 -> .83/.98; CD123 baso .48 -> .49/.61. +DANN: eos .55, baso .15 -> DANN harmful.
- In-domain cost: retrieval R@10 .28, student-vs-teacher .49 (vs .54 without alignment).
- Plot: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/plots/pbc-anchor-lineage-auroc_2026-11-09_1650.png
- => class-anchored alignment transfers lineage-level marker structure to clinical smears robustly; finer marker biology beyond the anchors (eosinophil CD16-negativity, basophil CD123) does not transfer reliably (seed-dependent / chance). IG and erythroblast checks are weaker evidence ("less neutrophil/lymphocyte-like" also passes them).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N021 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N021-slug/ + external absolute paths -->

## Next
