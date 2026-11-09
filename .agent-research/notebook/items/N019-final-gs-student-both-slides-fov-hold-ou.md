# N019 · Final GS student (both slides, fov hold-out) + clinical-transfer test on PBC smears: pre-registered lineage-consistency AUROCs

type: exp · date: 2026-11-09 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Final GS student (train_gs_bind, both slides, fov hold-out, 3,000 steps, r18 v4 teacher; job 1285259) applied to all BSCCM Wright cells + 17,092 PBC smears (job 1285260; /hnvme/workspace/v181bb10-bsccm/agent-research-large/N019-final-gs-student-both-slides-fov-hold-ou/{bsccm,pbc}_gs_student.npz (>5MB, moved off the notebook at lint 2026-11-09; symlinked back), .agent-research/results/N019-final-gs-student-both-slides-fov-hold-ou/lineage_consistency.json).
- In-domain (BSCCM held-out FOV Wright cells, classes from the LF model): AUROC CD16 gran>lym .989, CD45 lym>gran .987, CD3 lym .943, grp CD3/19/56 lym .934, CD14 mono .913, HLA-DR mono>gran .781 (bsccm_inDomain_lineage_auroc.json; partly circular: classes come from the LF side of the chain).
- CLINICAL TRANSFER (PBC, pre-registered contrasts): CD14 monocyte .709; CD16 neutrophil>lymphocyte .476; CD16 neutrophil>eosinophil .107 (INVERTED); grp CD3/19/56 lymphocyte .312, CD3 .365, CD45 lym>neut .329 (inverted); CD123 basophil .488; HLA-DR mono>neut .487. => the BSCCM-trained GS student does NOT transfer to clinical smears (domain gap: lysed/fixed coverslip cells, no RBC background, different stain/optics). Next: unsupervised domain adaptation with unlabelled PBC images (disjoint halves for adaptation vs evaluation).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N019 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N019-slug/ + external absolute paths -->

## Next
