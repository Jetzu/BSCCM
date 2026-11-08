# N008 · External lineage anchor: PBC (Acevedo) MGG classifier on frozen DinoBloom features applied to BSCCM Wright cells, scale-matched

type: analysis · date: 2026-11-08 · status: open
goal_effect: pending · scale: ?

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- PBC (Acevedo 2020, Mendeley snkd93bnjr, CC BY 4.0) 17,092 images (zip kept packed; 21 macOS fork/.DS files excluded), frozen DinoBloom-B, LogReg 5-fold CV acc 0.988 (in-domain).
- Applied to BSCCM Wright cells (scale-matched: median cell area fraction PBC 0.105 vs BSCCM 0.108 -> full 398 px crop): predicted platelet 72-75 %, erythroblast 13 %, neutrophil 3-9 %, lymphocyte/monocyte < 0.5 %, eosinophil/basophil 0 (gated truth same blood, b0: lymph 32 / gran 52 / mono 16 %). => complete domain collapse; NOT usable as an anchor as is.
- Cause (pbc_vs_bsccm.png): BSCCM Wright cells are lysed/fixed on coverslips -> white background without RBCs (PBC: salmon background full of RBCs), soft focus, colour fringes from axial chromatic aberration (sequential R/G/B LEDs; lateral R/B->G shift = 0.0 px by phase correlation on 200 cells), weaker nuclear contrast. Background RGB PBC [0.996,0.90,0.80] vs BSCCM [0.98,0.96,0.96].
- Implication for Objective 3 (transfer to clinical smears): the BSCCM->clinical gap is large in BOTH directions; frozen DinoBloom features do not bridge it. Needs explicit domain adaptation (source degradation / stain+background normalisation / semi-supervised OT) before any clinical claim.
- IN-DOMAIN ANCHOR (scripts/gs_clusters.py, gs_cluster_vs_lf.py; results/N008-external-lineage-anchor-pbc-acevedo-mgg/gs_kmeans12_montage.png, gs_cluster_vs_lf.json): k-means(12) on frozen DinoBloom-B Wright embeddings; clusters labelled by eye BEFORE the crosstab (Claude, non-expert; stated in chat 2026-11-08): lymph-like {6,7,9 (dumbbell/apoptotic)}, granulocyte-like {1,3,4,8,10,11}, smooth-large/unclear {0,2,5}. LF-predicted cls3 of the SAME cells (LogReg on frozen sup-LF features, gated-cell test acc 0.945): granular clusters 96-100 % gran; c6/c7 61-70 % lymph; c9 91 % lymph; c2/c5 77-91 % mono; c0 87 % gran. Overall agreement of pre-stated labels (unclear -> mono) with LF prediction: 13,483/15,712 = 0.858. GS-cell LF-pred composition lym/gran/mono 0.19/0.60/0.21 vs gated 'all' cells (same blood, b0) 0.32/0.52/0.16 (lymphocytes under-represented in GS cells: lysis/apoptosis exclusion).
- Batch-specific GS clusters (c4 30 % b0, c11 17 % b0) -> stain differs between the two stained slides.
- Next options: (a) degrade PBC towards BSCCM (blur, background whitening via WBC mask, desaturation) and retrain; (b) in-domain morphology anchor: cluster BSCCM Wright DinoBloom embeddings, label clusters by morphology from montages (Claude, flagged as non-expert; mentors to confirm).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N008 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N008-slug/ + external absolute paths -->

## Next
