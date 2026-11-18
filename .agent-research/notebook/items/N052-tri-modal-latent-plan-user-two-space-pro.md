# N052 · Tri-modal latent plan: user two-space proposal (LF+marker matrix, LF+Giemsa, flow alignment) vs LF-anchored hub; hidden-information marker completion

type: decision · date: 2026-11-18 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal
Record the user's tri-modal research plan (2026-11-18), Claude's assessment, and the user's ruling: test BOTH designs, and test the
encoders ("defo do the encoders, it is not a lot and maybe worth it").

User plan: (1) shared latent space LF + markers: marker-matrix MLP encoder (the cell's row of the cells x markers matrix, missing
entries masked) + a new LF image encoder (fine-tuned/retrained DINOv3 or another encoder: DinoBloom/SuperBloom, ResNet, Cytoland,
live-cell-dynamics models; LED channel selection/weighting); (2) second space LF + Giemsa (same structure, DinoBloom for Giemsa);
(3) align the two spaces with flow matching (alternatives listed: SigLIP, student-teacher, pixel-level OT, SOTAlign, diffusion;
Giemsa<->markers via steering, GNN, discrete flow matching, unbalanced OT, Brownian bridge); (4) maybe decode markers from the
matched space. RQs: RQ1 predict markers from Giemsa; RQ2 align three modalities; RQ3 identify cell distributions in smears.
Geometry checks: CKA, mutual kNN, spectral gap; positive/negative AUROC proxy.

User's "hidden information" idea (2026-11-18): a cell labelled only for CD19 whose embedding sits near CD45-bright cells should be
decodable as CD19+ AND CD45+ -> marker completion through the shared space; the marker-matrix encoder may add information beyond the
LF image when observed marker values are fed in.

## Plan
Claude's assessment (given to the user before the ruling; evidence in brackets):
1. With ONE shared LF encoder the two spaces are already aligned through the LF image (GS<->LF and LF<->MK are both same-cell pairs);
   aligning two separately trained spaces WITHOUT pairs meets the identifiability limit of set-level matching across different cells
   (N034-B: |r| <= .15). Prefer binding both modalities to the LF hub (ImageBind-style); the LF bridge also supplies pairs for any
   cross-space flow.
2. The binding loss must be predictive, not only contrastive: SigLIP/SOTAlign spaces find the same cell (~20x chance) but carry little
   marker information (N009, N045: autofl .38/.18 and .44/.21 vs distillation .72/.60; PBC lineage AUROC ~.55 vs .95-1.00).
   Latent distillation (GS encoder regresses the LF latent of its own cell) is the latent form of the current best bridge.
3. Marker-matrix encoder: most cells carry one scalar (+ autofl; 9 % pooled cells with 4 groups), so a marker embedding is thin; most
   useful as decoder target / for masking and RQ3. Our predictor already decodes all 11-13 markers for every cell from the LF image
   (image-mediated completion); the new part of the user's idea is CONDITIONAL completion (LF + observed markers), testable with truth
   on single-marker cells (autofl measured) and pooled cells (4 groups).
4. Encoders: expected gains modest (N042 low-pass costs <= .011; N050 ConvNeXt errors correlate .94-.96 with ResNet; labels limit),
   but cheap -> test (DINOv3 needs a 24-channel adapter; learned LED gate answers "which LEDs matter", cf. N018 information map).
5. Flow/unbalanced OT fits where no pairs exist: clinical smears (PBC, MLL23) and population proportions (RQ3).
6. Metrics: GOAL v10 (LF->MK, best .6804, target .70); Giemsa: same-cell autofl truth, student-vs-teacher r x teacher-truth r,
   clinical contrasts (PBC, MLL23); geometry diagnostics; AUROC needs per-marker positive definitions (N049: positives ill-defined).

Ruling (user 2026-11-18): test BOTH the user's design and the hub; do the encoders.
-> N053 (encoders on the v10 GOAL), N054 (two-space + flow vs hub vs TT6c on Giemsa metrics + completion tests C1-C3).
   E3 (unpaired flow/OT BSCCM-Giemsa -> clinical smears) follows N054 on its best GS encoder.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N052 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N052-slug/ + external absolute paths -->

## Next
