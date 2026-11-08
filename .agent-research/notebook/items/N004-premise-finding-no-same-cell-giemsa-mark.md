# N004 · Premise finding: no same-cell Giemsa-marker pairs; label-free is the pivot

type: decision · date: 2026-11-08 · status: open
goal_effect: pending · scale: ?

## Goal
Record the project premise (as given by the user, 2026-11-08) and the data finding that changes it.

## Premise (project text, Marr lab "Project 5: Multimodal Single-Cell Alignment", supervisor Carsten Marr; mentors Furkan Dasdelen, Fatih Ozlugedik)
- Claim: BSCCM "links all three modalities on the same white blood cells": Giemsa morphology (G), label-free LED-array imaging (L: brightfield/DPC/darkfield), six-channel fluorescence surface-marker abundance (M).
- Obj 1: shared embedding of G, L, M + "information map" (which markers predictable from which modality).
- Obj 2: label-free focus: markers from unstained cells; virtual Giemsa staining from L; rank LED patterns by information.
- Obj 3: transfer the Giemsa model to clinical smears (MLL23, BMC, Acevedo) with L as training-only information.
- Methods: DinoBloom/SuperBloom encoders (frozen+adapters or fine-tuned), MLP on arcsinh markers, pairwise SigLIP + marker regression heads; any-to-any retrieval, per-marker regression, cell-type classification, donor-level splits, missing-modality robustness; stain augmentation, TTA, unpaired OT alignment (SOTAlign) for transfer; patient-level comparison with flow cytometry; CLL vs reactive lymphocytosis.
- Baselines quoted: [2] arXiv 2605.14717: 91.3 % 3-class accuracy, Pearson 0.72 (this is the MEAN over 4 marker groups; per-marker CD16 r = 0.819; uses BSCCMNIST 28x28 DPC only; test n = 1,418) [source: subagent read of the paper, N002 context].

## Finding (verified from BSCCM_index.csv + BSCCM_surface_markers.csv, Dryad v13, 2026-11-08)
1. Histology (G) exists for 15,712 cells; ALL have `antibodies == 'unstained'` (batch 0: 10,015; batch 1: 5,697). 0 antibody-stained cells have histology. => NO cell has both G and M.
2. L (23 LED patterns + DPC phase + 6-ch fluorescence images) exists for all 412,941 cells => L is the only bridge: G-L pairs 15,712; L-M pairs ~350k.
3. M is partially observed: single-antibody cells (CD45, CD16, CD14, HLA-DR, CD123, CD3, CD19, CD56) measure ONE marker; 37,284 'all' cells measure 4 spectral groups (CD123/HLA-DR/CD14, CD3/CD19/CD56, CD45, CD16); unstained cells = autofluorescence only.
4. Cell-type labels: 10 marker gates, only batch-0 'all' cells (10,137 single-gate cells).
5. No donor IDs. Structure = 2 batches (cell isolations) x antibody condition x slide replicate; 22 imaging dates; 65,096 FOVs.

## Decision
- L is the pivot ("binding") modality; the G-M link is transitive (G->L->M) and must be treated as a statistical-matching problem (validity rests on G ⊥ M | L, untestable from these data alone).
- Masked marker targets; FOV-level train/val/test split (80/10/10, seed 0) + batch-transfer test; no "donor-level" split possible.
- Literature review (N002) BEFORE committing the architecture; establish how mosaic/bridge integration is validated in the field.

## Evidence
- results/N001-bsccm-training-format-pack-fixed-record/audit.txt, targets_build.txt (crosstabs, counts).

## Outcome

## Outputs

## Next
- N002 literature review; N003 dataset audit (G-L match quality, dpc meaning, batch effects).
