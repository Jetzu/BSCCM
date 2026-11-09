# Knowledge

## Headline
- BSCCM has NO cell with both Giemsa/Wright (G) and surface markers (M): all 15,712 histology cells are unstained. Label-free LED images (L) exist for all 412,941 cells and are the only bridge (G-L 15.7k pairs, L-M ~350k). G->M is transitive; validity rests on G ⊥ M | L (statistical-matching CIA) [N004].
- Markers are partially observed (single-antibody cells = 1 marker; 37k 'all' cells = 4 spectral groups); cell-type labels only for 10k batch-0 'all' cells; no donor IDs (2 batches) [N004].
- Training format: /hnvme/workspace/v181bb10-bsccm/pack/*.npy fixed-record memmaps (row = global_index), targets.npz (masked asinh(x/50) markers, gate classes, FOV split) [N001].

- LF->MK baseline (resnet18, 23 LED + phase, N006): mean r (12 markers) FOV 0.707 | batch-held-out 0.494 / 0.471 (seed noise ~0.002) -> GOAL metric = batch-held-out mean (best 0.4835). Signal is mostly cell type: 3-class oracle >= model on gated cells (CD16 .95 vs .88); within-class r 0.08-0.45. CD19/CD56/CD123-group collapse across batches [N006].
- GS<->LF pairs are real: Wright-morphology clusters (DinoBloom k-means) agree with LF-predicted lineage at 0.858 cell level [N008]; a PBC-trained classifier collapses on BSCCM Wright cells (72 % "platelet"): BSCCM GS is far from clinical smears [N008].
- Binding v0 (frozen sup-LF + frozen DinoBloom-B): GS<->LF exact-cell retrieval near chance on the held-out slide; autofl on real GS cells: LF teacher r .61, distilled GS .27, head transfer -.05, direct GS ceiling .37 -> distillation >> head transfer (matches literature) [N009].

- SLIDE-POSITION ARTEFACT in the fluorescence readouts: position alone predicts autofl (r .84) and CD19 (.84), CD56 (.53), CD123 (.49) (region-blocked, median over slides); CD16/CD14/CD45 nearly clean. LF features encode position (r .92). Labels detrended -> targets_v4 [N012, N013]. Rule: clean labels only; inputs keep real artefacts.
- On v4 labels the LF teacher transfers across isolations: batch-held-out mean r .555 (v3 .483), FOV .635, gap .08 (was .22); CD19 held-out .28-.32 (was -.05) [N013]. GOAL re-based on v4 (best .5545).
- Detrended autofl on GS cells: Wright-image signal transfers across the two stained slides (ridge within .57/.68, cross .55/.47) [N012].

## Topics

## Refuted
