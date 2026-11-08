# Knowledge

## Headline
- BSCCM has NO cell with both Giemsa/Wright (G) and surface markers (M): all 15,712 histology cells are unstained. Label-free LED images (L) exist for all 412,941 cells and are the only bridge (G-L 15.7k pairs, L-M ~350k). G->M is transitive; validity rests on G ⊥ M | L (statistical-matching CIA) [N004].
- Markers are partially observed (single-antibody cells = 1 marker; 37k 'all' cells = 4 spectral groups); cell-type labels only for 10k batch-0 'all' cells; no donor IDs (2 batches) [N004].
- Training format: /hnvme/workspace/v181bb10-bsccm/pack/*.npy fixed-record memmaps (row = global_index), targets.npz (masked asinh(x/50) markers, gate classes, FOV split) [N001].

- LF->MK baseline (resnet18, 23 LED + phase, N006): mean r (12 markers) FOV 0.707 | batch-held-out 0.494 / 0.471 (seed noise ~0.002) -> GOAL metric = batch-held-out mean (best 0.4835). Signal is mostly cell type: 3-class oracle >= model on gated cells (CD16 .95 vs .88); within-class r 0.08-0.45. CD19/CD56/CD123-group collapse across batches [N006].
- GS<->LF pairs are real: Wright-morphology clusters (DinoBloom k-means) agree with LF-predicted lineage at 0.858 cell level [N008]; a PBC-trained classifier collapses on BSCCM Wright cells (72 % "platelet"): BSCCM GS is far from clinical smears [N008].
- Binding v0 (frozen sup-LF + frozen DinoBloom-B): GS<->LF exact-cell retrieval near chance on the held-out slide; autofl on real GS cells: LF teacher r .61, distilled GS .27, head transfer -.05, direct GS ceiling .37 -> distillation >> head transfer (matches literature) [N009].

## Topics

## Refuted
