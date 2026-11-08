# Knowledge

## Headline
- BSCCM has NO cell with both Giemsa/Wright (G) and surface markers (M): all 15,712 histology cells are unstained. Label-free LED images (L) exist for all 412,941 cells and are the only bridge (G-L 15.7k pairs, L-M ~350k). G->M is transitive; validity rests on G ⊥ M | L (statistical-matching CIA) [N004].
- Markers are partially observed (single-antibody cells = 1 marker; 37k 'all' cells = 4 spectral groups); cell-type labels only for 10k batch-0 'all' cells; no donor IDs (2 batches) [N004].
- Training format: /hnvme/workspace/v181bb10-bsccm/pack/*.npy fixed-record memmaps (row = global_index), targets.npz (masked asinh(x/50) markers, gate classes, FOV split) [N001].

## Topics

## Refuted
