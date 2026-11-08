# Ideas

idea · mechanism · cheapest test · status open|testing|promising|dead(why)

- simulated-mosaic transitivity test · split L channels into a pseudo-"G" (BF+DPC+phase) and a pseudo-pivot (darkfield) on stained cells; train only pseudoG-pivot pairs on set A and pivot-M on set B; test pseudoG->M where truth exists vs direct supervised · measures the transitive gap with ground truth · CPU/1 GPU, existing pack · open
- population-level check on batch 0 · same cell isolation gives 10,015 Giemsa cells and 10,210 'all' cells; predicted marker distribution on Giemsa cells vs measured distribution (W1 per marker, gate-class fractions) · zero new data · open
- lineage-consistency anchor · DinoBloom classifier trained on EXTERNAL Giemsa datasets (Acevedo/Matek) labels BSCCM Giemsa cells (neutrophil/eosinophil/mono/lympho); check predicted markers are lineage-consistent (CD16 hi neutrophils, CD14/HLA-DR monos, CD3/19/56 lymphos) · independent of L->M chain · open
- autofluorescence as extra label-free channel · 6-ch fluorescence images exist for Giemsa cells too (unstained = autofluorescence) · part of L · open
- wet-lab ground truth ask · sequential antibody fluorescence THEN Wright/Giemsa on the same slide (BSCCM hardware) for a few thousand cells would give true G-M pairs · supervisors · open
