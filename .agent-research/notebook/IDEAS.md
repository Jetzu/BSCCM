# Ideas

idea · mechanism · cheapest test · status open|testing|promising|dead(why)

- simulated-mosaic transitivity test · split L channels into a pseudo-"G" (BF+DPC+phase) and a pseudo-pivot (darkfield) on stained cells; train only pseudoG-pivot pairs on set A and pivot-M on set B; test pseudoG->M where truth exists vs direct supervised · measures the transitive gap with ground truth · CPU/1 GPU, existing pack · open
- population-level check on batch 0 · same cell isolation gives 10,015 Giemsa cells and 10,210 'all' cells; predicted marker distribution on Giemsa cells vs measured distribution (W1 per marker, gate-class fractions) · zero new data · open
- lineage-consistency anchor · DinoBloom classifier trained on EXTERNAL Giemsa datasets (Acevedo/Matek) labels BSCCM Giemsa cells (neutrophil/eosinophil/mono/lympho); check predicted markers are lineage-consistent (CD16 hi neutrophils, CD14/HLA-DR monos, CD3/19/56 lymphos) · independent of L->M chain · open
- autofluorescence as extra label-free channel · 6-ch fluorescence images exist for Giemsa cells too (unstained = autofluorescence) · part of L · open
- wet-lab ground truth ask · sequential antibody fluorescence THEN Wright/Giemsa on the same slide (BSCCM hardware) for a few thousand cells would give true G-M pairs · supervisors · open
- shared-latent head transfer vs distillation · latent cell state Z; if GS and LF encoders estimate the same Z coordinates (JEPA/PRH theorem, arXiv 2610.39302), a marker head trained on LF applied to rotated GS embeddings can exceed the LF->MK ceiling on Z components LF sees noisily; distillation cannot · simulated mosaic (sharp vs weak LED subsets) with truth + autofluorescence on real GS cells · open
- cross-illumination JEPA for LF · views = disjoint LED-pattern subsets of the same cell (physical "good views", not synthetic aug) vs augmentation-only SSL · prediction: higher GS<->LF mutual-kNN and better few-pair Procrustes alignment · 2 SSL runs ~20 GPU-h · open
- autofluorescence as real same-cell molecular ground truth on GS cells · autofl measured on all unstained cells incl. the 15.7k GS cells -> GS->(LF)->autofl checkable vs truth · minutes once heads exist · open
- PRH diagnostic · mutual-kNN + whitened Procrustes GS(DinoBloom)<->LF with 100/1k/10k pairs vs SigLIP adapter · minutes · open
- spatial artefact audit for ALL targets · position-only GBM (region-blocked) per slide x target; if markers also carry position signal, add position-detrended targets and a position-adversarial term to the LF encoder · CPU minutes · testing (N012b)
- slide-position covariates in the model · feed (FOV x,y,z) as nuisance input to heads only (not encoder) so position-driven variance is absorbed explicitly; evaluate on detrended targets · cheap · open
- re-unmix batch-1 'all' grp:CD123/HLA-DR/CD14 from raw 6-ch with per-batch single-slide spectra · would recover the masked v5 labels · dead (N023: the group's dye is ~4 % of the 426-446 signal in 'all' cells and its spectrum has cosine ~.95 with CD16/CD45 -> ill-conditioned)
- reliability-weighted loss per (batch, target) · down-weight labels with low dye SNR / weak subset detection (batch-1 HLA-DR SNR 1.2, CD56 dim both batches) instead of masking · weights from N023 label_qc.csv, 2 GOAL runs · open
- v5 teacher -> GS chain · autofl per-slide normalisation removes slide fingerprints the teacher could learn; may fix the asymmetric GS cross-slide autofl (b1->b0 .32 vs .65) · teacher FOV run on v5 + 2 students · open
- student ensembles / multi-seed for teacher comparisons · GS-student seed spread (PBC eos CD16 .96 vs .36, MLL23 .85 vs .75) exceeds teacher effects -> rank teachers on >= 3 student seeds or a seed-ensembled student · ~1 GPU-h per teacher · open
