# N077 data scout: public label-free leukocyte data that could help BSCCM LF->marker decoding

Date 2026-11-23. Read-only scout. Searched with WebSearch and WebFetch. Nothing was downloaded.
Quotes are verbatim from the page or tool extract. Anything not checked first-hand is marked [unverified].

## Bottom line
- Only one public set gives label-free images with **many markers per cell** on human whole-blood leukocytes: **S-BIAD452** (Lippeveld/Saeys, ImageStream). It is the top pick for joint decoding.
- No other LED-array/FPM leukocyte set with markers exists apart from BSCCM's own **BSCCM-coherent**.
- No large public label-free WBC set (>BSCCM) turned up for SSL. The large label-free blood sets are RT-DC flow brightfield, which is low-res and shows deformed cells (big gap).
- Reference ceiling: DeepIFC (60x IFC, 527k PBMC) still gets only CD19 r=.61 and CD56 r=.53, against CD45 .90. So weak markers stay weak even at 3x magnification. Our CD19 .32 has headroom, but ~.6 is a realistic external ceiling.

## Ranked list (usefulness for multi-marker LF->marker)

### 1. S-BIAD452 — human WBC imaging flow cytometry (Lippeveld, Saeys, Filby et al.)  [HIGH]
- URL: https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD452 ; files: https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/452/S-BIAD452/Files/human_wbc_ifc/
- Study text: "Four human whole blood samples were assayed on an ImageStream-X MK II imaging flow cytometer." "Two samples were stained for the identification of 8 white blood cell types."
- Single 9-colour tube, so **all markers are on every cell**: "CD15 FITC, Siglec-8 PE, CD14 PE-CF594, CD19 PerCP-Cy5.5, CD3 BV421, CD45 V500, CD4 BV605, CD56 APC, CD8 APC-Cy7" (API record). Channel names in filelist.json: BF1, BF2, SSC, CD15, Siglec8, CD14, CD19, CD3, CD45, CD4, CD56, CD8. There are also unstained/eosinophil-panel samples.
- Overlap with ours: CD45, CD3, CD19, CD56, CD14 (5/8). Missing: CD16, CD123, HLA-DR.
- Size: filelist.json lists 45 entries totalling 42,233,708,651 B (~42 GB): .cif 19.2 GB, zarr single-cell stores 17.1 GB, SCIP features.parquet 4.1 GB, plus .daf files with manual gating.
- Cells: the paper's labelled WBC set holds ~98k cells: "CD4+ T-cells 17,358; CD8+ 8,022; neutrophils 59,034; monocytes 2,655; B-cells 4,256; CD56+ NKT 2,214; other NKT 1,318; eosinophils 3,156" (bioRxiv 10.1101/680975 extract). B and NK cells are few.
- Modality: "using the 60x magnification", "0.25 μm per pixel". Two BF bands (420–480 nm, 570–595 nm) and SSC darkfield (laser side scatter).
- Licence: not stated on the record. BIA policy says "all new data directly submitted to the BioImage Archive will be made available under a CC0 or CC-BY-4.0 licence" (https://www.ebi.ac.uk/bioimage-archive/help-policies). Either works for academic use.
- How to use:
  - (a) Multi-task training with a second input stem: an IFC stem feeds a shared trunk and shared heads for the 5 overlapping markers. IFC cells carry joint labels, so they teach co-expression and exclusivity (CD3/CD19/CD56).
  - (b) A source of joint marker vectors for a structured or joint decoder prior.
  - (c) Extra SSL data.
- Modality gap: medium-high. Magnification is 3x ours. TDI-flow brightfield lacks the angular LED diversity. Laser SSC is not LED darkfield. The panel is per-sample compensated fluorescence intensity, not BSCCM's fluorescence.
- Cheapest test: pull one sample's zarr (a few GB) and resample BF1+SSC to the BSCCM pixel size. Then (i) run a frozen BSCCM encoder plus a ridge probe on CD19/CD56/CD3. (ii) Reverse direction: train the IFC multi-task model, then linear-probe its trunk on BSCCM CD19/HLA-DR against the in-domain SSL encoder. Under 1 GPU-h.

### 2. BSCCM-coherent (in-domain, same instrument)  [HIGH-MEDIUM, small]
- URL: https://waller-lab.github.io/BSCCM ; Dryad https://datadryad.org/dataset/doi:10.5061/dryad.sxksn038s ; paper https://arxiv.org/html/2402.06191v1
- Paper: "contains 566 different illumination patterns, each corresponding to illumination by a different single LED". "BSCCM-coherent contains only all-antibodies and no antibody conditions". "each dataset also contains fluorescence images in 6 channels".
- Size: 30 GB, 4,304 cells, 128x128 12-bit (tiny version: 100 cells, 0.6 GB). Licence: the arXiv page is CC BY 4.0; the dataset licence is not stated on the project page [unverified, Dryad default CC0].
- Use: there is no modality gap. Under incoherent illumination, any pattern is the sum of single-LED images. That allows:
  - learned-illumination search (Cooke/Horstmeyer, https://arxiv.org/abs/2004.04306: "optimized patterns result in up to a 45% performance improvement");
  - testing whether high-angle LEDs beyond our 24 channels carry CD19/HLA-DR signal;
  - FPM/QPI phase reconstruction as an extra input channel.
- Limits: only 4.3k cells and 'all' (pooled) labels only. Good for an evaluation or design study, too small to train a backbone.
- Cheapest test: synthesize our 24 channels from the single-LED stack, then compare a ridge/small-CNN pooled-channel r on 24 channels against all 566 LEDs (or PCA-64).

### 3. DeepIFC (Timonen et al., Cytometry A 2023)  [MEDIUM as reference/teacher, data NOT public]
- https://github.com/timonenv/DeepIFC ; https://www.biorxiv.org/content/10.1101/2022.08.10.503433v2.full
- "The DeepIFC workflow and models trained on PBMC data are available under a permissible license." (repo: Apache-2.0). No data deposit found.
- Data: PBMC "n=527,107 cells", "Images were acquired at 60× magnification", BF Ch01/Ch09 + darkfield Ch12. Panel: CD45, CD14, CD19, CD8, CD3, CD56, 7-AAD.
- Per-marker r: CD45 .90, CD14 .88, CD3 .87, CD19 .61, CD56 .53, CD8 .41.
- Use: external ceiling and benchmark. The weights could serve as a frozen IFC teacher for S-BIAD452 pseudo-labels, though the U-Net was trained on a different instrument [unverified fit]. Ask the authors for the data.

### 4. BBBC045 (Nassar et al. 2019, ImageStream)  [MEDIUM-LOW]
- https://bbbc.broadinstitute.org/BBBC045 . Channels: "fluorescent, brightfield and darkfield". Classes: "B- and T-lymphocytes, eosinophils, monocytes and neutrophils". Labels: "The ground truth has been obtained by gating based on the appropriate corresponding fluorescent channels."
- 13 subjects, ~2.3 GB (591.6 MB populations, 1.58 GB montages, 170.6 MB CP features). CC0.
- Per-cell antibodies: FITC CD3/CD14/CD15/CD19 per search summary [unverified whether one marker per tube]. Gives class labels only.
- Use: a second IFC domain for the stem in #1, and a cheap transfer-evaluation set.

### 5. Cebinelli et al., BD FACSDiscover A8 + CellView, PB leukocytes (Mendeley)  [UNKNOWN, potentially MEDIUM]
- https://data.mendeley.com/datasets/rhjd7kg5by/2 (also 28sczzzddy, d432w939nf, s7vgdkx5dg, ky4svt2nyw). "acquired on a BD FACSDiscover™ A8 instrument equipped with BD CellView™ technology". CC BY 4.0.
- 10-marker panel per cell: CD5, CD4, CD8, CD14, CD15, CD19, CD45, CD56, PD-1, PD-L1. Overlap: CD45, CD14, CD19, CD56.
- The file lists rendered empty. Whether label-free image channels (light-loss/FSC/SSC images) are included is **[unverified]**; more likely FCS plus image-derived features.
- Cheapest test: open the Files tab by hand (5 min). If images are present, it is a second jointly labelled multi-marker source.

### 6. Guck lab deformability cytometry, blood cells + aggregates (OSF 3zkvw, 2026)  [LOW-MEDIUM; SSL/aux]
- https://osf.io/3zkvw ; paper https://arxiv.org/html/2609.07410 . "This dataset contains Deformability Cytometry data (.rtdc files) with images of blood cells and cell aggregates."
- Brightfield "250 × 80 pixel resolution, recorded at 3600 frames per second". Ground truth comes from antibody panels (lymphocytes CD19/CD3/CD56; neutrophils CD15/CD16/CD66; basophils CD123/CD203c; monocytes CD14).
- Files: train_data.zip 1.94 GB, test_data.zip 0.66 GB, WBCtest.zip 35 MB. About 15.6k WBCs train and 1.7k test. No licence set (node_license null).
- Gap: large (flow, deformed cells, low-res grayscale). Use as SSL or a class-auxiliary task only. Check whether the .rtdc files keep fl1-3 intensities (would give per-cell marker regression).

### 7. Immunological-synapse IFC (Dryad 10.5061/dryad.ht76hdrk7)  [LOW-MEDIUM, HLA-DR/CD3 only]
- https://datadryad.org/dataset/doi:10.5061/dryad.ht76hdrk7 . "over 2.8 million images"; "one bright field channel and multiple fluorescent channels". Markers: CD3, HLA-DR, phalloidin, pCD3ζ, viability. 124.94 GB in 4 zips. Licence: Dryad terms [unverified, Dryad is normally CC0].
- Cells: memory CD4 T cells plus an EBV B-LCL line, mostly conjugates. Activated or cell-line biology means a large biological gap. A possible weak HLA-DR auxiliary signal.

### 8. Smaller or partial sets  [LOW]
- **soRT-FDC** (DCOR, CC0): https://dcor.mpl.mpg.de/dataset/figshare-11302595-v1 . Blood RT-DC brightfield with "for identification of neutrophils (CD66+/CD14−)". DNN training files are included. Few markers.
- **IFC-seq human PBMC** (ImageStream, 82,109 cells, BF + CD3/CD8, 32x32 px): https://github.com/theislab/ifcseq . "all data used in this publication online at" (Google Drive). No licence.
- **LymphoMNIST**: https://arxiv.org/html/2503.12622v2 ; https://zenodo.org/records/17352233 (Apache-2.0). About 80k 64x64 **mouse** DIC images; "Splenocytes were FACS-sorted into T4, T8, and B subsets." Class only; species and modality gap.
- **Zenodo 5391155**: custom IFC of T cells (healthy vs Sézary), 2.2 GB, CC BY 4.0. T cells only, no markers stated.
- **ATOM time-stretch** (IEEE DataPort): https://ieee-dataport.org/documents/human-somatic-label-free-bright-field-cell-images . "asymmetric-detection time-stretch optical microscopy". Mostly cell lines with one PBMC class, 1.4 GB, subscription needed.

### 9. Marker co-expression priors without images (for joint decoding)  [MEDIUM idea, cheap]
- BSCCM is mostly single-stain, so the joint distribution of our 8 markers is weakly constrained. A public multi-colour flow panel gives an empirical joint prior. Uses:
  - a copula or flow prior on the joint head;
  - consistency checks against the 4 pooled 'all' channels;
  - synthetic joint targets.
- Candidate: OMIP-069, 40-colour PBMC, Cytek Aurora, FlowRepository FR-FCM-Z2QV ("40 color OMIP new donors unmixed"). Fluorofinder lists CD3, CD14, CD16, CD19, CD123. CD45, CD56 and HLA-DR are **[unverified]**. FlowRepository had a certificate error and a storage-limit notice.
- Gap: PBMC (no granulocytes) and a different intensity scale, so use rank/quantile-normalised values only.
- Cheapest test: fit a Gaussian copula on flow ranks, then check whether it predicts the observed correlation structure of BSCCM 'all'-slide pooled channels.

## Other LED-array / FPM / QPI data (checked, not useful for markers)
- FPM raw stacks exist only for tissue or targets: Siegen leech HDR-FPM on Zenodo (25.4 GB); an IEEE DataPort colour-FPM blood smear (files possibly absent); Glasgow lung/USAF (on request). There are no leukocyte+marker sets.
- QPI/ODT lymphocytes: Yoon 2017 (n=149/95/112, B/CD4/CD8, mouse) has no public data found. Deep-UV label-free hematology (Robles lab) has no data found. The HKU multi-ATOM T-cell data returned 403.
- arXiv 2605.14717 uses BSCCMNIST (DPC only, 28x28): "BSCCM provides paired DPC images, WBC labels, and quantitative protein-expression measurements". It reports composite-label r (CD16 .819, CD45 .799, CD3/CD19/CD56 .768, CD123/HLA-DR/CD14 .339). These are composite channels, **not comparable** to our per-marker numbers.
- Severin/Snijder PBMC "Phenoplex" (scDINO) is multiplexed immunofluorescence, not label-free, so it was excluded.

## Recommended order
1. S-BIAD452 one-sample pull: frozen-encoder ridge probe, then a 2-stem multi-task screen (1 seed) on the 5 overlapping markers. Watch BSCCM CD19/CD56 and joint metrics.
2. BSCCM-coherent: illumination/angle-content study for CD19/HLA-DR (CPU-feasible).
3. Flow-panel copula prior for the joint head (CPU only).
4. Check the Cebinelli Files tab by hand. If images are present, rank it next to #1.
