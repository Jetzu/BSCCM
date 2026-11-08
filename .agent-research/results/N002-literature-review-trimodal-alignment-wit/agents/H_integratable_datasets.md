# H — Public datasets that could be integrated with BSCCM (stained-morphology <-> surface-marker ground truth)

Agent H, N002 literature review, 2026-11-08. Scope: public data only (no new acquisition).
Abbreviations: GS = Romanowsky-stained (Wright/Giemsa/MGG) image; LF = label-free image (brightfield/darkfield/phase/scatter); MK = surface-marker fluorescence.

## 0. Bottom line

1. **I found no public dataset in category 1** (the same human blood or marrow WBC imaged with a Romanowsky stain *and* measured for surface markers) **and none in category 2** (marker-sorted WBC subsets that were then smeared or cytospun, Romanowsky-stained and imaged). Papers that did this kind of experiment show the images only in figures (for example cytospins of sorted progenitors in Nat Immunol 2024 s41590-024-01782-4, and zebrafish index-sort plus MGG in Nat Commun 2017 s41467-017-02305-6). The only same-cell "stain + marker" public data I found is off-domain: SegPath, which uses H&E tissue restained with IF (section 4).
2. **What does exist publicly:**
   - (a) Same-cell LF + MK data from other instruments: imaging flow cytometry, RT-FDC, confocal/HCS.
   - (b) Marker-sorted populations imaged label-free.
   - (c) Smear-only GS datasets with expert *morphology* labels.
   - (d) One same-smear GS <-> deep-UV LF set.
   - (e) Patient-level smears whose labels were set by flow cytometry (the flow data itself is not released).
3. **Consequence for the pitch:** GS <-> MK can only be validated in two ways:
   - **Transitively inside BSCCM.** BSCCM's 15.7k GS cells also have LF images, so an LF->MK model trained on BSCCM's ~350k marker cells can impute markers for those GS cells.
   - **Externally at class level.** The external LF+MK sets show that LF->MK generalizes across instruments. The external GS sets give independent morphology-class labels to check against the imputed marker classes.
4. **Portals and catalogs searched:**
   - Zenodo, Figshare, Dryad.
   - IDR: grepped all 217 rows of `studies.tsv`; no blood or WBC study.
   - BioImage Archive: search API, 12 + 15 hits reviewed.
   - TCIA/IDC, Kaggle, Mendeley, GEO (imaging only), OSF, DCOR (the RT-DC repository), BBBC, ETH Research Collection.
   - The HHMI `AI-HHMI/mia-agentic-search` dataset catalog: 2,273 dataset entries; all 154 blood/immune-related entries were reviewed by name.

## 1. Ranked summary (by how directly each dataset helps BSCCM's missing GS <-> MK link)

| # | Dataset | Category | Same-cell modalities | Marker overlap with BSCCM (CD45, CD16, CD14, HLA-DR, CD123, CD3, CD19, CD56) | Scale | Size | License | Open download | Link it validates |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **S-BIAD452** Lippeveld/Filby human WBC imaging flow cytometry (IFC) | 3 | 2x BF + SSC (darkfield) + 9 markers | CD45, CD14, CD19, CD3, CD56 (5/8) + CD15, Siglec8, CD4, CD8 | 98,013 labelled WBCs in paper; 5 runs deposited | ~23 GB originals + zarr + 3.8 GB features | none stated (BIA default) [unverified] | yes | LF->MK cross-instrument; class-level GS via Siglec8/CD15 |
| 2 | **S-BIAD505** Lippeveld/Filby whole-blood confocal (CellDiscoverer 7) | 3 | BF + oblique + phase-gradient contrast + DAPI + 3 markers per panel | CD45, CD3, CD19 (+CD4, CD8, Siglec8) | 4 panels x 125 fields; cell count [unverified] | 26 GB (one .czi) | none stated [unverified] | yes | LF->MK with optics closest to BSCCM; nucleus morphology -> MK |
| 3 | **BBBC045** Nassar et al. 2019 | 3 | BF + DF + 1 FITC marker per tube | CD3, CD14, CD19 (+CD15) | 13 donors, 5 classes | 0.59 GB .cif / 1.58 GB montages | CC0 | yes | LF->MK, class level |
| 4 | **ETH 343106** Severin et al. 2022 PBMC (Opera Phenix) | 3 | BF + DAPI + marker channels | CD3, CD14, CD16 (+CD4, CD8, CD11c; full list in Table S1 [unverified]) | 15 donors, 89,483 curated cells, 8 classes | 1.78 GB | CC BY-SA 4.0 | yes | LF/nucleus -> MK (PBMC only) |
| 5 | **Dryad 2fk71** Toepfner et al. 2018 RT-DC whole blood | 2-like (label-free, not GS) | BF images of MACS-sorted subsets [sorted runs in archive: unverified] | sorted on CD3, CD19, monocyte, neutrophil, basophil, eosinophil kits | 21 healthy donors + patients | 31.8 GB | CC0 | yes | marker-defined classes -> LF (includes basophils) |
| 6 | **figshare 11302595 / DCOR** soRT-FDC (Nawaz et al. 2020) | 3 | BF (0.68 um/px) + CD66 + CD14 fluorescence, same cell | CD14 | ~22k events per file, several blood files | 10 GB | CC0 | yes | LF->MK (neutrophil vs monocyte) |
| 7 | **PBC** (Acevedo 2020), **MLL23**, **AML-Cytomorphology_LMU**, **Bodzas 2023** | GS only (4-ish) | MGG/Wright RGB, expert morphology labels; Bodzas blast lineage set by flow | none at cell level | 16k-41k cells each | 0.28-46 GB | CC BY 4.0 / CC BY 4.0 / CC BY 3.0 / CC0 | yes | class-level GS <-> (imputed) MK |
| 8 | **OSF AYW4J** Ojaghi et al. 2020 deep-UV + Giemsa | 3 (GS <-> LF) | deep-UV (260/280/300 nm) + BF of the same smears after Giemsa [unverified] | none | 13 smears | ~10 GB | none stated | yes | GS <-> LF (UV, not visible LED array) |
| 9 | **Zenodo 5391154** Otesteanu et al. 2021 | 2-like | BF IFC of FACS-sorted CD3+CD4+ T cells | CD3 | 9 donors, >100k images | 2.2 GB | CC BY 4.0 | yes | one marker-defined class -> LF |
| 10 | **SegPath** (Komura et al. 2023) | 1, off-domain (tissue) | H&E + IF restain of the same section | CD45RB, CD3/CD20 | large | e.g. 23.7 GB for CD3/CD20 | CC BY-NC-SA 4.0 | yes | precedent only: stain -> marker in tissue |

## 2. Detailed entries

### 2.1 S-BIAD452 — "Imaging flow cytometry dataset of healthy human white blood cells" (Lippeveld, Knill, Ladlow, Filby, Michaelis, Saeys, Peralta)
- **URL:** https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD452
  - HTTP: https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/452/S-BIAD452/Files/human_wbc_ifc/
  - FTP and Globus also available.
- **Paper:** Lippeveld et al., Cytometry A 97(3) 2020, doi:10.1002/cyto.a.23920 (bioRxiv 10.1101/680975). Processing paper: SCIP, Cytometry A 2024, doi:10.1002/cyto.a.24896.
- **Modalities (from the deposit's `filelist.tsv`, verified):** ImageStream-X MkII, 12 channels per cell: `BF1, CD15, Siglec8, CD14, CD19, SSC, CD3, CD45, BF2, CD4, CD56, CD8`.
  - Brightfield bands: BF1 = 420-480 nm LED, BF2 = 570-595 nm LED.
  - SSC = 758 nm laser side scatter (darkfield).
  - Fluorochromes: CD15-FITC, Siglec8-PE, CD14-PE-CF594, CD19-PerCP-Cy5.5, CD3-BV421, CD45-V500, CD4-BV605, CD56-APC, CD8-APC-Cy7.
- **Magnification:** 60x. The bioRxiv text says 0.25 um/px; the ISX nominal value at 60x is 0.33 um/px [unverified which applies].
- **Files (verified sizes):**
  - 5 runs: `sample0` (3 parts), `sample1_early` (3), `sample1_late` (3), `sample2_early`, `sample2_late`.
  - Compensated `.cif`: about 17.7 GB total. IDEAS `.daf` gating files: about 1.4 GB. IDEAS feature `.fcs`: about 0.4 GB.
  - Focused single-cell `zarr` arrays (size not listed) and SCIP `features.parquet` (3.8 GB).
- **Cells:** The paper's 8-class WBC set has 98,013 cells:
  - neutrophils 59,034; CD4+ T 17,358; CD8+ T 8,022; B 4,256; eosinophils 3,156; monocytes 2,655; "CD56+ NKT" 2,214; "other NKT" 1,318.
  - Donors: the paper reports 2 WBC-panel samples plus 2 eosinophil-panel samples. The deposit holds 5 runs, all with the WBC panel; how runs map to donors is [unverified].
  - The manual gates in the `.daf` files define the labels.
- **License:** none stated on the accession. The BIA FAQ says unlicensed accessions fall under the EMBL-EBI Terms of Use, and direct submissions are CC0 by default [unverified for this accession]. **No application is needed.**
- **Integration recipe (validates LF->MK across instruments, and class-level GS):**
  1. Read the zarr arrays (or the `.cif` files via saeyslab `cifconvert`/SCIP).
  2. Treat BF1, BF2 and SSC as LF inputs and the markers as targets.
  3. Resample to a common grid (ISX 0.33 um/px vs BSCCM 0.166 um/px) and crop to BSCCM's ~21 um field of view.
  4. Pair BSCCM brightfield with BF1/BF2, and BSCCM darkfield with SSC; both darkfield channels are scatter images, though the physics differ.
  5. Train BSCCM LF->MK heads for the 5 shared markers (CD45, CD14, CD19, CD3, CD56). Report zero-shot and few-shot transfer to S-BIAD452, then the reverse direction.
  6. The extra markers give marker-defined classes that are also unambiguous on Wright stain: Siglec8+ eosinophils, CD15+ neutrophils, CD4/CD8 splits.
  7. Use these classes as the external reference for the class-level GS <-> MK concordance test on BSCCM's GS cells (section 3).
- **Domain shifts:**
  - Cells in flow versus BSCCM's static imaging.
  - 60x high NA versus 20x/0.5 NA.
  - Spectral-band LED brightfield and laser SSC versus LED-array BF/DF/DPC.
  - Whole-blood preparation differences.
  - Only 2-3 donors.
  - 3 of BSCCM's markers are missing: CD16, HLA-DR, CD123. So nothing checks the CD16/HLA-DR/CD123 heads, or basophils and pDCs.

### 2.2 S-BIAD505 — "Confocal fluorescence microscopy dataset of human whole blood" (Lippeveld, Filby)
- **URL:** https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD505
  - HTTP: https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/505/S-BIAD505/Files/human_wbc_microscopy/
  - One file, `Experiment-800.czi` (26 GB, verified), plus per-panel file lists mapping panel, replicate, position and well to scenes.
- **Modalities (per BIA metadata):** Zeiss CellDiscoverer 7, 7 channels x 3 focal planes, 1144x1144 px per field.
  - 4 fluorescence channels: DAPI, eGFP, PE, APC.
  - 3 transmitted-light channels: brightfield, oblique, phase-gradient contrast.
- **Panels:**
  - A: CD45-eGFP / CD3-PE / CD4-APC.
  - B: CD45 / CD3 / CD8.
  - C: CD45 / CD3 / CD19.
  - D: CD45 / Siglec8-PE / CD4-APC. The panel-D description also mentions CD15, which conflicts with its protocol.
  - Each panel: 5 wells x 25 positions.
- **Not reported:** objective, pixel size and cell count [unverified]. The SCIP paper uses this set.
- **License:** none stated (same caveat as 2.1). **Open download.**
- **Integration recipe:**
  - This is the optically closest public analogue to BSCCM: static cells, transmitted light, and oblique/phase-gradient contrast that roughly matches BSCCM's asymmetric-illumination DPC.
  - Use it as a second, slide-based LF->MK replication (CD45, CD3, CD19 shared).
  - **GS-relevant extra:** DAPI gives nuclear morphology, which is most of the information in a Wright stain (lobation, chromatin). A DAPI-nucleus -> marker model can be compared against the nuclear (azure/hematoxylin-like) channel obtained by colour-deconvolving BSCCM's Wright RGB images. This gives a partial, class-level GS -> MK check [speculative].
- **Domain shifts:** confocal sectioning versus widefield; unknown magnification; only CD45 and CD3 shared across all panels; no granulocyte markers except Siglec8.

### 2.3 BBBC045 — "Human White Blood Cells" (Nassar et al., Cytometry A 95:836, 2019, doi:10.1002/cyto.a.23794)
- **URL:** https://bbbc.broadinstitute.org/BBBC045
  - Files: `Stained_Populations.zip` (0.59 GB `.cif`), `Stained_Montages.zip` (1.58 GB TIFF montages, 900 cells per montage), `Stained_CP_Features.zip` (0.17 GB).
  - Code: github.com/mariamnassar/imagingFlowCytometry.
- **Modalities:** ImageStream; brightfield + darkfield (SSC) + fluorescence.
  - Ground truth comes from FITC-labelled anti-CD3, CD14, CD15 and CD19, stained one tube per marker, plus one unstained tube per donor (PMC6767740).
  - 5 classes: B, T, eosinophils, monocytes, neutrophils. How eosinophils were gated is [unverified].
  - Pixel size not stated; ImageStream100 at 40x is about 0.5 um/px [unverified].
- **Scale:** 13 donors. Montage count suggests ≤~190k cells [unverified estimate]. The paper's 85-donor unstained validation set is not in the deposit [unverified].
- **License:** CC0 (verified on the BBBC page). **Open download.**
- **Recipe:** the simplest external benchmark for "a BSCCM-trained LF classifier, or LF->MK-derived classes, works on another instrument" (5 classes, BF+DF).
- **Domain shift:** flow, low resolution, one marker per tube (no co-expression), older instrument.

### 2.4 ETH Research Collection 10.3929/ethz-b-000343106 — "Deep phenotyping PBMC Image Set" (Severin, Snijder; Sci Adv 2022, doi:10.1126/sciadv.abn5631)
- **URL:** https://www.research-collection.ethz.ch/handle/20.500.11850/343106 — one ZIP, 1.78 GB, **CC BY-SA 4.0**, open access.
- **Modalities (per paper):**
  - Opera Phenix 20x; BF + DAPI + 3 fluorescence channels (green / PE / APC).
  - Single-round multiplexed panel of 8 surface markers. Named in the text: CD3, CD4, CD8, CD14, CD16, CD11c; full list in Table S1 [unverified].
  - CNN crops of 50x50 px = 39.5 um, i.e. ~0.79 um/px.
- **Scale:** 89,483 curated single-cell images from 15 healthy donors, 8 classes. Whether the ZIP holds exactly these crops is [unverified].
- **Recipe:** PBMC-only LF/nucleus -> MK replication that includes CD16 (one of BSCCM's markers that S-BIAD452 lacks).
- **Domain shift:** coarse pixels (~5x BSCCM), no granulocytes, fixed and stained plate format.

### 2.5 Dryad doi:10.5061/dryad.2fk71 — RT-DC "morpho-rheological phenotyping of blood" (Toepfner et al., eLife 2018;7:e29213)
- **URL:** https://datadryad.org/dataset/doi:10.5061/dryad.2fk71 — 31.8 GB in 31 zip parts plus `Data.xlsx`. **CC0**, verified via the Dryad API.
- **Modalities:** RT-DC brightfield image + contour for every event, cells deformed in a 20 um channel; ShapeOut/dclab readable. Pixel size not stated [unverified; AcCellerator ~0.34 um/px].
- **Labels:** subset identity came from MACS isolation (purity checked by FACS):
  - kits for neutrophils, monocytes, **basophils**, Pan-T/CD3, Pan-B/CD19, eosinophils;
  - NK cells only via CD56 staining.
  - Whether the sorted-population measurements are inside the zip is [unverified].
- **Cohort:** 21 healthy donors, plus AML (7), ALL (4), EBV (5) and other patient groups.
- **Recipe:** marker-defined class prototypes for LF morphology. It is the only public LF source with a sorted **basophil** class, the cell type BSCCM's CD123 marker tags.
- **Domain shift:** severe — cells deformed in flow, single low-NA brightfield, no static-cell texture.

### 2.6 soRT-FDC data (Nawaz, Urbanska, Herbig et al., Nat Methods 17:595, 2020)
- **URL:** https://figshare.com/articles/dataset/Sorting_real-time_fluorescence_and_deformability_cytometry_soRT-FDC_-_manuscript_data/11302595 (mirror: https://dcor.mpl.mpg.de/dataset/figshare-11302595-v1). **CC0**, 10 GB, open.
- **Modalities:** same-cell RT-FDC brightfield (0.68 um/px in the Figure 4 files; 0.34 um/px in the Figure 2 files, from DCOR metadata) + PE-CD66a/c/e (FL-2) + APC-CD14 (FL-3). DNN training files hold about 22k events each.
- **Images:** that the `.rtdc` files contain images is inferred from file size [unverified].
- Related: figshare 7771184 (RT-FDC reference data, CC0, 2.7 GB) has `leukocytes.rtdc` with no markers, and a CD34 HSPC file.
- **Recipe:** a small LF->MK sanity check (neutrophil vs monocyte).
- **Domain shift:** as 2.5.

### 2.7 GS-only smear datasets (expert morphology labels, no per-cell markers)
These are the only public sources of GS images with any cell-level ground truth. They are class-level bridges.

| Dataset | URL | Content | Size | License |
|---|---|---|---|---|
| PBC (Acevedo et al. 2020) | https://data.mendeley.com/datasets/snkd93bnjr/1 | 17,092 normal cells, CellaVision DM96, MGG, 360x363 JPG; 8 classes incl. basophils and immature granulocytes | 281 MB | CC BY 4.0 (verified) |
| MLL23 | https://zenodo.org/records/14277609 | >40k single cells, 18 classes, Munich Leukemia Laboratory | 9.7 GB | CC BY 4.0 |
| AML-Cytomorphology_LMU (Matek 2019) | https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/ | 18,365 cells, 100 AML + 100 control patients, M8 scanner 100x oil, 14.14 px/um | 11 GB | CC BY 3.0 |
| Bodzas et al. 2023 (Sci Data) | https://doi.org/10.6084/m9.figshare.c.6612970.v1 | 16,027 cells, 100x oil, MGG; myeloblast vs lymphoblast lineage set by patient-level flow cytometry | 46 GB | CC0 (verified) |

- **Recipe (class-level GS <-> MK concordance inside BSCCM):**
  1. Train a GS morphology classifier on PBC/MLL23, with stain/colour augmentation.
  2. Apply it to BSCCM's 15.7k Wright images.
  3. On the same cells, impute markers from their LF images with BSCCM's LF->MK model.
  4. Test agreement against the expected mapping:
     - neutrophil -> CD16hi CD45+
     - monocyte -> CD14+ HLA-DR+
     - lymphocyte -> CD3+ / CD19+ / CD56+
     - basophil -> CD123+ HLA-DR-
     - eosinophil -> CD16- with high side scatter
  5. The external IFC sets (2.1, 2.3) calibrate LF->MK accuracy.
- **Domain shifts:** MGG vs Wright; 100x vs 63x/1.4 NA; CellaVision or scanner colour; label noise. Bodzas, Matek and MLL23 are enriched for pathology.

### 2.8 OSF AYW4J — "Label-free Hematology Analysis Using Deep-Ultraviolet Microscopy" (Ojaghi et al., PNAS 2020; Robles lab)
- **URL:** https://osf.io/ayw4j/ — folder "Wide-field UV and BF Images Used in Hematologist Panel Review", 26 TIFF files, ~10 GB, verified via the OSF API.
  - Samples: Healthy x4, ITP x5, sickle-cell (SCD) x4.
  - The UV and BF images are of the same smears. BF = after Giemsa staining [unverified].
  - **No license stated** on the OSF node.
- **Related:** the follow-up paper (Kaza et al., BME Frontiers 2022; 23 donors, 40x/0.5 NA UV, 0.165 um/px) points to the same OSF node. Its virtual-stain training pairs (~74k) were pseudocolourised UV, **not** registered Giemsa.
- **Recipe:** a second GS <-> LF bridge, for testing whether a GS <-> LF mapping learned on BSCCM transfers in concept. Per-cell registration is needed.
- **Domain shift:** UV absorption contrast is not visible LED-array contrast; few samples; pathology.

### 2.9 Zenodo 5391154 — Sézary/healthy T-cell IFC (Otesteanu et al., Cell Rep Methods 2021)
- **URL:** https://zenodo.org/records/5391154 — `ImageFlowCytometryTcellData.zip`, 2.2 GB, CC BY 4.0, open.
- **Content:** in-house brightfield-only IFC of FACS-sorted live CD3+CD4+CD8- T cells; 4 healthy donors + 5 Sézary patients; >100k images.
- **Use:** one marker-defined class (CD4 T) as an LF reference, plus a malignant-morphology contrast. Low value for BSCCM.

### 2.10 Off-domain same-cell "stain + marker" precedent: SegPath (Komura et al.)
- **URL:** e.g. https://zenodo.org/records/7412529 — CD3/CD20 lymphocytes, 23.7 GB; sibling records 7412739 (CD45RB leukocytes) and 7412580 (CD235a).
- **Content:** H&E tissue patches with masks from IF restaining of the same section (per the paper title "Restaining-based annotation..."; method detail [unverified from the record page]).
- **License:** CC BY-NC-SA 4.0.
- **Use:** methodological precedent and possible pretraining for "stained nuclear morphology -> lymphocyte marker". It is **not** blood and **not** Romanowsky.

## 3. Patient-level (category 4): nothing clean
- I found no public cohort that releases both smears **and** flow cytometry (FCS) files per patient. Flow-only sets exist (e.g. Mendeley/Data in Brief, Marburg), but they have no images.
- Smear sets whose labels came from flow:
  - Bodzas 2023: blast lineage only (2.7).
  - Kaggle "Acute Lymphoblastic Leukemia (ALL) image dataset" (Aria et al. 2021; https://www.kaggle.com/datasets/mehradaria/leukemia): 89 patients, ~3.2k images; Early-pre-B / pre-B / pro-B / hematogone labels said to be set by flow cytometry [unverified]. Needs a Kaggle login.
- Smear cohorts with diagnosis but no flow:
  - BoneMarrowWSI-PediatricLeukemia (Zenodo 22818005, served via NCI IDC): 245 pediatric ALL/AML/CML marrow smears, Pappenheim, 0.11 um/px, 47k cell boxes, lab values, CC BY 4.0.
  - S-BIAD440: MLL MDS/anaemia whole-blood slides, 354 slides, 0.93 TB.
- All of these are patient- or diagnosis-level and leukemia-dominated. They are weak for BSCCM's healthy-WBC surface-marker question.

## 4. Checked and not usable (so nobody re-chases them)
- **Data not public:**
  - Doan et al. 2020, ALL bone-marrow IFC with BF/DF + CD45/CD10/CD34/CD19 at 40x, 0.3 um/px: data "available upon reasonable request".
  - DeepIFC (Timonen et al. 2023): PBMC IFC with CD45/CD14/CD19/CD8/CD3/CD56, 60x, 0.33 um/px, 527k cells; only model weights are on GitHub.
  - Ugele et al. 2018: holographic microscopy of MACS-sorted leukocytes with a CD123/CD14/CD3/CD16/CD66 purity panel; data restricted (Siemens).
  - Fanous et al. 2022 (PICS, cSLIM of 18 Wright smears): on request.
  - Ryu et al. 2021 (ODT of MACS-sorted marrow WBCs: CD3/CD19/CD14/CD33): no data statement.
  - Huang et al. 2024 (smartphone autofluorescence -> virtual Giemsa): no deposit found.
  - AACR 2025 abstract 5087 (unstained smear brightfield -> CD3): conference abstract, no data.
- **Off-target:**
  - Dryad ht76hdrk7 (IFC of CD4 T / B-LCL conjugates with CD3, HLA-DR, brightfield; 125 GB): an immunological-synapse study, not WBC typing.
  - S-BIAD2344 Blood Cell Painting (CC0, 50M PBMCs): dyes only, no CD markers.
  - Cebinelli et al. 2026 (Mendeley rhjd7kg5by / s7vgdkx5dg; BD FACSDiscover A8 CellView with a 10-marker panel incl. CD45/CD14/CD19/CD56, CC BY 4.0): whether image files are included is [unverified], and the file list was empty via the API.
  - C-NMC 2019 (TCIA): labels by expert oncologists, not flow.
  - IDR: no blood or WBC studies.

## 5. Suggested pitch (one paragraph)
"BSCCM is the only public resource that pairs Romanowsky morphology (15.7k cells) with LED-array label-free images, and label-free images with an 8-marker panel (~350k cells), across the same donor pool. We close the missing GS <-> MK edge transitively (GS->LF->MK) and check each edge externally:
- LF->MK on two independent instruments: S-BIAD452 (ImageStream, same-cell BF/SSC + 9 markers, 5 shared with BSCCM) and S-BIAD505 (slide-based BF/oblique/phase-gradient + CD45/CD3/CD19);
- class-level GS morphology against expert-labelled MGG atlases (PBC, MLL23).
No public dataset measures Romanowsky morphology and surface markers on the same blood cell, so BSCCM plus this chain is the closest achievable ground truth."
