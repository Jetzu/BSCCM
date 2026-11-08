# Strand D: hematology-specific literature for BSCCM tri-modal alignment and Giemsa-to-marker prediction

Compiled 2026-11-08 by a literature subagent. Each claim carries a URL. Where only an abstract, a search snippet or a secondary source was available, that is said. `[unverified]` means the full text was not read, or the claim is my inference and not stated in a source. `[inference]` marks my own reasoning.

---

## 0. TL;DR

1. **Few papers use BSCCM, and none use its Wright/histology modality or a donor/slide-held-out split.** Only three papers actually work with the data:
   - the Waller lab's own information-theory paper (DenseNet121 that predicts the distribution of all 8 markers);
   - Nazir and Behera, ICPR 2026 (DPC to 3 classes plus 4 marker regressions) and ISBI 2026;
   - a dictionary-learning preprint (unsupervised lymphoid vs myeloid).

   The best published per-marker Pearson r from label-free DPC, on BSCCMNIST 28x28 crops with a cell-level split, are CD16 0.82, CD45 0.80, CD3/CD19/CD56 (merged channel) 0.77 and CD123/HLA-DR/CD14 (merged) 0.34. These numbers mix between-lineage and within-lineage variance.
2. **B vs T vs NK from label-free or stained morphology is only moderately possible.** Typical results are 75-90 % F1 or accuracy for B vs T, and CD4 vs CD8 is near 60-85 %. The modalities behind these numbers are imaging flow brightfield/darkfield, quantitative phase, and 3D refractive index (mouse). I found **no human study that pairs Romanowsky/MGG morphology with same-cell B/T/NK immunophenotype at scale.** The closest are a 2015 antibody-microarray capture study (Romanowsky staining after capture) and a canine lymphoma study.
3. **Genetics from smears works for AML; lymphoid neoplasms are mostly detectable only when cell counts are high.**
   - NPM1 AUROC 0.92 (Eckardt 2022).
   - cAItomorph (Marr lab, 6,115 patients): CLL sensitivity 0.80, but other B-cell neoplasms 0.26 and "reactive changes" 0.45. Lymphoma probability correlates with lymphocyte ratio, which is a direct warning for any CLL-vs-reactive claim.
4. **Domain shift between blood-smear datasets is severe and is not only color.**
   - Supervised cross-dataset accuracy collapses to about 0-50 % (Salehi 2022). Imbalanced-DG macro-F1 on an unseen MLL domain (INT-20) is about 0.40-0.46.
   - Frozen foundation models trained/probed on Acevedo (CellaVision) score macro-F1 0.41-0.70 on MLL23 (Metafer), compared with 0.98-0.997 in-domain.
   - DinoBloom is invariant to synthetic stain perturbations and still collapses on Metafer. Label-shift-aware feature re-standardization (CBR) helps a little; batch-statistics TTA hurts under realistic class priors.
5. **Hematology foundation models:** DinoBloom (MICCAI 2024) is the reference. Successors from the Marr lab are CytoSAE, RedDino, cAItomorph and GenBloom (genetics-aligned patient encoder, MICCAI 2026). **No model called "SuperBloom" could be found.** External: CytoDINO, CytoDiffusion, Uni-Hema, DeepHeme/CAREMIL.
6. **Expected marker visibility** [inference supported by the biology below]:
   - Lineage-defining markers that coincide with Romanowsky classes should transfer well: CD14 (monocytes), CD16 (neutrophils+, eosinophils-), CD123 (basophils), CD45 (intensity tracks lineage).
   - Lymphocyte sub-lineage markers (CD3, CD19, CD56) should be weak-to-moderate, carried mostly by large-granular-lymphocyte morphology (NK/CD8) and by cell size.
   - Activation-state markers (HLA-DR on T cells, CD16 on monocytes, CD56 on T cells) are expected to be largely invisible.
7. **The strongest external ground truth is same-sample flow cytometry at patient level.** The MLL has both cytomorphology and a flow archive of more than 18k B-NHL cases. Natural experiments are also available:
   - CD19 CAR-T or anti-CD20 B-cell aplasia as a CD19≈0 negative control;
   - CLL;
   - LGL leukemia;
   - CMML;
   - basophilia (CML);
   - the morphology-plus-marker-labelled MLL23 classes (hairy cells, LGL, reactive lymphocytes).

---

## (a) Papers that use BSCCM (citations of arXiv 2402.06191)

Search covered:
- Semantic Scholar citations API: 4 citing papers;
- arXiv full-text search for "BSCCM": 3 hits;
- web search;
- the BSCCM project page, which lists no third-party uses ([waller-lab.github.io/BSCCM](https://waller-lab.github.io/BSCCM)).

| Paper | What they used | What they achieved | Caveats |
|---|---|---|---|
| **Pinkard, Liu, Nyatigo, Fletcher, Waller 2024, the dataset paper** ([arXiv 2402.06191](https://arxiv.org/html/2402.06191v1)) | 412,941 cells; 22 LED patterns (BF/DF/DPC) plus single-LED images; 8 antibodies (CD19, HLA-DR, CD45, CD16, CD56, CD3, CD123, CD14), several sharing fluorophores, unmixed into fewer channels; Wright stain on a subset, same coverslip, after LED and fluorescence imaging | Defines 3-class and 10-class labels from manually gated batch-0 all-antibody cells using CD45, CD16 and CD3/CD19/CD56. CD123/CD14/HLA-DR was judged too noisy. Reports no marker-prediction baseline. Only a DenseNet201 crop filter (~80 %) | Limitations listed in the paper: CD19 at 35 % concentration in batch 1; bleedthrough and underdetermined unmixing; slide-to-slide and batch intensity drift of unknown cause; apoptotic dumbbell lymphocytes often excluded |
| **Pinkard, Kabuli, Markley, Chien, Jiao, Waller, "Information-driven design of imaging systems"** ([arXiv 2405.20559](https://arxiv.org/html/2405.20559)) | BSCCM single-LED, DPC and BF contrasts | Information estimates generally tracked prediction performance. DPC gave more accurate protein predictions than single-LED, even though single-LED had higher (inflated) information estimates. The authors attribute this to optical imperfections and task-specific information | No numeric per-marker results in the parts I could read |
| **Nazir & Behera, ICPR 2026, "Towards Label-Free Single-Cell Phenotyping Using Multi-Task Learning"** ([arXiv 2605.14717](https://arxiv.org/html/2605.14717); code [github.com/saqibnaziir/Single-Cell-Phenotyping](https://github.com/saqibnaziir/Single-Cell-Phenotyping)) | BSCCMNIST 28×28, DPC only (4 channels). 3 classes: Lymph/Gran/Mono. Test set 1,418 cells (456/736/226). Hybrid CNN+ViT with gating; an LLM writes summaries | Accuracy 91.3 % (best baseline: AttentionCNN 89.6 %). Pearson r: CD16 0.819, CD45 0.799, CD3/CD19/CD56 0.768, CD123/HLA-DR/CD14 0.339 (Table 4). Aggregate r 0.726 | Split granularity (cell vs slide/donor) not stated. Marker list inconsistent between Sec. 3 and Table 4. The abstract says CD16 r = 0.72 while Table 4 says 0.819. The r values include between-lineage variance. Low resolution (28 px) |
| **Nazir & Behera, ISBI 2026, "Hybrid Inception-ViT Networks for Fine-Grained Single-Cell Image Classification"** ([Edge Hill record](https://research.edgehill.ac.uk/en/publications/hybrid-inception-vit-networks-for-fine-grained-single-cell-image-/); DOI 10.1109/ISBI61048.2026.11515896) | BSCCM, 3 classes | Class recalls: lymphocyte 90.31 %, granulocyte 97.97 %, monocyte 81.21 % | Abstract only |
| **Altuntac 2026, learned TV/non-negative dictionaries** ([arXiv 2604.05211](https://arxiv.org/html/2604.05211)) | BSCCM-tiny (1,000 cells), DPC L/R/T/B plus BF | Unsupervised lymphoid vs myeloid: ARI 0.575, NMI 0.471 | Evaluated on only N=28 expert-labelled cells. The ARI bootstrap CI [−0.11, 0.71] includes 0 |
| Turner et al., SPIE BiOS 2026, "Cellular fluorescence microscopy aberration correction with focus-diverse machine learning" (DOI 10.1117/12.3080622, via the Semantic Scholar citation list) | Cites BSCCM | Whether it uses the data is `[unverified]` (paper not found) | |

**Implications for the project** [inference]:
- (i) No published BSCCM work uses the Wright modality, aligns modalities, or splits by slide/batch/donor. This is open ground, and also means there is no external number to compare against.
- (ii) Nazir's per-marker r gives a rough ceiling and floor for the LED-array to marker bridge. Recompute it within lineage, because within-lineage r is what "digital immunophenotyping" needs.
- (iii) In the all-antibody slides, CD3/CD19/CD56 and CD123/HLA-DR/CD14 are merged channels (consistent with Nazir's Table 4 and the dataset paper's note on shared fluorophores). Per-marker identity comes only from single-antibody slides.

---

## (b) Label-free WBC imaging and marker or phenotype prediction

### b1. LED-array / Fourier ptychography / learned illumination
- **Learned illumination for classification:** Muthumbi et al. 2019 ([arXiv 1709.07223](https://ar5iv.arxiv.org/html/1709.07223)) jointly optimized LED-array patterns with a CNN for malaria, gaining up to 5-10 % accuracy over standard lighting. Cooke et al. ([arXiv 2004.04306](https://arxiv.org/abs/2004.04306)) learned LED patterns that improve **virtual fluorescence** inference from unstained transmission images. The benefit grows with the information content of the target. Precedent for LED-array to marker prediction.
- **FPM WBC classification:** Hassini et al., Sci Rep 2025 ([Europe PMC record](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Enhancing%20classification%20of%20rare%20white%20blood%20cells%20in%20FPM%22&resultType=core&format=json)) used FPM intensity plus phase on smears for 5-class WBC classification. A physics-inspired GAN supplied synthetic basophils (+5 % precision). Older: Chung et al. 2015 FPM WBC counting ([PLoS One](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0133489)).
- **Lineage from brightfield before marker onset (Marr lab):** Buggenthin et al., Nat Methods 2017 ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Prospective%20identification%20of%20hematopoietic%20lineage%20choice%20by%20deep%20learning%22&resultType=core&format=json)) predicted hematopoietic lineage choice from brightfield patches plus motion, up to three generations before molecular markers appear.

### b2. Quantitative phase, refractive index tomography and holography

| Study | Modality / species | Ground truth | Result |
|---|---|---|---|
| Yoon et al., Sci Rep 2017 ([bioRxiv full text](https://www.biorxiv.org/content/10.1101/107805v1.full-text)) | 3D RI tomography, **mouse** | Sorted B, CD4 T, CD8 T | B vs T test accuracy 89.8 %; CD4 vs CD8 84.4 %; 3-class 75.9 % (k-NN) |
| Ryu et al., BME Frontiers 2021 ([PMC10521749](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10521749/)) | RI tomograms + 3D CNN, human bone marrow | Sorted populations | Myeloid vs lymphoid >99 %; 4-class (mono, myelocyte, B, T) >96 % (abstract) |
| Shu et al., Adv Intell Syst 2021 ([arXiv 2012.08518](https://arxiv.org/abs/2012.08518)) | QPM, human, 6 donors | **Negative isolation kits** per population | 4-class average 90.5 %. Cross-donor F1: B 83.7 %, T 81.4 %; mono 91.6 %, gran 94.3 %. CD4/CD8 F1 80.4/77.5 (same donor). Quotes BF/DF baseline B/T F1 79.4/75.7. The authors note that kit impurity limits accuracy |
| Fanous et al., Sci Rep 2022, PICS ([DOAJ](https://doaj.org/article/c0f0a46455f14df3bb6bb19063dd21b4)) | cSLIM phase → **virtual Wright stain**, then segmentation and classification | Pathologist labels, 18 smears | Majority-vote F1 80 % |
| Ugele et al., Adv Sci 2018 ([PMC6299719](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6299719/); [Fraunhofer](https://publica.fraunhofer.de/handle/publica/254753)) | Flow digital holographic microscopy | Clinical diagnosis | 9 leukocyte types separated. 20 blinded clinical samples (AML/ALL/CLL/MPN): 70 % correct, 20 % close |

**Caveat** [inference, supported by Shu's own discussion and by Islam et al. below]: most of these studies isolate or sort each population and image it separately. The class is then confounded with acquisition session. BSCCM's single-antibody-per-slide design has a milder version of the same issue for marker regression.

### b3. Deep-UV microscopy and pseudo-/virtual Giemsa (Robles lab)
- **Ojaghi et al., PNAS 2020** ([Europe PMC abstract](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1073/pnas.2001404117&resultType=core&format=json)):
  - deep-UV multi-wavelength imaging of unstained smears;
  - five-part WBC differential, RBC/Hb characterization and platelets;
  - pseudocolorization that mimics Giemsa/conventional stains for blood smears and bone marrow aspirates;
  - hematologists evaluated it blindly (healthy, thrombocytopenia, sickle cell);
  - per-class accuracy numbers `[unverified]`, full text not reached.
- **Kaza, Ojaghi, Robles, BME Frontiers 2022** ([PMC10521747](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10521747/); [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.34133/2022/9853606&resultType=core&format=json)): single-wavelength deep-UV, with separate networks for virtual staining (Giemsa-like) and segmentation, then a five-part differential classifier. Numbers `[unverified]`.
- Related: a neutropenia detection SPIE paper ([SPIE](https://biomedicaloptics.spiedigitallibrary.org/conference-proceedings-of-spie/11651/116510O/Application-of-deep-ultraviolet-microscopy-in-hematological-assessment-of-neutropenia/10.1117/12.2576929.full)) and a patent on UV-to-RGB colorization ([USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12136284)).
- **Relevance** [inference]: BSCCM has same-cell registered LED-array and Wright images, so it is a natural paired dataset for **virtual Wright staining from LED-array images**. This is the label-free counterpart of the Robles/Popescu work. A virtual-stain model would let the ~350k marker-stained LED-array cells be rendered as pseudo-Giemsa (see the protocol section).

### b4. Stain-free imaging flow cytometry with CD-marker ground truth
- **Nassar et al., Cytometry A 2019** ([PMC6767740](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6767740/); [PDF](https://carpenter-singh-lab.broadinstitute.org/files/anne/files/123_Nassar_CytometryA_2019.pdf)):
  - ImageStream brightfield + darkfield;
  - **one FITC antibody per tube (CD3, CD14, CD15, CD19), the same single-antibody design as BSCCM**;
  - 13 stained donors, leave-one-donor-out CV;
  - 4 main types, F1 about 97 %; **B vs T F1 about 78 %**;
  - applied to 85 unstained donors;
  - B vs T features: overall shape, plus brightfield intensity and granularity.
- **Lippeveld et al., Cytometry A 2020** ([bioRxiv 680975](https://www.biorxiv.org/content/10.1101/680975v1.full)):
  - ImageStream BF + SSC/darkfield;
  - 9-colour gating panel (CD15, Siglec8, CD14, CD19, CD3, CD45, CD4, CD56, CD8);
  - 8 classes, 2 donors, **cell-level 5-fold CV (no donor hold-out)**;
  - balanced accuracy 0.778 (gradient boosting) vs 0.703 (DL);
  - neutrophil/monocyte/eosinophil recall >0.95, but CD4 0.61, CD8 0.59; DL B-cell recall 0.39-0.63; T vs NKT confused.
  - The authors conclude that BF/DF may lack the information for fine subtypes.
- **Doan et al., Cytometry A 2020** ([PMC7213640](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7213640/)): BF/DF features alone identified ALL cells at >88 % accuracy. Ground truth was an ALL-discriminating antibody panel.
- **Ghost cytometry, Ugawa et al., eLife 2021** ([eLife 67660](https://elifesciences.org/articles/67660)):
  - "in-silico-labelled" WBC classification from stain-free diffraction, scatter and BF signals;
  - ground truth from **CD45, CD123, CD14, CD16, CD34**, 4 of which are in the BSCCM panel;
  - 6 classes (neutrophil, lymphocyte, monocyte, eosinophil, basophil, spiked CD34+);
  - **inter-donor** macro-F1 0.88-0.90.
- **Goda/Nitta lab:**
  - Intelligent image-activated cell sorting (Nitta et al., Cell 2018; [EurekAlert](https://www.eurekalert.org/news-releases/553214)) demonstrated platelet aggregates, not WBC markers.
  - Virtual OTS-QPI (brightfield to phase GAN) supports WBC/leukemia classification ([APL Photonics 2020](https://doaj.org/article/dcede6f2a599400687723bbb57bc949b)).
  - I found **no Nitta/Goda paper predicting WBC CD markers** `[unverified absence]`.
- **Sorted-population caution:** Islam et al. 2025 ([arXiv 2503.12622](https://arxiv.org/html/2503.12622v2)) report 98 % T4 vs B and 93 % zero-shot T8 vs B from DIC images. The cells were FACS-sorted mouse splenocytes, seeded and imaged after sorting, and the paper does not report balance of acquisition conditions across classes. Treat as a likely-confounded upper bound.

---

## (c) Immunophenotype or genetics from Giemsa/MGG/Pappenheim smears

### c1. Genetics and diagnosis from smears (patient-level MIL)
- **Eckardt et al., Leukemia 2022** ([PMC8727290](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8727290/)): bone marrow smears, 1,251 AML patients and 236 controls. AML vs healthy AUROC 0.97; **NPM1 mutation AUROC 0.92**. Occlusion maps point to condensed chromatin and perinuclear clearing in NPM1-mutant blasts.
- **Kockwelp et al., Blood Adv 2024** ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22predicts%20therapy-relevant%20genetics%20in%20acute%20myeloid%20leukemia%22&resultType=core&format=json)): Pappenheim bone marrow smears, more than 2M cells from 408 patients, plus a temporal test set of 71 patients. Therapy-relevant genetics predicted with "high AUROC" (values `[unverified]`).
- **Hehr et al. (Marr lab), PLOS Digit Health 2023, SCEMILA** ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Explainable%20AI%20identifies%20diagnostic%20cells%20of%20genetic%20AML%20subtypes%22&resultType=core&format=json)): 129 AML patients across 4 genetic subtypes plus 60 controls, about 80k cells. AML vs healthy separated perfectly; APL F1 0.86.
- **Dasdelen et al. (Marr lab), cAItomorph** ([arXiv 2509.20402](https://arxiv.org/abs/2509.20402); published as a Leukemia 2026 letter, DOI 10.1038/s41375-026-02934-1, [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22hematological%20malignancy%20prediction%20from%20peripheral%20blood%20smears%20in%20a%20large%20diagnostic%20laboratory%20cohort%22&resultType=core&format=json)):
  - **6,115 MLL patients plus 495 controls**, Wright-Giemsa smears scanned on Metafer (10x overview, 40x cells), 478±55 cells per patient;
  - ground truth from bone marrow workup including **immunophenotyping**;
  - 8-class accuracy 0.72 (top-2 0.87); healthy sensitivity 0.98;
  - **CLL sensitivity 0.80 (12/15) vs 0.26 for other B-cell neoplasms**; lymphoma 0.40; reactive changes 0.45;
  - lymphoma probability correlates with blood lymphocyte ratio;
  - on the external APL-AML set ("completely out-of-domain" staining), acute-leukemia sensitivity 0.76.
- **Singi et al. (MSK), Commun Med 2026, CAREMIL** ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Interpretable%20Multiple%20Instance%20Learning%20for%20Hematologic%20Diagnosis%20from%20Peripheral%20Blood%20Smears%22&resultType=core&format=json)): DeepHeme cell encoder + attention MIL on peripheral blood smears. AUROC AML 0.999, MDS 0.891, **hairy cell leukemia 0.945**. Hairy cells, a CD19+ neoplasm, are therefore morphologically recognizable.

### c2. CLL vs reactive and atypical lymphocytes
- **Barcelona group** (the Acevedo dataset group; CellaVision DM96):
  - Alférez et al. 2016 ([zendy](https://zendy.io/title/10.1111/ijlh.12473)): normal vs abnormal-lymphoid vs reactive, 97.67 %; abnormal subtypes (CLL, HCL, MCL, FL, prolymphocytes) 91.23 %.
  - Alférez 2015 AJCP ([PubMed 25596242](https://www.pubmed.gov/25596242/)): validation set 85.33 %.
  - COVID reactive lymphocytes: sensitivity 90.5 %, specificity 99.4 % ([Bioengineering 2022 / J Clin Pathol 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9137554/)).
  - Numbers from abstracts or snippets.
- **cAItomorph** (above): CLL is detected mainly when lymphocyte counts are high. Any CLL vs reactive-lymphocytosis evaluation must be **matched or stratified on lymphocyte fraction**, otherwise "marker" predictions may just be lymphocyte counting [inference].
- **Biology of the comparison:**
  - CLL requires ≥5×10⁹/L clonal B cells co-expressing CD5, CD19 and CD23 (iwCLL 2018, [PDF](https://www.chip.dk/media/vqwfefz2/iwcll-guideline-hallek-et-al-2018.pdf)).
  - Reactive (EBV/IM) atypical lymphocytes are predominantly **activated CD8+ T cells**, with very few CD19+ cells. This comes from secondary sources ([ASH image bank](https://imagebank.hematology.org/image/64924/downey-cells-in-infectious-mononucleosis-decoding-the-role-of-lymphocyte-abnormalities-in-diagnosis), [Wikipedia](https://en.wikipedia.org/wiki/Reactive_lymphocyte)); primary immunophenotyping paper not retrieved.
  - So predicted CD19 vs CD3 within lymphocytes is exactly the discriminating axis, which makes it a good **biologically grounded test**.

### c3. Can B, T and NK lymphocytes be told apart morphologically?

| Evidence | Modality | B vs T | Other |
|---|---|---|---|
| Nassar 2019 | IFC BF/DF, human, leave-donor-out | F1 ~78 % | |
| Lippeveld 2020 | IFC BF/SSC, human, cell-level CV | B recall 0.39-0.63 (DL) | CD4 0.61 / CD8 0.59 recall; T vs NKT confused |
| Shu 2021 | QPM, human, cross-donor | F1 84/81 % | CD4/CD8 F1 80/78 % (same donor) |
| Yoon 2017 | 3D RI, mouse | 89.8 % | CD4/CD8 84.4 % |
| Canine lymphoma, Wright-Giemsa cytology, ResNet-50 ([Springer 2026](https://link.springer.com/article/10.1186/s44356-026-00056-5)) | **Romanowsky**, dog lymph node | 66-80 % test accuracy (from search snippet `[partially verified]`) | lymphoma vs reactive 85-95 % |

**Classical morphology:**
- Strokotov et al. 2009 ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Is%20there%20a%20difference%20between%20T-%20and%20B-lymphocyte%20morphology%22&resultType=core&format=json)): B cells are on average **larger**, but the difference is smaller than the variation between cells.
- Large granular lymphocytes are 2-6 % of leukocytes, and NK/ADCC activity is "exclusively associated" with LGL-enriched fractions. About half of LGL are FcγR+ T cells (Timonen et al., J Exp Med 1981, [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1084/jem.153.3.569&resultType=core&format=json)).
- In the antibody-microarray study below, Romanowsky-stained **granular lymphocytes** were CD3+, CD8+, CD16+ and CD56+, and CD4−, CD19−.
- **Conclusion** [inference]: on Giemsa, NK and cytotoxic CD8/CD56/CD16 cells are partly recognizable through granules (LGL morphology). B vs T is a weak, mainly size- and chromatin-based signal. CD4 vs CD8 is essentially invisible. Expect cell-level CD19/CD3 prediction AUROC well below the lineage markers. Patient-level fractions may still be estimable by aggregation; this must be tested.
- I found **no human study** with Giemsa/MGG images paired to same-cell CD3/CD19/CD56. MSK's CAR-T work maps six lymphocyte *morphotypes* to outcome in about 600 patients, but presentation-level information only ([VJHemOnc](https://www.vjhemonc.com/video/hkvznpq7b9c-deep-learning-based-profiling-of-car-t-immune-dynamics-from-routine-peripheral-blood-smears/)) `[unverified details]`.

### c4. Datasets or protocols with the same cell in morphology and immunophenotype
- **BSCCM itself**: LED-array + fluorescence on stained cells; LED-array + Wright on (per the project's data check) unstained cells. Giemsa and markers never co-occur, so the bridge is two-hop.
- **Antibody-microarray capture then Romanowsky/cytochemistry**, Khvastunova et al., Sci Rep 2015 ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/srep12573&resultType=core&format=json)):
  - cells are captured on anti-CD spots, then stained, giving a "sorted smear";
  - morphology matches standard smears;
  - spot proportions correlate highly with flow cytometry;
  - granular lymphocytes are CD3+CD8+CD16+CD56+, CD4−CD19−;
  - **the most direct same-cell Romanowsky + CD precedent found**, and a feasible lab protocol for building a validation set.
- **Destain-and-restain:**
  - Immunocytochemistry on destained MGG/Giemsa air-dried smears is unreliable. Kushwaha et al. 2022 ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Evaluation%20of%20immunocytochemistry%20on%20destained%20giemsa%22&resultType=core&format=json)): CD45/LCA optimal in 14.3 % of air-dried vs 85.7 % of wet-fixed smears.
  - Veterinary post-Romanowsky ICC sometimes works ([Springer 2022](https://link.springer.com/article/10.1007/s11259-022-09953-0)).
  - The classic MAC (morphology-antibody-chromosomes) method does MGG plus immunophenotype on the same cell ([protocol PDF](https://api.naturalis.fcnym.unlp.edu.ar/server/api/core/bitstreams/01b06ddb-0574-479f-a373-c59126e2bb48/content)).
  - **Recommendation** [inference]: if a same-cell clinical-stain validation set is built, do immunofluorescence first and then MGG/Pappenheim, in BSCCM's order. Do not destain.
- **Imaging flow cytometry** (Nassar, Lippeveld, Doan) gives same-cell BF/DF + CD markers but no Romanowsky stain.
- **Separate-slide IF on blood smears** is clinically feasible: immunofluorescence blood smears identify MPN ([PMC11347374](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11347374/)), using one MGG smear plus separate antibody-labelled smears.
- **CAR-T expansion** produces atypical lymphocytes with characteristic morphology, alongside **B-cell aplasia** (Faude et al., Blood Adv 2021, [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=PMCID:PMC8095130&resultType=core&format=json)). Useful as a patient-level CD19≈0 control.

---

## (d) Hematology foundation models and domain shift

### d1. Foundation models

| Model | Group | Key facts |
|---|---|---|
| **DinoBloom** S/B/L/G, MICCAI 2024 ([arXiv 2404.05022](https://arxiv.org/html/2404.05022); [code](https://github.com/marrlab/DinoBloom)) | Marr lab | DINOv2 adapted for single cells (global-local crop loss removed). 13 datasets, ~380k WBC, including BMC 171k, AML Hehr 81k, a 40k "LA" peripheral blood set (MLL23 per Sharma & Tapadiya below), Matek, Raabin and others. **Acevedo held out**: 1-NN wF1 89.1 (G) vs 77.8 (DINOv2-G); linear probe 91.8 vs 90.1. BMC 1-NN wF1 ~80 vs 49 (CTransPath). AML-Hehr ABMIL wF1 93.1. The paper does not describe stain augmentation |
| **CytoSAE**, MICCAI 2025 ([arXiv 2507.12464](https://arxiv.org/html/2507.12464v1)) | Marr lab | Sparse autoencoder on DinoBloom-B patch tokens, trained on MLL23. Concepts validated on Acevedo, Matek19, BMC, AML Hehr. Patient "concept barcodes" |
| **RedDino** ([arXiv 2508.08180](https://arxiv.org/pdf/2508.08180)) | Zedda, Marr co-author | RBC model, >3M RBC crops. Surprisingly the **best frozen encoder for Acevedo→MLL23 WBC transfer** (below) |
| **GenBloom**, "Genetically Aligned Patient Representations", MICCAI 2026 ([arXiv 2605.29980](https://arxiv.org/abs/2605.29980); [github.com/marrlab/GenBloom](https://github.com/marrlab/GenBloom)) | Marr lab | iBOT-style vision pretraining of a patient aggregator on >1,500 patients, then a **supervised-contrastive alignment of smear representations to karyotype and mutation profiles**. Outperforms slide-level histopathology FMs. Closest methodological precedent for aligning smear morphology with another modality. No flow data |
| "SuperBloom" | – | **Not found** in web, arXiv or GitHub searches; possibly an internal name `[unverified]` |
| CytoDINO ([arXiv 2512.17930](https://arxiv.org/html/2512.17930)) | external | DINOv3 adapted to bone marrow; wF1 88.2 vs 85.7 (DinoBloom-S) |
| CytoDiffusion, Nat Mach Intell 2025 ([arXiv 2408.08982](https://arxiv.org/html/2408.08982v2)) | Cambridge | Diffusion classifier. Domain shift Raabin→LISC balanced accuracy 85.9 vs 74.4 (EfficientNetV2-M) vs 62.3 (ViT-B). Expert-indistinguishable synthetic cells (52.3 %) |
| Uni-Hema ([arXiv 2511.13889](https://arxiv.org/html/2511.13889)), DeepHeme (Goldgof/MSK, used in CAREMIL) | external | Unified detection, segmentation and VQA; a bone-marrow cell encoder |
| HASSL ([arXiv 2607.04353](https://arxiv.org/abs/2607.04353)) | Marr co-author | Hierarchy-aware SSL on 2.3M cells from 20 datasets. Not hematology-specific |

### d2. Measured domain shift between the target clinical datasets

**Dataset acquisition facts:**
- **Acevedo/PBC:** CellaVision DM96, Hospital Clinic Barcelona, normal donors, 17,092 cells, 360×363 px ([Data in Brief](https://www.pubmed.gov/32346559/)).
- **Matek-19 / AML-Cytomorphology_LMU:** Precipoint M8, 100x oil, 14.14 px/µm, 18,365 cells, 100 AML + 100 controls ([TCIA wiki](https://wiki.cancerimagingarchive.net/x/fgWkAw)).
- **MLL23:** 41,906 cells, 288×288 px = 25 µm (11.5 px/µm), 18 classes including **LGL, hairy cells, reactive lymphocytes, other neoplastic lymphocytes**. Metafer scanner per [Sharma & Tapadiya](https://arxiv.org/html/2608.25148) and the cAItomorph methods. Sources: [GitHub](https://github.com/marrlab/MLL23); Boushehri et al., Sci Data 2025 ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41597-025-06223-x&resultType=core&format=json)). The repo ships a label-harmonization notebook for Matek19/Acevedo20.
- **BMC:** 171,374 bone-marrow cells, 945 patients, 21-22 classes, MGG/Pappenheim ([Matek 2021 Blood](https://dx.doi.org/10.1182/blood.2020010568); stain per the DinoBloom paper).

**Results:**
- **Salehi et al., MICCAI 2022** (Marr lab, [ar5iv 2207.00501](https://ar5iv.labs.arxiv.org/html/2207.00501)). Accuracy, 13 harmonized classes:
  - Supervised oracle trained on Matek: 96.1 % on Matek, but 29.5 % on INT-20 and 8.1 % on Acevedo.
  - ResNeXt trained on INT-20: 88.7 % in-domain, 49.0 % on Matek, 16.9 % on Acevedo.
  - The unsupervised cross-domain autoencoder AE-CFE plus random forest beat the oracles off-domain (e.g., INT-20→Matek 73.2 %), but stayed low (21-48 %) for other pairs.
- **Umer et al., ICLR-W 2023** (Marr lab, [arXiv 2303.07771](https://arxiv.org/pdf/2303.07771v3)): train on Matek + Acevedo, test on unseen INT-20 (MLL in-house). Macro-F1 in-source 0.77, unseen ERM 0.40, CORAL 0.43, their loss 0.43-0.46.
- **Sharma & Tapadiya 2026, "Can You Trust Frozen Hematology Foundation Models under Acquisition Shift?"** ([arXiv 2608.25148](https://arxiv.org/html/2608.25148)):
  - Source Acevedo (CellaVision), 5 shared classes, 15 frozen encoders with linear probes.
  - In-domain macro-F1 0.98-0.997. Cross-dataset drop 34-72 %.
  - MLL23 (Metafer): RedDino 0.704, DinoBloom-S 0.671, DINOv2-B 0.650, DinoBloom-B 0.553, DinoBloom-L 0.552; supervised ResNet-18 0.255.
  - Matek: 0.36-0.65. Raabin: 0.10-0.45.
  - ECE 0.004 in-domain vs 0.35 off-domain ("confidently wrong").
  - **DinoBloom is invariant to synthetic stain/colour perturbations yet collapses on Metafer**, so the shift is not only color.
  - MLL23 is part of DinoBloom's pretraining, so exposure and shift are confounded.
  - Mitigation: label-shift-aware **class-balanced re-standardization (CBR)**, +0.059 macro-F1 and harmful in 0/18 scenarios. Global target standardization: +0.032, harmful in 6/18. **SHOT/IM test-time adaptation: −0.029, harmful in 10/18.**
- **Tsutsui et al., ICASSP 2023** ([arXiv 2303.01777](https://arxiv.org/abs/2303.01777)): ResNet WBC generalization is poor *because of batch normalization*; alternative normalizations help.
- **WBCBench 2026** (ISBI challenge, [site](https://xudong-ma.github.io/WBCBench2026-Robust-White-Blood-Cell-Classification/)): 55,012 images, 13 classes, synthetic scanner shift (noise, blur), patient-split test.
  - Winner: DinoBloom + DINOv3 hierarchical ensemble + pseudo-labelling, macro-F1 0.777.
  - Third: ResNet ensemble + Macenko normalization + 8× TTA, 0.708.
- **Older:** Raabin→LISC accuracy fell from 93.97 % to 51.57 % ([bioRxiv 2021.05.12.443717](https://www.biorxiv.org/content/10.1101/2021.05.12.443717v3)). SADA stain-aware alignment ([arXiv 2412.02976](https://arxiv.org/abs/2412.02976)), numbers `[unverified]`.
- **Stain augmentation (Marr lab, histopathology):** HistAuGAN structure-preserving multi-domain stain augmentation, MICCAI 2021 ([arXiv 2107.12357](https://arxiv.org/abs/2107.12357)).

**BSCCM-specific domain gap** (from the dataset paper, [arXiv 2402.06191](https://arxiv.org/html/2402.06191v1)):
- BSCCM Wright cells are RBC-lysed, **paraformaldehyde-fixed cells settled on poly-L-lysine coverslips**, Wright-stained in Coplin jars after LED and fluorescence imaging. They are imaged at **63×/1.4 oil** with **sequential R/G/B LED illumination on a monochrome camera**, 398×398 px crops.
- Clinical smears differ:
  - wedge smears with an RBC background;
  - methanol fixation (standard for Romanowsky, e.g. [Creighton protocol](https://creighton.edu/sites/default/files/2024-03/Staining_with_Wrights_Stain.pdf)); fixation shifts Giemsa colour balance ([StainsFile](https://www.stainsfile.com/protocols/romanowsky-giemsa-general-oversight-stain/));
  - MGG/Pappenheim/Wright-Giemsa;
  - colour CCD scanners (CellaVision, Metafer, M8).
- Cell spreading differs between settled suspensions and smears `[inference]`. The BSCCM histology pixel size is not stated. My estimate is ~0.053 µm/px (3.45 µm camera pixel / 63×), giving ~21 µm crops `[inference, verify from metadata]`.
- **Expectation** [inference]: BSCCM→clinical will be a larger shift than any clinical→clinical pair above, because preparation, fixation, optics and colour formation all differ.

---

## (e) Marr-lab related work (chronological, hematology)
- Buggenthin 2017 (lineage before markers, brightfield) ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Prospective%20identification%20of%20hematopoietic%20lineage%20choice%20by%20deep%20learning%22&resultType=core&format=json)).
- Matek 2019 NMI, AML blasts at human level, AML-Cytomorphology_LMU ([TCIA](https://wiki.cancerimagingarchive.net/x/fgWkAw)).
- Sadafi 2020 attention-MIL for blood disorders ([arXiv 2007.11641](https://export.arxiv.org/api/query?search_query=au:Marr_Carsten&sortBy=submittedDate&sortOrder=descending&start=30&max_results=30)).
- Matek 2021 Blood, BMC ([DOI](https://dx.doi.org/10.1182/blood.2020010568)).
- Wagner 2021 HistAuGAN.
- Salehi 2022 AE-CFE cross-domain.
- Kazeminia 2022/2023 anomaly-aware and topological MIL (rare anemia).
- Umer 2023 imbalanced DG.
- Sadafi 2023 continual cross-domain WBC classification ([arXiv 2308.12679](https://arxiv.org/abs/2308.12679)).
- Hehr 2023 SCEMILA.
- Koch, Wagner et al. 2024 DinoBloom.
- Kazeminia 2024 self-supervised MIL for AML.
- Koch 2024 UNICORN, multi-stain integration with missing-modality handling ([arXiv 2409.17775](https://arxiv.org/abs/2409.17775)); a design pattern for "missing modality" tri-modal data.
- Mert et al. 2024 β-VAE joint morphology + scRNA latent for AML WBC differentiation ([arXiv 2408.06720](https://arxiv.org/abs/2408.06720)); the closest Marr-lab precedent for a **shared morphology-molecular embedding at single-cell level**.
- Deutges 2024/2025 neural-cellular-automata WBC classification and segmentation.
- Boushehri 2025 MLL23 dataset.
- Dasdelen 2025 CytoSAE.
- Carreras Boada 2025 CytoDiff synthesis.
- Zedda 2025 RedDino.
- Ebrahimi 2025 continual MIL.
- Dasdelen 2025/26 cAItomorph.
- Dasdelen 2026 GenBloom.
- Zedda 2026 QG-MIL.
- Dasdelen 2026 patient-embedding re-mixing.

Full list from the [arXiv author query](https://export.arxiv.org/api/query?search_query=au:Marr_Carsten&sortBy=submittedDate&sortOrder=descending&max_results=30). The MLL collaborators (Pohlkamp, Haferlach, Spiekermann) give the lab access to paired cytomorphology and flow cytometry. Zhao et al. 2020 trained on **18,274 MLL flow cytometry cases** of mature B-cell neoplasms, wF1 0.94 ([PubMed 32519455](https://pubmed.ncbi.nlm.nih.gov/32519455/)).

---

## Synthesis

### S1. Which BSCCM markers should be visible in Giemsa morphology?

Classes refer to Romanowsky/Giemsa morphology classes.

**CD45**
- Expressed on: all leukocytes. Intensity is roughly lymphocytes ≥ monocytes > granulocytes; dim on blasts and basophils (standard flow knowledge).
- Morphological handle: the lineage class.
- Expected Giemsa→marker performance: **high between lineages, ~nil within lineage**. Within-lineage variance is mostly technical in BSCCM, which reports slide drift.
- Evidence: Nazir r 0.80 (DPC).

**CD16**
- Expressed on: neutrophils (bright), NK (CD56dim), nonclassical/intermediate monocytes. **Eosinophils negative**; immature granulocytes lower ([Beckman](https://www.beckman.com/resources/cell-types/blood-cells/leukocytes/eosinophils); [JSLRT 2015](https://www.jstage.jst.go.jp/article/jslrt/55/2/55_55/_article)).
- Morphological handle: neutrophil vs eosinophil granules; band vs segmented nucleus; LGL granules; CD16+ monocytes are smaller (13.8 vs 18.4 µm) with lower scatter (Passlick 1989, [Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22Identification%20and%20characterization%20of%20a%20novel%20monocyte%20subpopulation%20in%20human%20peripheral%20blood%22&resultType=core&format=json)).
- Expected performance: **high** for granulocytes; **moderate** for NK via LGL; **low-moderate** for monocyte subsets.
- Evidence: Nazir r 0.82; ghost cytometry used CD16 as ground truth.

**CD14**
- Expressed on: monocytes (classical bright, nonclassical dim); dim on neutrophils.
- Morphological handle: monocyte morphology.
- Expected performance: **high** at lineage level; **low-moderate** for CD14 level within monocytes.
- Evidence: in BSCCM the merged CD123/HLA-DR/CD14 channel has r 0.34 (noisy).

**CD123**
- Expressed on: **basophils** (HLA-DR−) and **pDC** (HLA-DR+), both bright; low on monocytes and eosinophils ([PMC4935682](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4935682/)).
- Morphological handle: basophil granules are unmistakable in Giemsa; pDC are rare and plasmacytoid.
- Expected performance: **high for basophils**, but rare (<1 %), so few positives. pDC unknown.
- Caveat: the dataset paper flagged this channel as noisy. Activated basophils down-regulate CD123.

**HLA-DR**
- Expressed on: monocytes, B cells, DCs, activated T; basophils negative; ~2× higher on CD16+ monocytes (Passlick).
- Morphological handle: monocyte morphology; B vs T is subtle; activated or reactive lymphocyte morphology.
- Expected performance: **moderate** (lineage-driven); **low** for T-cell activation.

**CD3**
- Expressed on: T cells and NKT.
- Morphological handle: small agranular lymphocyte by default; some CD8 T are LGL.
- Expected performance: **low-moderate** at cell level (B vs T 75-89 % label-free, worse on Giemsa `[inference]`).

**CD19**
- Expressed on: B cells, including CLL, HCL and most B-NHL.
- Morphological handle: slightly larger size; in neoplasia, CLL (small mature cells, smudge cells) and hairy cells are distinctive.
- Expected performance: **low** in normal blood; **moderate-high** for neoplastic B with characteristic morphology (HCL AUROC 0.945 at patient level, CAREMIL).

**CD56**
- Expressed on: NK, NKT, some CD8 T.
- Morphological handle: LGL azurophilic granules (Timonen; Khvastunova).
- Expected performance: **moderate** (granularity); confounded with cytotoxic T.

**Lineage-defining vs activation markers:**
- CD14, CD16 on granulocytes, CD123 on basophils, and CD45 (via lineage) coincide with Romanowsky classes. They should transfer, and are the right anchors for checking the Giemsa→marker bridge.
- CD3, CD19 and CD56 are lineage-defining but morphologically subtle in normal blood.
- **Activation markers should be treated as not-expected-visible** unless proven: HLA-DR on T cells, CD16 on monocytes, CD56 on T cells. Exception: global "reactive/atypical" morphology (activated CD8 T in viral infection; CAR-T expansion).
- **Report marker performance within morphological class.** A pooled Pearson r is dominated by lineage, so a high pooled r for CD16 or CD45 mostly restates the differential count [inference].

### S2. External ground truth that could validate Giemsa→marker predictions
1. **Same-sample flow cytometry, patient level (primary).** Compare per-patient predicted fractions with flow-measured fractions using ICC and Bland-Altman:
   - CD19+ B (cAItomorph has ~480 cells per patient, so binomial SE ≈1.3 % at a 10 % B fraction `[inference]`);
   - CD14+ monocytes;
   - CD16+CD56+ NK;
   - CD123+HLA-DR− basophils.

   The MLL has both modalities routinely (cAItomorph ground truth includes immunophenotyping; the flowCat archive has 18k B-NHL cases). Acevedo, Matek and BMC have **no** paired flow data published. For Acevedo/Matek, only analyzer differentials (CBC 5-part) are plausible.
2. **Natural experiments / biological controls (patient level):**
   - CD19 CAR-T or anti-CD20 B-cell aplasia: predicted CD19≈0 (Faude 2021).
   - CLL (≥5×10⁹/L CD5+CD19+CD23+ B; iwCLL 2018) vs EBV/viral reactive lymphocytosis (CD8 T-dominant).
   - T-LGL/NK-LGL leukemia: CD16/CD56/CD8 high ([AJCP 2024](https://academic.oup.com/ajcp/article/162/5/433/7686433)).
   - CMML (CD14 monocytes).
   - CML (basophilia → CD123).
   - HCL (CD19+, HLA-DR+).
3. **Class-conditional sanity matrix on MLL23 labels (cell level, no new data):**
   - basophils → CD123↑ HLA-DR↓;
   - eosinophils → CD16↓;
   - band/segmented neutrophils → CD16↑;
   - myelocytes/promyelocytes → CD16↓;
   - monocytes → CD14↑ HLA-DR↑;
   - LGL → CD16/CD56↑;
   - hairy and neoplastic lymphocytes → CD19↑ HLA-DR↑;
   - reactive lymphocytes → CD3↑ (activated CD8);
   - myeloblasts → CD45 dim.

   Cheap, and catches gross failures [inference].
4. **Same-cell Romanowsky + marker validation set (gold standard, requires wet lab):**
   - antibody-microarray capture then Romanowsky (Khvastunova 2015); or
   - BSCCM's order (IF first, then **MGG/Pappenheim** on a clinical scanner) on cytospins or smears.
   - Do not rely on destain-then-ICC (Kushwaha 2022).
   - Imaging flow cytometry can validate the **label-free** leg (BF/DF + CD markers, Nassar/Lippeveld design), not the Giemsa leg.

### S3. Recommended transfer protocol (BSCCM-Wright → MLL23 / Matek / Acevedo / BMC) [inference, grounded in d2]
1. **Make physical scale explicit.** Recover BSCCM histology µm/px from metadata. Resample all domains to a common µm/px, e.g. ~0.08-0.1 µm/px to match the clinical scanners (MLL23 0.087, Matek 0.071 µm/px). Crop a fixed physical field of ~25 µm, as MLL23 does.
2. **Neutralize the background difference.** BSCCM has no RBC background. Either segment the WBC and mask the background in all domains (the Marr lab's R-CNN focus in Salehi 2022 and Mert 2024), or paste realistic RBC backgrounds into BSCCM crops. Ablate both.
3. **Colour and optics.** Stain normalization (Macenko/Reinhard) as one arm, and strong HED/HSV jitter or HistAuGAN-style augmentation as another. Synthetic stain jitter alone is **insufficient**: DinoBloom is stain-invariant yet collapses on Metafer. Also augment blur, noise, JPEG and resolution (the WBCBench recipe), and simulate the sequential-LED RGB vs Bayer-camera difference.
4. **Backbone and adaptation.**
   - Start from frozen DinoBloom-S/B and RedDino features. On Acevedo→MLL23 the *smaller* DinoBloom-S and RedDino transferred best.
   - Avoid BN-statistic test-time adaptation under clinical class priors (SHOT/IM hurt in 10/18 scenarios).
   - Use class-balanced re-standardization or LayerNorm ViTs.
   - Recalibrate with source temperature, then re-check ECE on the target.
5. **Two-hop bridge, made explicit.**
   - Learn Giemsa ↔ LED-array on the ~15.7k same-cell unstained pairs.
   - Learn LED-array → markers on the ~350k stained cells, with **slide- or batch-held-out** splits, because no published BSCCM work does this and slide drift is documented.
   - Optionally train a **virtual-Wright** generator on the paired cells (deep-UV/PICS precedent) to create pseudo-Giemsa for all marker-stained cells. Treat the generated cells as augmentation, never as evaluation.
6. **Anchor the shared embedding with cross-domain labels.** The 5 shared WBC classes exist in BSCCM (3/10-class labels) and in all clinical sets (harmonization notebook in the MLL23 repo). Use them as weak supervision or contrastive anchors, e.g. GenBloom-style supervised-contrastive alignment. Marker heads then predict **within-class residuals**.
7. **Evaluate in three tiers:**
   - (i) within-class marker correlation on held-out BSCCM slides and batches;
   - (ii) the class-conditional sanity matrix on MLL23/Matek/Acevedo/BMC (BMC is bone marrow, so expect immature-myeloid CD16↓ and blasts CD45 dim);
   - (iii) patient-level fractions vs MLL flow cytometry, plus CLL vs reactive lymphocytosis **matched on lymphocyte fraction**. Test whether predicted CD19 adds information beyond lymphocyte count; cAItomorph's CLL signal tracks lymphocyte ratio.

---

## Gaps and unverified items
- No human Giemsa/MGG dataset with same-cell CD3/CD19/CD56 was found. BSCCM is the only same-cell Wright + label-free dataset, and it has no same-cell Wright + marker.
- Per-class accuracies for Ojaghi 2020, Kaza 2022 and Kockwelp 2024 were not read `[unverified]`.
- "SuperBloom" was not found.
- Nitta/Goda WBC CD-marker prediction was not found.
- BSCCM histology pixel size is estimated, not sourced.
- The CD8 dominance of IM atypical lymphocytes comes from secondary sources.
- The canine B vs T figures come from a search snippet.
- MSK CAR-T morphotype details come from a talk only.
- Semantic Scholar showed only 4 BSCCM citations (rate-limited on re-query); Google Scholar was not accessible, so citing papers outside S2 and arXiv may be missed.
