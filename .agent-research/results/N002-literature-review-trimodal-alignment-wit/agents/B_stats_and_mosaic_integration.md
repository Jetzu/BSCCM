# Strand B: Statistics of the G–L–M mosaic and how single-cell "mosaic/bridge integration" validates bridged predictions

Agent B, N002 literature review, 2026-11-08. Every claim has a URL. A claim marked **[unverified]** was not checked against the primary text. A result marked **(derivation)** is my own short derivation from cited standard results; it is not quoted from a paper.

Notation for BSCCM:
- **G** = Wright/Giemsa RGB image, measured on 15.7k cells.
- **L** = label-free LED-array images, measured on all cells.
- **M** = surface-marker vector: ~350k cells, mostly one antibody per cell; 37k cells have all 8 antibodies.
- Files: A = (G, L) on 15.7k cells; B = (L, M) on ~350k cells. No cell has both G and M.

---

## 0. Bottom line

1. **This design is the textbook "statistical matching / data fusion" problem.** Two files share common variables Z (here L). X is observed only with Z, Y only with Z, never X with Y. The joint law of (G, M) is **not identified** from files A and B. Any method that uses only the two files implicitly imposes conditional independence **G ⊥ M | L** (the CIA). The CIA cannot be tested from those files.
   - Sources: [Conti, Marella & Scanu 2012](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf); [D'Orazio, Di Zio & Scanu](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf); [Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf).
2. **The G→L→M predictor estimates E[ E[M|L] | G ]: "the marker profile implied by the label-free phenotype of a cell that looks like this in Giemsa".** It does not estimate E[M|G].
   - Under the CIA the two coincide.
   - Without the CIA, any Giemsa information about markers that does not pass through L (I(G;M|L) > 0) is invisible to the predictor.
   - Under the CIA, the data-processing inequality caps I(G;M) at min(I(G;L), I(L;M)) ([DPI](https://en.wikipedia.org/wiki/Data_processing_inequality)).
3. **Several claims can still be made without G–M ground truth.**
   - (a) The L→M error a² is measurable on held-out L–M cells.
   - (b) The G-side loss b² = E[(m(L) − f(G))²] is measurable on the 15.7k G–L cells, because their L is known.
   - (c) A worst-case guarantee needs no CIA, only transportability of M|L: √MSE ≤ a + b (derivation, §1.6).
   - (d) Population-level ("fourth-level") checks are possible.
   - (e) Sensitivity bounds are possible. The width of the bounds shrinks as L predicts G or M more strongly.
   - Sources: [Conti 2012](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf); [Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf).
4. **Some claims need new same-cell G–M (ideally G–L–M) data:**
   - cell-level accuracy of markers predicted from Giemsa;
   - any test of the CIA;
   - any "Giemsa feature X is associated with marker Y" discovery;
   - calibrated per-cell uncertainty.

   A small auxiliary file helps a lot. In Rässler's simulation, a complete-record file of 5% (250 records), and even 1% (50 records), restored nominal coverage ([Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)). Prediction-powered inference turns such a small gold-standard set plus the 15.7k predictions into valid confidence intervals ([Angelopoulos et al. 2023](https://arxiv.org/abs/2301.09633)).
5. **Single-cell mosaic and bridge methods validate their bridged predictions almost only on data where the "missing" pair was in fact measured on the same cells.** They then hide that pairing ("artificial unpairing" or a simulated mosaic). Cell-level accuracy for a modality pair that was truly never co-measured is essentially never shown. Where it is attempted, the evidence is one of three kinds:
   - label transfer or biological plausibility, as in the [Seurat v5 CyTOF bridge](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/) and the [StabMap multi-hop real data](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/);
   - a "proof-of-concept" with no ground truth, as in [BABEL ATAC→protein](https://pmc.ncbi.nlm.nih.gov/articles/PMC8054007/);
   - a **small targeted wet-lab experiment** on the query modality, as in the immunofluorescence of predicted proteins on chromatin-imaged cells in [Yang et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782789/).

   When no cell links the modalities ("diagonal"), methods degrade or produce plausible but wrong alignments ([MIDAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11471558/); [Xu & McCord 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9206644/)).
6. **Recommended protocol (§3):**
   - Tier 1: in-silico checks, with no new data.
   - Tier 2: CIA sensitivity analysis and bounds.
   - Tier 3: weak external ground truth (morphology classes, sorted populations).
   - Tier 4: a small same-cell G–L–M experiment analysed with PPI and a conditional-independence test.

---

## Part 1 — Statistics

### 1.1 The framework: statistical matching (data fusion)

**Definition.** Statistical matching estimates the joint distribution, or its parameters, of (Y, Z) when ([Conti, Marella & Scanu 2012, JOS 28:69–88, p.69](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)):
- "Y and Z are not jointly observed";
- Y is in sample A and Z in sample B;
- the samples do not overlap, so record linkage is impossible;
- "A and B both observe a set of additional variables X".

The authors' symbols map to ours as **X ↔ L**, **Y ↔ G**, **Z ↔ M**.

**The two classical approaches** ([Conti et al. 2012, p.69–70](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)):
1. Methods built on the conditional independence assumption Y ⊥ Z | X (Okner 1972; criticised by Sims 1972 and Rodgers 1984).
2. Methods that use auxiliary information about the Y–Z relationship, for example "an additional file C where (X,Y,Z) are jointly observed" ([Singh et al. 1993](https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X199300114475)).

**Non-identifiability.** "The model is not identifiable given the available data, unless some untestable models are assumed, as conditional independence between Y and Z given X. Most of the literature on this topic is based on the conditional independence assumption" ([D'Orazio, Di Zio & Scanu, p.2](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf)). Conti et al. add that "the CIA is usually a misspecified assumption, while external auxiliary information is hardly ever available" ([p.70](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)).

**Matching noise.** This is the discrepancy between the population joint distribution and the joint distribution in the synthetic fused file ([Conti et al. 2012, p.70](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf), citing Paass 1986). It is the formal name for the "matching error" in the brief.

**Canonical references:**
- D'Orazio, Di Zio & Scanu, *Statistical Matching: Theory and Practice* (Wiley 2006) — cited as the reference text in the two papers above; book not read **[unverified, book text]**.
- Rässler, *Statistical Matching: A Frequentist Theory, Practical Applications and Alternative Bayesian Approaches* ([Springer LNS 168, 2002](https://link.springer.com/978-0-387-95516-2)).
- Ridder & Moffitt, "The econometrics of data combination" ([Handbook of Econometrics 6B, 2007](https://ideas.repec.org/h/eee/ecochp/6b-75.html)). Per its abstract, it treats "(conditional) bounds on the joint distribution" for independent samples and notes that exclusion restrictions narrow them. Full text not read **[unverified beyond abstract]**.

### 1.2 What two-file methods do implicitly

- **Rässler's abstract** ([Austrian J. Stat. 33:153–171, 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)): "It is well known that those approaches establish conditional independence of the (specific) variables not jointly observed given the common variables, although they may be conditionally dependent in reality. However, if the common variables are (carefully) chosen in a way that already establishes conditional independence, then inference about the actually unobserved association is valid. In terms of regression analysis, this implies that the explanatory power of the common variables is high concerning the specific variables. Unfortunately, this assumption is not testable yet."
- **Simulation evidence** (same paper, §5): the nearest-neighbour match "always establishes conditional independence. Thus, this matching procedure only works, when the conditional independence assumption is satisfied". Even slight deviations made the actual coverage of confidence intervals far below nominal.
- **Fusion covariance** (Rässler & Fleischer 1998, cited in [Rässler 2004 §3.3](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)): the covariance in a fused file is cov~(X,Y) = cov(E[X|Z], E[Y|Z]). It equals the true cov(X,Y) only if E[cov(X,Y|Z)] = 0.
  - For BSCCM: any G-feature/marker correlation computed on imputed markers re-expresses the G–L and L–M associations. It cannot reveal new G–M biology.
- **Single-cell parallel.** totalVI warns that correlation matrices computed from expected values (posterior means) of imputed features "induced spurious correlations". It uses posterior-predictive samples instead ([Gayoso et al. 2021, Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7954949/)).

**Implication for any neural G→L→M pipeline (derivation).** If the predictor is trained only on A and B, its target is E_{L|G}[ E[M|L] ]. Gradient descent on the two files cannot reach a different target, because no loss term involves (G, M) jointly. This is the CIA solution in the sense of Rässler's statement above.

### 1.3 Partial identification: Fréchet bounds and uncertainty measures

**Nonparametric (Fréchet) bounds.** Given the conditional marginals F(y|x) and G(z|x), every compatible conditional joint H satisfies max(0, F+G−1) ≤ H(y,z|x) ≤ min(F, G) (Fréchet–Hoeffding). The two extremes correspond to "maximal negative and maximal positive association" ([Conti et al. 2012, eq. 6](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)).

**Categorical cell-probability bounds** ([Conti et al. 2012, eq. 11](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf); unconditional version in [D'Orazio et al., §3](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf)):

  Σ_i p(x_i) · max{0, p(y_j|x_i) + p(z_k|x_i) − 1} ≤ p(y_j, z_k) ≤ Σ_i p(x_i) · min{p(y_j|x_i), p(z_k|x_i)}.

- The ISTAT paper prints "max" in the upper bound. That is a typo: Conti et al. eq. 11 gives U = min.
- Bounds become sharper when the categories are ordered (eq. 12–13).
- The R package [StatMatch](https://cran.csiro.au/web/packages/StatMatch/refman/StatMatch.html) implements these bounds (`Frechet.bounds.cat`) **[unverified: package docs not read]**.
- **BSCCM use:** discretise L into cell-state clusters, G into morphology classes and M into gates. The bounds above then give the identified range of P(morph = j, gate = k) without any CIA.

**Gaussian case.** For jointly normal (X,Y,Z), the only unidentified parameter is ρ_YZ ([Conti et al. 2012, eq. 18](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)):

  ρ_YZ ∈ [ ρ_XY·ρ_XZ − √((1−ρ²_XY)(1−ρ²_XZ)), ρ_XY·ρ_XZ + √((1−ρ²_XY)(1−ρ²_XZ)) ].

- The CIA value ρ_XY·ρ_XZ is the midpoint.
- Rässler's worked example finds the CIA value at "exactly the middle" of the admissible interval ([Rässler 2004, p.155–156](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).
- Kadane (1978, reprinted JOS 2001) first estimated these bounds; Moriarity & Scheuren (2001) generalised them (both cited in [Conti et al. 2012](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)).
- Rodgers (1984) showed that "only an extremely high correlation narrows the range of the unconditional association considerably" (quoted in [Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).

Illustrative widths from eq. 18 (computed for this report; scalar summaries of G and M):

| ρ(G-score, L) | ρ(L, M) | CIA value of ρ_GM | admissible range |
|---|---|---|---|
| 0.70 | 0.70 | 0.49 | [−0.02, 1.00] |
| 0.90 | 0.90 | 0.81 | [0.62, 1.00] |
| 0.95 | 0.90 | 0.855 | [0.72, 0.99] |
| 0.95 | 0.95 | 0.90 | [0.81, 1.00] |

**The uncertainty measure of Conti et al.**
- The measure is the average width of the Fréchet class.
- With continuous conditional marginals and no extra information, it equals **1/6 for any F and G** ([Conti et al. 2012, eq. 10](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)).
- There is a discontinuity: "if either Y or Z is fully determined by X ... the uncertainty measure becomes equal to zero" (p.76).
- Logical constraints (structural zeros, ratio bounds) shrink the measure. In their simulation it fell from ≈0.167 to ≈0.11 (Table 1).
- They close with Manski's "law of decreasing credibility": "the credibility of inference decreases with the strength of the assumption maintained" (p.86).
- Practical reading: the measure, and the Gaussian width, are small only when L nearly determines M, or nearly determines the parts of G that matter. In that case the CIA is almost automatically true.

**Regression coefficients with combined data.** [D'Haultfœuille, Gaillac & Maurel (arXiv 2412.04816)](https://arxiv.org/abs/2412.04816) partially identify linear-regression coefficients when the outcome and some covariates sit in unmatchable datasets. The bounds are sharp and simple, and get tighter when variables are observed in both datasets. Abstract only **[unverified formulas]**.

### 1.4 What is checkable without auxiliary data: Rässler's four validity levels

[Rässler (2004, §3)](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf) defines four levels of validity for a fused file:

| Level | Name | Requirement | Checkable from two files? |
|---|---|---|---|
| 1 | Preserving individual values | the imputed value equals the true value ("hit rate") | No. Rässler calls hit rates misleading: a method that preserves marginals gets "high" hit rates even when "worst". Rubin's example: impute-all-heads beats correct random imputation on hit rate (60% vs 52%) but gives p = 0.96 instead of 0.6. |
| 2 | Preserving the joint distribution | f~(X,Y,Z) = f(X,Y,Z) | Only under the CIA (Sims 1972; Rodgers 1984; Rässler 2002). |
| 3 | Preserving the correlation structure | cov~ = cov | Only if E[cov(X,Y|Z)] = 0. |
| 4 | Preserving marginals | f~(Y) = f(Y) and f~(Y,Z) = f(Y,Z) | **Yes.** "In the typical fusion situation only the fourth level can be controlled." |

Beyond split-half or simulation studies, "all tests actually applied in practice only indicate the fourth-level validity" ([Rässler 2004, p.159–160](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).

For BSCCM:
- Level 4 corresponds to comparing the distribution of predicted M on G-cells with the observed M on L–M cells.
- The L–(predicted M) joint on G-cells can be compared with the L–M joint on L–M cells.
- Both checks are necessary, not sufficient.

### 1.5 Relaxing the CIA

1. **Auxiliary file with (X,Y,Z) jointly observed.** [Singh et al. 1993 (Survey Methodology)](https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X199300114475) showed by simulation that "auxiliary information can overcome the CIA limitation".
   - [Rässler 2004 (§5, Table 2)](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf): with "an additional file of size 5%, i.e., a file of only 250 observations completely observed in X, Y, and Z, the simulated actual coverage in most of the cases is higher than its nominal value". Using auxiliary data "works fine ... also if only 1% (i.e., 50 observations) are completely observed".
   - This is the strongest literature argument for a small same-cell G–L–M experiment.
2. **Logical constraints and structural zeros** ([Conti et al. 2012 §3](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)). For BSCCM, haematology knowledge can supply these:
   - a morphologically unambiguous eosinophil or basophil constrains the CD14/CD16 gates;
   - lymphocyte morphology excludes high CD14.

   Encoding such rules shrinks the Fréchet bounds. **[unverified: the specific marker rules must be checked against the BSCCM antibody panel]**
3. **Rubin's file concatenation plus multiple-imputation sensitivity analysis** ([Rubin 1986, JBES 4:87–94](https://econpapers.repec.org/RePEc:bes:jnlbes:v:4:y:1986:i:1:p:87-94)). The method is described in [NRC (1991) "Rough sensitivity analysis"](https://www.nationalacademies.org/read/1853/chapter/43):
   - impute first assuming ρ(YZ·X) = 0;
   - rerun "assuming ρ(YZ·X) = 0.5";
   - the differences show "variability due to model selection".
   - Generalised by [Moriarity & Scheuren 2003](https://ideas.repec.org/a/bes/jnlbes/v21y2003i1p65-73.html), and by Rässler's NIBAS, which places a prior on the unidentified conditional correlation ([Rässler 2004 §4](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).
4. **Design-based avoidance.** Rässler recommends "split questionnaire surveys". These create missingness patterns under which the parameters are identified ([Rässler 2004 §6](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)). The BSCCM analogue is a planned subset in which all three modalities are measured.

### 1.6 Information ceiling and identified bounds for the transitive predictor

All results in this subsection are **derivations from standard facts**. I did not find them stated in this form for this setting. They need red-team review before use.

Notation:
- m(L) = E[M|L], fitted on file B.
- f(G) = any G-only predictor, trained on file A to approximate m(L).
- a² = E[(M − m(L))²], the irreducible L→M error. It can be estimated on held-out L–M cells. A fitted m̂ gives an upper estimate.
- b² = E[(m(L) − f(G))²]. It is **measurable on the 15.7k G cells**, because their L is observed.

**(i) Data-processing inequality.** For a Markov chain G → L → M (which is exactly the CIA), I(G;M) ≤ I(L;M) and I(G;M) ≤ I(G;L). Equality in I(L;M) ≥ I(G;M) holds iff I(L;M|G) = 0 ([DPI](https://en.wikipedia.org/wiki/Data_processing_inequality), citing Cover & Thomas). Any predictor is a function of G, so I(f(G); M) ≤ I(G; M).
- So under the CIA, Giemsa can never predict markers better than L does.
- Without the CIA, I(G;M) can exceed I(L;M). The transitive predictor cannot exploit that excess, because its target is fixed at E[m(L)|G] (§1.2).

**(ii) Targets.** The right cascade regresses m(L) on G, giving E[m(L)|G]. Plugging in a predicted L, m(L̂(G)), is a different estimator:
- for nonlinear m it differs by a Jensen gap: E[m(L)|G] ≠ m(E[L|G]);
- for proper uncertainty, sample L ~ p(L|G) and then M ~ p(M|L), in the spirit of multiple imputation ([Rubin 1986](https://econpapers.repec.org/RePEc:bes:jnlbes:v:4:y:1986:i:1:p:87-94); [Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).

**(iii) Worst-case MSE without the CIA.** Write M − f(G) = (M − m(L)) + (m(L) − f(G)). By Minkowski's inequality in L²:

  |a − b| ≤ √E[(M − f(G))²] ≤ a + b, for **every** joint law of (G, L, M) consistent with files A and B.

- Under the mean-CIA, E[M − m(L) | L, G] = 0, so the cross term vanishes and the MSE equals a² + b².
- Only transportability is needed: M|L must be the same in the G-cell population as in the L–M population.
- The upper bound (a+b)² is therefore a reportable, CIA-free guarantee.
- The lower end shows that the CIA could also *understate* performance, if G carries extra marker information.

Example (standardised units). a² = 0.2 and b² = 0.1 give CIA MSE = 0.30. The range under any joint law is [0.017, 0.58].

**(iv) Covariance and correlation bounds for a scalar Giemsa score S and a marker M.**
- cov(S, M) = cov(S, m(L)) + cov(S − E[S|L], M − m(L)).
- The first term is identified: S, L and m(L) are all available on the G cells.
- The second term is the only unidentified piece.
  - Cauchy–Schwarz bounds it by √(E Var(S|L) · E Var(M|L)).
  - The sharp bound comes from conditional Fréchet–Hoeffding couplings within each L (comonotone and countermonotone quantile couplings), using [Conti et al. eq. 6](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf) applied per value of L.
- In the Gaussian, linear case this reduces exactly to eq. 18 above.

**(v) The Bayes-optimal G→M risk, E Var(M|G), is unidentified.**
- Its upper end is at most (a+b)², by (iii).
- Its lower end can be far below the CIA value if G resolves states that L does not.
- So "Giemsa predicts marker k with R² = r" is not a claim one can make at the cell level without G–M data.

**Assumption to flag.** All of (iii)–(v) assume that M|L transports from the L–M cells to the G cells. In BSCCM the G cells were never antibody-stained. If staining or processing shifts L, or if the G cells come from a different isolation, transport can fail.
- The L-covariate shift is testable: a two-sample classifier on L between G cells and L–M cells, with importance weights.
- Shift of M|L is not testable without M on the G population **[unverified: whether the BSCCM G subset spans both batches]**.

### 1.7 Prediction-powered inference and designing a small gold-standard experiment

**PPI** ([Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, arXiv 2301.09633](https://arxiv.org/abs/2301.09633); published in Science 2023 per the [Berkeley news release](https://vcresearch.berkeley.edu/news/how-use-ai-discovery-without-leading-science-astray), venue details not read **[unverified volume/DOI]**):
- Inputs: a small labelled set (X_i, Y_i), i = 1..n; a large unlabelled set X̃_j, j = 1..N; a predictor f.
- Mean estimator: θ̂_PP = (1/N) Σ f(X̃_j) − (1/n) Σ (f(X_i) − Y_i). The second term is the "rectifier".
- Main-text assumptions (§1.2):
  - the labelled and unlabelled sets are "independently and identically distributed samples from a common distribution";
  - f "is independent of the observed data".
- Extensions:
  - known **covariate shift** with a known density ratio w(x), §4.2.1;
  - label shift, §4.2.2;
  - finite populations, Appendix B.
- The guarantee holds "without making any assumptions on the machine-learning algorithm". More accurate predictions give smaller intervals.

**Related methods:**
- **PPI++** ([arXiv 2311.01453](https://arxiv.org/abs/2311.01453)) adds a power-tuning parameter λ. The resulting estimator is "never worse than classical inference" using the labelled data alone.
- **Cross-PPI** ([Zrnic & Candès, PNAS 2024; arXiv 2309.16598](https://arxiv.org/abs/2309.16598v3)) trains f on the labelled data via cross-fitting. It is more powerful than splitting off training data. Read at search-result level **[unverified details]**.
- **Active inference** ([Zrnic & Candès, arXiv 2403.03208](https://arxiv.org/abs/2403.03208)):
  - labels each point with probability π(x), proportional to the model's uncertainty;
  - estimator θ̂ = (1/n) Σ [f(X_i) + (Y_i − f(X_i)) ξ_i / π(X_i)], an AIPW form;
  - remains unbiased;
  - the abstract claims it can "save over 80% of the sample budget" in experiments.

**Mapping PPI onto BSCCM:**
- PPI is useful only for **population-level estimands**, for example:
  - the mean of CD-marker k within a Giemsa morphology class;
  - the fraction of gate-positive cells among Giemsa "monocytes";
  - regression coefficients of marker on Giemsa features.
- The rectifier **requires same-cell (G, M) gold-standard pairs**. No gold standard means no PPI for G→M.
- PPI is applicable today for L→M estimands, since L–M pairs are abundant, but that does not validate G→M.
- Design for validity: make the gold-standard cells a *random subsample* of the same preparation whose remaining cells get G (+L) only. That makes the i.i.d. assumption true by design.
  - If the gold standard must come from a new donor or isolation, the target population changes.
  - The covariate-shift extension needs Y|X to be invariant and w(x) to be known.
- Active selection matters only if per-cell M measurement is costly, as in sorting. In imaging, every stained cell on a slide is measured anyway.

**Rough sample sizes** (computed for this report; Fisher-z, α = 0.05):

| Quantity | n = 100 | n = 200 | n = 400 |
|---|---|---|---|
| 95% CI for a cell-level Pearson r ≈ 0.7 | [0.58, 0.79] | [0.62, 0.76] | [0.65, 0.75] |

- PPI gain: if corr(f, M) = 0.8 on the gold set and N ≫ n, the effective sample size is about 1/(1−0.8²) ≈ 2.8× n for a mean (derivation from the variance form above).
- **Testing the CIA directly.** On G–L–M cells, test whether G adds predictive information about M beyond L, i.e. G ⊥ M | L. Conditional-independence testing has no universally valid and powerful test ([Shah & Peters 2020, Ann. Stat. 48:1514](https://arxiv.org/abs/1804.07203v6)). Their Generalised Covariance Measure (GCM) is the recommended practical test:
  - regress M on L and S(G) on L;
  - test the covariance of the residuals;
  - this is exactly the unidentified term in §1.6(iv).
  - A 1-df approximation needs n ≈ 120 to 200 cells to detect an incremental R² of 0.02, when the full-model R² is 0.5 to 0.7, at power 0.8. This is a rough calculation that ignores cross-fitting overhead.
- **Per-cell uncertainty must be calibrated on held-out ground truth.** Model-based intervals in cross-modal prediction under-cover badly under shift:
  - sciPENN: 80% prediction intervals covered a median 44.6% (MALT), 71.7% (Monocyte) and 63.7% (H1N1) of held-out protein values;
  - totalVI: 18.3%, 21.2% and 11.1% respectively ([Lakkis et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9979929/)).

### 1.8 What is testable within the current data vs what needs new ground truth

| Question | Testable now? | How / why |
|---|---|---|
| L→M accuracy (per marker, per batch) | Yes | Held-out L–M cells; batch-held-out splits. |
| Joint marker structure P(M₁..M₈ \| L) | Partly | Only on the 37k full-panel cells. The single-antibody cells are themselves a within-M statistical-matching problem, and the 37k cells play the role of the auxiliary file ([Singh et al. 1993](https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X199300114475)). |
| G→L (or G→m(L)) fidelity, b² | Yes | On the 15.7k G–L cells, compare f(G) with m(L_i). |
| CIA-free worst-case MSE of G→L→M | Yes, with a transport assumption | (a+b)², §1.6(iii). |
| Population-level ("level 4") consistency | Yes | Predicted-M distribution on G cells vs observed M (L-reweighted) ([Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)). |
| Identified ranges for G–M associations | Yes | Fréchet or Gaussian bounds; Rubin-style ρ sweeps. |
| Cell-type agreement Giemsa-morphology vs predicted-marker gates | Partly | Needs independent morphology labels on G cells (expert or external classifier). Tests type-level structure, not within-type marker levels. |
| CIA (G ⊥ M \| L) | **No** | Untestable from A and B ([Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf); [D'Orazio et al.](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf)). Needs G–L–M cells. |
| Cell-level G→M accuracy, calibrated intervals | **No** | Needs same-cell G–M. |
| New G–M biology ("Giemsa feature X tracks marker Y") | **No** | Fused-file covariance = cov(E[G|L], E[M|L]) ([Rässler 2004 §3.3](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)), so this is circular. |

---

## Part 2 — Single-cell mosaic and bridge integration: how bridged predictions are validated

### 2.1 Taxonomy and where BSCCM sits

The field distinguishes integration settings by input structure:
- **vertical**: same cells;
- **horizontal**: shared features;
- **diagonal**: no shared cells or features;
- **mosaic**: datasets sharing only some modalities.

This scheme follows Argelaguet et al. 2021 ([doi:10.1038/s41587-021-00895-7](https://doi.org/10.1038/s41587-021-00895-7); primary text not read **[unverified wording]**). It is used in [scMoMaT](https://pmc.ncbi.nlm.nih.gov/articles/PMC9873790/) and in the [MultiVI discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406609/). A 2025 Registered-Report benchmark adds a fourth category, "cross" integration, and benchmarks 40 methods (12 of them mosaic) ([Liu et al., Nat Methods 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12615258/)).

**BSCCM is a "chain mosaic" with hub modality L.** Two datasets in that benchmark have the same chain topology:
- **D49** = (RNA+ADT, RNA+ATAC, ADT): ADT and ATAC are never on the same cell;
- **D40** = (RNA, ADT, RNA+ADT).

Only scMoMaT, Multigrate and StabMap could run on such data ([Liu et al. 2025, mosaic section](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12615258/)).

In Seurat-v5 terms, BSCCM maps as follows:
- query modality = G;
- **bridge** = the G–L pairs;
- reference = L annotated with M ([Hao et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/)).

### 2.2 Method-by-method

| Method | How it bridges non-co-measured modalities | Key assumption | How bridged predictions were validated |
|---|---|---|---|
| **Seurat v5 bridge integration** ([Hao et al., Nat Biotech 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/)) | Each multi-omic bridge cell is an "atom" of a dictionary. Each unimodal dataset is represented in atom space, then aligned with mnnCorrect. | "makes no assumptions about the relationships between modalities, as these are learned automatically from the multi-omic dataset". Implicitly, the reference annotation is a function of position in the bridged space. | (1) Split a 10x multiome into a ground-truth benchmark (pairing hidden) and a bridge; Jaccard similarity of matched cells. (2) Same with Paired-Tag replicates. (3) Downsampled bridge: "at least 50 cells ('atoms') representing a given cell type" sufficed. (4) Removing pDC from the bridge cut pDC annotation from 94.4% to 83.5% and mean prediction score from 0.907 to 0.514. (5) The CyTOF→RNA bridge via CITE-seq (161,764 cells, 30 shared proteins) was validated **only by biological plausibility** (Foxp3 in Tregs, a GzmB-low MAIT phenotype). |
| **StabMap** ([Ghazanfar et al., Nat Biotech 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/)) | Builds a mosaic data topology graph and projects cells onto reference PC/LD scores by iterated linear regressions along shortest paths ("multi-hop"). This is literally a transitive predictor. | The MDT is connected; shared features are linearly predictive of the reference scores. | (1) PBMC multiome split into RNA-only and ATAC-only; kNN (k=5) cell-type classification accuracy, neighbourhood Jaccard, rank of true match. (2) Gastrulation query with 50–5,000 genes. (3) **Multi-hop sim**: random thirds RNA-only/ATAC-only/multiome; embedding "compromised" with fewer than ~1,000 multiome bridge cells; Cobolt "performed consistently better in recapturing cell type labels". (4) 8-dataset chain: accuracy from dataset 1 to dataset 8 improves with more informative features. (5) Real multi-hop (CyTOF–ECCITE–Multiome; IMC–CITE–Xenium) assessed only by LISI mixing and plausibility. Limitation: linear modelling. |
| **MultiVI** ([Ashuach et al., Nat Methods 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406609/)) | VAE; paired cells use both likelihoods, unpaired cells one; adversarial mixing; decoders impute the missing modality. | Shared latent cell state generates all modalities. | "Artificially unpaired" 1–99% of a 10x PBMC multiome. With 75% unpaired, imputation Spearman 0.57 (raw RNA), AUPRC 0.41 (ATAC); 0.86 and 0.80 against smoothed data. Uncertainty correlated with error, but ~0.5% of values were high-confidence/high-error. DOGMA stimulated cells with ATAC hidden: 0.93. Tri-modal test: 75% unpaired, so **8% of cells kept all three modalities**; Spearman 0.78 (RNA), 0.73/0.76 (ATAC). B-cell held-out modality: 75% of 1,621 DE genes recovered. |
| **totalVI** ([Gayoso et al., Nat Methods 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7954949/)) | Joint RNA+protein VAE; missing proteins imputed by decoding under another batch's covariates. | Batch-independent latent space; "quality of imputation relies on how well batches mix". | Held out the proteins of one CITE-seq sample (SLN111-D2); RMSLE over 30 resampled runs; better than Seurat v3 for ~68% of 80 significantly different proteins. Posterior-predictive checks and held-out calibration error. Warns that correlations of imputed means are spurious. |
| **GLUE / scGLUE** ([Cao & Gao, Nat Biotech 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9546775/)) | Per-modality VAEs coupled by a prior-knowledge "guidance graph" of features plus adversarial alignment. | The guidance graph links features (e.g. peak–gene); robust to 90% corrupted edges. | SNARE/SHARE/10x-multiome treated as unpaired; FOSCTTM (fraction of samples closer than the true match). Error rises steeply below ~1,000 cells. Triple-omics with **no** paired ground truth: validated by marker overlap of label-transferred types (three-way Fisher, 12/14 types) and an "integration consistency score" (self-consistency with the graph). |
| **MIDAS** ([He et al., Nat Biotech 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11471558/)) | Mosaic VAE with product-of-experts, self-supervised modality alignment, information-theoretic disentangling. | Shared biological latent across modality combinations. | 14 incomplete datasets made by removing modality×batch blocks from trimodal DOGMA-seq/TEA-seq; "scMIB" with 20 metrics, including RNA/ADT Pearson r against the true held-out features. Micro-F1 > 0.885 versus full-data labels **except "diagonal"** tasks, which were poor "due to the lack of cell-to-cell correspondence"; transfer learning helped. |
| **scMoMaT** ([Zhang et al., Nat Commun 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9873790/)) | Matrix tri-factorisation over all cell×feature blocks. | Shared cell and feature factors. | Simulated 6 batches × 3 modalities, 8 seeds, including a scenario where "no simultaneously profiled cell batch exists"; ground-truth clusters and marker genes; GC/ARI/NMI; real data with published labels. |
| **Cobolt** ([Gong et al., Genome Biol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8715620/)) | Multimodal VAE with product of experts. | Shared latent. | Ignored the pairing in a second 10x multiome (different donor); SNARE-seq split 20% paired / 80% unpaired; fraction of single-modality cells whose true pair is among their k-NN. scMM and BABEL did worse with few paired cells. |
| **UINMF** ([Kriebel & Welch, Nat Commun 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8828882/)) | NMF with an extra metagene matrix for unshared features. | Shared features exist across all datasets. | SNARE-seq treated as unpaired; FOSCTTM improved from 0.2977 (iNMF) to 0.251. |
| **MultiMAP** ([Jain et al., Genome Biol 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8686224/)) | UMAP-style shared manifold; uses features not present in all datasets. | Shared manifold. | MNIST images split into halves (synthetic multimodal); kNN (k=5) label-transfer balanced accuracy; robust when only 1/10 clusters are shared. |
| **MOFA+** ([Argelaguet et al., Genome Biol 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7212577/)) | Bayesian group factor analysis over views×groups; "missing values are allowed". | Linear low-rank factors. For a chain design this is a *structural* identifying assumption (G ⊥ M given latent factors, not given L) **(derivation)**. | Simulated data drawn from its own generative model; real-data case studies. Not designed or validated for never-co-observed view pairs. |
| **CLUE** ([Tu et al., NeurIPS 2022](https://neurips.cc/virtual/2022/poster/53026)) | Cross-encoders trained on paired data give modality-specific embeddings, combined per cell; handles modality-incomplete data. | Paired training data for each cross-encoder **[unverified details: full paper not reachable]**. | NeurIPS 2021 competition: winner of the modality-matching task ([openproblems.bio](https://openproblems.bio/neurips_2021)); ground truth is same-cell pairing. |
| **sciPENN** ([Lakkis et al., Nat Mach Intell 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9979929/)) | RNA→protein network; imputes proteins missing from partially overlapping panels; quantile intervals. | Reference–query similarity. | External CITE-seq test sets with proteins held out (MALT, Monocyte, H1N1); Pearson r and RMSE; **prediction-interval coverage far below nominal** (§1.7). |
| **BABEL** ([Wu et al., PNAS 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8054007/)) | Two encoders and two decoders trained on all four paths; auxiliary protein decoder. | Shared latent. | Cluster-held-out test (Pearson 0.62 ATAC→RNA vs kNN baseline 0.27); external GM12878 cells. Unpaired clinical BCC: **pseudobulk concordance** with tissue-matched scRNA-seq (r = 0.70). **ATAC→protein (never co-trained, our exact transitive case) shown only as a "proof-of-concept"** (SI Fig. S9; not read **[unverified what S9 used as reference]**). |
| **MaxFuse** ([Chen et al., Nat Biotech 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11638971/)) | Iterative co-embedding, smoothing and matching for **weakly linked** modalities (few or weakly correlated linked features). | Some linked features. | CITE-seq ground truth (228 proteins) with pairing hidden: label agreement at two granularities (level-1 accuracy 93.9%), FOSCTTM, FOSKNN; antibody panels shrunk progressively. |
| **Seurat v4 reference mapping** ([Hao et al., Cell 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/)) | RNA query mapped to a CITE-seq reference; labels and proteins transferred. | Reference covers the query states. | Query CITE-seq with proteins **withheld** from mapping, used "as an independent assessment"; where Seurat and scArches disagreed, 79.4% of cells' measured proteins favoured the Seurat label. |
| **Cross-modal autoencoders, imaging↔RNA** ([Yang et al., Nat Commun 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782789/)) | Per-modality autoencoders into a shared latent space; optional anchor or discriminative losses with prior knowledge. | Shared latent; prior knowledge where available. | Paired RNA/ATAC benchmark, then **wet-lab validation**: CORO1A and RPL10A were predicted from chromatin images, then stained by immunofluorescence on the *same imaged cells*; the predicted direction was confirmed (p < 2.2e-16). |
| **NeurIPS 2021 benchmark** ([Luecken et al. 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/158f3069a435b314a80bdcb024f8e422-Abstract-round2.html); text read via [Helmholtz PDF](https://push-zb.helmholtz-munich.de/deliver.php?id=41546)) | Defines three tasks: modality prediction (RMSE, Pearson), matching, joint embedding. | — | Same-cell joint measurement is "a promising form of high-dimensional ground truth, akin to matched documents in machine translation". Splits of increasing difficulty: "within sample, within site across donor, within donor across site, and across donor and site". Positive and negative controls plus baselines bracket the scores. |

Additional benchmarks:
- Hu et al., Nat Methods 2024: 14 prediction and 18 integration algorithms on 47 datasets. Search-result level only **[unverified]**.
- Liu et al. 2025 ([PMC12615258](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12615258/)) evaluates mosaic imputation by holding one modality out of a query batch. Metrics: sMSE, preservation of feature correlations and DE statistics, plus downstream clustering and classification. It notes that "methods excelling in clustering and classification metrics do not necessarily perform well in preserving structural features", and that methods good on simulated data "exhibited reduced performance when applied to real datasets".

### 2.3 Patterns across the literature (what counts as validation)

1. **Simulated mosaic from fully co-measured data.** "Artificial unpairing" or deleted modality blocks is the dominant protocol ([MultiVI](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406609/), [StabMap](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/), [MIDAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11471558/), [Cobolt](https://pmc.ncbi.nlm.nih.gov/articles/PMC8715620/), [Seurat v5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/), [GLUE](https://pmc.ncbi.nlm.nih.gov/articles/PMC9546775/), [MaxFuse](https://pmc.ncbi.nlm.nih.gov/articles/PMC11638971/)). Metrics:
   - FOSCTTM, true-match rank or Jaccard;
   - held-out feature correlation;
   - label-transfer accuracy.
2. **Held-out real modality** on a separate sample or batch ([totalVI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7954949/), [sciPENN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9979929/), [Seurat v4](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/)).
3. **When the bridged pair is truly never co-measured**, papers fall back to weaker evidence:
   - label-transfer plausibility and canonical-marker checks ([Seurat v5 CyTOF](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/));
   - mixing scores ([StabMap real multi-hop](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/));
   - marker-overlap tests ([GLUE triple-omics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9546775/));
   - pseudobulk concordance ([BABEL](https://pmc.ncbi.nlm.nih.gov/articles/PMC8054007/));
   - a targeted same-cell wet-lab experiment ([Yang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782789/)).
4. **Tri-modal mosaic benchmarks retain some triple-measured cells.**
   - [MultiVI](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406609/) kept 8%.
   - The [MIDAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11471558/) "diagonal" variants without cell correspondence performed worst.
   - [Xu & McCord (Nat Commun 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9206644/) split scRNA-seq genes into two pseudo-modalities with known truth. All six diagonal methods tested "fail at least in one scenario". "Any artificial alignment that satisfies a mathematical optimum can stand out as the best solution". They recommend shared cells or labels as anchors, and simulated benchmarks.
5. **Quantitative design lessons:**
   - Bridge size: ≥50 bridge cells per cell type ([Seurat v5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/)).
   - Bridge size: more than ~1,000 multiome cells for stable multi-hop ([StabMap](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/)).
   - Neural aligners degrade below ~1,000–2,000 cells ([GLUE](https://pmc.ncbi.nlm.nih.gov/articles/PMC9546775/)).
   - BSCCM's 15.7k G–L bridge is large overall. A cell type at 0.5% frequency gives only ~78 G cells; check per-type counts.
   - Missing cell types in the bridge lower both accuracy *and* confidence. This is a usable out-of-support detector ([Seurat v5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/)).

---

## Part 3 — Best practice and a concrete validation protocol for BSCCM

### 3.1 Estimand and claim language (pre-register)

| Claim | Status | Basis |
|---|---|---|
| "f(G) predicts the label-free-implied marker profile m(L) with error b²" | Allowed now | Measured on the G–L cells. |
| "Under the CIA the expected marker MSE is a² + b²; under any joint law it is ≤ (a+b)², assuming M\|L transports" | Allowed now | §1.6. |
| "Predicted marker distributions on Giemsa cells match observed marker distributions" | Allowed now | Level 4 only. |
| "Giemsa predicts marker k at cell level with r = …" | **Not allowed** without Tier 4 data | — |
| "Giemsa reveals marker information beyond the label-free images" | **Not allowed** without Tier 4 data | — |
| "Giemsa feature X associates with marker Y" | **Not allowed** without Tier 4 data | Circular by [Rässler §3.3](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf). |

### 3.2 Tier 1 — in-silico checks with existing data (no new wet lab)

1. **L→M quality and a²:**
   - per marker;
   - per batch, train on isolation 1 and test on isolation 2 (Luecken-style difficulty splits, [Luecken et al.](https://push-zb.helmholtz-munich.de/deliver.php?id=41546));
   - with positive and negative controls and a kNN baseline ([BABEL](https://pmc.ncbi.nlm.nih.gov/articles/PMC8054007/)).
2. **G→m(L) quality, b², on the 15.7k G–L cells:**
   - compare f(G_i) with m(L_i);
   - compare f(G_i) with the cascade m(L̂(G_i)) to quantify the Jensen gap.
   - This is exactly the "split the multiome into separate modalities" benchmark of [Seurat v5](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/) and [MultiVI](https://pmc.ncbi.nlm.nih.gov/articles/PMC10406609/), with G–L as the bridge.
   - Report FOSCTTM-style matching of G to L as well ([GLUE](https://pmc.ncbi.nlm.nih.gov/articles/PMC9546775/)).
3. **Transport diagnostics:**
   - a classifier two-sample test on L for G cells vs L–M cells;
   - if shifted, importance-reweight a² and the Level-4 checks;
   - flag G cells outside the support of L–M using prediction confidence ([Seurat v5 pDC-depletion result](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/)).
4. **Simulated chain mosaic inside L–M data** (the field's standard, §2.3).
   - Choose a proxy G′ from L, e.g. brightfield-only or degraded channels, and a hub L′ from the other channels.
   - Split cells into disjoint sets with (G′, L′), (L′, M) and a hidden (G′, M) test set.
   - Train the transitive predictor on the first two sets and evaluate on the third. Also train a direct G′→M oracle.
   - The **transitive gap** (oracle minus transitive) can then be plotted against the measured CIA violation, i.e. the GCM statistic for G′ ⊥ M | L′.
   - Caveat: this calibrates *method behaviour*. It does not say whether the CIA holds for real Giemsa ([Xu & McCord](https://pmc.ncbi.nlm.nih.gov/articles/PMC9206644/); [Liu 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12615258/) on simulations vs real data).
5. **Within-M mosaic.** In the 37k full-panel cells, mask antibodies to mimic the single-antibody design. Check recovery of marker–marker structure with posterior-predictive samples, not posterior means ([totalVI](https://pmc.ncbi.nlm.nih.gov/articles/PMC7954949/)).
6. **Level-4 population checks** ([Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)) on G cells, after L-reweighting:
   - predicted-marker marginals and gates vs observed;
   - L × predicted-marker joints vs L × M.
7. **Controls:**
   - permuted-G predictions, which should drop to the marginal baseline;
   - m(L_i) itself as the CIA upper bound for any transitive model ([Luecken positive/negative controls](https://push-zb.helmholtz-munich.de/deliver.php?id=41546)).

### 3.3 Tier 2 — sensitivity analysis and bounds

- Report §1.6(iii) [|a−b|, a+b] per marker.
- Report categorical Fréchet bounds for (morphology class × marker gate) given L-clusters ([Conti et al. eq. 11](https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf)). Tighten them with defensible structural zeros from haematology.
- Run a Rubin-style multiple-imputation sweep over the unidentified partial association ρ(S, M | L) ∈ {−r, 0, +r} ([NRC 1991](https://www.nationalacademies.org/read/1853/chapter/43); [Rässler NIBAS](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)). Show which downstream conclusions are robust across the sweep.

### 3.4 Tier 3 — weak external ground truth on G cells

- **Morphology labels.** Wright/Giemsa morphology defines the classical five WBC classes by eye.
  - Have the G cells labelled by an expert or by an independent Giemsa classifier trained on external data **[unverified: choose and cite a public WBC morphology dataset]**.
  - Compare with types gated from predicted markers.
  - This validates type-level structure only, in the way label transfer is used across the field ([MultiMAP](https://pmc.ncbi.nlm.nih.gov/articles/PMC8686224/), [StabMap](https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/), [MaxFuse](https://pmc.ncbi.nlm.nih.gov/articles/PMC11638971/)).
  - It does not validate within-type marker levels.
- **Sorted populations.** Sort cells by marker gate (e.g. CD14⁺, CD16⁺), then Giemsa-stain and image them. This gives population-level P(G | gate) as ground truth, and PPI-style population estimands become checkable. This is standard haematology practice, but I did not find a citation in this pass **[unverified citation]**.

### 3.5 Tier 4 — a small same-cell G–L–M experiment (the only route to cell-level claims)

- **Design.** Antibody-stain the cells, then do LED-array and fluorescence imaging, then fix and Wright/Giemsa-stain the same slide, then do RGB imaging. Register cells across rounds.
  - Feasibility of Giemsa after immunofluorescence is **[unverified]**. A vendor note says immunocytochemistry on MGG/Diff-Quik-stained smears is "difficult to establish" ([Sigma-Aldrich tech note](https://sigmaaldrich.com/LC/en/tech-docs/paper/166954)). The proposed order (fluorescence first, Giemsa last) avoids that direction, but needs a pilot.
  - The precedent is [Yang et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7782789/): immunofluorescence on the same imaged cells validated predictions bridged from a never-co-measured modality.
- **Size.**
  - Rässler's simulations restored coverage with 1–5% complete records (50–250) ([Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).
  - For cell-level r ≈ 0.7 within ±0.05, use ~400 cells. For a CIA (GCM) test detecting incremental R² ≈ 0.02, use ~120–200 cells (§1.7).
  - Stratify to ≥50 cells per major type ([Seurat v5 bridge guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/)).
  - A few hundred to ~1,000 triple-measured cells across both cell isolations is a defensible target. This is a rough planning estimate.
- **Analysis:**
  1. Test the CIA directly with GCM ([Shah & Peters 2020](https://arxiv.org/abs/1804.07203v6)).
  2. Report cell-level accuracy for the transitive model, and for a model fine-tuned on G–M (cross-fitted).
  3. Build PPI/PPI++ confidence intervals for population estimands, using the 15.7k G predictions as the unlabelled set ([PPI](https://arxiv.org/abs/2301.09633), [PPI++](https://arxiv.org/abs/2311.01453), [cross-PPI](https://arxiv.org/abs/2309.16598v3)).
  4. Calibrate per-cell intervals on these cells, because model-based intervals under-cover ([sciPENN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9979929/)).
  5. Add the triple cells as an auxiliary file in a multiple-imputation fusion ([Singh et al. 1993](https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X199300114475); [Rässler 2004](https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf)).
  6. Make the triple cells a random subsample of a preparation whose other cells receive G+L only, so the PPI i.i.d. assumption holds by design.

---

## 4. Open or unverified items

- D'Orazio–Di Zio–Scanu 2006 book; Rässler 2002 book; Ridder & Moffitt chapter: abstracts or secondary descriptions only.
- Exact formulas in D'Haultfœuille–Gaillac–Maurel: abstract only.
- Argelaguet 2021 exact taxonomy wording; CLUE full method; Hu et al. 2024 benchmark; BABEL SI Fig. S9 reference data.
- The PPI Science volume/DOI.
- StatMatch `Frechet.bounds.cat` behaviour.
- Feasibility of Giemsa after antibody/fluorescence on BSCCM slides.
- Whether the G subset spans both BSCCM isolations.
- Which morphology→marker structural zeros apply to the actual 8-antibody panel.
- All §1.6 results are my derivations and need red-team review: the Minkowski bound, the conditional Fréchet–Hoeffding covariance bound, and the GCM sample-size approximation.

## 5. Sources read (primary text unless noted)

- Conti, Marella, Scanu (2012) JOS — https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/uncertainty-analysis-in-statistical-matching.pdf
- D'Orazio, Di Zio, Scanu — complex-survey statistical matching (ISTAT) — https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf
- Rässler (2004) Austrian J. Stat. — https://www.stat.tugraz.at/AJS/ausg041+2/041+2Raessler.pdf
- NRC (1991) rough sensitivity analysis (Rubin's procedure) — https://www.nationalacademies.org/read/1853/chapter/43
- Rubin (1986) JBES (abstract) — https://econpapers.repec.org/RePEc:bes:jnlbes:v:4:y:1986:i:1:p:87-94
- Singh et al. (1993) Survey Methodology (abstract) — https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X199300114475
- Moriarity & Scheuren (2003) — https://ideas.repec.org/a/bes/jnlbes/v21y2003i1p65-73.html
- Ridder & Moffitt (2007) (abstract) — https://ideas.repec.org/h/eee/ecochp/6b-75.html
- D'Haultfœuille, Gaillac, Maurel (abstract) — https://arxiv.org/abs/2412.04816
- Data processing inequality — https://en.wikipedia.org/wiki/Data_processing_inequality
- PPI — https://arxiv.org/abs/2301.09633 ; PPI++ — https://arxiv.org/abs/2311.01453 ; Active inference — https://arxiv.org/abs/2403.03208 ; Cross-PPI — https://arxiv.org/abs/2309.16598v3
- Shah & Peters (2020) — https://arxiv.org/abs/1804.07203v6
- Hao et al. 2024 (Seurat v5 bridge) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10928517/
- Ghazanfar et al. 2024 (StabMap) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10869270/
- Ashuach et al. 2023 (MultiVI) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10406609/
- Gayoso et al. 2021 (totalVI) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7954949/
- Cao & Gao 2022 (GLUE) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9546775/
- He et al. 2024 (MIDAS) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11471558/
- Zhang et al. 2023 (scMoMaT) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9873790/
- Gong et al. 2021 (Cobolt) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8715620/
- Kriebel & Welch 2022 (UINMF) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8828882/
- Jain et al. 2021 (MultiMAP) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8686224/
- Argelaguet et al. 2020 (MOFA+) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7212577/
- Wu et al. 2021 (BABEL) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8054007/
- Chen et al. 2024 (MaxFuse) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11638971/
- Lakkis et al. 2022 (sciPENN) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9979929/
- Hao et al. 2021 (Seurat v4) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/
- Yang et al. 2021 (cross-modal autoencoders, imaging↔RNA) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7782789/
- Xu & McCord 2022 (diagonal integration pitfalls) — https://pmc.ncbi.nlm.nih.gov/articles/PMC9206644/
- Liu et al. 2025 (multitask benchmark) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12615258/
- Luecken et al. 2021 (NeurIPS sandbox) — https://push-zb.helmholtz-munich.de/deliver.php?id=41546 ; competition results — https://openproblems.bio/neurips_2021
- CLUE (abstract level) — https://neurips.cc/virtual/2022/poster/53026
