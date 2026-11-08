# N002 / Strand E: creative and assumption-breaking ML ideas for G-L-M alignment

Agent: E (creative ML). Date: 2026-11-08. Scope: recent ML ideas (mostly 2023-2026) that are not yet used, or barely used, in single-cell cytology. For each one: how it works, why it fits BSCCM's G-L-M structure, how to adapt it, the cheapest test, and the main risk. The ranking is in section 3.

Conventions. Every paper cited here has a URL that I opened in this session (an arXiv abstract or HTML page, a proceedings page, or an official package page). Claims I could not check against an opened page are tagged **[unverified]**. Statements that are my own derivations or proposals, not taken from a paper, are tagged **[proposal]**.

Data facts used below. They come from the BSCCM paper (https://arxiv.org/html/2402.06191v1) and the project brief:
- 22 multi-LED patterns plus 1 single-LED pattern.
- 8 antibodies: CD19, HLA-DR, CD45, CD16, CD56, CD3, CD123, CD14. Staining conditions are unstained, single antibody, and all antibodies. In the all-antibody condition, two groups were merged into one spectrum each for demixing: CD14/HLA-DR/CD123 and CD3/CD19/CD56.
- Two biological replicates (batches).
- The histology subset is described as **Wright's stain**. The paper does not say Giemsa.
- Histology cells were matched to LED/fluorescence cells by a global translation and rotation grid search, then nearest-cell matching. The result is stored as `has_matched_histology_cell`. **So G-L pairs are matched at the cell level, not registered pixel to pixel, and some pairs may be wrong.**

---

## 0. Summary

The most useful reframing is this. **BSCCM's G-L-M problem is a classic *statistical matching / data fusion* problem**: two files, (G,L) and (L,M), share only L. Under that framing, every transitive pipeline makes the same untestable assumption, whatever the architecture. That assumption is conditional independence, M ⟂ G | L (the CIA). Without it, the G-M joint distribution can only be **bounded**, never point-identified (StatMatch docs: https://cran.mi2.ai/web/packages/StatMatch/refman/StatMatch.html; Fan, Sherman & Shum 2014: https://ideas.repec.org/a/wly/emetrp/v82y2014i2p811-822.html).

Current multi-view identifiability theory explicitly leaves this case open. Yao et al. list learning from "overlapping but not simultaneously observed marginals" as future work (https://ar5iv.labs.arxiv.org/html/2311.04056).

The highest-value move is therefore an **assumption-light evaluation layer**:
- partial-identification bounds on the G→M error,
- union-bound conformal intervals that need no G-M pairs,
- a mosaic simulation benchmark with known ground truth, built from BSCCM itself.

The modelling ideas (distillation, semi-supervised alignment, contrastive hub binding, bridges, any-to-any models) are then ranked against that layer.

**Top 5 by expected value:**

| # | Idea | What it delivers | Cost | Novelty in cytology |
|---|---|---|---|---|
| 1 | **Partial identification + union-bound conformal + in-dataset mosaic simulation** (statistical matching, Fréchet/rearrangement bounds, surrogate-index logic) | A credible, assumption-explicit evaluation of the never-co-observed G-M link | CPU, hours | Not found in imaging/cytology |
| 2 | **Generalized distillation (LUPI) with a "modality-general" teacher** (Lopez-Paz 2016, Modality Focusing Hypothesis 2023) | The core G→M predictor for clinics (L and M used only in training) | 1 GPU, under 1 h per arm | Used at slide level in pathology. Not found for single-cell hematology |
| 3 | **Semi-supervised G-L alignment with abundant unpaired data** (SOTAlign 2026, STRUCTURE 2025, UML 2025), plus the **"L as stain-invariance teacher"** argument (Daunhawer 2023) | Better G-L alignment from 15.7k pairs, and wider stain/scanner coverage for clinical transfer | 1 GPU, hours | Not found in cytology |
| 4 | **Dual-target illumination ranking** with information imbalance, concrete selectors, and PID | Ranks LED patterns by information about M *and* about G. "Bridge patterns" explain the transitive signal | CPU, hours | Information imbalance and concrete selectors not found for LED-array design |
| 5 | **Hub-and-spoke contrastive embedding with Rank-N-Contrast on partially observed continuous markers** (ImageBind/C-MCR with CentroBind caveats; RNC 2023) | The shared G/L/M embedding, with marker order preserved | 1 GPU, hours | RNC for single-cell markers not found |

Near misses (section 2.7 onward): label-proportion / PPI validation against clinical flow cytometry, noisy-correspondence audit of G-L pairs, bridges for virtual Wright staining, any-to-any masked/diffusion models, and direct unpaired G-M alignment (vec2vec/GW) as a triangulation control.

---

## 1. Reframing: what any G-L-M method can and cannot identify

### 1.1 Statistical matching and the CIA

The setup is: file A holds (X,Y), file B holds (X,Z), and the goal is the joint (Y,Z). D'Orazio/Di Zio/Scanu's framework says (Y,Z) is **not identifiable** from such data unless an untestable model is assumed, usually conditional independence given X. The StatMatch R package estimates the Y-vs-Z table under the CIA, or derives **Fréchet bounds** on the cell probabilities from the X-vs-Y and X-vs-Z tables. The bounds can be unconditional or conditional on X strata (https://cran.mi2.ai/web/packages/StatMatch/refman/StatMatch.html; book cited there: D'Orazio, Di Zio & Scanu 2006, *Statistical Matching: Theory and Practice*, Wiley).

Mapped onto BSCCM: X = L, Y = G, Z = M. Every "transitive" method assumes M ⟂ G | L, whether it is composition f_LM ∘ f_GL, ImageBind-style emergent alignment, distillation, or any-to-any masked modelling trained on these data. They differ only in how they approximate p(L|G) and p(M|L).

Econometrics treats the continuous version: outcomes and conditioning covariates come from separate datasets. Point identification fails, and **sharp bounds** follow from the monotone rearrangement inequality (Fan, Sherman & Shum, *Econometrica* 2014, https://ideas.repec.org/a/wly/emetrp/v82y2014i2p811-822.html).

The **surrogate index** (Athey, Chetty, Imbens & Kang, NBER w26463, https://www.nber.org/papers/w26463) has the same structure. A long-term outcome is predicted from short-term surrogates. Validity rests on the Prentice surrogacy assumption, that the outcome is independent of treatment given the surrogates. The paper characterizes the bias when the assumption is violated and gives ways to validate it with additional observed outcomes. In BSCCM, L plays the surrogate, G the "treatment/covariate" side, and M the outcome.

### 1.2 What representation-learning theory gives, and where it stops

- **Daunhawer et al., ICLR 2023** (https://ar5iv.labs.arxiv.org/html/2303.09166). Two modalities x_i = f_i(c, s, m_i) with distinct diffeomorphic mixing. Under content invariance and style-change assumptions, smooth minimizers of the asymptotic symmetric InfoNCE block-identify the shared content c (up to an invertible map). Minimizers *cannot* depend on the modality-specific latents m_1, m_2. The encoder dimension must equal n_c.
  - Applied here: G-L contrastive learning recovers c_GL, and L-M contrastive learning recovers c_LM.
- **Yao et al.** (https://ar5iv.labs.arxiv.org/html/2311.04056) give an "identifiability algebra": intersections of content blocks are block-identifiable (Cor. 3.9). But it is proven for **jointly observed views**, and the paper names "overlapping but not simultaneously observed marginals" as open.
  - So the natural conjecture, "the transitive G-M signal is c_GL ∩ c_LM", is **not covered by existing theory**. Proving or refuting it would be a research contribution. [proposal]
- **Sturma, Squires, Drton & Uhler** (https://ar5iv.labs.arxiv.org/html/2302.00993) do identify the *joint* distribution of **unpaired** domains from their marginals. This needs a linear SEM, linear mixing, non-Gaussian pairwise-distinct errors, and two partial pure children per shared node. Shared components are matched across domains with Kolmogorov-Smirnov tests on the recovered sources. This shows that identifying G-M without pairs requires strong structural assumptions. I doubt they hold for Wright-stain images versus marker vectors. [proposal]

### 1.3 Two hidden assumptions to test first [proposal]

1. **Transportability of L.** G cells are antibody-unstained and M cells are antibody-stained. Any transitive estimate needs p(M|L) to be the same in both groups, and either equal p(L) or an explicit reweighting.
   - The p(L) part is **testable**: train a classifier to separate L-embeddings of G-set cells from M-set cells, per batch. AUC well above 0.5 means reweighting is needed, e.g. density-ratio weighting as in weighted conformal prediction (Tibshirani et al., https://arxiv.org/abs/1904.06019).
2. **Correct G-L pairing.** Pairs come from slide-level alignment plus nearest-cell matching (BSCCM paper). A fraction of wrong pairs biases everything downstream (section 2.11).

---

## 2. Idea catalogue

### 2.1 [Rank 1] Partial-identification bounds and union-bound conformal for G→M

**Mechanism.** These are my derivations, built on the statistical-matching and rearrangement literature above [proposal].

Notation, for marker j:
- m̂(L) is a cross-fitted L→M regressor, fitted on L-M cells.
- ĥ(G) is a cross-fitted G→m̂(L) regressor, fitted on G-L cells. This is the transitive or distilled predictor.

Then exactly, for any m̂ and ĥ:

M − ĥ(G) = ε + δ, with ε = M − m̂(L) and δ = m̂(L) − ĥ(G).

So MSE(ĥ) = E[ε²] + E[δ²] + 2E[εδ].
- a² = E[ε²] can be estimated on held-out L-M cells.
- b² = E[δ²] can be estimated on held-out G-L cells.
- Only E[εδ] needs the unobserved G-M coupling.

What follows from this:
- **CIA point estimate:** a² + b². Under M ⟂ G | L, with m̂ = E[M|L], the cross term vanishes.
- **No-coupling bounds:** stratify L into K bins (e.g. k-means on an L-embedding). Within each bin, Cauchy-Schwarz gives |E[εδ|k]| ≤ a_k b_k, so MSE ∈ [Σ_k π_k (a_k − b_k)², Σ_k π_k (a_k + b_k)²]. These bounds are valid for any coupling and tighten as bins get finer.
- **Sharper bounds:** within each bin, replace Cauchy-Schwarz with the rearrangement (comonotone / counter-monotone) bounds on E[εδ|k], computed from the empirical quantile functions. This is the device behind Fan-Sherman-Shum's sharp bounds.
- **Cell-type version:** conditional Fréchet bounds on the table "G-predicted type × marker-gated type", stratified by L clusters. This is exactly what `Frechet.bounds.cat` computes (StatMatch link above). It turns into bounds on G-based cell-typing accuracy against gated truth.
- **Union-bound conformal interval:**
  - On G-L calibration cells, take the split-conformal quantile q_δ of |δ| at level α₁.
  - On L-M calibration cells, take q_ε of |ε| at level α₂.
  - Since |ε+δ| ≤ |ε| + |δ|, the interval ĥ(G) ± (q_δ + q_ε) covers M with probability ≥ 1 − α₁ − α₂ for a new cell from the common population.
  - No assumption on the G-M coupling is needed, only exchangeability within each file and transportability (section 1.3).
  - Under the CIA the interval can be tightened by convolving the two residual distributions. Report both.
  - The closest published two-stage conformal method (Zhang, Amin & Perakis, https://arxiv.org/html/2510.04406v1) assumes the intermediate variable is observed **jointly** with both ends in every calibration triplet. Our disjoint-calibration version is therefore not covered by it.

**Why it fits.** This is the only route I found to an evaluation of G→M that needs **zero** G-M pairs and states its assumptions. It also gives a decision rule:
- If b ≪ a (G recovers almost all of L's marker information), the G→M error is pinned near the L→M error *whatever the coupling*.
- If b ≈ a, the honest answer is "unknown", and no model can fix that without G-M data.

**Adaptation.** Per-marker tables with:
- R² under the CIA,
- the R² interval implied by the bounds,
- conformal interval widths,
- the cell-type Fréchet bounds.

Run leave-one-batch-out, given only 2 batches. Use density-ratio weights on L if the transportability test fails.

**Cheapest experiment.** CPU, a few hours, using any existing frozen embeddings plus MLP or ridge heads:
1. Cross-fit m̂ and ĥ.
2. Compute a_k and b_k with K ∈ {1, 10, 50}.
3. Compute bounds and conformal widths.
4. Run the transportability classifier.
5. Validate bound coverage on the mosaic simulation in 2.2, where truth is known.

**Risk.**
- The bounds may be wide. That is the correct answer, not a failure.
- Heavy-tailed marker intensities make Cauchy-Schwarz loose, so use rearrangement bounds or rank-transformed markers.
- The bounds are for the *composite* predictor ĥ. They say nothing about whether G carries marker information that L lacks (G-unique signal). Only G-M data could show that.

### 2.2 [Rank 1, same experiment] An in-dataset mosaic simulation benchmark with ground truth [proposal]

**Mechanism.** Build artificial "never co-observed" files from data where the truth *is* observed, then measure how wrong transitive methods are and whether the bounds from 2.1 cover the truth.

Two BSCCM constructions:
- **(a) Marker-marker.** In the all-antibody condition, several markers are co-observed on the same cell. Take two separately resolved channels; per the BSCCM paper, CD14/HLA-DR/CD123 and CD3/CD19/CD56 are merged, so check which channels are separate.
  - Fit L→A on single-stain-A cells and L→B on single-stain-B cells.
  - Estimate their joint (correlation, 2×2 gate concordance) under the CIA, and the bounds.
  - Compare with the truth on all-antibody cells.
  - This is literally two files sharing only L, with the real L.
  - Caveat: demixing differs between single and all-antibody conditions, so use rank-based or gating metrics.
- **(b) Pseudo-G.** On L-M cells:
  - Declare a subset S_G of LED patterns as "pseudo-G" and the rest S_L as the hub.
  - Split cells into disjoint sets: (pseudo-G, hub) and (hub, M).
  - Train any transitive method, and evaluate pseudo-G→M on held-out cells where all three are observed.
  - **Dial the CIA violation** by moving marker-informative patterns from the hub into S_G only. The hub then misses information that pseudo-G has, and the CIA fails in a known direction.

This gives a ground-truth leaderboard for *every* method in this review: distillation, contrastive hub, any-to-any, unpaired OT.

**Analogues in biology.** Hub-mediated "mosaic" or "bridge" integration is established in single-cell omics:
- StabMap projects cells along shortest paths through a mosaic data topology, including datasets that share no features (https://www.biorxiv.org/content/10.1101/2022.02.24.481823v1).
- Seurat bridge integration uses a multiomic bridge dataset as a dictionary (Hao et al., https://www.biorxiv.org/content/10.1101/2022.02.24.481684v1.full.pdf+html).

I did not verify how either validates its multi-hop mappings [unverified]. I found no imaging or cytology analogue.

**Risk.** Pseudo-G shares physics with L, so it is easier than real Wright stain. It calibrates *methods* and *bounds*, not the true G-M link.

### 2.2b Within-BSCCM population check [proposal]

G-set and M-set cells come from the same donors and batches. So the distribution of ĥ(G) on G cells must match the observed marker distribution of M cells, per batch and per morphology cluster.

Test with an energy distance or MMD. This is necessary but not sufficient. It runs in minutes and catches gross failures such as batch leakage or calibration drift.

### 2.3 [Rank 2] Generalized distillation (LUPI) with a modality-general teacher

**Mechanism.** Lopez-Paz, Bottou, Schölkopf & Vapnik, ICLR 2016 (https://ar5iv.labs.arxiv.org/html/1511.03643):
1. Train a teacher f_t on privileged features x*.
2. Form soft labels σ(f_t(x*)/T).
3. Train the student on regular features x with (1−λ)·CE(y, ·) + λ·CE(soft, ·).

The theory suggests a benefit when the teacher learns faster or with less error. The paper's causal discussion says privileged information helps when it carries information about p(y|x).

The **Modality Focusing Hypothesis** (Xue et al., ICLR 2023, https://ar5iv.labs.arxiv.org/html/2206.06487) states that "distillation performance is dependent on the proportion of modality-general decisive features preserved in the teacher network" (γ). A *more accurate* teacher can produce a *worse* student when its accuracy comes from modality-specific features. The paper gives no method to measure γ on real data.

**Why it fits.** At deployment (clinical smears) only G exists. L and M are training-only, i.e. privileged information. The G-L pairs have no M labels, so this is pure distillation (λ = 1): the teacher T(L) is trained on about 350k L-M cells, and the student S(G) on 15.7k G-L pairs. The student learns E[T(L)|G], which is the composite ĥ from 2.1. Framing it as LUPI brings in MFH's warning: **teacher accuracy on L→M is the wrong model-selection criterion.** What matters is how much of the teacher's signal lives in the G-L shared content.

**Adaptation** [proposal]. Arms:
- **(i)** Raw teacher T on full L.
- **(ii)** "Modality-general" teacher T_gen. First learn the G-L shared component of the L-embedding, e.g. top CCA directions, or the L-branch of a G-L contrastive model, which by Daunhawer's theorem keeps c_GL. Then train T_gen on that component using all L-M cells.
- **(iii)** Feature-level distillation, where the student predicts the teacher's penultimate L features. This is JEPA-like latent prediction; see 2.12.
- **(iv)** Heteroscedastic teacher (mean and variance), so the student also inherits the ε-scale needed for 2.1.

MFH predicts that (ii) gives a better student than (i) even though T_gen is a worse L→M model. That is a falsifiable, cheap test.

**Prior adaptation.**
- Cross-modal KD with a privileged modality exists in computational pathology at slide level:
  - MoMKD, genomics → histology, CVPR 2026 (https://arxiv.org/abs/2602.21395).
  - Multi-modal knowledge decomposition, which explicitly separates modality-general from modality-specific teacher features for IHC-biomarker prediction from H&E (https://www.emergentmind.com/papers/2508.17213).
- LUPI was used in adipocyte virtual staining [unverified: seen only in a search snippet].
- Not found for single-cell hematology.

**Cheapest experiment.** One GPU, under 1 h per arm:
- G features: frozen DinoBloom. It is a hematology DINOv2 model trained on 13 public datasets with over 380k WBC images, in four sizes (https://arxiv.org/abs/2404.05022).
- L encoder: small, trained on L-M.
- MLP heads.
- Metrics: b² from 2.1, bound intervals, and pseudo-G truth from 2.2(b).

**Risk.**
- The student's ceiling is the CIA composite.
- A teacher fitted on 2 batches can encode batch artefacts that leak into the student. Use leave-one-batch-out.
- MFH is supported by synthetic and proxy evidence, not a measurable γ.

### 2.4 [Rank 3] Semi-supervised G-L alignment with unpaired data, and L as a stain-invariance teacher

**Mechanism.**
- **SOTAlign** (ICML 2026; https://arxiv.org/html/2602.23353v1):
  - Fit a linear teacher on a few pairs (Procrustes, CCA, or linear contrastive; CCA worked best).
  - Train linear alignment layers with SigLIP on the pairs plus a **KLOT** regularizer on unpaired batches. KLOT is the KL divergence between entropic-OT plans from the teacher's affinities and the learned affinities, with an explicit gradient.
  - Default setup: 10k pairs and 1M unpaired samples. Gains are largest at **10³-10⁴ pairs** and continue up to about 500k unpaired samples. All methods fail at 100 pairs.
  - Encoders: frozen DINOv3 and NV-Embed.
- **STRUCTURE** (NeurIPS 2025; https://arxiv.org/abs/2506.16895): preserve each unimodal space's neighbourhood geometry during alignment, and align the layers with the highest cross-modal similarity rather than the last layer. Good alignment from "tens of thousands" of pairs.
- **UML** (https://arxiv.org/html/2510.08492v1): one shared network over modality-specific encoders, trained on *unpaired* data. Under a linear model, Fisher information along shared directions adds up across modalities, so unpaired auxiliary data improves the target modality. Average supervised gain: 81.54 → 83.99 over nine datasets.

**Why it fits.** BSCCM has 15.7k G-L pairs, which is SOTAlign's sweet spot. It also has two large unpaired pools:
- about 397k L cells without G,
- public Wright/Giemsa/MGG WBC images from many sites (DinoBloom's 13 datasets, over 380k cells).

The public G pool has a second effect: it widens G's support over stains and scanners. Clinical transfer needs exactly that. Tsutsui et al. show WBC classifiers degrade substantially across imaging conditions, and that ResNet with batch norm is especially bad (https://arxiv.org/abs/2303.01777v2).

**"L as stain-invariance teacher"** [proposal, based on Daunhawer Theorem 1 and von Kügelgen et al. 2021, https://arxiv.org/abs/2106.04619]:
- Stain hue, stain batch, smear thickness, and scanner colour response are **G-specific latents**: L was imaged before staining and never sees them.
- In Daunhawer's idealized setting, minimizers of G-L contrastive learning cannot depend on modality-specific latents. A G encoder trained against L should therefore *discard stain style* while keeping morphology shared with L.
- Two caveats:
  1. The theorem assumes diffeomorphic mixing, continuous latents, global optima, and known n_c.
  2. Invariance is learned only over stain variation *present in training*, which is one lab. So add stain augmentation (von Kügelgen: augmentations isolate content from style) and the unpaired multi-site G pool.

**Cheapest experiment.** One GPU, a few hours:
- Frozen DinoBloom-B for G and a frozen L encoder.
- Pair budgets {500, 2k, 5k, 15.7k}.
- Arms: CCA, Procrustes, InfoNCE heads, plus STRUCTURE, plus KLOT with (a) unpaired BSCCM-L only and (b) also public G.
- Metrics: held-out G→L R@1/R@10, b² (2.1), and **cross-site kNN cell-type accuracy** on public datasets. The last one asks whether the aligned G embedding keeps cell-type separation under real stain shift.

**Risk.**
- Public sets differ in class mix and include pathological or bone-marrow cells. SOTAlign reports that gains depend on the shift between paired and unpaired data.
- Check dataset licences.
- KLOT hyperparameters were tuned on CC3M in the paper.

### 2.5 [Rank 4] Dual-target ranking of illumination patterns

**Mechanism.**
- **Information imbalance** Δ(A→B) (Glielmo et al., PNAS Nexus 2022; https://pmc.ncbi.nlm.nih.gov/articles/PMC9802303): asks how well nearest neighbours in distance space A predict ranks in space B.
  - Asymmetric, roughly in (0,1).
  - About 0 means A contains B's neighbourhood information; about 1 means the spaces are independent.
  - Used to select the feature subset that minimizes imbalance toward a target.
- **Concrete autoencoder** (Abid, Balin & Zou, ICML 2019; https://arxiv.org/abs/1901.09346): a differentiable selector of K input features with a temperature-annealed concrete relaxation. On gene expression it beat the expert-curated L1000 landmark genes.
- **PID** (Liang et al., NeurIPS 2023; https://ar5iv.labs.arxiv.org/html/2302.12247): redundancy, uniqueness, and synergy of two input groups with respect to a *finite* Y.
  - The CVX estimator needs discretized inputs.
  - The BATCH estimator handles continuous high-dimensional X with discrete Y.
  - Case studies include TCGA histology plus genomics.
- **IDEAL** (Pinkard et al., NeurIPS 2025; https://arxiv.org/abs/2405.20559v5): the BSCCM authors' own task-agnostic upper-bound estimate of I(object; measurement), used to design imaging systems. This is the natural baseline.

**Why it fits** [proposal]. The project's question "which patterns carry information" has **two targets**: M (on L-M cells) and G (on G-L cells, with target = distances in a frozen G embedding). Three outcomes are possible for a pattern:
- informative for **both**: a "bridge pattern" that carries the transitive G-M signal;
- informative for M only: contributes to a²-side signal that G cannot recover (the b-gap in 2.1);
- informative for G only: carries morphology irrelevant to markers.

The rankings can be compared with Kendall-τ, and against IDEAL's task-agnostic ranking. A further cheap score is mutual-kNN alignment (PRH metric, k = 10; https://arxiv.org/html/2405.07987) between a pattern subset's embedding and the G or M representation on paired cells.

**Cheapest experiment.** CPU only, about 1 day:
- Per-pattern features: frozen generic CNN embedding per channel, or radial intensity statistics.
- Δ(L_S → M_j) per marker j, using cells stained for j.
- Δ(L_S → G-emb) on G-L cells.
- Greedy forward selection up to 6 patterns.
- Supervised concrete selector over the 23 patterns with K = 1..6, with heads for M and for G-emb.
- PID on pairs of pattern groups with Y = 3-class gated labels.

**Risk.**
- Rankings depend on featurization.
- Partially observed markers cut samples per marker.
- kNN distances may be dominated by cell size or brightness, so normalize.
- CVX PID needs binning.

### 2.6 [Rank 5] Hub-and-spoke contrastive embedding with Rank-N-Contrast for partially observed continuous markers

**Mechanism.**
- **ImageBind** (CVPR 2023; https://arxiv.org/abs/2305.05665): binds six modalities using only image-paired data, and claims emergent zero-shot ability between non-image pairs.
- **C-MCR** (NeurIPS 2023; https://arxiv.org/abs/2305.14381): connects an (A,B) contrastive space and a (B,C) space through overlapping modality B, with no A-C pairs. It reached state of the art on audio-image tasks by connecting CLIP and CLAP via text.
- **Critiques:**
  - CentroBind (https://arxiv.org/html/2410.02086v2) names over-reliance on the anchor modality and no shared information among non-anchored modalities as problems. Its non-anchor retrieval on MUStARD went from top-1 0.051 with fixed-anchor binding to 0.232, but on 690 samples.
  - EmergentBridge (https://arxiv.org/html/2604.11043v5) reports that never-paired modality pairs stay "weakly aligned".
- **Rank-N-Contrast** (NeurIPS 2023; https://arxiv.org/html/2210.01189):
  - For anchor i and positive j, the denominator runs over all k whose label distance to i is ≥ d(y_i, y_j).
  - This guarantees representations ordered by target value.
  - Two-stage training: RNC encoder, then a frozen-encoder regressor.
  - Reported MAE gains, e.g. AgeDB 6.63 → 6.14, and better robustness and transfer.

**Why it fits.** L is the only modality paired with both others, so it is the natural hub. G-M is the "emergent" pair. The literature says to expect weak emergent alignment unless marker-relevant structure sits in the G∩L content, which is consistent with section 1.

The markers are continuous, and mostly only one is observed per cell. RNC is a natural objective for continuous markers and can be computed **per marker within cells stained for that marker** (masked RNC) [proposal]. That avoids imputing missing markers.

**Adaptation** [proposal]:
- InfoNCE(G, L) on G-L pairs.
- Masked per-marker RNC on the L-embedding using L-M cells. The "M embedding" is then the predicted marker vector, or a set-transformer over (marker-id, value) tokens.
- Alternative: C-MCR-style connection of two separately trained spaces through L.
- Evaluate the emergent G-M link with 2.1 and 2.2, not with anecdotes.

**Cheapest experiment.** One GPU:
- L encoder with masked RNC vs MSE on per-marker targets.
- L→M R² at 10% and 100% of the data.
- Then G-M via the hub on the pseudo-G benchmark.

**Risk.**
- RNC denominators grow as O(N²) per batch.
- Masks shrink effective batches. Build batches stratified by stained marker.
- Emergent G-M may be weak by construction.

### 2.7 Population-level clinical ground truth: label proportions and prediction-powered inference

**Mechanism.**
- **Learning from label proportions (LLP).** Train or evaluate with bag-level proportions only. Example: PD-L1 proportion estimation with interval-valued proportion labels and a proportion loss (Okuo et al. 2024; https://arxiv.org/html/2405.04815v1).
- **PPI** (https://arxiv.org/abs/2301.09633): valid confidence intervals for population quantities from a small gold-labelled set plus many model predictions.
  - PPI++ adapts to prediction quality and always improves on classical intervals (https://arxiv.org/abs/2311.01453).
  - Generalized PPI handles three forms of covariate shift and classifier evaluation (TPR, FPR, AUC) (https://arxiv.org/abs/2602.10332).

**Why it fits** [proposal]. In clinics, smears often come with a CBC differential, and sometimes flow-cytometry immunophenotyping, *from the same blood sample* [unverified: availability depends on the partner hospital]. That gives **bag-level G-M ground truth**: per-smear fractions of CD3+, CD19+, CD16+, CD14+ cells, etc.

This is the only realistic external validation of G→M in the target domain:
- PPI gives cohort-level CIs.
- LLP can fine-tune the student on the clinical domain.
- Weighted conformal (https://arxiv.org/abs/1904.06019) can adapt intervals with a G-embedding density ratio. The L-side term of 2.1 cannot be reweighted at the clinic because L is unobserved there. Treat it as label shift (cell-type priors) [proposal].

**Risk.**
- Proportions are weak labels: correct fractions are possible with wrong cells.
- Smear and flow populations differ (RBC lysis, smear selection bias).

### 2.8 Direct unpaired G-M alignment: triangulation and negative control, not the main route

**Mechanism.**
- **vec2vec** (https://arxiv.org/html/2505.12540): input and output adapters around a shared backbone, trained with adversarial (output and latent), reconstruction, cycle, and vector-space-preservation losses, with no pairs.
  - Data efficiency (gte→gtr): **10k embeddings → top-1 0.01; 50k → 0.81**.
  - CLIP translations are much weaker.
  - GAN seed sensitivity: e5→gtr reached at least 80% top-1 in only 3 of 15 seeds.
  - Removing the vector-space-preservation loss drops top-1 to 0.
- **Blind matching as a quadratic assignment problem** (CVPR 2025; https://arxiv.org/abs/2503.24129).
- **Spectral unpaired shared representations** (NeurIPS 2025; https://arxiv.org/abs/2505.21524).
- **Neural Gromov-Monge Alignment**, with identifiability conditions via decoder pullback geometry (NeurIPS 2026 workshop; https://neurips.cc/virtual/2026/159364).
- **Keypoint-guided OT** with GW variants (JMLR per page; https://arxiv.org/abs/2303.13102).
- **SCOT** GW alignment in single-cell omics (https://pmc.ncbi.nlm.nih.gov/articles/PMC8812490).

**Why it fits.** A direct G↔M estimator fails in different ways from the transitive one. If the two agree, that is triangulation evidence [proposal]. Class anchors are available on both sides:
- marker-gated types on the M side (10k cells),
- morphology types on the G side (hematologist labels, or a DinoBloom kNN classifier trained on public labelled smears).

These anchors can serve as keypoints for keypoint-guided OT or fused GW.

**Expected outcome.** I expect failure as a *cell-level* method. G has 15.7k cells, which is in vec2vec's failing regime. M has about 6-8 effective dimensions with one marker per cell, a geometry with many symmetries.

**Cheapest experiment.** CPU: fused GW with class anchors between G-embeddings and all-antibody M vectors. Measure agreement of the coupling with the transitive coupling, and test on pseudo-G (2.2b).

**Risk.** Ambiguous couplings, and only about 3-10 anchors.

### 2.9 Any-to-any masked modelling and multi-noise diffusion with missing modalities

**Mechanism.**
- **4M** (NeurIPS 2023; https://arxiv.org/abs/2312.06647): tokenizes all modalities and predicts masked subsets with one encoder-decoder.
- **MultiMAE** (https://arxiv.org/abs/2204.01678): masks across patches and modalities and trains on pseudo-labelled modalities.
- **UniDiffuser** (ICML 2023; https://arxiv.org/abs/2303.06555): uses per-modality timesteps, so one model fits marginals, conditionals, and joints. My inference, not stated on that page: a fully noised modality acts as "unobserved", so partially observed cells fit naturally.

**Why it fits.** Each cell has a subset of {23 LED channels, Wright RGB, ≤ 8 markers}. One model can learn every conditional, including p(M|G). But whatever coupling it learns comes from inductive bias, i.e. an implicit CIA. So it needs the 2.1/2.2 evaluation, not a claim of "emergent" ability [proposal].

There is a bonus: masked-channel reconstruction across the 23 patterns gives a redundancy-based pattern ranking (compare 2.5).

**Prior adaptation.**
- Masked channel modelling for multiplexed single-cell tissue imaging at MICCAI 2025 (https://papers.miccai.org/miccai-2025/paper/4789_paper.pdf). The PDF text was not readable, so details are [unverified] from a search snippet.
- Tissue-level any-to-any models exist.
- Nothing found for LED-array, histology, and marker data together.

**Risk.** Compute; a silent CIA; and the important L→M link may be underfit relative to a dedicated regressor. EV medium, cost high.

### 2.10 Bridges and stochastic interpolants for virtual Wright staining from L

**Mechanism.**
- **I2SB** (https://arxiv.org/abs/2302.05872): a paired, simulation-free Schrödinger bridge between degraded and clean images.
- **UNSB** (ICLR 2024; https://arxiv.org/abs/2305.15086): an unpaired Schrödinger bridge posed as a sequence of adversarial problems.
- **OTCS** (NeurIPS 2023; https://arxiv.org/abs/2311.01226): a conditional score model for unpaired or partially paired data, using a coupling from L2-regularized unsupervised or semi-supervised OT plus resampling-by-compatibility.
- **Multimarginal stochastic interpolants** (https://arxiv.org/abs/2310.03695): transport on a simplex across K marginals, extracting multi-way correspondences.

**Why it fits.** G-L pairs are **cell-matched but not pixel-registered** (BSCCM paper), and the stain was applied after LED imaging. Strictly paired bridges (I2SB) assume more alignment than the data have. Semi-paired couplings (OTCS) or a UNSB with a pair-consistency term fit better [proposal]. There are also about 397k unpaired L cells.

**Prior work.**
- Deep-UV "Giemsa-like" virtual staining of blood smears used *UV pseudocolour* targets, not real Giemsa images (Kaza et al., BME Frontiers 2022; https://pmc.ncbi.nlm.nih.gov/articles/PMC10521747/).
- Schrödinger-bridge virtual staining is emerging in histopathology, e.g. SheafStain (https://arxiv.org/html/2606.11846v1) [unverified: details seen only via search].
- No LED-array → Wright stain work found.

**Cheapest experiment.** One GPU:
- I2SB vs an OTCS-style semi-paired model at 64×64.
- Score with distances in DinoBloom feature space.
- **Leak test** [proposal]: apply the 2.3 student to virtual G̃(L) for L-M cells. The virtual stain must not let the student predict M better than real G does (compare with b² on real pairs). Otherwise G̃ is smuggling L-only marker information.

**Risk.** Hallucination. Pseudo-pairing (G̃, M) for 350k cells is tempting but **inflates** G→M unless G̃ is restricted to G-L shared content.

### 2.11 Noisy-correspondence audit of G-L pairs (cheap, do early)

**Mechanism.** NCR (NeurIPS 2021; https://proceedings.neurips.cc/paper/2021/hash/f5e62af885293cf4d511ceef31e61c80-Abstract.html, code https://github.com/XLearning-SCU/2021-NeurIPS-NCR). It treats mismatched pairs as "noisy correspondence", separates clean from noisy pairs using networks' memorization tendency, and rectifies them with co-teaching. The method description comes from the paper listing via search; the proceedings page refused connection.

**Why it fits.** Pairs come from a global slide alignment plus nearest-cell matching. A few percent of wrong pairs would bias every G-side number: b², the distillation targets, and virtual-stain training.

**Cheapest experiment.** CPU or GPU, about 1 hour:
- Fit a G-L contrastive head and inspect the per-pair loss distribution for bimodality.
- If the metadata contain inter-cell distances, check first-to-second nearest-neighbour distance ratios.

### 2.12 Cross-modal JEPA and per-modality MoE (low EV here)

- **M3-JEPA** (ICML 2025; https://arxiv.org/abs/2409.05929): latent-space prediction across modalities with a multi-gate mixture-of-experts predictor and contrastive plus regularization losses.
- **I-JEPA** (ICCV 2023; https://arxiv.org/abs/2301.08243).
- **LIMoE** (https://arxiv.org/abs/2206.02770): modality-specific experts emerge in sparse contrastive MoE, which needs entropy regularization for balance.

For BSCCM, "predict L-latents from G" is the same as feature distillation (2.3, arm iii). MoE adds capacity that 15.7k pairs cannot use. Include only as a distillation arm.

### 2.13 Test-time training for clinical shift (later, cheap)

**TTT-MAE** (https://arxiv.org/abs/2209.07522) updates the encoder per test input with an MAE loss and improves robustness under distribution shift. It is a natural later add-on to the DinoBloom G encoder for unseen hospitals. It changes nothing about the G-M identification problem.

### 2.14 Platonic Representation Hypothesis: use as a diagnostic, not a premise

- PRH (https://arxiv.org/html/2405.07987) measures mutual k-NN overlap (k = 10). The best cross-modal score reported is only about 0.16. The authors note that lossy or stochastic observations cap alignment at the level of shared information.
- In scientific foundation models (https://arxiv.org/html/2512.03750v1), alignment is strong in-distribution, but models collapse onto a low-information representation out of distribution.
- In bioimaging, untrained or simple baselines often match state-of-the-art SSL (ICML 2026; https://arxiv.org/abs/2603.13377).

Recommendation:
- Compute mutual-kNN and CKNNA between frozen G, L, and M representations on the paired sets, to estimate *how much* is shared and to choose layers (STRUCTURE's insight).
- Do **not** assume cytology encoders converge to a shared geometry, which is what vec2vec-style unpaired translation would need.

---

## 3. Ranking rationale (expected value = value to the project goals × probability of a useful result ÷ cost)

1. **Partial identification + union-bound conformal + mosaic simulation (2.1, 2.2, 2.2b).**
   - Directly answers the goal "establish credible evaluation of the never-co-observed G-M link".
   - Near-certain to produce a useful result: even wide bounds are a finding.
   - CPU-hours. Not found anywhere in imaging or cytology.
   - Also the yardstick for ranking items 2-5.
2. **Generalized distillation with a modality-general teacher (2.3).**
   - This *is* the deployable G→M model for clinics.
   - MFH gives a falsifiable prediction: T_gen beats T as a teacher.
   - Cheap with frozen DinoBloom.
3. **Semi-supervised G-L alignment with unpaired L and public G, plus the L-as-stain-teacher argument (2.4).**
   - Targets the weakest link (only 15.7k G-L pairs) and clinical domain shift at once.
   - 15.7k pairs is exactly where SOTAlign and STRUCTURE report their largest gains.
4. **Dual-target illumination ranking (2.5).**
   - Directly answers the "rank illumination patterns" goal.
   - Novel twist: ranking against *both* targets identifies bridge patterns. Also explains the b-gap from 2.1.
   - CPU only.
5. **Hub contrastive + masked RNC (2.6).**
   - Needed for the "one shared embedding" goal.
   - Lower novelty, and the literature predicts weak emergent G-M alignment, so value depends on 1-3.

Valuable but outside the top 5:
- LLP/PPI clinical validation (2.7): high value, but blocked on external data.
- Noisy-correspondence audit (2.11): very cheap, should run before 2-3.
- Bridges for virtual staining (2.10): needed for the virtual-staining goal, but high risk of leakage.
- Any-to-any models (2.9).
- Direct unpaired G-M alignment (2.8): negative control only.

## 4. Caveats and open questions

- **Wright vs Giemsa.** The BSCCM paper says Wright's stain. Clinical sites use Wright, Wright-Giemsa, or May-Grünwald-Giemsa, which is itself a domain-shift axis. Treat "G" as "Romanowsky-type stain", and record the stain type for every external dataset.
- **Only 2 batches.** All exchangeability-based guarantees (conformal, PPI) are within-batch-population guarantees. Report leave-one-batch-out results.
- **G-unique marker information** (e.g. granule chemistry visible only in stain) is *unknowable* without G-M pairs. All transitive methods underestimate it, and the bounds in 2.1 cannot detect it. The only remedy is new data: a small set of cells imaged under Wright stain *and* antibody fluorescence, or the bag-level clinical data from 2.7.
- **Theory gap.** Yao et al. leave identifiability under "overlapping but not simultaneously observed marginals" open. A proof or counterexample for the G-L-M chain would be publishable on its own [proposal].
- **Not verified in full text:** the venues of PRH, MultiMAE, TTT-MAE, DinoBloom, and Yao et al. (their arXiv pages list none or I did not check); the details of the MICCAI 2025 multiplex paper and SheafStain; the adipocyte LUPI paper; and the validation designs of StabMap and Seurat bridge integration.

## 5. Sources opened in this session

- BSCCM dataset: https://arxiv.org/html/2402.06191v1
- IDEAL / information-driven design: https://arxiv.org/abs/2405.20559v5
- Platonic Representation Hypothesis: https://arxiv.org/html/2405.07987 , https://arxiv.org/abs/2405.07987
- vec2vec: https://arxiv.org/html/2505.12540 , https://arxiv.org/abs/2505.12540
- Relative representations: https://arxiv.org/abs/2209.15430
- ASIF: https://arxiv.org/abs/2210.01738
- Latent space translation via semantic alignment: https://arxiv.org/abs/2311.00664
- SOTAlign: https://arxiv.org/html/2602.23353v1
- STRUCTURE: https://arxiv.org/abs/2506.16895
- UML / Better Together: https://arxiv.org/html/2510.08492v1
- ImageBind: https://arxiv.org/abs/2305.05665
- C-MCR: https://arxiv.org/abs/2305.14381
- CentroBind: https://arxiv.org/html/2410.02086v2
- EmergentBridge: https://arxiv.org/html/2604.11043v5
- Symile: https://arxiv.org/abs/2411.01053 (higher-order total-correlation contrastive; the abstract claims robustness to missing modalities, but whether it can learn a G-M term with no G-M co-observation is [unverified]; probably not applicable)
- GRAM: https://arxiv.org/abs/2412.11959 (n-modality Gramian-volume alignment; whether every modality must be co-observed is [unverified])
- Daunhawer et al.: https://ar5iv.labs.arxiv.org/html/2303.09166
- Yao et al.: https://ar5iv.labs.arxiv.org/html/2311.04056
- Sturma et al.: https://ar5iv.labs.arxiv.org/html/2302.00993
- von Kügelgen et al.: https://arxiv.org/abs/2106.04619
- Generalized distillation: https://ar5iv.labs.arxiv.org/html/1511.03643
- Modality Focusing Hypothesis: https://ar5iv.labs.arxiv.org/html/2206.06487
- MoMKD: https://arxiv.org/abs/2602.21395
- MKD (IHC biomarkers): https://www.emergentmind.com/papers/2508.17213
- StatMatch: https://cran.mi2.ai/web/packages/StatMatch/refman/StatMatch.html
- Fan, Sherman & Shum 2014: https://ideas.repec.org/a/wly/emetrp/v82y2014i2p811-822.html
- Surrogate index: https://www.nber.org/papers/w26463 , https://opportunityinsights.org/paper/the-surrogate-index/
- PPI: https://arxiv.org/abs/2301.09633
- PPI++: https://arxiv.org/abs/2311.01453
- Generalized PPI: https://arxiv.org/abs/2602.10332
- Weighted conformal under covariate shift: https://arxiv.org/abs/1904.06019
- Two-stage conformal: https://arxiv.org/html/2510.04406v1
- PID: https://ar5iv.labs.arxiv.org/html/2302.12247
- Multimodal learning without labeled multimodal data: https://arxiv.org/abs/2306.04539
- Rank-N-Contrast: https://arxiv.org/html/2210.01189
- 4M: https://arxiv.org/abs/2312.06647
- MultiMAE: https://arxiv.org/abs/2204.01678
- UniDiffuser: https://arxiv.org/abs/2303.06555
- I2SB: https://arxiv.org/abs/2302.05872
- UNSB: https://arxiv.org/abs/2305.15086
- OTCS: https://arxiv.org/abs/2311.01226
- Multimarginal stochastic interpolants: https://arxiv.org/abs/2310.03695
- Keypoint-guided OT: https://arxiv.org/abs/2303.13102
- SCOT: https://pmc.ncbi.nlm.nih.gov/articles/PMC8812490
- Blind match: https://arxiv.org/abs/2503.24129
- Spectral unpaired shared representations: https://arxiv.org/abs/2505.21524
- Neural Gromov-Monge Alignment: https://neurips.cc/virtual/2026/159364
- DinoBloom: https://arxiv.org/abs/2404.05022
- WBC classification under domain shift: https://arxiv.org/abs/2303.01777v2
- TTT-MAE: https://arxiv.org/abs/2209.07522
- I-JEPA: https://arxiv.org/abs/2301.08243
- M3-JEPA: https://arxiv.org/abs/2409.05929
- LIMoE: https://arxiv.org/abs/2206.02770
- Information imbalance: https://pmc.ncbi.nlm.nih.gov/articles/PMC9802303
- Concrete autoencoder: https://arxiv.org/abs/1901.09346
- Masked LLP: https://arxiv.org/html/2405.04815v1
- Deep-UV virtual staining of blood smears: https://pmc.ncbi.nlm.nih.gov/articles/PMC10521747/
- StabMap: https://www.biorxiv.org/content/10.1101/2022.02.24.481823v1
- Seurat bridge integration: https://www.biorxiv.org/content/10.1101/2022.02.24.481684v1.full.pdf+html
- Universally converging representations of matter: https://arxiv.org/html/2512.03750v1
- Deep Learning for BioImaging: https://arxiv.org/abs/2603.13377
- NCR code: https://github.com/XLearning-SCU/2021-NeurIPS-NCR (the proceedings page refused connection)
