# F: "Proving and Improving the Platonic Representation Hypothesis with JEPAs" (Balestriero & LeCun, arXiv 2610.39302v1), close read and fit to BSCCM (LF / GS / MK)

Agent F, N002 literature review, 2026-11-08.

**Sources.** I read the paper from the arXiv **TeX source of v1**, the only version (submitted 29 Oct 2026, CC BY 4.0, cs.AI + cs.CV + stat.ML). I cross-checked theorem, section, equation, table and figure numbers against the **HTML rendering**. No news summaries were used. The paper is a 10-day-old preprint and has not been peer-reviewed.
- Abstract: https://arxiv.org/abs/2610.39302
- HTML (used for anchors): https://arxiv.org/html/2610.39302v1
- PDF: https://arxiv.org/pdf/2610.39302
- Code, linked in the paper header: https://github.com/galilai-group/prh-proof. The URL resolves (HTTP 200), but **I did not read the code**.
- Demo: https://huggingface.co/spaces/galilai-group/prh_demo (HTTP 200, not used).
- Context paper, the original PRH (Huh et al. 2024): https://arxiv.org/abs/2405.07987. I read §4 and §6 of HTML v5: https://arxiv.org/html/2405.07987v5

**How claims are cited.** Each claim links to a section anchor, for example [§2.1 Thm 1](https://arxiv.org/html/2610.39302v1#S2.SS1). For equation numbers, Eq. 1, 2 and 7 were checked against the HTML anchors. Eq. 3–6 are the remaining display equations of §2 (anchors S2.E3–E6) in the order they appear.

**Labels.** Unlabelled statements in §1–§4 are the paper's own content. Within the BSCCM fit (§5), the labels mean:
- **[analysis]**: my own derivation or inference, not stated in the paper.
- **[unverified]**: not checked in any primary source.
- **(project notebook)**: a BSCCM fact taken from this repo's N001/N003/N002-SYNTHESIS records.

---

## 0. Verdict

1. **What is proven.** The theorem is about **population minimisers** of an idealised JEPA. It assumes a **finite world**, rendered **injectively** (losslessly) in every modality. If every modality uses the **same positive-pair distribution over world states**, and the minimiser is unique up to rotation, then the optimal encoders agree **up to one orthogonal map in O(k)**. Uniqueness up to rotation is proven only for squared-error prediction with a hard whitening constraint, plus an eigengap. It is not proven for LeJEPA's SIGReg, which is what the experiments actually use. ([§2.1 Thm 1, Eq. 2; Prop 1](https://arxiv.org/html/2610.39302v1#S2.SS1); [§7 Known limitations](https://arxiv.org/html/2610.39302v1#S7))
2. **For BSCCM the hypotheses fail.** LF and GS are lossy, non-injective views of a cell, each seeing different physics. Separately trained JEPAs with their own augmentations induce **different** positive kernels: the paper's Thm 4 says a bad-view JEPA "does not choose its world kernel". Both would be **bad-view JEPAs** by the paper's own definition, because their positives are corruptions of a single observation. Bad-view JEPAs provably spend embedding directions on modality nuisance, which no rotation can align (Thm 3, Prop 3).
   - **(a)** The theorem therefore does **not** say that separately trained LF-JEPA and GS-JEPA will match up to a rotation fitted on a few pairs. The paper's own closest analogue is two encoders trained from scratch, one per modality, on Flickr. They reach only **R@1 = 4.4 %** through a rotation fitted on 1,014 pairs ([Table 1](https://arxiv.org/html/2610.39302v1#S3.T1)).
3. **[analysis] A degeneracy the paper does not discuss.** Suppose good views pair only observations of the *same* state, as in "two captions of one image" (Fig. 3a: "join every description of a scene and never another"). Then the world kernel is the identity, every eigenvalue equals 1, and Prop 1's eigengap condition fails. The theory then guarantees only that the embedding is *some* function of the state, not one that is unique up to rotation. What makes two encoders agree in practice is which *different* states the positives (and the network's inductive bias) link together. The paper places this outside the formal model ([App A.4, "Special cases and scope"](https://arxiv.org/html/2610.39302v1#A1.SS4)). For cells, this means "two views of the same cell" cannot by themselves force rotation-equivalence.
4. **(b) No help with the transitive link.** At best, rotation alignment carries the part of the information that GS and LF *share*. In the model, anything unique to a modality is "nuisance". So GS→LF→MK through a rotation estimates MK only from GS∩LF information that lives in the leading shared spectral directions. That is no more than the distillation route in SYNTHESIS item 2, and it says nothing about the conditional-independence assumption (CIA, MK ⟂ GS | LF). The one practical gain is regularisation: the paper reports that a rotation fitted on in-domain pairs transfers out of domain better than an unconstrained linear map or an MLP fitted on the same pairs (COCO CIDEr .802 vs .635 / .585, [§5.1, Table 10](https://arxiv.org/html/2610.39302v1#A3.T10)). That is cheap to test across our two stained slides.
5. **(c) Little for clinical smears.** There are no same-cell pairs across labs, the class mix differs, and Thm 2 assumes matched marginals. The usable idea is to treat lab, stain and scanner as **nuisance** and to build views that **resample** that nuisance independently, which is a principled form of stain/optics augmentation. The paper's pair-free route (Wasserstein-Procrustes, §5.2) needs large pools, is very seed-variable (16.4 % ± 10.4 % R@1 on 15k-image Flickr pools), and is at chance on ~1k-sample clouds.
6. **(d) Cheapest test.** First, a zero-training screen with frozen encoders: Procrustes accuracy against the number of pairs, evaluated on a held-out slide. Then a ~1-seed screen of LF-JEPA and GS-JEPA trained separately, compared with a SigLIP two-tower at matched pair counts. My prior is that SigLIP wins whenever there are ≥ ~1k pairs, and the rotation can win only at ≤ ~100–300 pairs or on cross-slide out-of-distribution robustness. Section 5(d) has the design and kill criteria.

---

## 1. The theorem, precisely

### 1.1 Setting ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1))
- **World and renderings.** The world is a **finite** set of states z ∈ Z with p(z) > 0. There are M ≥ 2 modalities, and modality m observes the world through an **injective rendering** x = g_m(z), such as the pixels of a photograph or the tokens of a caption.
- **Positive pairs.** A JEPA for modality m is trained on positive pairs (g_m(z), g_m(z')). The pairs are drawn as (z, z') ~ π_m, a **symmetric positive-pair distribution whose two marginals both equal p**.
- **Objective (Eq. 1).** L_m(f) = E_{(z,z')~π_m} ℓ(f(g_m z), f(g_m z')) + Ω(f).
  - Ω is an anti-collapse term that depends on f only through the distribution of its embeddings; it may be a hard constraint.
  - Both ℓ and Ω are invariant to **O(k)**.
  - **The predictor is the identity.** The paper says JEPAs that "predict one embedding from the other through a learned predictor, often conditioned on an action, which our analysis does not cover". This excludes I-JEPA/V-JEPA-style predictors.
- **Which objectives fit.** Spectral contrastive learning and whitened MSE fit exactly. VICReg and Barlow Twins fit only "in a whitened limit", and LeJEPA's SIGReg fits "in expectation over its directions" ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1); [App A.1](https://arxiv.org/html/2610.39302v1#A1.SS1)).
- **Lemma 1.** An injective rendering only changes coordinates: L_m(f) = L_{π_m}(f∘g_m) ([App A.2](https://arxiv.org/html/2610.39302v1#A1.SS2)).

### 1.2 Theorem 1, the Platonic Representation Theorem ([§2.1, Eq. 2](https://arxiv.org/html/2610.39302v1#S2.E2))
> Let M ≥ 2 modalities share one positive-pair distribution, π_1 = … = π_M = π, and let the minimizers of L_π be unique up to rotation. Then there is one embedding of the world h*: Z → R^k such that every minimizer f*_m of L_m, for every modality m, satisfies f*_m ∘ g_m = Q_m h* for some Q_m ∈ O(k), so any two modalities agree up to the single rotation Q_m Q_{m'}^T.

- **Proof.** It is one line: every f*_m∘g_m minimises the same L_π ([App A.2](https://arxiv.org/html/2610.39302v1#A1.SS2)).
- **What "converges".** The theorem concerns **global optima of the population objective**. It says nothing about SGD dynamics or finite samples. The authors' wording: "in this idealized JEPA setting—one finite world rendered injectively, shared positives, population optima, and minimizers unique up to rotation" ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1)).
- **Identifiability group.** O(k), the **orthogonal group**, which includes reflections. It is not GL(k) and not an affine map. The encoders themselves can be any nonlinear functions; only the map between the two *embedding spaces* is orthogonal.
- **Contrast with next-token prediction.** Next-token prediction is identified only up to **GL(k)**, i.e. up to an invertible linear map ([§2.4 Prop 4(iii)](https://arxiv.org/html/2610.39302v1#S2.SS4)).
- **Equal dimension k is required.** When dimensions differ (768 for DINOv2, 384 for their text JEPA), the authors fit "a linear map with orthonormal rows, a case the theory does not cover" ([§2.2, after Prop 2](https://arxiv.org/html/2610.39302v1#S2.SS2)).
- **The headline assumption.** The key assumption is π_1 = π_2: both modalities must declare **the same pairs of world states** to be positives, with the same probabilities. The marginal p (the population of states) must also match.

### 1.3 Proposition 1, when uniqueness up to rotation holds ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1))
- **Conditions.** ℓ(a, b) = ‖a − b‖², and Ω enforces whitening (E_p h = 0, E_p hh^T = I_k), with 1 ≤ k < |Z|.
- **Result.** min L_π = 2 Σ_{i≤k} (1 − μ_i). If μ_k > μ_{k+1}, the minimisers are exactly **Q Φ_π**, where Φ_π(z) = (v_1(z), …, v_k(z)) / √p(z) is the **spectral embedding**: the top non-trivial eigenvectors of K_π = D^{-1/2} Π D^{-1/2}, as in Laplacian eigenmaps.
- **Scope.** The result is proven for second-order (whitening) anti-collapse only. "Prop 1 does not cover uniqueness under [SIGReg's] full-distribution constraint, which we leave as an assumption" ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1)).

### 1.4 What "good views" are ([§2.3](https://arxiv.org/html/2610.39302v1#S2.SS3); formal statement in [App A.4](https://arxiv.org/html/2610.39302v1#A1.SS4))
- **Definition.** "Call a JEPA *good-view* when its positives are two observations of world states that the world itself relates, e.g., two photographs or two descriptions of one scene, and *bad-view* when they are corruptions of a single observation, e.g., two crops of one photograph or two maskings of one sentence."
- **Formal model.** Each observation is x = g_m(z, ν), with a modality-specific **nuisance ν drawn independently of z** and the rendering injective.
  - **Good views:** (g_m(z, ν), g_m(z', ν')) with (z, z') ~ π and **independent** ν, ν'.
  - **Bad views:** two corruptions of one observation, which **share its nuisance**.
- **Practical recipe they recommend.** "views drawn from different observations of the same world state, each trimmed (cropped, for images) and lightly masked, trained with gradient clipping" ([§3.2](https://arxiv.org/html/2610.39302v1#S3.SS2)). Trimming is "the knob that matters": using whole captions as views loses about a quarter of the alignment (R@1 .185 vs .246).
- **The image side of their experiments is bad-view.** The authors call their image encoders, and DINOv2, bad-view JEPAs, because the positives are crops of one photograph. Good image views require several observations of one scene, such as video frames. That was tested only on MSR-VTT, and "scaling good image views is left to future work" ([§2.3](https://arxiv.org/html/2610.39302v1#S2.SS3)).

### 1.5 [analysis] A degeneracy relevant to cells
- **The derivation.** Fig. 3(a) says that good views on Flickr/COCO "join every description of a scene and never another" ([Fig. 3, §2.3](https://arxiv.org/html/2610.39302v1#S2.F3)). Treat each scene as a world state. Then π(z, z') = p(z)·1[z = z'], so K_π = I and μ_1 = … = μ_{|Z|−1} = 1.
- **The consequence.** For k < |Z| − 1, the gap conditions of Prop 1 and of Prop 3(i) (μ_k > max(μ_{k+1}, 0)) fail. Every centred, whitened function of z then reaches loss 0. Good-view encoders are functions of the state (ρ = 1), but they are **not unique up to rotation**, so Thm 1 gives no guarantee.
- **Where the shared geometry really comes from.** It must come from positives that link *different* states, such as trimming or masking overlaps across scenes (the "trimming" knob), and from the network's inductive bias. The paper puts both outside the formal model: "inductive biases that merge observations of different states, or act softly, are outside the model" ([App A.4](https://arxiv.org/html/2610.39302v1#A1.SS4)). It also acknowledges in general that "finite-sample training relies on the network's inductive bias" ([§7](https://arxiv.org/html/2610.39302v1#S7)).
- **For BSCCM:** a "same cell, two views" scheme does not, by the theorem, produce a representation that is rotation-unique across modalities. What would be shared is determined by **which different cells get linked**.

---

## 2. Bad views and mismatched positives: the bounds

### 2.1 Theorem 2, kernel mismatch bounds misalignment ([§2.2, Eq. 3](https://arxiv.org/html/2610.39302v1#S2.SS2))
- **Statement.** Let π_1 and π_2 share the marginal p, let Φ_1 and Φ_2 be their spectral embeddings, and let δ = μ_k − μ_{k+1} > 0 be the eigengap of K_{π_2}. Then

  min_{Q∈O(k)} ( E_p ‖Φ_1(z) − Q Φ_2(z)‖² )^{1/2} ≤ (2√2 / δ) · min( √k ‖K_{π_1} − K_{π_2}‖_2 , ‖K_{π_1} − K_{π_2}‖_F ).
- **Proof.** It follows from the Davis–Kahan variant of Yu, Wang & Samworth 2015 ([App A.3](https://arxiv.org/html/2610.39302v1#A1.SS3)).
- **Scope and interpretation.** It is proven **for spectral solutions only** ([§7](https://arxiv.org/html/2610.39302v1#S7)). By the authors' heuristic block version, directions separated from the rest by large spectral gaps are protected, while "directions near the end of the spectrum need not align at all" ([§2.2](https://arxiv.org/html/2610.39302v1#S2.SS2)).
- **Empirical check.** On 300 random finite worlds, the error stays below the bound and reaches at most 18 % of it ([Fig. 2](https://arxiv.org/html/2610.39302v1#S2.F2)). The bound is valid but loose.

### 2.2 Proposition 2, k pairs identify the rotation ([§2.2](https://arxiv.org/html/2610.39302v1#S2.SS2))
- **(i)** Every rotation of a whitened embedding is still whitened. So "no statistic of the first two moments of unpaired samples" identifies the rotation.
- **(ii)** If **n ≥ k** paired states have embeddings that span R^k, the rotation is the unique minimiser of the orthogonal Procrustes problem.
- **With mismatched positives,** "more pairs average out the residual misalignment of Thm 2".
- **Unpaired data.** The distribution of a finite world is not rotation-invariant, so higher-order geometry *can* reveal Q without pairs; this underlies §5.2 ([App A.3 proof of Prop 2](https://arxiv.org/html/2610.39302v1#A1.SS3)).

### 2.3 Theorem 3, nuisance costs alignment ([§2.3, Eq. 4](https://arxiv.org/html/2610.39302v1#S2.SS3))
- **Definitions.** Take centred, whitened encoders f_1 and f_2 into R^k. Let C_f = Cov(E[f(x) | z]) and let the **scene-explained fraction** be ρ(f) = tr C_f^{1/2} / k ∈ [0, 1].
- **Statement.** Over paired observations with independent nuisances:

  min_Q E ‖Q f_1(x_1) − f_2(x_2)‖² ≥ 2k (1 − min_m ρ(f_m)),

  and ρ(f) = 1 if and only if f(x) is a function of the world state alone.
- **In words.** "every direction an encoder spends on its own modality's nuisance is lost for alignment, whatever whitened embedding the other modality learns."
- **The proof's key assumption.** x_1 and x_2 are **conditionally independent given z** ([App A.4](https://arxiv.org/html/2610.39302v1#A1.SS4)).

### 2.4 Proposition 3, good views ignore nuisance and bad views capture it ([§2.3](https://arxiv.org/html/2610.39302v1#S2.SS3); formal in [App A.4](https://arxiv.org/html/2610.39302v1#A1.SS4))
- **(i)** If μ_k > max(μ_{k+1}, 0), a good-view JEPA learns a function of the state alone (ρ = 1). Good-view JEPAs with the same world kernel π then agree up to one rotation each.
- **(ii)** Consider a bad-view JEPA with product corruption Γ = A ⊗ B_m. Let α_k be the correlation two corruptions keep along the k-th state direction, and β_1 the largest correlation they keep along any nuisance direction. The JEPA then spends **r** of its k directions on nuisance, so its alignment error with *any* whitened encoder of another modality is ≥ 2r. Moreover, **r ≥ 1 if and only if β_1 > α_k**.
  - Intuition: crops of one photograph keep viewpoint and lighting, and spans of one caption keep its word choice.

### 2.5 Theorem 4, views and data set the world kernel ([§2.3, Eq. 5](https://arxiv.org/html/2610.39302v1#S2.SS3); formal in [App A.4](https://arxiv.org/html/2610.39302v1#A1.SS4))
- **Statement.** For any law W_m of positive pairs (good, corrupted or mixed), let π_{W_m} be the induced world kernel.
  - (i) The spectral solution lies within τ_m of a rotation of Φ(π_{W_m}). τ_m = (2√2 / δ_m) · min(√k ε_m, √2 ε_m^HS), where ε measures how much the positives mix state with nuisance.
  - (ii) The cross-modal error is within τ_1 + τ_2 of d, the misalignment of the two induced kernels' embeddings, which Thm 2 bounds.
- **Key sentence.** "a bad-view JEPA does not choose its world kernel: π_{W_m} records which states its positives connect, which depends on how the data cover each state and its neighbors. Bad-view JEPAs of two modalities therefore agree only insofar as their data make the positives connect the same states, while good views fix the kernel by design."
- **Data requirement.** Weak bad views need dense data: the reach needed scales as Δ ∝ log N / N ([§2.3](https://arxiv.org/html/2610.39302v1#S2.SS3), [Fig. 4](https://arxiv.org/html/2610.39302v1#S2.F4)).

### 2.6 Proposition 4, next-token prediction as a bad-view JEPA ([§2.4, Eq. 6](https://arxiv.org/html/2610.39302v1#S2.SS4))
- **Decomposition.** The softmax loss splits into a prediction term −⟨h(c), w_y⟩ plus a log-partition "anti-collapse" term.
- **Positives.** They are a prefix and its continuation within one document.
- **Identifiability.** Minimisers are identified only up to GL(k).
- **Status.** The claim that next-token prediction captures nuisance is labelled "a prediction to test rather than a corollary" ([§2.4](https://arxiv.org/html/2610.39302v1#S2.SS4)).

### 2.7 How tight the bounds are on real encoders ([App C, Table 6](https://arxiv.org/html/2610.39302v1#A3.T6))
- **Measured ρ.** At k = 32 on Flickr, ρ = .804 (good-view), .655 (bad-view) and .608 (next-token).
- **Bound vs measurement.** The Thm 3 lower bounds against DINOv2 are 12.6, 22.1 and 25.1. The measured errors are 33.5, 39.4 and 40.3, against 64 for uncorrelated encoders.
- **Reading.** The bound holds, ranks the encoders correctly, and is loose by a factor of ~1.6–2.7.

---

## 3. Practical recipe and numbers

### 3.1 Training recipe ([§3 Setup](https://arxiv.org/html/2610.39302v1#S3); [App B](https://arxiv.org/html/2610.39302v1#A2))
- **Objective.** Image and text encoders are trained **separately**, each with **LeJEPA**: invariance to the mean of two global views, plus SIGReg with λ = 0.02 (1024 slices). The projector is 384→512→2048→2048→512 with BatchNorm, built on the stable-pretraining library.
- **Image encoder.** ViT-S/16 with 2 global crops (224², scale 0.3–1.0) and 6 local crops (96², scale 0.05–0.3), plus flip, colour jitter, grayscale, blur and solarisation.
- **Text encoder.** 12-layer BERT, width 384 (33M parameters). Global views are spans covering 50–100 % of a caption, local views 15–50 %, with 15 % masking.
- **Optimisation.** AdamW, lr 5·10⁻⁴, wd 0.05, batch 256, 200 epochs, one GPU per run.
- **Larger runs** use lr 10⁻⁴ with gradient clipping 1.0.
- **Collapse risk.** Without clipping, 3 of 36 default runs collapse abruptly between epochs 20 and 60. BERT-L on web text collapsed even with clipping ([§3.2](https://arxiv.org/html/2610.39302v1#S3.SS2); [App C](https://arxiv.org/html/2610.39302v1#A3)).

### 3.2 Aligning independently trained encoders
- **Map.** One orthogonal map (orthogonal Procrustes, Schönemann 1966), fitted on paired validation samples. Evaluation is image→caption retrieval R@1/R@5 among 1,000 test captions (chance 0.1 %), plus the fit-free mutual 10-NN (chance 10/N) ([§3 Metrics](https://arxiv.org/html/2610.39302v1#S3)).
  - When dimensions differ, the map is semi-orthogonal (orthonormal rows), with d_a·d_b − d_b(d_b+1)/2 free parameters. That is 0.22M for 768→384 and 1.9M for the 5,120-dimensional captioner features ([§2.2](https://arxiv.org/html/2610.39302v1#S2.SS2)).
  - Each side is centred by its own unpaired mean ([App C, "Transfer to COCO"](https://arxiv.org/html/2610.39302v1#A3)). Whitening before the rotation "barely changes retrieval" on projector outputs (R@1 .030 vs .034) ([App C, "Projector outputs"](https://arxiv.org/html/2610.39302v1#A3)).
- **How many pairs.** Prop 2 needs n ≥ k. In practice:
  - **1,014 pairs vs 30k pairs:** R@1 rises from .246 to .346 (good-view BERT-S vs frozen DINOv2 ViT-B) ([§3.4, Table 2](https://arxiv.org/html/2610.39302v1#S3.T2)). "Retrieval is limited by how well the rotation is estimated."
  - **Captioner CIDEr by pairs:** 100 pairs .244; 300 pairs .317; 1,014 pairs .396; 30k pairs .431 ([Table 9](https://arxiv.org/html/2610.39302v1#A3.T9)).
  - **Fewer pairs than dimensions** is underdetermined, but unpaired refinement helps. Refinement alternates Sinkhorn optimal transport with Procrustes on the transport plan, using the 1,014 validation images and captions with their pairing hidden. It raises 100 pairs to .294 and makes 30 pairs reach .169 ([App C, "Few pairs and no pairs"](https://arxiv.org/html/2610.39302v1#A3)).
  - **10 pairs:** Procrustes alone gives R@1 ≈ .01. The pair-free method of Schnaus et al. seeded with those 10 pairs gives .32 for the good-view JEPA (one seed) ([App C, "Alignment without any pair, in detail"](https://arxiv.org/html/2610.39302v1#A3)).
- **Absolute levels, which matter for setting expectations:**
  - **Both encoders trained from scratch** (ViT-S on 29k Flickr images vs the good-view text JEPA): mutual 10-NN **.062**, R@1 **.044** (1k candidates, 1k-pair rotation) ([Table 1](https://arxiv.org/html/2610.39302v1#S3.T1)). Training the image JEPA 5× longer raises mutual 10-NN from .065 to .077 ([§3.4](https://arxiv.org/html/2610.39302v1#S3.SS4)).
  - **With a frozen DINOv2 ViT-B:** the good-view JEPA reaches R@1 .246 (1k pairs) and .346 (30k pairs). The best web-text JEPA (BERT-B, 940M images) reaches **.464**, against **.489 for CLIP B/32's text encoder** (trained jointly on 400M pairs) and .481 for all-mpnet-base-v2 ([Table 2](https://arxiv.org/html/2610.39302v1#S3.T2)).
  - **With DINOv3 ViT-7B (CLS + mean patch):** R@1 .531 through the rotation alone ([App C, Table 12](https://arxiv.org/html/2610.39302v1#A3.T12)).
  - **Where alignment lives:** after the best rotation, image–text correlation is **0.70 on the first text PC and decays to ~0.1 by the 60th PC** ([§3.1, Fig. 6](https://arxiv.org/html/2610.39302v1#S3.F6)). The caption notes that "noise in low-variance directions would produce the same trend".
- **Good views on the image side** (MSR-VTT, frames of a clip as views) help only a little:
  - Mutual 10-NN: **.067** with good views on both sides, .059 with good text views only, .043 with good image views only, .039 with neither.
  - "the image-side effect is small and reverses on R@5" ([§3.3](https://arxiv.org/html/2610.39302v1#S3.SS3); [App C, Table 13](https://arxiv.org/html/2610.39302v1#A3.T13)).

### 3.3 The "42× fewer samples" claim ([§4.2](https://arxiv.org/html/2610.39302v1#S4.SS2); [Fig. 1](https://arxiv.org/html/2610.39302v1#S0.F1))
- **Exact basis.**
  - Both runs use the same 109M BERT-B and texts from the same corpus, Recap-DataComp.
  - The good-view JEPA reaches WIT mutual 10-NN **.109 after 21.6M samples**, on the texts of 2.16M images.
  - Next-token prediction reaches **.111 after 900M samples**, on the texts of 940M images.
  - 900 / 21.6 ≈ 42, which also corresponds to "14 times fewer text tokens" and "435 times fewer unique images".
- **Caveats, all stated in the paper:**
  - "we scored no earlier next-token checkpoint". The 42× compares one good-view point with the *final* next-token checkpoint, not a measured crossing, so the true ratio may be smaller **[analysis]**.
  - Each run used "the recipe of its scale": lr 10⁻⁴ / batch 256 versus 2·10⁻⁴ / batch 2,048.
  - Two of the three texts per image were written by **LLaVA (LLaMA-3-8B with a CLIP ViT-L/14 vision encoder)**. Without any image captioner, good views reach .060 on WIT instead of .107 ([§7](https://arxiv.org/html/2610.39302v1#S7); [App C, Table 7](https://arxiv.org/html/2610.39302v1#A3.T7)).
  - The SIGReg weight at 940M images was chosen after a seed at λ = .02 scored only .121 on WIT, so "that weight was selected on WIT" ([App C, "Scaling to the texts of 940M images"](https://arxiv.org/html/2610.39302v1#A3)).
  - At matched samples (21.6M, 33M encoder), the gap is .108 vs .071.
  - At the 50M scale, comparisons with language models are "within noise", while the gaps between view types are far outside noise ([App C, "Uncertainty on WIT"](https://arxiv.org/html/2610.39302v1#A3)).
- **Related claims.**
  - A 109M good-view JEPA scores WIT mutual 10-NN .184 and CKA .507 with DINOv2-B, above Qwen2.5-72B (.154) and Llama-3-70B (.151) ([§4.3, Table 3](https://arxiv.org/html/2610.39302v1#S4.T3)). The four-seed mean is .168.
  - A scaling law, R@1 = a + d log M + b log(1 + q^α M^β N^γ / c), predicts 85 unseen runs with RMSE .0093, versus a seed-to-seed difference of .0096. It "fails further out" and is "an empirical summary of one recipe, not a consequence of Thm 4" ([§4.4, Eq. 7](https://arxiv.org/html/2610.39302v1#S4.SS4)).

### 3.4 Captioning demo ([§5.1](https://arxiv.org/html/2610.39302v1#S5.SS1); [Table 9](https://arxiv.org/html/2610.39302v1#A3.T9), [Table 10](https://arxiv.org/html/2610.39302v1#A3.T10))
- **Design.**
  - A caption decoder is trained **on text only**: it reconstructs a caption from that caption's frozen good-view JEPA embedding, with Gaussian noise added.
  - At test time, the image is embedded with frozen DINOv2, rotated, and decoded. The rotation is the only component fitted on image–caption pairs.
  - The noise level is set by **matching the residual cosine** of the rotation: σ = (c⁻² − 1)^{1/2}. With c ≈ .3, σ ≈ 3, and transfer peaks there.
  - CIDEr is reported on a 0–1 scale, so .838 corresponds to 83.8 in the usual ×100 convention.
- **Flickr test** (1,000 images, small decoder):

| Configuration | CIDEr |
|---|---|
| Rotation from 30k pairs | **.431** (BLEU-4 .185) |
| Rotation from 1,014 pairs | .396 |
| Rotation from 300 pairs | .317 |
| Rotation from 100 pairs | .244 |
| Retrieve nearest caption, 1,014 pairs | .158 |
| Retrieve nearest caption, 30k pairs | .188 |
| Random rotation | .043 |
| Oracle (decode the true caption embedding) | .628 |

- **COCO Karpathy test** (5,000 images), rotation fitted on **Flickr pairs only**:
  - CIDEr **.668**, BLEU-4 .199 (beam 3), .630 greedy.
  - Random rotation .038; retrieving the nearest Flickr caption .169.
- **Best captioner** (Qwen2.5-1.5B decoder trained on text only, BERT-S JEPA trained on texts of 100M web images, frozen DINOv2-g + DINOv3-L concatenated, consensus of 32 samples, one map fitted on 30k Flickr pairs): **Flickr .519, COCO .838**. The demo decoder gives .538 and .832.
  - CLIP-based captioners report .345 (ZeroCap), .493 (MAGIC), .918 (CapDec) and .912 (DeCap). Unlike this paper, those do not use in-domain Flickr pairs; they use CLIP's 400M pairs instead.
- **Rotation vs a free map** (the most transferable finding for us), all fitted on the same 30k pairs:

| Map | Flickr (in-domain) | COCO (out of domain) |
|---|---|---|
| Rotation | .504 | **.802** |
| Unconstrained linear map | .531 | .635 |
| 2-layer MLP | .498 | .585 |
| Rotation, with consensus decoding | .538 | .832 |
| Linear map, with consensus decoding | .561 | .652 |
| MLP, with consensus decoding | .511 | .597 |

  The authors' summary: "a map free to fit the pairs fits Flickr, while a rotation, which can only turn one space onto the other, carries over to new images."
- **Transfer across datasets** ([App C, "Transfer to COCO"](https://arxiv.org/html/2610.39302v1#A3)): a rotation fitted on 30k Flickr pairs gives R@1 11.3 % on 5,000 COCO images (chance 0.02 %). A rotation fitted on 2,500 COCO pairs gives 15.8 % on the other 2,500.
- **Video:** MSR-VTT CIDEr .246, from the mean of 8 rotated frame embeddings ([§5.1](https://arxiv.org/html/2610.39302v1#S5.SS1)).

### 3.5 Alignment with no pairs ([§5.2, Table 4](https://arxiv.org/html/2610.39302v1#S5.T4); [App C](https://arxiv.org/html/2610.39302v1#A3))
- **Method.** The code of Schnaus et al. 2026, unchanged: Wasserstein-Procrustes with 30 clusters, 30 restarts and OT refinement. DINOv2-B vs the text encoder, each fitted on disjoint unpaired halves of COCO train2014.
- **COCO results** (FOSCTTM, then zero-shot CIFAR-10 / CIFAR-100 / ImageNet-100):

| Text encoder | FOSCTTM | C-10 | C-100 | IN-100 |
|---|---|---|---|---|
| 109M good-view JEPA | .018 | 86.0 | 67.2 | 64.4 |
| Qwen3-8B (rerun) | .018 | 83.4 | 61.7 | 47.6 |
| CLIP text encoder | .006 | 84.0 | 67.0 | 68.7 |
| Next-token prediction | .287 | 25.5 | — | — |
| Bad-view JEPA | .026 | — | — | — |

  The next-token run "fails in half of its runs". The bad-view JEPA "aligns well too" on COCO.
- **Flickr** (smaller unpaired pools, 15,007 + 15,007): zero-pair R@1 is **16.4 % ± 10.4 %** (median 16.1 %). Every encoder that is not a good-view JEPA scores ≤ 0.3 %.
- **Small clouds:** with the 1,014-sample clouds, Gromov–Wasserstein stays at chance ([App C, "Few pairs and no pairs"](https://arxiv.org/html/2610.39302v1#A3)).
- **Failure mode:** failures "come from the initialization". The method's own unpaired objective is anti-correlated with test FOSCTTM (Spearman −.78) and can flag bad seeds.

---

## 4. Limitations

### 4.1 Stated by the authors ([§7, "Known limitations"](https://arxiv.org/html/2610.39302v1#S7), unless another link is given)
- "Our theorems are population statements over a finite world with injective renderings and matched marginals; real modalities are lossy, and finite-sample training relies on the network's inductive bias."
- "Uniqueness up to rotation is proved for second-order anti-collapse, not for LeJEPA's full-distribution constraint … Thm 2 is proved for spectral solutions only, and **no experiment meets the hypotheses of Thm 1 exactly**, since our runs use LeJEPA and the image side mostly uses crops as positives."
- "The main caveat is where good views come from: they require knowing which texts describe the same scene." At scale, the texts are LLaVA/CLIP recaptions, so "paired information thus enters through the definition of positives, one fitted linear map and, on Recap, captioner-generated texts".
- "hyperparameters were tuned only for the good-view JEPA."
- The theory does not cover a learned predictor ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1)).
- The model "still assumes that every observation determines its state, while a caption sees only part of a photograph; Thm 2 covers that departure only through the kernel mismatch" ([§2.3](https://arxiv.org/html/2610.39302v1#S2.SS3)).
- The image side is weak: an image JEPA trained with good views on MSR-VTT's 7k clips "is far weaker than DINOv2" ([§2.3](https://arxiv.org/html/2610.39302v1#S2.SS3)).
- The scaling law is empirical and fails out of range ([§4.4](https://arxiv.org/html/2610.39302v1#S4.SS4)).
- SIGReg instability: collapses occur, and the WIT score is sensitive to the SIGReg weight at scale ([App C](https://arxiv.org/html/2610.39302v1#A3)).

### 4.2 [analysis] Additional sceptical points
1. **Identity-kernel degeneracy (§1.5).** Same-scene-only good views do not satisfy the paper's own uniqueness condition.
2. **"Convergence" is far from complete in practice.** The best fit-free mutual 10-NN is ~.25 on Flickr and .184 on WIT, against a maximum of 1. The best rotation R@1 is ~.46–.53. The Thm 3 bound and the measured error leave ~half the embedding unaligned even for good views (§2.7).
3. **The good-view advantage on the scaled text side comes mostly from grouping texts by image,** with captions written by a vision-language model. Both are forms of paired supervision.
4. **Most comparisons are text-side, against a frozen DINOv2.** Two encoders trained from scratch align far less (Table 1: R@1 .044).
5. **Single version, very recent, not reviewed.** I did not run the released tests or the code.

---

## 5. Fit to BSCCM

Project facts used here, from the project notebook (N001, N003, N002 SYNTHESIS):
- **LF:** 412,941 cells × 23 LED patterns at 128×128, plus a DPC phase image.
- **GS:** 15,712 Wright-stained RGB images at 398×398. All are paired with LF, but come from **only 2 stained slides** (61 % / 41 % of cells matched). LF→GS matching is by rigid alignment plus nearest neighbour, with ~7 % mismatches (a crude estimate).
- **MK:** surface-marker abundances, partially observed, for ~350k cells paired with LF. Never co-observed with GS.
- Cells are lysed, PFA-fixed, settled, imaged under LF first and then Wright-stained.

### 5.0 The theorem's assumptions mapped to LF vs GS
| Paper assumption | LF vs GS in BSCCM | Status [analysis] |
|---|---|---|
| Finite world, **injective** rendering of each state ([§2.1](https://arxiv.org/html/2610.39302v1#S2.SS1)) | LF (scattering/phase at 23 illumination angles) and GS (dye uptake: chromatin, granules, cytoplasm) are each **lossy** views. Neither determines the other. GS is also taken *after* lysis, fixation and staining. | Violated. Only Thm 2 / Thm 4 cover this, and only through kernel mismatch. |
| **Same positive-pair distribution π over world states** in both modalities | Separate JEPAs with their own augmentations induce kernels from their own corruption overlaps and data (Thm 4). They will differ. | Violated unless positives are defined by one shared external cell graph (see below). |
| Matched marginals p | An LF-JEPA on 413k cells (all batches) vs a GS-JEPA on 15.7k cells (2 slides) see different cell populations. | Violated unless LF is restricted or reweighted to the GS slides. |
| Population optimum, unique up to O(k) (eigengap, whitening objective) | LeJEPA/SIGReg is not covered; real training is finite-sample. | Assumed, as in the paper. |
| Identity predictor | Fine for LeJEPA/VICReg/Barlow without a predictor. I-JEPA-style predictors are not covered. | Choose accordingly. |
| Good views: two observations with **independent nuisance** | LF: illumination subsets of one acquisition share focus, position, slide, ROI shift and normalisation. GS: one image per cell, so only crops/jitter are possible. | Both modalities would be bad-view or at best intermediate. |

### (a) Do LF-JEPA and GS-JEPA, trained separately, converge up to a rotation, so that a few pairs suffice?
- **Short answer: no, not by this theorem.** Its conclusion needs (i) lossless renderings of the same world state, (ii) **the same** positive-pair kernel over world states in both modalities, (iii) matched marginals, and (iv) a unique optimum. BSCCM LF/GS meets none of these exactly.
  - Under the paper's own Thm 4, two JEPAs that each use their own augmentations "agree only insofar as their data make the positives connect the same states".
  - Under Prop 3(ii) and Thm 3, any direction that encodes acquisition nuisance cannot be aligned by a rotation. In GS this includes stain intensity, focus and slide; in LF it includes ROI shift, LED119 background artefacts and per-slide scaling.
- **What the theory still offers:**
  1. A bound: Thm 2 / Thm 4 give a qualitative prediction. Alignment will be partial, concentrated in the leading, well-separated spectral directions, and worse where the two kernels or nuisance fractions differ.
  2. Identifiability: Prop 2 says that *if* rotation-equivalence held, ≥ k pairs in general position would fix the map. Here pairs are not the bottleneck, since we have ~15.7k of them. The bottleneck is the residual after the best rotation.
  3. A diagnostic: the scene-explained fraction ρ, which gives a lower bound on the alignment error (Thm 3).
- **The paper's empirical analogue sets a low prior.** Two encoders trained from scratch on 29k Flickr items reach R@1 4.4 % among 1,000 through a 1k-pair rotation, and 3.2 % on MSR-VTT ([Tables 1](https://arxiv.org/html/2610.39302v1#S3.T1), [13](https://arxiv.org/html/2610.39302v1#A3.T13)). Separately trained LF and GS encoders on ~15.7k GS cells are, if anything, data-poorer on the GS side **[analysis]**.

**What "good views" would mean for a cell [analysis]**, ranked from most to least "good" in the paper's sense:
1. **Two observations of the same cell with independent nuisance.** In BSCCM, the only such pairs are **cross-modal**: LF and GS of the same cell. Using those as positives is joint training (CLIP/SigLIP), not separate training, so the theorem's selling point disappears.
   - A within-modality version would need the same cell re-imaged with independent acquisition, for example a second focus, session or microscope. I do not believe BSCCM contains that [unverified: the N001 pack has one LED stack and one histology image per cell].
   - **Two stains** of the same cell would be another cross-modal pair. Sequential restaining also alters the cell, and it is not in BSCCM.
2. **Disjoint LED subsets of one LF stack**, for example brightfield-type vs darkfield-type LEDs vs DPC, plus independent crops and noise. These are physically different contrasts, but they share every acquisition nuisance: position, focus, slide, debris and normalisation. By Prop 3(ii) they behave as good views only if the shared nuisance correlation β_1 stays below the k-th state correlation α_k. That is plausible for some nuisances but not for slide or batch effects. **Intermediate** [unverified: which of the 23 LED indices are brightfield vs darkfield].
3. **GS colour-channel splits or crops:** bad views.
4. **"Related cells" as positives,** e.g. same gate class, nearest neighbours in MK space, or nearest neighbours in a frozen LF embedding. This is what actually gives the kernel off-diagonal structure and hence a unique spectral embedding (§1.5). But it puts paired or labelled information into the positives, as the authors also note for their own setup.
   - A *shared* kernel for LF and GS can only be built on the 15.7k paired cells, using one modality (LF) to define it. Both JEPAs would then learn the spectral embedding of the LF graph. Thm 1 would make them rotation-equivalent by construction, but this is **graph distillation of LF geometry into GS**. It is information-equivalent to, and likely weaker than, directly regressing GS onto frozen LF features (SYNTHESIS design item 2).

**Bottom line for (a).** Fitting only an orthogonal map on a few pairs is **not justified by the theorem** for LF/GS. It is a cheap hypothesis to *measure* (see (d)), not one to rely on.

### (b) Can it justify or improve the transitive GS→LF→MK link?
- **Justify: no.**
  - In the model, an observation is (shared state z) + (independent modality nuisance). There is no notion of *biologically real but modality-unique* signal. Anything GS sees that LF does not is "nuisance" relative to the shared world, and Thm 3 says it is unalignable.
  - So even in the ideal case, a rotation-aligned GS embedding contains only the shared component h*(z). The composite GS→Q→LF-space→MK estimates MK from (GS ∩ LF)-shared information, restricted to the leading spectral directions, where alignment lives (Fig. 6: correlation .70 → ~.1 by PC 60).
  - That is **at most** what E[E[MK | LF] | GS] (distillation) can recover, and in practice less, because the rotation discards MK-relevant LF information outside the shared top-k subspace.
  - It is silent on the CIA (MK ⟂ GS | LF), which is untestable without GS–MK pairs. The original PRH paper makes the same point explicitly: alignment between lossy views is "capped by the mutual information between the input signals" ([PRH §6](https://arxiv.org/html/2405.07987v5)).
- **Improve: possibly, in two narrow ways.**
  1. **Regularising the GS→LF map.** An orthogonal map has k(k−1)/2 degrees of freedom, and the paper's best out-of-domain transfer came from the rotation, not from linear or MLP maps (COCO .802 vs .635 / .585, Table 10). With only 2 stained slides, a rotation (after centring and whitening each side) may generalise across slides better than ridge or an MLP. It is a cheap test, with an expected effect only under shift [analysis].
  2. **Noise matched to residual alignment.** The rule σ = (c⁻² − 1)^{1/2} ([§5.1](https://arxiv.org/html/2610.39302v1#S5.SS1)) suggests training the LF→MK head on LF embeddings **with noise matched to the GS→LF residual cosine**. The head then becomes robust to the alignment error it will see at GS inference. This transfers from their decoder trick and is cheap to try [analysis].
- **Diagnostic value.** The ρ / Thm 3 machinery can quantify how much of the LF pivot embedding is cell-explained rather than acquisition-explained. It needs repeated observations per state: illumination-subset views per cell (which inflate ρ, because they share nuisance) or cell-class labels (only 10k batch-0 cells, project notebook). A low ρ predicts that the GS→LF link cannot be tight [analysis].

### (c) Can it help transfer from BSCCM GS to unpaired clinical smears?
- **The setting differs from the theorem's.** This is the **same modality under a different nuisance distribution** (lab, stain protocol, scanner, magnification, preparation: lysed and settled vs smeared, with RBC background), with no same-cell pairs and different class proportions. Thm 2 assumes matched marginals p, and the OT-based pair-free methods implicitly assume similar distributions, so label shift violates both [analysis].
- **Rotation between two separately trained encoders (BSCCM-GS vs smear) is a weak lever:**
  - No paired cells exist.
  - Class-centroid anchors give only C ≈ 5–10 constraints, fewer than k, so Prop 2(ii) is underdetermined.
  - Pair-free Wasserstein-Procrustes needs large pools, is seed-unstable (Flickr 16.4 % ± 10.4 %) and is at chance on ~1k-point clouds ([§5.2](https://arxiv.org/html/2610.39302v1#S5.SS2), [App C](https://arxiv.org/html/2610.39302v1#A3)).
  - Training **one** encoder on both domains avoids the rotation question entirely.
- **What does transfer: the nuisance view of augmentation.**
  - By Prop 3 and Thm 4, positives that **resample the nuisance independently** produce state-only embeddings. The paper's numerical checks: "positives that resample the nuisance align exactly" ([App A.6](https://arxiv.org/html/2610.39302v1#A1.SS6)).
  - Operationally: build GS views with *independently drawn* stain-colour, blur, scale, illumination and JPEG/scanner perturbations that cover the clinical range, rather than mild perturbations of one image. The theory says mild perturbations keep the nuisance (β_1 > α_k).
  - This principled argument for heavy, independent nuisance resampling agrees with SYNTHESIS item 7 (colour + optics augmentation; avoid BN-statistics TTA).
  - Limit: it only removes nuisances that can be simulated. Preparation differences (lysed vs smeared morphology) cannot be.
- **Cross-domain "good views" using cell-type labels** (pair a BSCCM cell with a public-smear cell of the same class) would make the kernel class-level. That removes within-class variation, which is exactly what MK prediction needs. Possibly useful for classification transfer, harmful for marker regression [analysis].

### (d) Cheapest experiment on BSCCM
Aim: measure whether separately trained LF and GS encoders are "rotation-equivalent" enough that an orthogonal map on few pairs competes with joint training. The design keeps the paper's protocol (centring, orthogonal Procrustes, R@1 among 1,000 candidates, fit-free mutual 10-NN). Following CLAUDE.md: screen on 1 seed, save every value to csv, plot with `bin/plot-compare`. **Only the main agent submits GPU jobs.**

**Split.**
- Hold out by **slide**: fit maps on pairs from stained slide A and test on slide B, then swap (2 folds). The slide is the dominant shared nuisance.
- The ~7 % LF–GS mismatch caps R@1. Report the ceiling, e.g. using only high-confidence matches [unverified: per-slide pair counts; the user's n = 10,000 may exceed the pairs on one slide, so use "all pairs on the training slide" as the top level].

**Phase 0: zero-training screen (minutes, CPU or 1 GPU).**
- **Frozen GS encoders:** DINOv2 ViT-B/14 (and ViT-S) on GS RGB, CLS + mean-patch features. DinoBloom too if available locally [unverified availability].
- **Frozen LF encoders:** DINOv2 on 3-channel renderings of LF (e.g. mean brightfield LEDs, DPC, mean darkfield LEDs), plus a random-init CNN as the floor.
- **Procedure:** centre, PCA-whiten to d ∈ {16, 64, 128}, then fit an orthogonal map on n ∈ {10, 30, 100, 300, 1000, 3000, all-train-slide} pairs, with 5 random subsets for each n.
- **Comparators:** ridge regression GS→LF, a 2-layer MLP, and a SigLIP head (linear projections trained contrastively on the same n pairs over the frozen features).
- **Metrics:**
  - GS→LF R@1 and R@10 on held-out-slide pools of 1,000 (chance 0.1 %).
  - Mutual 10-NN, which needs no fit (chance 10/N).
  - Per-PC correlation after rotation, the Fig. 6 analogue.
- This gives the "generic encoders" PRH baseline on cells.

**Phase 1: separate JEPAs vs joint (1 seed).**

| Arm | Data | Positives |
|---|---|---|
| L-bad: LF-JEPA (LeJEPA or whitened-MSE, no predictor) | all 413k cells, LED channels only, **no fluorescence** (it contains the marker signal, SYNTHESIS item 1) | two augmented crops of the full 23-channel stack |
| L-illum: LF-JEPA | same | **disjoint LED subsets** (brightfield-type vs darkfield-type vs DPC) of the same cell, each independently cropped and noised. This is the "intermediate good view". |
| G-bad: GS-JEPA | 15.7k GS cells | crops + flip/rotation + **independent** stain/optics resampling (the §5(c) nuisance-resampling idea) |
| (opt.) G-kernel | GS cells on the training slide | positives = GS cells whose paired LF embeddings (frozen L-illum) are mutual k-NN. This is the shared-π construction of Thm 1. **Uses pairs**; compare it with direct GS→LF-feature regression. |
| J: SigLIP two-tower LF↔GS | training-slide pairs, n as above | cross-modal pairs; from scratch, and initialised from L-* / G-* |

- **Maps on top of L-* and G-*:** Procrustes (semi-orthogonal if dimensions differ) vs ridge vs MLP, each at every n. For n < d, add the paper's unpaired OT refinement: alternate Sinkhorn with Procrustes on the held-out-pairing clouds ([App C](https://arxiv.org/html/2610.39302v1#A3)).
- **Controls:**
  - Random-init encoders with the same maps (floor).
  - Shuffled pairs, which should give chance.
  - **An LF-internal ladder:** two LF-JEPAs trained on *disjoint LED groups* of the same 413k cells, aligned with n pairs. Here the world (the cell), the population and the acquisition are fully shared, so it is the most favourable case for Thm 1. If even this pair of encoders does not saturate near n ≈ d, then LF↔GS certainly will not.
- **Downstream proxy for the transitive link:**
  - For held-out-slide GS cells, compare MK predicted via GS → Q → LF-space → MK-head (head trained on LF–MK cells) with MK predicted from the **same cells' own LF**: per-marker r. This is the "b" term in the SYNTHESIS error decomposition.
  - Run it with and without the σ-matched noise trick of §5(b).
  - Reference: LF→MK on held-out LF–MK cells, the "a" term.
- **Nuisance diagnostic:** ρ for the L-* arms, estimated as in [App C, "Nuisance on real encoders"](https://arxiv.org/html/2610.39302v1#A3). Use the covariance of per-cell means over illumination-subset views, corrected for within-cell spread. Note it is inflated by shared nuisance.

**Hypotheses and decision rules [analysis].**
- **H1 (PRH-JEPA useful):** separate JEPAs + Procrustes at n ≤ 300 **beat SigLIP trained on the same n pairs**, and/or at full n the rotation **loses less than ridge or MLP when moving from slide A to slide B**.
- **H0 (expected from the paper's from-scratch numbers):** SigLIP or distillation dominates at n ≥ ~1k, and separate + rotation stays at low single-digit R@1.
- **Kill the "rotation-only" lane** if separate + Procrustes R@1 at full n is < 0.5× SigLIP's, **and** it never beats SigLIP at n ≤ 300, **and** the cross-slide advantage over ridge is within seed noise. Close it as a null result and keep the rotation only as an optional regulariser.
- **Promote (≥ 3 seeds)** only if H1 holds on both slide folds.

**Placement and rough cost** [estimate from typical throughput, not measured]:
- Phase 0 runs on CPU (48 cores) or 1 GPU in < 1 h, because the frozen-feature extraction is ~16k + ~16k images.
- L-* arms: ResNet-18 or ViT-S at 128 px on 413k cells, ~20–50 epochs, ~2–6 GPU-h each on one H100.
- G-*: ≤ 1 GPU-h. SigLIP arms: ≤ 1 GPU-h each.
- **Total ≈ 10–15 GPU-h, below the 25 GPU-h approval line.** Use the `preempt` partition, preempt-safe, with wandb registration as per CLAUDE.md.

---

## 6. Recommendations for the N002 synthesis
1. **Do not adopt "train separately, then rotate" as the primary LF↔GS alignment.** The theorem's hypotheses (injective renderings, shared positive kernel, matched marginals) are violated, and the paper's own from-scratch cross-modal numbers are low (R@1 .044, [Table 1](https://arxiv.org/html/2610.39302v1#S3.T1)). Keep SigLIP plus distillation through a frozen LF pivot (SYNTHESIS items 1–2) as the main line.
2. **Adopt two cheap, theory-motivated add-ons:**
   - an **orthogonal-map ablation** for GS→LF, evaluated across slides;
   - **σ-matched noise** on the LF→MK head (σ = (c⁻² − 1)^{1/2}).
3. **Use the nuisance view (Thm 3 / Prop 3) to motivate augmentations:**
   - **resample** stain, optics and scale nuisance independently across views rather than lightly perturbing one image;
   - measure ρ as a cell-vs-acquisition diagnostic for the LF pivot.
4. **Read the theory as supporting a point already in SYNTHESIS:** rotation or transitive links carry only shared information, so GS-only signal and CIA violations are invisible to them (this paper and PRH §6).
5. **Run the §5(d) Phase 0 screen first.** It costs minutes and decides whether any of this is worth GPU time.

---

## 7. Context: the original PRH (Huh et al. 2024, https://arxiv.org/abs/2405.07987)
- **Hypothesis.** Representations trained on different data and modalities "are converging to a shared statistical model of reality in their representation spaces" ([§1](https://arxiv.org/html/2405.07987v5)).
- **Supporting argument.** An idealised world of discrete events with "a bijective, deterministic" observation function, and a contrastive learner trained on co-occurrence whose kernel approaches pointwise mutual information ([§4](https://arxiv.org/html/2405.07987v5)).
- **Its own limitation, which bears on BSCCM.** "Two different models cannot converge to the same representation if they have access to fundamentally different information … our mathematical argument … only strictly holds for bijective projections … This will not hold true for either lossy or stochastic observation functions." A lower-information representation aligns with a higher-information one only "up to a level capped by the mutual information between the input signals" ([§6, "Different modalities may contain different information"](https://arxiv.org/html/2405.07987v5)).
- **What the JEPA paper changes.** It replaces the contrastive/PMI argument with a JEPA/spectral one and adds bounds for mismatched positives (Thm 2, Thm 4). It still assumes injective renderings in the main theorem and names lossy modalities as the open limitation ([§6 Related Work](https://arxiv.org/html/2610.39302v1#S6), [§7](https://arxiv.org/html/2610.39302v1#S7)).

## 8. Verification ledger
- **Verified in the primary text** (TeX source of v1 plus HTML anchors):
  - all theorem and proposition statements, Eq. 1–7 content and the proofs' assumptions;
  - every number in §2.7, §3 and §4.1;
  - the §7 limitation quotes.
- **Verified in PRH HTML v5:** the §4 bijective-observation definition and the §6 quotes.
- **[analysis], mine:**
  - §1.5 identity-kernel degeneracy, derived from Prop 1 plus the Fig. 3(a) caption;
  - the claim that the 42× ratio may be overstated;
  - every BSCCM mapping, prediction and cost in §5;
  - the noise-matched LF→MK-head idea;
  - the G-kernel construction.
- **[unverified]:**
  - the released code and tests (not run);
  - BSCCM per-slide GS pair counts;
  - which LED indices are brightfield vs darkfield;
  - whether BSCCM has any re-imaged cells;
  - DinoBloom availability;
  - GPU-hour estimates.
