# N088 synthesis: three literature reviews (A label-free/representations, B sparse multi-target, C alignment/shift), 104 references
## What the literature says about our open questions
- Autoencoder latents cluster cell identity worse than SSL in every microscopy comparison (cytoself without its identity pretext 3-5x
  worse; Lu et al. 2019 AE 42.5 % vs SSL 88.0 % kNN) and small bottlenecks keep illumination/background variance (A). -> N087 AE arm is a
  test, not a hope.
- Post-projector features are worse than backbone features; SIGReg makes the projector output isotropic (no clusters by design) (A).
- Distribution alignment (DANN/OT/CORAL) provably fails when class proportions differ between domains (Zhao et al. ICML 2019) (C) ->
  explains N014/N026/N056.
- The within-slide LF<->Giemsa retrieval (.38-.40) may be a field-of-view shortcut (~3 paired cells per field) (C) -> audit.
- The only unused information channel in BSCCM training is each cell's own measured marker (B); external joint data is the only source
  of cross-marker structure (B).
- Feature-preserving fine-tuning (LP-FT, lower encoder LR, WiSE-FT) is the one lever both A and C rate highest for a 2-draw shift and
  that was never applied to our in-domain SSL encoders (Kumar et al. ICLR 2022; Wortsman et al. CVPR 2022). In-house evidence: Cytoland
  encoder LR x0.1 .6671 vs x1.0 .6464 (N053).
## Portfolio (GPU-h; status)
1. N090 gentle fine-tuning of SSL ResNet-50: encoder LR x0.1 / x0.3 (2 seeds x 2 directions, ~3 h; launched), then LP-FT + WiSE-FT
   interpolation (Codex code). Pre-registered bar +.0054 over control .6706 (2-seed mean).
2. N087 (running): decoder sweep, image AE, marker prior, flow decoder (done: no co-expression gain), equal-label-count control (B).
3. Missing-marker completion from the measured marker (B-P1): free CPU gate in the sandbox, then ~5-6 h.
4. Field-of-view shortcut audit (C-P1, CPU), then nuisance-aware LF<->Giemsa alignment (~2.5 h) only if the audit confirms it.
5. Nested (Matryoshka) projection head (A-P2, sandbox first, ~2-3 h) replaces N087-D3 (narrow backbone, predicted harmful).
6. Cell-type mixture head with S-BIAD452 profiles (B-P2, ~3 h) after the equal-label-count control.
7. DINO-style in-domain SSL (A-P3, ~6 h) and cross-slide positives in SSL (C-P3, ~4 h): diverse ensemble members.
