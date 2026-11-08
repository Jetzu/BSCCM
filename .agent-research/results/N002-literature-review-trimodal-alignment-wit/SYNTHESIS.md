# N002 synthesis — literature review (6 strands, 2026-11-08)
Reports: agents/A_ml_pivot_binding.md, B_stats_and_mosaic_integration.md, C_image_molecular_microscopy.md, D_hematology.md, E_creative_ml.md; N003/agents/bsccm_paper_code.md.
Modality names from here on: GS (Wright stain; "Giemsa" in the brief), LF (23 LED patterns + phase `dpc`), MK (surface-marker abundances).

## Established framing
- Data structure = statistical matching / mosaic ("bridge") integration: files (GS,LF) and (LF,MK) share only LF. Every transitive method (ImageBind emergent binding, pivot translation, distillation, any-to-any models) assumes MK ⟂ GS | LF (CIA); untestable from BSCCM alone (B, E).
- A GS->MK model estimates E[E[MK|LF] | GS] ("the marker profile implied by the label-free phenotype"); information only in GS (not LF) is unrecoverable (A, B, E).
- Field practice for validation: hide pairing in fully co-measured data (simulated mosaic), label transfer, population-level checks; truly never-co-measured links only get plausibility/proof-of-concept or a small targeted wet-lab set (B). 50-250 fully observed records restore valid inference in matching simulations (B, Rässler).
- Pivot-only links are markedly weaker than directly trained ones (LanguageBind pivot-only image->audio R@1 10.6 vs 36.9 direct; pivot MT -6..-9 BLEU) (A).

## Design implications
1. LF encoder is the foundation: pretrain self-supervised on all 413k cells (LED channels only; fluorescence images must be excluded — they contain the marker signal), then FREEZE it as the pivot (ImageBind; LanguageBind drift lesson) (A, C).
2. GS->MK via generalized distillation through the pivot (teacher LF->MK, student on GS using the 15.7k pairs) beats chaining (MT teacher-student +3..+5.6 BLEU) (A, E); test a "modality-general" teacher restricted to GS-shared LF information (Modality Focusing Hypothesis) (E).
3. Losses: SigLIP (small batches, per-pair targets); MK as set of (marker-id, value) tokens; similarity only over co-observed markers; soft targets / Rank-N-Contrast with ties under the noise floor (A, E). One joint model with masked loss over mismatched panels (ROSIE/HistoPlexer) (C).
4. Avoid as primary: mixture multimodal VAEs, objectives needing GS-LF-MK triples, distribution-matching/cycle losses, unpaired GS-MK alignment (vec2vec top-1 0.01 at 10k) (A, E).
5. Evaluation without GS-MK pairs: a^2 = LF->MK error (held-out stained cells), b^2 = GS-student vs LF-teacher error (held-out GS cells); true error in [(a-b)^2, (a+b)^2] with no CIA, = a^2+b^2 under CIA [derivation, red-team needed]; union-bound conformal interval; verify on in-BSCCM mosaic simulation with known truth and a CIA-violation dial (B, E). Report per-marker r WITHIN cell class vs noise ceiling (C, D).
6. Biology: expected GS-visible markers = lineage-matched (CD14 mono, CD16 neutro/eos-absent, CD123 baso, CD45); weak CD3/CD19/CD56; activation markers not expected (D). Label-free B vs T 75-90 % in literature (D).
7. Transfer: BSCCM GS cells are lysed, PFA-fixed, settled on coverslips, 63x sequential-LED RGB — large gap to clinical smears; DinoBloom collapses across scanners (MLL23 macro-F1 0.41-0.70 vs 0.98 in-domain); avoid BN-statistics TTA (SHOT hurt 10/18); scale matching, background masking, colour+optics augmentation; SOTAlign semi-supervised with unpaired public smears (D, A, E). SuperBloom not public; newest Marr-lab = GenBloom (MICCAI 2026) (D).
8. Ground truth: no public human same-cell Giemsa + CD dataset; destain-then-immunostain works poorly (D). Wet-lab ask: fluorescence (antibody) imaging then Wright on the same coverslip (BSCCM already does LED -> Wright) for a few hundred-1000 cells -> PPI + CIA test (B).

## Data corrections (N003 / paper+code read)
- Stain = Wright's (Sigma 45253). GS-LF pairs = per-slide rigid alignment + nearest neighbour; only 2 slides stained (61 % / 41 % matched); our mismatch estimate ~7 % (crude).
- `dpc` = quantitative phase (Tian & Waller 2015), radians.
- LED normalisation: (raw-30)/(bg-30) with one flat field per channel + per-slide scale; LED119 background has strong artifacts; batch-1 unstained rep-1 (= batch-1 histology slide) has a shifted camera ROI.
- Marker units arbitrary, NOT comparable across batches -> normalise per batch, calibrate against same-batch unstained cells. Single-antibody cells: use *_single_antibody_model_unmixed; 'all' cells: 4 pooled full-model groups only.
- Gates: 0 lymphocytes, 1-3 granulocytes, 4 RBC, 5-6 unknown, 7-9 monocytes (batch-0 'all' only). Donor count unknown -> hold out a whole batch.
- Prior BSCCM works (3 groups) never use GS, never align modalities, never hold out slides; [2]'s 0.72 is a 4-group mean at 28x28 DPC, likely a cell-level split.
