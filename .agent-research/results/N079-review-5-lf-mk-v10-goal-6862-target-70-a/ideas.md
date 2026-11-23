# N079 ideas: synthesis (agents/: outsider.md, codex_ideas.md, codex_diag_redteam.md, lit_panel_merging.md; N077 data_scout.md)

Verified first-hand: S-BIAD452 panel/channels/format (BioStudies API, zarr attrs, IDEAS FCS columns, label correlations CD3-CD4 +.75,
CD3-CD19 -.63, CD15-CD45 -.82); IFC pixel 0.1 um^2 (IDEAS area 139 um^2 = 42 px BF diameter) and the BSCCM/IFC scale 1.75 by the same
BF method; red-team cell-type-only control .7299 vs composed .7372 (n079_rt_repro.csv); K033 step confound (code). Lit: Infinity Flow
design (backbone + one exploratory marker per well, nonlinear imputation) and Mocking et al. 2023 (imputation needs independent
validation) confirmed; exact quotes [partial].

Ranked by P(success) x gain / cost, with the user priority (joint per-cell decoding) as a first-class objective:
| # | idea | source | scale | gain | cost | decision |
|---|---|---|---|---|---|---|
| 1 | joint-decoding SANDBOX on S-BIAD452: BSCCM-like single-marker regime vs full joint supervision vs single+pooled; held-out donor; cell- vs cluster-level, pair-correlation, gating F, kNN-imputation baseline | outsider 3a, Codex, lit A/C | paradigm | measures the user's priority with ground truth; tells which training regime/model decodes joint profiles | ~1-2 GPU-h | N080 (Codex write) |
| 2 | two-stem joint-source training BSCCM + S-BIAD452 (shared trunk, shared heads + per-dataset affine, IFC-only heads) | Codex 1, lit B, outsider 3b | module | +.003-.009 GOAL (5/11 targets), joint observations | ~2-4 GPU-h | N081 (Codex write) |
| 3 | step-matched learning curve (data vs optimisation) | red-team F2 | knob | decides the lever | ~1.5 GPU-h | N082 (queued) |
| 4 | dense fluorescence supervision BSCCM / nine-image IFC fluorescence pretraining | N074 (running), Codex 2 | paradigm | +.002-.008 | 3 GPU-h | N074 pending; IFC variant after N080 |
| 5 | in-domain SSL scaling | N078 (running) | module | +0-.010 | 19 GPU-h | pending |
| 6 | joint-correctness card on BSCCM 'all' cells (4x4 pooled corr, within-gate r, gating F) | lit A | analysis | evidence only | 0 GPU-h | with N080 metrics |
| 7 | image-conditioned joint uncertainty (rank-2 covariance) | Codex 4 | module | joint likelihood | 2 GPU-h | option inside N080 (--cov-rank) |
| 8 | marker-adapted ViT / physical 3-ch rendering + adapter | Codex 3, outsider 1 | module | +0-.008 | 3-4 GPU-h | parked (N053 ViTs worse) |
| 9 | latent cell-state mixture model | outsider 2 | paradigm | joint coherence | 1.5 d | parked until N080 shows a joint gap |
| 10 | BSCCM-coherent 566-angle teacher distillation | Codex 5 | module | +0-.002 | CPU gate | parked |
| 11 | hybrid handcrafted morphology bridge | Codex 6 | module | +0-.004 | CPU gate | parked |
| 12 | leave-one-marker-out transfer test | lit D | analysis | representation check | 1 GPU-h | parked |

Chosen portfolio: N080 (paradigm), N081 (module), N082 (knob), plus pending N074 (paradigm) and N078 (module). Total new GPU ~6 h.
Unchosen -> notebook/TODO.md. Honest expectation: the registered .70 still needs ~+.014 over the best ensemble; external joint data is the
only untested source of NEW information; joint per-cell decoding can only be judged on S-BIAD452.
