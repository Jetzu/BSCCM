# N047 diagnosis (review 2): the frozen pooled-v5 GOAL is saturated for this predictor family

Gap: .67 - .6527 = .0173 (target is an unconfirmed Claude proposal).

1. Predictor family saturated against these labels (d1_big_ensemble.py -> .csv): mean of ALL 10 inductive held-out models trained
   so far (N031, N037, N040, N041, N046; base + zoom2 variants) = .6571 (b10 .6648, b01 .6494): +.0044 over the best, still inside the
   flat zone; the N040 student pair alone .6564. Model-side changes since N031 add <= .004 at ensemble level (single models up to
   +.010, but the gain is absorbed by ensembling).
2. The eval labels cap the score (N038, reproduced): scored within FOVs the same predictions reach .7205 (best) / .7259 (10-model);
   34-49 % of CD19/CD56/CD3 label variance is FOV-level background-estimation error (authors' background_lowess), predictions carry none.
   A label fix with per-cell background (N039 v8) removes the offsets but changes those labels (r .48-.70 with v5) and failed QC.
3. Clean in-distribution reference (N039): a v5-trained model that has SEEN the test draw reaches .678 (data-matched .659); the
   held-out loss sits in train-draw-1 -> test-draw-2 (.034: HLA-DR + pooled targets = draw-2 label-definition differences: CD19 at 35 %
   in pooled slides, autofluorescence-driven HLA-DR/CD19/CD14 strata, a 5x dimmer pooled CD45 slide). An inductive model cannot learn
   draw-2 label definitions it never sees.
4. More supervision helps only in-distribution-like ways: label-count curve +.035 on weak targets (50 -> 100 %), but pseudo labels
   from the same draw add +.0037 (ensemble) and do not compound (N046).
Bottleneck hypothesis: the remaining gap to .67 under the frozen definition is mostly eval-label noise (FOV-level offsets) and
draw-specific label definitions, i.e. the metric, not the predictor. Falsifiers: (a) a model-side change that lifts the 10-model
ensemble's pooled score by > .0054 without lifting within-FOV r more than pooled r; (b) the within-FOV/pooled gap shrinking for a
predictor that reproduces FOV-level offsets from LF images (would mean the offsets are image-visible, not label noise).
Implication (review #2 without progress): the GOAL as defined is likely near its ceiling for inductive methods; candidates for
re-scoping must go to the user: within-FOV r as co-primary, a v8b label re-base (dim-dye targets only, new pre-registration), or a
reliable-target headline (exclude CD19/CD56/CD3 pooled offsets from the mean).

## REVISION after the Codex diagnosis red-team (codex/diag_rt/diag_redteam.md; numbers reproduced to 1e-12 by Codex, my d1 rerun matches)
- The 10-model ensemble is a WEAK saturation test: members share encoder, recipe, seed and labels; mean residual correlation .953 (b10)
  / .973 (b01), CD19/CD56/CD3 .98-.99 -> equal-correlation n_eff ~1.03-1.05. The +.0043 says these models make the SAME errors, not
  that every inductive predictor is capped. Diversity (different heads, losses, inputs, measurement models) is untested.
- Within-FOV r (.72) is a conditional diagnostic, not an achievable pooled score: centring removes FOV offsets AND real between-FOV
  composition (raw between-FOV shares CD19 .66/.57, CD56 .62/.64, CD3 .55/.45; the permutation-excess part .34-.49 is the defensible
  'offset' estimate). Attribution of the gap to label noise vs ordinary cross-draw error is NOT identified by current evidence.
Revised: the gap is unexplained between (i) label/FOV measurement effects and (ii) shared errors of one predictor family; the next
portfolio must contain DIVERSE predictors and a direct decomposition (outsider Step 0: transfer factor vs within-draw ceiling).

## REVISION 2 -- the noise is partly OURS (assumption breaker agents/assumption_breaker.md + agents/ab/; reproduced first-hand
by d2_cofactor_check.py, independent code, no detrend)
- v3-v5 single labels = asinh(x / q10) with q10 = the 10th percentile of the authors' unmixed value per slide (N005). For dim dyes q10 is
  tiny (CD3 .06-.84), so asinh expands the background noise of negative cells. Fixed cofactor 50 instead: CD3 r of the SAME N031
  predictions .53 -> .78 (b10) / .60 -> .77 (b01), CD56 .26 -> .37 / .30 -> .46; FOV excess CD3 .49 -> .12 / .34 -> .03, CD56 .65 -> .51 /
  .62 -> .37; the other six markers unchanged. With detrend + z as v5 (agents/ab/a6): N031 ensemble .6828, 10-model .6870 (re-base).
- FOV offsets are NOT image-visible (agents/ab/a_fov_offsets, a3: LF FOV features R^2 <= .03) -> FOV-context head refuted.
- The b01 draw gap is a draw-specific READOUT: an oracle per-target remap over the 13 heads closes 91 %; autofluorescence slope of the
  HLA-DR/CD19/CD14 labels flips sign between draws (draw-1 stains 2.5-3.9x dimmer); every zero-label correction tested is negative.
Final: (1) a label-curation error (asinh cofactor) inflates noise in CD3/CD56 labels -> fix = principled cofactor (label-only rule);
(2) the remaining cross-draw gap is draw-specific label readout, not learnable inductively from one draw; (3) the predictor family is
saturated against frozen v5 (shared errors, residual r .97).
