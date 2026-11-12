# N038 ideas: synthesis and chosen portfolio

Sources: agents/outsider.md, agents/lit_labelfree_subsets.md, agents/lit_labelnoise_shift.md, agents/assumption_breaker.md (+ agents/ab/),
codex/ideas/ideas_codex.md, codex/diag_rt/diag_redteam.md; diagnosis.md (3 revisions). Top literature claims verified first-hand by a
web-fetch pass (BSCCM paper: CD19 at 35 % in batch-1 'all', pooled CD123/CD14/HLA-DR "too noisy", biological variation between
isolations; DeepIFC held-out CD19 r .61 / CD56 .53 / CD45 .90; Pinkard 2405.20559 GMM heads per marker, no numeric marker results).

## What the review established
1. The binding limit is the EVAL LABELS, not (only) the predictor: within-FOV GOAL .7211 vs pooled .6527 (reproduced, d2_within_fov);
   CD19/CD56/CD3 labels carry 34-49 % FOV-level variance (authors' smoothed `background_lowess` in the shared dim dye channel) that
   LF predictions do not and should not contain. A pooled .67 on v5 labels needs ~60-90 % of the remaining draw gap closed.
2. A real, smaller generalisation gap for rare populations: tail weighting lifts CD123 in-distribution (.42 -> .65/.53 -> .68; FOV
   tail05 .680) but not across draws (b10 .6306 vs .6343). HLA-DR in- vs cross-draw gap positive in both draws.
3. Draw-1 single HLA-DR/CD19/CD14 labels within lymphocytes follow autofluorescence, against their own dye ([unverified] by me).
4. Literature: dim/rare markers plateau at r ~.4-.6 even at 60x (DeepIFC); resolution and positives-per-subset are the levers with
   controlled evidence; targeted (physics) augmentation is the only inductive family with measured OOD gains; no paper shows a
   noise-aware loss raising r against NOISY test labels (Noise2Noise logic: cleaner training labels cannot lift the eval-label cap,
   consistent with N037 flat).
5. Metric noise: seed sd .0027 (GOAL.md updated), FOV-bootstrap sd of the GOAL .0077; CD123/CD19 hinge on the top 1 % of cells.

## Ideas, deduplicated and ranked (P(success) x gain / cost)
| # | idea | sources | scale | gain (reasoned) | cost | rank |
|---|------|---------|-------|-----------------|------|------|
| A | Label v8: per-cell-background re-derivation of single-antibody labels (+ draw-1 own-dye QC), pre-registered QC, retrain; score frozen v5 GOAL AND v8 (re-base proposal to user) + proper in-distribution ceiling (r50 FOV on v5, data-matched) | assumption breaker I1/I2, lit-noise 1/4/5, codex audit | module | v5 GOAL +0-.005; v8-scored ~.72 (re-base, not progress) | 1 CPU-day + ~2.5 GPU-h | 1 |
| B | Dense label completion (noisy student): ensemble pseudo-labels all 11 targets on every training-draw cell, augmented student on real + down-weighted pseudo labels; label-count learning curve first | outsider 4, lit-subsets 1 | paradigm | +.003-.012 if the 50 % -> 100 % learning curve gains >= .03 on weak targets | ~3 GPU-h | 2 |
| C | Physics-based source-only nuisance randomisation (per-LED-family gain/offset, shot noise, angle-proportional defocus shifts) + consistency loss, bundled with tail05 rare weighting: does invariance let the in-distribution CD123/HLA-DR gain transfer? | outsider 2/3, codex 4, lit-noise 3, lit-subsets 5 | module | +.003-.010 | ~1.5 GPU-h | 3 |
| D | Weak-marker architecture: separate BF/DPC, DF-low, DF-high stems with late fusion + private late blocks for CD19/CD56/CD123/HLA-DR, as an extra ensemble member; low-pass-input probe first (image- vs label-limited) | codex 1/2, lit-subsets 2 | module | +.004-.015 if the low-pass probe shows weak targets are image-limited | ~3 GPU-h | 4 |
| E | Slide-replicate nuisance subspace: project between-slide (same draw) feature directions out of frozen features, refit head | outsider 1a | module (diagnostic) | +.0-.017 (depends on slide-level f) | < .5 GPU-h | parked |
| F | Correlation-direct / HL-Gauss / mixture head for weak markers | codex 5, lit-noise 2 | knob-module | +.001-.006, likely flat | .5 GPU-h | parked |
| G | Photon summaries as auxiliary label-side targets | codex 6 | module | +.002-.008 | .8 GPU-h | parked |
| H | Reconstructive SSL (predict masked LF detail / held-out LEDs) | codex 3 | paradigm | +.003-.011 | 2-3.5 GPU-h | parked (literature: SSL weak lever) |
| I | Synthetic-aperture probe on the 4,304 batch-1 single-LED cells | lit-subsets 3 | diagnostic | hardware ceiling only | 2-4 GPU-h | parked |
| J | Shared-dye structured head (grp CD3/19/56 = sum of singles) | assumption breaker I3 | module | +.004-.009 if positives-limited | 1 GPU-h | merged into B's learning curve |
| K | Down-weight the dim batch-1 'all' replicate-0 slide for grp targets | assumption breaker I4 | knob | +.001-.003 | .5 GPU-h | parked |

## Chosen portfolio (4 exp items, concurrent where the cluster allows; total ~10 GPU-h < 25, no approval needed)
- N039 (module) = A. Biggest lever on what the GOAL can measure; also produces the clean ceiling the diagnosis lacked.
- N040 (paradigm, wild-ish bet) = B. Changes the training signal from sparse 1-of-11 supervision to dense semi-supervised.
- N041 (module) = C. The only literature-backed inductive shift lever; bundled with tail05 to test revision-1 directly.
- N042 (module) = D, gated on its low-pass probe (cheap first stage).
No knob item. Every attempt measures the frozen GOAL (v5, common-11, both directions) and is labelled with effect.sh; flat zone
+-2 x .0027; paired FOV-bootstrap differences reported alongside.

## User decision to request (not blocking)
Re-base the GOAL on v8 labels if N039's pre-registered QC passes (precedent N023 v4 -> v5): the frozen v5 GOAL keeps ~35-50 %
non-cell variance in CD19/CD56/CD3. Alternatives: report within-FOV r as co-primary, or restrict the headline to reliable targets.
