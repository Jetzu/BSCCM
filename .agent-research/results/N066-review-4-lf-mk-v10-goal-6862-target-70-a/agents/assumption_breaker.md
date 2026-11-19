# N066 assumption breaker (2026-11-19)

Read: GOAL.md, KNOWLEDGE.md, N066 {attempts, declined, diagnosis, n066_*.csv}, N059 {diagnosis, ideas}, scripts/train_lf.py,
scripts/make_targets_v3/v4/v5.py, N051 build scripts, bsccm/{gpu_norm,data,heads}.py. New CPU analyses on the frozen 6-way ensemble
(N065 E3c + N053 members, GOAL .6929, same construction as n066_perslide.py) and v10 labels: scripts `/tmp/ab066/a3.py`..`a11.py`
(ephemeral: copy into `agents/ab/` to keep). No GPU used. "6-way" numbers reproduce n066_perslide.csv (b10 .7036, b01 .6822).

## 1. New measurements (all CPU, minutes)

| # | test | result | source |
|---|---|---|---|
| M1 | Label-free per-slide standardisation of predictions (median/IQR, like the v5 label rule) on the two b01 'all' test slides | b01 .6822 -> .6830; grp:CD16 +.0077, grp:CD3/19/56 +.0016, grp:CD45 -.0006; GOAL +.0004 | a3.py |
| M2 | Cross-output: single-marker heads scored against pooled labels on 'all' test cells | grp:CD45 via single:CD45 head .797/.666 vs own head .779/.660 (b10/b01); grp:CD16 z-avg .874/.773 vs .868/.784; grp:CD3/19/56 sum of single heads .737/.762 vs .812/.795 -> net about +.001 GOAL | a3.py |
| M3 | Spearman per b01 'all' slide, grp:CD45 | rep0 .596 vs rep1 .819 (Pearson .578/.881): rep0 loses RANKING, not only amplitude | a3.py |
| M4 | FOV-cluster bootstrap (1000x) of the GOAL level | SD .0089 (6-way), .0090 (N053 3-way) = 3.3x the seed noise .0027; paired 6-way minus 3-way +.0069, SD .0016; E3c minus 3-way SD .0032. b01 CD19 rests on 82 FOVs / 722 cells | a4.py |
| M5 | Label starvation: per-target (b10 - b01) vs log(n_draw1/n_draw0) | r .18 over 11 targets, .004 over the 8 singles. CD45 reverses (-.100) at a 2.0x ratio; CD19 favours b10 at a 0.6x ratio | a5.py, n066_perslide.csv |
| M6 | Is `<grp>_full_model_unmixed` (the pooled-label unmixing) present on single-antibody slides? | yes, on EVERY cell. Raw r with the single-model value on the matching slide: CD45 .984/.841, CD16 .983/.896 (d0/d1). Never used as a training label | a2.py, a5.py |
| M7 | Same-cell autofl axis of single labels: cov(marker, autofl label)/var(autofl), d0/d1 | CD45 .02/.79, HLA-DR +.37/-.14, CD3 -.35/+.40, CD56 -.46/-.21. In the negative bulk only: CD56 -.28/-.22, HLA-DR -.02/-.20, CD45 .01/.01 | a7.py, a9.py |
| M8 | HLA-DR b01 on the autofl axis | removing the prediction's projection on the predicted-autofl output (label-free): HLA-DR b01 .445 -> .522, b10 .525 -> .538, but GOAL .6929 -> .6831 (hurts CD56/CD19/CD14). Removing the training-draw leak: .6905; negative-bulk leak: .6932 (flat). Test-label oracle using the cell's own autofl LABEL: CD56 b10 .49 -> .693, CD19 b01 .328 -> .570 = unmixing anti-correlation INSIDE the test label (unreachable) | a6.py, a7.py, a9.py |
| M9 | grp:CD123/HLA-DR/CD14 calibration | 65 % of draw-0 unstained and 59 % of draw-0 'all' cells sit exactly at the clip floor -> unstained MAD = 0 -> asinh((x-med)/1e-6): a BINARY label (T = 0 vs 17.9-19.8). Trained only in b01 (masked in draw 1); not in GOAL | a10.py, make_targets_v3.py |
| M10 | Censored (clip-floor) label fraction d0/d1 | single:CD3 .094/.427, grp:CD16 .307/.082, grp:CD45 .095/.179, CD56 .087/.115, others <= .014. MSE treats censored values as exact | a11.py |
| M11 | Predicted marginals per slide of one draw (same blood) | shifts <= .35 SD, concentrated on the two CD16-antibody slides of draw 0 (single:CD16 pred -.35 on the CD16 slide, -.33 on 'all'). 19/22 scores are single-slide Pearson, so this cannot move them | a8.py |
| M12 | Is the 10-epoch schedule converged? | held-out r_mean per epoch still rising: b01 tpfov .6705/.6714/.6716 (ep 8-10), b10 .679/.6818/.6836. The only longer run (N014, 15 ep) was RANDOM-init r50 (init_ssl ''), so the SSL recipe's length was never varied | runs/n063_tpfov_s2_b0{1}/result.json, runs/n014_r50_b0to1/result.json |
| M13 | Supervision differs by direction | w_cls .5 in every GOAL run, but gates exist only for the 10,161 draw-0 'all' cells (K002): class CE (cls3 + cls10) and the binary M9 target are active only in b01 | train_lf.py, a1.py |

## 2. Assumptions, the cheapest test that could overturn each, and the gain

| area | assumption | status | cheapest overturning test | gain if overturned |
|---|---|---|---|---|
| data | Each target is learned only from its own slide; other slides give nothing | partly FALSE: M6, the pooled unmixing exists on matching single slides (+10,951 grp:CD45 and +9,397 grp:CD16 cells in draw 0 = 2.1x/1.9x) | b01 1 seed with those cells added to grp targets via per-slide affine adapters (0.3 GPU-h) | small: M2 shows the single heads already carry that information to pooled labels; <= +.003 |
| data | b01 is label-starved (outsider) | NOT supported per target (M5) | outsider C2 (b10 subsampled to draw-0 counts, 0.7 GPU-h) is still the clean test | decides whether the dense-teacher idea is worth building |
| data | Training uses every useful training-draw cell | true apart from 15.7k GS cells with autofl only | none needed | ~0 |
| labels | Clip-floor (censored) values are exact targets | untested; censoring mismatched between draws (M10) | Tobit/censored MSE, b01+b10 1 seed (1 GPU-h) | <= +.0045 (bounded by the N059 monotone-recalibration oracle) |
| labels | Auxiliary targets are clean | FALSE for grp:CD123/HLA-DR/CD14 (binary, M9), b01 only | in the M13 test below | unknown sign |
| labels | Single-label autofl content can be removed post hoc | killed: M8 (-.0024 / +.0003) | done | 0 |
| repr. | Per-target heads are independent | tested: cross-output adds about +.001 (M2); the N062 panel head already shares | done | 0 |
| repr. | Rotation and flip carry no extra signal (translation-only aug; data.py forbids naive rotation) | untested; outsider idea 2 covers it (LED permutation, Top = Full - Bottom) | outsider C3/TTA (0.05 GPU-h) | +.002-.006 if no symmetry is learned |
| loss | One recipe for both directions | FALSE: class CE and the binary target are b01-only (M13) | b01 arms w_cls 0 and drop the binary target, 2 seeds (about 1.2 GPU-h) | -.005..+.005 on b01 (b01 is the deficit direction) |
| model | 10 epochs is converged | never tested for SSL r50 (M12) | 20 ep b01 (0.6 GPU-h) + b10 (1.3 GPU-h), 1 seed vs the s2/s3 controls | +.000..+.006 single, about half that in the ensemble |
| model | Outputs should not depend on the slide's antibody condition | weak evidence of slide nuisance (M11), and only on slide means | marginal-invariance (outsider idea 3) | ~0 via Pearson within slide |
| metric | b01 pooled-group deficit is a slide offset that standardisation fixes | killed: M1 +.0004 | done | 0 |
| eval | Seed sd .0027 is the relevant noise | true for paired differences (M4 .0016-.0032); FALSE for the level: FOV SD .0089, with about 65 items selected on the same split-2 cells | rescore frozen members on held-out-draw splits 0+1 (about 10x cells, never used for selection, labels Pearson-valid: affine split-0 z, cross-fitted detrend); about 0.5 GPU-h inference | decides whether .6929 vs .70 is signal; no GOAL change (diagnostic, not a re-base) |
| eval | SSL/test protocol is inductive | true (N031); LF slide_scale uses each slide's own images (label-free, accepted) | none | 0 |

## 3. What explains the b01 deficit (grp targets, HLA-DR)

All three are properties of the draw-1 TEST labels. None is a missing image feature:
- grp:CD45 b01 (.660 vs b10 .779): one dim replicate. rep0 r .578 / Spearman .596 vs rep1 .881 / .819 (n066_perslide.csv, M3).
  Per-slide standardisation (-.0006) and cross-output (+.006) do not fix it.
- grp:CD16 b01 (.784 vs .868): label shape differs between the two slides (means -.12/+.21, sd 1.13/.75; predictions .17/.14).
  The best label-free fix gives +.0077 on this target (M1).
- HLA-DR b01 (.445 vs .525): the draw-1 label carries a negative autofl component (pred-autofl coefficient -1.52 in a test-fitted
  oracle, .534). A label-free projection recovers .522, but the same operation costs other targets more (M8). No inductive rule
  picks HLA-DR alone without using test labels.
- Supervision asymmetry (M13) is the one remaining b01-specific TRAINING difference that has not been tested.

## 4. Pre-mortem: "three months on, still < .70" (3 reasons)

1. **The ceiling sits in the evaluation labels.** b01's three deficits are draw-1 label artefacts (section 3). b10's weak targets
   carry unmixing anti-correlation inside the test label (CD56 b10: .49 -> .693 only with the cell's own measured autofl, M8).
   Every inductive post-hoc fix tried here moves GOAL by <= +.001. Training-side wins of +.002-.004 each were absorbed by ensembling
   (K039/K042).
2. **We chased the level inside its own sampling noise.** FOV SD .0089 for the level. b01 CD19 rests on 82 FOVs. About 65 items
   were scored and selected on the same split-2 cells. A .70 reached by selection would not replicate on fresh held-out cells.
   Or the population value is already about .70, and the programme stalled on a bar (.6916 = best + 2 x seed sd) that ignores
   level noise.
3. **Two draws give one observed shift.** The autofl-axis coefficients flip between draws (HLA-DR +.37/-.14, CD3 -.35/+.40,
   CD45 .02/.79; M7). Recipe choices are also confounded with direction (class CE, binary target; M13). A fix tuned on b01 fits
   one flip and has nothing to replicate on.

## 5. Ideas (respecting declined.md: no lineage priors, no new data, no transductive training, v10 eval labels fixed)

**I0 (prerequisite, diagnostic). Fresh-cell replication on held-out-draw splits 0+1.**
- Mechanism: run the frozen 6-way, E3c and N053 members on the 90 % of held-out-draw cells that no model selection has touched,
  and score them with the same v10 labels. Pearson is invariant to the split-0 affine z, and the detrend is cross-fitted.
- Why it moves the bottleneck: it measures selection bias and the true level. If the fresh level is >= .70, the problem is test
  sampling, and the user should hear that. If it is <= .685, stop. It also re-tests rep0/HLA-DR on 10x cells.
- Expected: level shift within +-.01. Cost about 0.5 GPU-h of inference. Build 1-2 h (an `--eval-split` flag on a finished
  checkpoint). Report it as a diagnostic line, NOT a re-base.

**I1. Direction-symmetric supervision (b01 recipe audit).**
- Mechanism: b01 trains the shared encoder on two losses that b10 never sees: a 10-class + 3-class CE on gates derived from
  draw-0 'all' fluorescence, and a binary floor/above-floor grp:CD123/HLA-DR/CD14 target (M9).
- Arms on b01 only: (a) w_cls 0; (b) (a) + drop the binary target; (c) the reverse, keep CE but replace the binary target with
  a censored version.
- Why: b01 is the whole gap (outsider), and these are its only untested training-side differences.
- Expected -.005..+.005 on b01, i.e. up to +.0025 GOAL; P(gain) about 0.3. Falsifier: 2 seeds x 2 arms on the composite r50
  member, about 1.2 GPU-h. Kill if both arms are within .0054 of the s2/s3 controls. Build: flags only.

**I2. Schedule length, bundled with censored loss (one attempt, two flags).**
- Mechanism: (a) 20 epochs instead of 10, because held-out r is still rising at epochs 8-10 (M12) and SSL-r50 length was never
  varied; (b) Tobit/censored MSE for labels at the clip floor, because censoring is mismatched between draws (CD3 9 % vs 43 %,
  grp:CD16 31 % vs 8 %; M10), and a point-mass target teaches a draw-specific floor.
- Why: both are readout-calibration and underfitting levers on the weak and pooled targets that N060-N065 did not touch.
  (b) also addresses part of the N059 nonlinear-amplitude finding.
- Expected: +.000..+.006 single model, +.000..+.004 ensemble.
- Falsifier: 1 seed per arm per direction, about 2 GPU-h for the 20 ep arms and 1 GPU-h for Tobit. Pre-register against the
  s2/s3 controls at +.0054. Build: 2-3 h for the Tobit term, using the `censored` key already in targets_v5cx_v10.npz.

**I3. Real extra pooled labels from matching single-antibody slides.**
- Mechanism: `CD45_full_model_unmixed` on the CD45 slide and `CD16_full_model_unmixed` on the CD16 slide are the same unmixing
  as the pooled test label (raw r .98 with the single model in d0, M6). Calibrate them with the pooled rule (unstained med/MAD,
  'all' z) and add them as grp targets with a per-slide affine adapter (TrainAdapters exists).
- This doubles draw-0 grp:CD45/grp:CD16 labels with REAL measurements, not pseudo-labels. Optionally add the
  CD3/CD19/CD56-slide values as lower-bound (partial-sum) labels for grp:CD3/CD19/CD56.
- Why: it is the only label-count lever that uses the eval label's own measurement model.
- Expected +.000..+.003; capped by M2, because the single heads already transfer. Falsifier: b01 1 seed, 0.3 GPU-h; kill below
  +.005 on the pooled targets. Build: half a day.

Not recommended (killed here): post-hoc per-slide standardisation (M1), cross-output stacking (M2), autofl-leak deconfounding of
predictions (M8).
