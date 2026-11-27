# N034 · LF -> fluorescence marker IMAGES by conditional flow matching (A: paired per-cell CFM + same-net regression baseline, stain-conditioned, batch-held-out) as the bench for B: set-level distribution matching with pairs hidden (user idea 2026-11-11)

type: exp · date: 2026-11-11 · status: done
goal_effect: flat · scale: paradigm · value: void

## Goal
User idea 2026-11-11: first predict the marker IMAGES (6-ch fluorescence) from the label-free images, with a distribution-matching
(flow-matching) model; user clarification: "actual distribution matching ... not one cell at once but more like a subspace of a slice"
(set-level matching over a subset of a slide).

## Plan
A (paired): conditional rectified flow matching LF(24 ch) + stain token -> 6-ch fluor image (128 px), CFG-dropout 0.1, EMA; the SAME
UNet as a deterministic MSE regressor = posterior-mean baseline. Train blood draw 1 split 0 (b1to0 direction, screen 1 seed); test
in-batch held-out FOVs (split 2) and the new blood draw 0. Targets: border-median background removed, slide-position gain removed
multiplicatively (HGB per slide x channel on log integral; labels-only curation, LF untouched), asinh(./a_c)/3.
Yardstick: per-stain ridge READOUT image -> curated v5 label fitted on REAL training-draw images, applied identically to real (reference),
samples, sample mean, regression. Hypotheses: (1) one sample r ~ r_mean^2-ish below the mean (a perfect sample of p(y|x) correlates
r^2 with the truth), sample mean ~ regression ~ scalar teacher on the readout scale -> flow matching does NOT add per-cell accuracy;
(2) it fixes the DISTRIBUTION: W1(sample readouts, real) and spread ratio near 1 vs regression shrinkage (sd ratio ~ r), calibrated
rank histogram; (3) realistic spatial patterns (membrane ring/caps) vs blurred regression. Kill: samples worse than regression on W1.
B (user's set-level idea), after A: pairs hidden, loss only between SETS of cells (subset of one slide): B1 same cells shuffled,
B2 different cells of the same slide/stain (= the Giemsa situation: same blood draw, different cells). Theory: B2 is only identified
through variation of set composition (learning from label proportions) or a shared stratifying observable; with sets drawn from one
population it can fix marginals, not per-cell values -> B quantifies how much a distribution-matching bridge could give for G->M.

## Changes
scripts/fm_prep.py (targets), fm_common.py (CondUNet, FluorSet, Prep, Heun sampler with CFG), fm_train.py (cfm|reg, preempt-safe,
wandb offline), fm_eval.py (readout yardstick, distribution + calibration + image metrics).

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->
- Smoke 1302023 (h11-08, 7 min, 1 H100; CPU partition had 1 node -> prep ran in this GPU job): fm_prep 92 s (81 GB read 58 s);
  position-gain trend r(trend, log integral) .10-.37 per slide x channel; a_c = [5.9 5.9 4.4 4.4 7.4 3.0] counts/px. CondUNet 27.7 M
  params, peak 69.7 GiB at bs 128. Train/eval pipeline end-to-end OK (cfm 1 epoch on 6k cells, reg on 3k, eval K 2).
- Full runs (preempt, 1 H100 each, gpu_job.sbatch + preempt-run, wandb offline -> helma2 cron): 1302059 n034_cfm (30 ep, b1 split 0),
  1302060 n034_reg (15 ep); out /hnvme/workspace/v181bb10-bsccm/runs/n034/{cfm,reg}. Estimate <= 4 + 2 GPU-h.
- Codex red-team of fm_* launched (codex/brief_redteam.md -> codex/redteam_N034.md).
- 1302059/1302060 cancelled after 2 min: 221 cells/s at 100 % GPU (7.5 h for 30 ep). Added --patch 2 (pixel-unshuffle stem /
  pixel-shuffle head: 128 px in/out, UNet at 64 px) + channels_last -> 1302067 n034_cfm (runs/n034/cfm_p2, 30 ep), 1302068 n034_reg
  (runs/n034/reg_p2, 15 ep); 536 cells/s at step 400 and rising, peak 16 GiB.
- B, scalar bench of the set-level idea (scripts/n034_setmatch_sim.py, login node CPU, 3 min; setmatch/setmatch_sim.csv; plot
  .agent-research/plots/setmatch-newdraw-r-vs-log2-setsize_2026-11-11_1111.png). Frozen SSL LF features (n007, no labels) -> 64 PCs,
  linear readout learnt ONLY from set-level 1-D OT (iterated within-bag sort-matching, 8 starts x 30 it, best set-level objective),
  train draw 1 split 0 (20k cells/marker), test new draw 0. Mean r over 7 targets (CD16, CD14, CD45, CD3, HLA-DR, CD19, autofl):
  paired .562; same cells / pairing hidden, set size 4 .522, 16 .432, 64 .191, 256 .029, 2048 -.020; DIFFERENT cells of the same
  slide (Giemsa situation) -.04..+.08 at every size (= chance). Strong markers survive small sets (CD16 .85 -> .84 / .83 / .48 at
  4 / 16 / 64; CD14 .80 -> .80 / .78 / .26). In-batch test identical pattern. Interpretation: set-level matching identifies per-cell
  values only through variation of set composition (small sets of the SAME cells); between different cells drawn from one population
  it carries no per-cell information -> it cannot close the Giemsa<->marker gap in BSCCM (2 blood draws). Practical corollary for
  the wet-lab request: if the same cells are imaged with markers and then Giemsa, FOV-level groups of <=16 cells already recover
  77-99 % of paired r -> cell-level registration is not required. [WITHDRAWN by rereview N3/N4/N5, see below: per-marker recovery at
  n=16 is 5-98 % (autofl 5 %, CD19 22 %, HLA-DR 49 %); the OT learner fails on large bags where bag-mean regression succeeds; B used
  legacy v5 labels and all-split testing -> rerun as v2.]
- Codex red-team (codex/redteam_N034.md): F1 BLOCKER new-draw split-0 fluorescence calibrates its own gains/labels while all splits were
  scored; F2 MAJOR HGB position gain does not generalise; F3 MAJOR default targets_v5 = unclean lineage (Giemsa cells in v3/v4 fits);
  F4 MAJOR yardsticks pass trivial controls (template pix_r .53; unconditional null passes marginal W1/rank); F5 MAJOR cfm 30 vs reg
  15 epochs; F6 MAJOR smoke never entered the label loop; F7/F8 MINOR (averaging space, r_real is not a bound; eval identity checks).
  Reproduced first-hand: F2 (median var ratio of log integral after correction: fit split 0 .961, held-out split 1 .995, split 2 .994;
  25 % of slide x channel worse on split 2) and F3 (v5 vs v5c differ on non-Giemsa rows: autofl rms .078, grp rms .043-.048; singles
  identical). F1/F4/F5/F6 verified from code/configs. Runs 1302067/1302068 cancelled (trained on the F2 targets).
  Fixes: fm_prep gain = smooth Huber model per slide x channel (quadratic position-in-FOV + quadratic stage x/y + linear z), fit split 0,
  KEPT only if held-out split-1 log-integral variance drops > 1 % -> 72/132 kept (median var ratio fit .978 / split 1 .977 / split 2
  .983, 6 % of kept worse on split 2; gains p5-p95 .89-1.13), others gain 1; a_c from training-draw split 0 (identical values).
  targets_v5c everywhere (F3). fm_eval: newdraw scored on split 2 only (F1: calibration cells never scored; LF normalisation per-slide
  scale remains transductive unlabelled input calibration as in all GOAL runs -- disclosed); controls stain-template + LF-shuffled
  sample, conditional W1 in regression-readout quintiles, CRPS, sharpness, per-channel disk pix_r for all arms (F4); r_avgread primary,
  r_realref renamed (F7); mode + fm_prep hash asserted, identities saved (F8); trainer fingerprints LF normalisation files (F8).
  Matched budgets 30/30 epochs (F5): 1302123 n034_cfm runs/n034/cfm_v2, 1302124 n034_reg runs/n034/reg_v2. Eval smoke with 80 cells/stain
  (label loop exercised, F6): 1302125 runs/n034_smoke2 -> OK: every stain x set entered the label loop with all keys (r_realref,
  r_avgread, r_reg, r_shuffled, w1/condw1/crps, rank hist), real-image reference r .53-.97, batch-1 'all' pooled CD123/HLA-DR/CD14
  correctly absent; 1-epoch smoke models are noise (pix_r 0), template pix_r .49-.84 (= the control floor to beat). Full runs
  ~860 cells/s -> ~2.1 h each.
- Full runs done (30 ep each, 2 h 11 min, 16 GiB): val loss converged (cfm EMA .3071 @5 -> .3009 @15 -> .2997 @30; reg .0672 @5 ->
  .0663 @15-20 -> .0665 @30). Eval 1302722 (24 min; runs/n034/eval_v2; 400 cells/stain/set, K 8, Heun 16, cfg 1; table
  n034_eval_v2.csv; figure fig_n034_eval_v2.png). Mean over 12 labels:
    newdraw (split 2): r_realref .726 | flow r_avgread .440, r_sample .314, r_mean .449 | regression .599 | shuffled-LF .009.
    inbatch          : r_realref .777 | flow .508 / .383 / .520 | regression .695 | shuffled .022.
  Distribution vs real readouts (sd units): newdraw marginal W1 flow .753 / reg .696 / shuffled .755; conditional W1 .792 / .739 /
  .862; CRPS flow .739 vs reg .778; inbatch W1 .345 / .222, condW1 .392 / .299 / .699, CRPS .414 / .384. Flow wins the marginal only
  on dim markers (inbatch W1 CD19 .11 vs .73, autofl .10 vs .51, CD123 .17 vs .19, HLA-DR .26 vs .32), where regression is biased;
  rank calibration poor (outside-fraction .32 inbatch / .71 newdraw vs ideal .22). Diagnosis (per_cell.npy): flow samples are
  UNDER-dispersed (single-sample sd / real sd .41-.89 inbatch) and dimmer for bright markers (bias -.42 CD45, -.79 CD16, -.6 grp
  in sd) [rereview correction: K-averaged marginal sd ratio .41-1.20 over 12 labels, .41-.82 over singles; grp CD16 1.20 over-spread];
  montage: realistic texture/caps for CD3+/CD14+/autofl but CD16+ neutrophil samples far too dim (regression gets it).
  Spatial pix_r (disk, per channel): regression > flow mean ~ template > single sample > shuffled for every stain -> no evidence
  of conditional spatial detail beyond the stain template yet. Sampler sweep (steps 16/48 x cfg 1/2, K 4, 200 cells, seed 1)
  1302931-1302934 to test whether the deficit is the sampler.
- Sampler sweep round 1 (n034_sampler_sweep.csv; mean of 12 labels; reg .612 newdraw / .664 inbatch on these cells):
    newdraw r_avgread / r_sample / CRPS / condW1: s16 g1 .416/.319/.765/.814; s16 g2 .463/.396/.676/.700; s48 g1 .521/.423/.677/.713;
    s48 g2 .548/.491/.597/.599 (reg CRPS .776, condW1 .741). inbatch s48 g2 .557/.513/.331/.215 (reg .664, CRPS .387, condW1 .311).
  The deficit was largely the sampler (16 Heun steps under-integrate: sd ratio .57 -> .78-.97). At 48 steps + CFG 2 the flow model
  BEATS regression on every distribution score (CRPS, marginal + conditional W1) while still trailing on per-cell r (.548 vs .612).
  Not saturated -> round 2: s96 g2, s48 g3, s96 g3, s96 g4 (1303342-1303345).
- Codex re-review (codex/rereview_N034.md): F2/F5/F6/F7 RESOLVED; F1/F3/F4/F8 PARTIAL (B still on v5 + all splits; evaluator does not
  check LF files; no spatial-sensitive validation). New: N1 MAJOR sampler chosen while viewing new-draw scores -> N034 new-draw numbers
  are DEVELOPMENT results (frozen confirmation = N035); N2 MAJOR rank calibration fails at the promoted settings (outside-fraction
  newdraw .77-.78 vs exchangeable .40 for K 4; within-cell sharpness .25) -> lower CRPS is real, calibrated uncertainty is NOT shown;
  N3 MAJOR B arms confound pairing with budget/locality; N4 MAJOR the OT learner is not an information bound -- REPRODUCED first-hand
  on the synthetic example (paired r .9998, OT .0321, bag-mean ridge .9908 for 64 same-cell bags of 256); N5 B on legacy labels /
  calibration cells; N6 MINOR shuffled control had 7-12 fixed points per 2-4k cells, ties could empty strata, no saved readout metadata
  (offline equal-K shuffled CRPS .95 vs flow .57 confirms the conditional CRPS gain); N7 MINOR montage +/- are brightness extrema.
  All macro numbers in this item reproduce to 1e-16. Fixes: fm_eval derangement, rank-based quintiles, LF-file hash assert, saved
  readout weights/seed/sizes, --newdraw-split; fm_figure 'bright/dim' + channel labels; B rewritten (v2, below); N1 -> N035.
- B v2 (scripts/n034_setmatch_sim.py v2; setmatch/setmatch_sim_v2.csv; plot .agent-research/plots/
  setmatch-v2-newdraw-r_2026-11-11_1707.png): targets_v5c, new draw split 2 only, learners OT / BAGMEAN / OT-init-from-
  bagmean, budget- and locality-matched arms. Mean new-draw r over 7 targets: paired .561, paired half-A .554 (budget is not the issue);
  same cells random sets, bagmean n 4/16/64/256/2048: .548/.503/.451/.309/.104 (OT-init .526/.470/.422/.342/.095; old OT .514/.424/.262/
  .016/-.035); same cells FOV-sorted bagmean .540/.483/.390/.226/.055 (locality does not help); same cells half budget .542/.493/.385/
  .251/.070; DIFFERENT cells random sets bagmean .089/.058/-.012/.027/-.016 at n 4-2048 (n 1: .032), different cells sharing an FOV group
  -.026/.012/-.069/-.044/.009 -- no positive recovery with any learner/design [check3 correction: mean |r| over targets .05-.15, single
  markers up to |r| .48 of BOTH signs (CD16 other_fov n 1 -.31 with 10k bags) = systematic FOV-level/position components, not chance and
  not per-cell information; a permutation null was not run]. Per marker, same-cell random bags n 16 vs paired: CD16 .84/.85, CD14 .79/.81, CD45 .65/.67, CD3 .51/.54, HLA-DR .39/.49,
  autofl .25/.41, CD19 .08/.16. Interpretation (red-team-consistent): the v1 large-set decay was mostly the OT learner; with real
  finite bag counts information still declines with set size; sets of DIFFERENT cells from one population (randomly or by FOV) carry
  no per-cell information in this bench.
- Check 3 (Codex unavailable -> Claude subagent, codex/check3_N034_claude.md): setmatch v2 code clean (A/B disjoint, n=1 identities
  2e-16, split-2 tests, budgets matched, all B v2 means reproduce). MAJOR wording errors fixed after first-hand reproduction: 'other'
  arms mean |r| .05-.15, max .48 (not '< .09 = chance'); flow distribution scores already win at s16 g2 / s48 g1 (K027 corrected);
  withdrawn registration claim removed from IDEAS; N035 cohort statement made consistent + commands/decision rule frozen. Derangement
  fix was not a permutation (duplicated LF) -> neighbour-swap loop with permutation/fixed-point asserts (unit-tested n 2/3/5/400 x 200
  seeds); hardened eval smoke 1304388. PROJECT-WIDE F6 reproduced (v4 label detrend overfits split-0 FOVs) -> Next / IDEAS.
- Round 2 (same csv; plots .agent-research/plots/n034-newdraw-readout-r-sampler_2026-11-11_1632.png and
  n034-newdraw-crps-sampler_2026-11-11_1632.png): SATURATED at 48 steps. newdraw r_avgread / CRPS / condW1: s48 g3 .545/.570/.551,
  s96 g2 .550/.592/.593, s96 g3 .548/.565/.546, s96 g4 .540/.571/.536 (sd ratio 1.08 = over-dispersed); inbatch best s48-96 g2
  (.557-.558, CRPS .331, condW1 .215; g3/g4 over-disperse in-batch, sd ratio 1.12-1.32). Per label (newdraw, s48 g2 vs reg): equal on
  lineage markers CD16 .86/.86, CD14 .86/.85, CD45 .76/.75, grp CD45 .68/.71, grp CD16 .76/.80, autofl .58/.58; clearly worse on the
  subtle lymphoid/antigen-presenting signals CD3 .25/.55, HLA-DR .36/.50, CD123 .22/.32, CD19 .26/.32, grp CD3/19/56 .66/.75.
  Marginal W1 of the LF-shuffled control equals the flow (.50 vs .49, s48 g3) = red-team F4 confirmed: only conditional W1 (.55 vs
  shuffled .81, reg .74) and CRPS (.57 vs reg .78) carry the conditional claim. Figure (s48 g3 montage): bright CD16/CD14 example samples
  realistic incl. ring/granular texture, regression blurred and too dim for CD14+; debris/neighbour structure reproduced.
  [coordination 15:17, session v181bb10-4a] two sessions are driving N034; 4a had submitted the identical round-2 commands
  (1303346-1303349, same output dirs) and cancelled them in favour of 1303342-1303345. 4a submits nothing further for N034 and does
  not commit until the user says which session continues.

## Outcome
<!-- on close: bin/effect.sh N034 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->
A (paired CFM image model, 1 seed, b1->b0 only; DEVELOPMENT numbers, sampler chosen on them -> confirmation N035): with an adequate sampler (Heun 48 steps, CFG 2-3) LF -> 6-ch fluorescence IMAGES are
realistic and distributionally better than the same network as regression [check3: already at s16 g2 or s48 g1; only s16 g1 loses
marginal/conditional W1, and even s16 g1 wins CRPS] (new draw CRPS .57 vs .78, conditional W1 .55 vs .74,
marginal spread ratio .8-.9 vs regression shrinkage on dim markers; but rank calibration fails, outside-fraction .77 vs .40), equal on lineage markers per cell, but worse on subtle markers (mean r
.55 vs .61; CD3 .25 vs .55). Hypothesis (1) confirmed in its main claim (no per-cell gain; its sub-predictions sample-mean ~ regression and r_sample ~ r^2 failed), (2) PARTIAL: distribution scores better only with >= 48 steps + CFG (16-step sampling
under-disperses and loses) and per-cell calibration fails, (3) partially: realistic texture/caps in samples, but pixel-level conditional spatial fidelity beyond the
stain template is not shown by pix_r. GOAL metric (scalar common-11, both directions) not measured -> flat --void, paradigm.
B (set-level, scalar bench, v2 after rereview): pairs hidden in sets of the SAME cells recover most of the paired signal with a
bag-mean learner (mean r .548/.503/.451 at n 4/16/64 vs paired .561; strong lineage markers ~97 %, autofl/CD19 only 50-60 % at n 16),
declining at large n (partly the regression set-up: 3-24 bags for 65 parameters at n >= 256); sets of DIFFERENT cells from one
population give no positive recovery with any learner or grouping (mean |r| .05-.15, both signs, FOV-level components) ->
within this bench (frozen transductive SSL features, linear readouts, one blood-draw population per set) distribution matching cannot
close the Giemsa<->marker gap of BSCCM; it needs composition variation across sets (many donors with bag-level phenotypes) or
same-cell groups. Not an information-theoretic proof (rereview N4); scope = these learners and bag designs.

## Outputs
<!-- files in .agent-research/results/N034-slug/ + external absolute paths -->
results/N034-*/: n034_eval_v2.csv, n034_sampler_sweep.csv, setmatch/setmatch_sim.csv, plot_*.csv, fig_n034_eval_v2.png,
fig_n034_s48_g3.{png,pdf}, smoke.sbatch, smoke2.sbatch, codex/ (redteam_N034.md, rereview_N034.md). Plots in .agent-research/plots/
(setmatch-..., n034-newdraw-...). Checkpoints /hnvme/workspace/v181bb10-bsccm/runs/n034/cfm_v2/ckpt.pt and runs/n034/reg_v2/ckpt.pt; evals
runs/n034/eval_v2, eval_sweep_*; targets /hnvme/workspace/v181bb10-bsccm/fm_prep.npz. Deliverables ~/bsccm_deliverables/figures/
fig4_marker_images_flow.{png,pdf}, fig5_setlevel_matching.png.

## Next
- Close the subtle-marker gap: train longer / wider, or condition on the scalar teacher's LF features (the regression net reads CD3
  from LF; the flow under-uses weak conditional signal) -- e.g. a CFM with a frozen LF-teacher embedding as extra condition.
- b0->b1 direction + >= 3 seeds before any claim; joint multi-marker sampling from the 'all' slides (co-expression consistency).
- Use the flow model where its strength matters: per-cell uncertainty (lower CRPS; calibration NOT yet achieved) and population
  statistics (fraction positive).
- PROJECT-WIDE (check3 F6, reproduced): the v4 HGB label detrend overfits split-0 FOVs -- FOV-mean variance share of v5c labels is
  lower on the fitted split 0 than on held-out splits 1/2 (CD19 draw 1 .34 vs .54/.57, autofl draw 0 .48 vs .64/.60, CD3 draw 0 .40 vs
  .53/.54; pre-detrend v3c equal across splits). The GOAL scores test-draw split-0 cells whose labels were detrended in-sample. Needs
  a smooth, cross-fitted, held-out-validated label detrend (as done for the image gain here) and a GOAL re-score -> own fix item.
