# N034 check 3 (Claude stand-in for Codex, CODEX-UNAVAILABLE): fixes after rereview_N034.md

Scope is brief_check3.md. The repo was read-only; this file is the only write. Checks ran in bsccm.sif (numpy 2.4.6, pandas 3.1.1)
on the stored CSV/npz files, with scripts in /tmp/$USER/n034c (c1-c7.py). No GPU, SLURM, training or inference was used.
Disclosure: finding 6 computes label-only summary statistics (within-FOV ICC) on every split, including draw 1 split 1, which is
N035's primary cohort. No model output on that cohort was looked at.

## Verified OK (no finding)
- `n034_setmatch_sim.py` v2 has no same-cell leak in `other`/`other_fov`. A/B are disjoint halves of `tr` (:36). `other` zips
  chunks of permuted A with chunks of permuted B within a slide (:46). `other_fov` splits each 2n chunk into disjoint [:n]/[n:]
  halves, and every cell lands in exactly one chunk (:47-50). The learners see only `y[label_rows]` as a sorted multiset or a mean
  (:55, :58).
- n=1 identities REPRODUCED. same_A n=1 equals paired_A (max |dr| 2.2e-16) and same_rand n=1 equals paired (5.6e-16), for all
  three learners. This also confirms the global-to-local `idx` mapping (:78, :87) under pandas 3.1.1. `ia = searchsorted(tr, A_)`
  (:80) is correct.
- Train is b1/s0/clean/masked; both tests are split 2 only (:73-75).
- Budget matching REPRODUCED. same_A/other/other_fov use 8823-8850 LF cells and have identical n_bags at every n. same_fov and
  same_rand use 17700. Locality is matched only approximately: the other_fov LF window is 2n cells against n for same_fov (MINOR,
  not a finding).
- Every mean in the item's 'B v2' bullet (:104-112) reproduces to 3 decimals. That covers paired .561 / paired_A .554, same_rand
  bagmean / ot_init / ot, same_fov, same_A, other and other_fov bagmean, and the n16 per-marker pairs (ratios: CD16 .99, CD14 .98,
  CD45 .98, CD3 .94, HLA-DR .80, autofl .61, CD19 .53). The Outcome B and K026 percentages ("~97 %", "50-60 %") are fine.
- `fm_eval.py` changes:
  - Rank quintiles (:122) are never empty for n>=5.
  - LF-hash keys match the trainer's (fm_train.py:38, same W).
  - Saved metadata is JSON-safe (only primitives in cfg).
  - `--newdraw-split` is asserted to be in {1,2} (:76) and recorded (:77).
- `fm_figure.py` bright/dim + channel labels are correct.

## Findings

1. **MAJOR: the "|r| < .09 / ±.3 = noise of few-bag fits / = chance" claim is false as written.**
   Locations: item :110 and :138 (Outcome B), KNOWLEDGE.md:48 (K026, tagged [verified]). REPRODUCED from setmatch_sim_v2.csv.
   - Mean |r| over the 7 targets for the different-cell arms exceeds .09 in most cells:
     - other bagmean: .098/.098/.096/.099 at n 4/16/64/2048.
     - other_fov bagmean: .091 (n1) and .116 (n64).
     - ot: up to .148. ot_init: up to .137.
   - Single-marker |r| reaches .48 (CD16 other_fov ot_init n64 −.483; inbatch −.474).
   - "Few-bag noise" cannot explain the large values. other_fov n=1 has 10,000 bags and still gives CD16 r −.307 newdraw / −.347
     inbatch (the same w scored twice, so this is one draw, not two).
   - Against an approximate random-direction null (SD ≈ r_paired/8 in the whitened 64-PC space), |z|>2 occurs in 15 % (bagmean),
     29 % (ot) and 24 % (ot_init) of rows, against ≈5 % expected.
   - other_fov ot is negative in 33/41 rows. Mean r is −.06 to −.15 at every n≥4 in both test sets.
   - The qualitative conclusion still holds: there is no useful positive recovery, and every mean is ≤.09 against .56 paired.
     The data does not show "~0 = chance".
   - Fix: state the true ranges, add a permutation null (shuffle bag-to-label assignment, several seeds), and drop "noise of
     few-bag fits". OT's excess |r| with unstable sign hints at partial marginal-shape identification; this is REASONED, not
     tested.

2. **MAJOR: K027 and Outcome A misstate when the flow model beats regression.**
   Locations: K027 (KNOWLEDGE.md:49: "needs >= 48 Heun steps + CFG 2-3 (16 steps under-disperse and lose everywhere)"), Outcome
   :133 ("distribution scores better only with >= 48 steps + CFG"). REPRODUCED from n034_sampler_sweep.csv / n034_eval_v2.csv
   macro means. Format below is CRPS, condW1, W1 for flow vs reg.
   - s16 g2 newdraw: flow wins all three (.676/.700/.656 vs .776/.741/.711).
   - s16 g1 newdraw: flow wins CRPS (sweep .765 vs .776; eval_v2 .739 vs .778, numbers already in the item at :79).
   - s48 g1, i.e. no CFG: flow wins all three in both sets (newdraw .677/.713/.673 vs .776/.741/.711; inbatch .353/.278/.232 vs
     .387/.311/.247).
   - What is actually true: 48 steps + CFG 2-3 gives the best distribution scores and r_avgread. Neither condition is required to
     beat regression.

3. **MAJOR: the withdrawn registration claim is still live.**
   Location: notebook/IDEAS.md:25, which is uncommitted and still says "FOV-level groups of <= 16 same cells recover 77-99 % of
   paired r -> ... needs no cell-level registration; add to the supervisor request · open". REPRODUCED (grep). The item withdrew
   this claim at :51-53; IDEAS was not updated. Mark it dead/withdrawn, or restate it per marker (CD19 53 %, autofl 61 %, HLA-DR
   80 % at n16, v2).

4. **MAJOR: the N035 pre-registration contradicts itself on the test cohort.**
   - The title (N035 :1) and INDEX.md:39 say "tested on draw 1 split 2".
   - The Goal (:8) says draw 1 split 2 was "never scored or inspected in N034". That is false: draw 1 split 2 is N034's `inbatch`
     set (fm_eval.py:80 with train_batch 1). It was scored in every N034 eval, and the frozen CFG 2 was chosen on it (N035 :11).
   - The Plan (:17-19) names draw 1 split 1 as PRIMARY.
   - A pre-registration that names two different primaries lets the cohort be picked after the results are in. Fix title, Goal and
     INDEX before any N035 eval. Also record the exact eval commands now (none exist; the results/N035 folder is empty):
     primary `--newdraw-split 1 --steps 48 --cfg 2 --K 8 --per-stain 400 --seed 0`, and secondary a second run with
     `--newdraw-split 2`. Also record the decision rule (macro mean over which 12 labels, what margin counts as "lower").
   - REPRODUCED from the text and the code.

5. **MINOR: draw 1 split 1 is untouched for model and sampler selection, but not for target calibration.**
   - fm_prep.py:61,67 accepts or rejects each slide×channel position gain by whether it lowers split-1 log-integral variance.
     So the 132 decisions behind the test cohort's own image targets were taken on that cohort's fluorescence. It is also N034's
     2048-cell val set; that was only looked at to call convergence, and no checkpoint was selected on it (fm_train.py has no
     best-val selection).
   - Both arms share these targets, so the pre-registered flow-vs-regression comparison is not biased toward either one. It is the
     cleanest cohort available. Call it "fluorescence-assisted, untouched for model/sampler selection", not "untouched".
   - Label QC heads-up, REPRODUCED: draw 1 split 1 CD14 labels have an anomalous within-FOV ICC (v5c .43 against .10 on s0 and .14
     on s2; already present in v3c, .53). A few aberrant FOVs probably dominate the cohort's CD14 numbers. Check them on labels only
     before unblinding.

6. **MAJOR, project-wide and beyond N034: the v4 label position-detrend shows the same in-sample signature as F2.**
   - make_targets_v4.py:20 fits an HGB with 200 iterations × 31 leaves, min 200 cells/leaf, on split 0.
   - After it, within-FOV label ICC (centred per slide) is lower on fitted split-0 FOVs than on held-out FOVs in 12/14
     marker×draw pairs:
     - CD3 b0 .27 vs .45/.46 (s1/s2); CD3 b1 .26 vs .36/.40.
     - CD19 b1 .25 vs .47/.47.
     - autofl b0 .21 vs .41/.36; autofl b1 .15 vs .29/.27.
     - HLA-DR b1 .105 vs .19/.14.
   - Before detrending (v3c) there is no such split pattern (e.g. CD3 b1 .37/.39/.41).
   - So training labels have FOV-level variance removed that held-out labels keep. Every readout fitted on split 0 and scored on
     split 1/2 is affected, including GOAL, N034-A and N035, and so is the other_fov arm of B.
   - The other_fov n=1 regression of a neighbour's label on a cell's own LF gives ≈0 for CD3/CD19 despite ICC .26-.47. That means
     the shared FOV component is not LF-visible composition, most likely an acquisition offset. This is REASONED.
   - Fix: validate the v4 detrend on held-out FOVs the same way F2 was validated (region-blocked or cross-fitted). Log it as its
     own item.
   - REPRODUCED (c3.py).

7. **MINOR: the "derangement" in fm_eval.py:86-87 has an edge-case hang and is not a permutation.**
   - n=1 loops forever: `pr[0] = pr[1 % 1]`. REPRODUCED. Unreachable at the current cohort sizes (smallest stain in any b×s is
     599 cells), but `--per-stain 1` hangs.
   - For n≥2 the loop always ends after one pass with zero fixed points (REPRODUCED over 2000 random permutations, max 1
     iteration). But it overwrites without swapping, so each former fixed point duplicates one LF and drops another: [0,2,1] →
     [2,2,1], and 1941 duplicates appeared across the 2000 permutations. That is harmless for the control (no cell gets its own
     LF); the "derangement" comment is wrong.
   - Fix: swap with a neighbour instead, or use a Sattolo cycle.
   - None of the hardened evaluator code has run yet: no eval.json under runs/n034* or n035 has `newdraw_split`. Smoke-test it
     before N035, since `json.dump` comes before the per_cell/montage saves.

8. **MINOR: smaller residual overstatements in the item.**
   - Next :155 says "calibrated per-cell uncertainty (CRPS)", which contradicts N2. Say "lower CRPS; calibration not achieved".
   - :84 and :122 still say "CD3+/CD14+/CD16+" for cells that are only brightness extrema (N7).
   - :16 still says "ceiling".
   - Outcome :133 "Hypothesis (1) confirmed": only the no-per-cell-gain part holds. The sample mean is below regression (.545 vs
     .612), not "≈". r_sample .491 is far from r_avgread² (.30).
   - The item's "old OT .514/.424/.262/.016/−.035" (:107) is OT on same_rand. fig5's "FOV sets (OT, v1 learner)" line is same_fov
     OT (.531/.445/.238/−.039/−.089), which is the actual v1 design. Pick one.
   - fig5 (~/bsccm_deliverables/figures/fig5_setlevel_matching.png) has x-axis "x", the filename as y-label, a truncated title,
     and no null band.
   - fm_figure panel (c) shows only scores the flow wins, with no rank calibration.
   - K026 is tagged [verified] before this cross-review. The "or varying composition" in its headline is theory, not tested.

9. **MINOR: setmatch robustness.**
   - `halves()` caches on `tuple(rows[:3])` (:34). There is no collision now because targets are disjoint stains, but it is
     fragile.
   - At n≥256 the same-cell arms have 3-24 bags for 65 parameters with lam=10/n. The large-n decline is partly a property of this
     regression setup, not only of information. The item notes "finite bag counts"; this deserves one more clause.

## Summary
- MAJOR: B v2's "|r| < .09 / = chance / ±.3 few-bag noise" (item :110/:138, K026) is false: mean |r| reaches .15, single |r| reaches .48 (10,000-bag n=1 included), dispersion is 2-3× a random-direction null, and other_fov OT is negative in 33/41 rows. K027's "needs 48 steps + CFG; 16 steps lose everywhere" is contradicted by s16-g2 and s48-g1. IDEAS.md:25 still carries the withdrawn "no registration needed" idea.
- MAJOR: N035 names two different primary cohorts (title/INDEX/Goal: draw 1 split 2, wrongly called "never scored"; Plan: split 1). Draw 1 split 1 is selection-clean but its fluorescence chose the gain corrections, so call it fluorescence-assisted. Freeze the exact eval commands and decision rule now.
- MAJOR (project-wide): the v4 HGB label detrend overfits split 0 (within-FOV ICC lower on fitted FOVs in 12/14 marker×draw pairs, e.g. CD19 .25 vs .47), which skews every split-0-fit/split-2-scored readout. The setmatch v2 code itself has no leak or indexing bug; fm_eval's "derangement" hangs at n=1 and duplicates LFs (MINOR).
