# N061 · Within-FOV pairwise-difference loss (shared FOV label offsets cancel) + co-stained autofluorescence supervision removed from the shared encoder, GOAL r50 recipe, 2 seeds

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: module · value: 0.6742

## Goal
N059 review: FOV-level label error still costs ~.04 (within-FOV .726) and slide-specific pooled-panel readouts (grp:CD45 dim rep0 slide) are averaged into the shared head; co-stained autofl supervision may steer marker features (draw-dependent autofl slope, K029).

## Plan
Writer Codex (results/N060-rare-positive-output-heads-for-dim-marke/codex/brief.md, one brief for N060-N062), Claude red-team + integration into scripts/train_lf.py.
GOAL recipe otherwise unchanged (SSL r50, base view, v10 labels, 10 epochs). 2 seeds x 2 directions per arm (per-item seed noise up
to .124 on CD123, N059 AB s8). PRE-REGISTERED bar: arm single-model GOAL (mean of 2 seeds) > r50 base .6730 + .0054, OR the arm as a
3rd member of N048 (base+zoom2) > .6862 + .0054 (= .6916). Report CD123-excluded GOAL alongside (CD123 carried 59 % of N053's gain).
Placement GPU 1 x H100 (r50 fine-tune).
Arms: (i) --fov-re 1 --w-autofl 0 (mixed-effects crowd layer: per-FOV offsets + per slide x class affine in the TRAINING loss
only; lit_crossbatch idea 1); (ii) --pair-loss + --w-autofl 0 (within-FOV pair differences; Codex idea 1); (iii) --w-autofl 0 alone
(ablation; Codex idea 3). Expected +.003-.007. Kill: no arm > base by .0054 and CD19/CD56 not up >= .01.

## Changes

## Evidence
- Code: Codex (results/N060-rare-positive-output-heads-for-dim-marke/codex/: PATCH.diff, bsccm_heads.py, CPU smoke 12/12). Claude red-team (scratchpad/n060rt/): P0 import
  fails under python3 -I (REPRODUCED: ModuleNotFoundError) -> module moved to bsccm/heads.py; P1-1 fov-re prior averaged over ~38k-FOV
  table = no shrinkage + slide x class affine can absorb class means (confirmed by reading) -> FOV offsets only, MAP prior
  lambda * sum b^2 / N_train (lambda = --fov-re 1.0); P2 two-part gate init logit(.02), gate loss over the 4 dim markers only;
  P1-2 seed-1 r50 control + sampler-only control added. Claude also added --pearson-w (lit_noisy_ceiling). evaluate() unchanged.
- Runs 1382631-64 (jobs.txt in N060 results): 9 arms x 2 seeds x 2 directions (hlgauss, twopart, pearson | fovre, pair, sampler,
  autofl0 | panel | r50 control seed 1), runs/<item>_<arm>_s<seed>_<b10|b01>. ~10 GPU-h.

- RESULTS (n060_score.py -> results/N060-rare-positive-output-heads-for-dim-marke/n060_scores.csv, n060_per_target.csv; 2 seeds x 2 directions each; control = r50 SSL
  seed 0/1 .6730/.6683, mean .6706). Single-model mean GOAL: twopart .6756, pair .6742, fovre .6740, pearson .6726, autofl0 .6713,
  panel .6704, sampler .6640, hlgauss .6591. As 3rd member with N048 base+zoom2: .6797-.6851 (all < .6862 best, bar .6916).
  Per target vs control (mean over seeds/directions): twopart CD123 +.037, grp:CD45 +.019; panel pooled groups +.014/+.015/+.031,
  singles -.01..-.02; fovre CD19 +.029 (seed 0); pair CD56 +.019, grp:CD45 +.017; hlgauss CD123 -.054, CD19 -.063.
  Plots: plots/goal-v10-n060-heads_2026-11-18_1704.png, goal-v10-n061-n062_2026-11-18_1704.png.

## Outcome
FLAT: pair .6742, fovre .6740, autofl0 .6713 (control .6706); the FOV sampler alone costs -.007. FOV-nuisance training helps CD19/CD56/grp:CD45 a little; label background error is not removable in training (K041).

## Outputs
<!-- files in .agent-research/results/N061-slug/ + external absolute paths -->

## Next
