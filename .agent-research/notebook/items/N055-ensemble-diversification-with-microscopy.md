# N055 · Ensemble diversification with microscopy-pretrained members: Cytoland VSCyto2D zoom-2 view and Cytoland on the cofactor-150 labels added to the N048 + Cytoland ensemble (pre-registered 4-member primary)

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: module · value: 0.6902

## Goal
Push the v10 GOAL (best .6862, N053; target .70) with more diverse, accurate ensemble members: the only member that decorrelated
errors while staying accurate was the microscopy-pretrained Cytoland ConvNeXtV2 (resid r .91, K036).

## Plan
Hypothesis: (a) a zoom-2 Cytoland view adds the scale complementarity the r50 family gets from zoom2 (K023: +.002-.003) on a
decorrelated backbone; (b) Cytoland trained on the cofactor-150 labels (the N048 label set, which scored best when re-scored on
v10) is a better member than the v10-trained one. Runs (enc-lr-mult .1, 10 epochs, seed 0, both directions): cyto_zoom2 (v10),
cyto_cf150 (base view). PRE-REGISTERED primary: E4 = N048 base + N048 zoom2 + Cytoland(v10, s0) + Cytoland zoom2 (v10);
success = E4 > .6862 + 2 x .0027 = .6916. Secondary (descriptive): every 3-/4-/5-member combination incl. cyto_cf150 and
Cytoland s1. Baseline .6862; same-size control = E4 with Cytoland s1 instead of zoom2 (.6880 measured). Kill: E4 <= .6880.
Placement: GPU 1 x H100 (ConvNeXtV2 fine-tune; GPU >> 2x CPU). Budget 4 runs x ~0.9 h = ~3.6 GPU-h.

## Changes

## Evidence
- Runs 1381382-85 (jobs.txt): Cytoland zoom2 (v10) single .6646; Cytoland on cf150 labels single .6704 (best single Cytoland;
  v10-trained s0 .6671). Scores n055_scores.csv (n053_score.py).
- PRE-REGISTERED E4 = N048 base + zoom2 + Cytoland s0 + Cytoland zoom2 = .6902 (b10 .7019 / b01 .6784) vs success bar .6916 ->
  NOT met (+.0040 over .6862 < 2 x noise); above the kill line .6880 (same-size control + Cytoland s1).
  Secondary (post hoc): N048 + Cytoland s0 + s1 + zoom2 = .6912; all six = .6906; N048 + cf150 Cytoland .6864 (= v10 member).
  Plot plots/goal-v10-n055_2026-11-18_1352.png (plot_n055.csv).

## Outcome
FLAT: more Cytoland members add +.002-.005 each step (scale complementarity of zoom2 ~ +.002 over a same-size seed member) but
the pre-registered 4-member ensemble stays below the bar (.6902 < .6916). Ensembling is saturating near .69; reaching .70 needs a
different lever (labels / draw-specific readout, K028-K030).

## Outputs
<!-- files in .agent-research/results/N055-slug/ + external absolute paths -->

## Next
