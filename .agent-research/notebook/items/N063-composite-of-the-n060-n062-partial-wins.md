# N063 · Composite of the N060-N062 partial wins, fixed a priori by head design and scored on fresh seeds 2-3: pooled groups from the panel head, all singles from a twopart + per-FOV-offset model; vs fresh r50 controls; also as Cytoland-ensemble member

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: module · value: 0.6902

## Goal
N060-N062 gains are complementary and match each head's design (IDEAS.md 2026-11-18): twopart CD123 +.037 / grp:CD45 +.019, panel
head pooled groups +.014-.031, fov-re CD19 +.029 (1 seed). Test whether combining them by DESIGN (not by per-target test results) lifts
the GOAL.

## Plan
PRE-REGISTERED composite (fixed before seeds 2-3 are seen): pooled targets (grp:CD3/CD19/CD56, grp:CD45, grp:CD16) from a
--panel-head model; all single targets (+ autofl) from a --head twopart --fov-re 1.0 --w-autofl 0 model (one model, both options).
Fresh seeds 2 and 3 for both models and for an r50 control (no reuse of seeds 0-1 that revealed the per-target pattern).
Primary: composite GOAL mean over seeds 2-3 vs control mean over seeds 2-3; success if > control + .0054. Secondary: composite (seed-averaged) added as a 4th member to the
N053 ensemble (N048 base + zoom2 + Cytoland); bar .6862 + .0054 = .6916. Expected +.006-.010 single. Kill: <= control + .0027.
Placement GPU 1 x H100; 3 models x 2 seeds x 2 directions = 12 runs, ~3.6 GPU-h.

## Changes

## Evidence
- Runs 1383361-72 (jobs.txt): tpfov / panel / ctrl x seeds 2, 3 x {b10, b01}; n063_score.py -> n063_scores.csv, n063_per_target.csv.
- PRIMARY (pre-registered): composite .6794 / .6774 vs fresh r50 control .6690 / .6665 -> +.0106 (seed means .6784 vs .6678) > .0054
  -> PASSED at the single-model level, both seeds. Components: twopart+fov-re model .6766, panel model .6712.
- Per target vs fresh control (replicates the seed-0/1 pattern): CD123 +.039, grp:CD45 +.031, grp:CD3/19/56 +.018, grp:CD16 +.013,
  CD56 +.011, CD3 +.008, CD19 +.006; CD45 -.005, HLA-DR -.006.
- SECONDARY: composite (seed-averaged) as 4th member of the N053 ensemble = .6902 (fresh control in the same slot .6855) -> +.0040
  over the GOAL best .6862, below the .6916 bar.
- Plot: plots/goal-v10-n063-composite_2026-11-18_1718.png (plot_n063.csv).

## Outcome
GOAL-level FLAT (.6902 < .6916) but the pre-registered single-model primary passed (+.0106 over fresh controls, both seeds): the
composite recipe is the new best SINGLE-model recipe (.6784 vs r50 .6678-.6730). Next: rebuild every ensemble member on this
recipe (zoom2 view, Cytoland encoder) -> N065.

## Outputs
<!-- files in .agent-research/results/N063-slug/ + external absolute paths -->

## Next
