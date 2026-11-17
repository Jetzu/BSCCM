# N043 · Paper figures in the house style (pfig): fig1 architecture, fig2 markers in a new draw, fig3 label limits, fig4 Giemsa + clinical, fig5 flow-matching marker images, fig6 set-level matching

type: task · date: 2026-11-17 · status: done
goal_effect: n/a · scale: ? · value: - · why: infra

## Goal
User request 2026-11-17: comprehensive BSCCM figures with the new plotting skill (paper-figures / pfig: Letter width, 4K PNG + SVG, Arial, palette d, hollow rings, stars, captions in .json).

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- scripts/make_figures_pfig.py -> figures/paper/fig{1..6}_*.{png,svg,json} (copied to ~/bsccm_deliverables/figures/paper_pfig/); every
  value read from saved predictions / result files (sources + test details in each .json); all six passed pfig.save checks and were
  inspected visually. Stats: two-sided, Holm per panel; paired Wilcoxon over the 22 marker x direction items or 12 labels where
  matched; Welch t for run-level groups.
- fig1 corrected vs the old fig1: the Giemsa-chain teachers are ResNet-18 (n029_r18_v4c_fov markers, n029_r18_v6c_fov autofluor.),
  not the ResNet-50 GOAL predictor; parameter counts read from checkpoints (r50 enc 23.65 M, r18 enc 11.24 M, DinoBloom-B ViT 85.7 M
  with 8 of 12 blocks frozen, UNet 1.96 M, CondUNet 27.7 M).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
DONE: six paper figures in the house style (key tests in the .json sidecars).
<!-- on close: bin/effect.sh N043 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N043-slug/ + external absolute paths -->

## Next
