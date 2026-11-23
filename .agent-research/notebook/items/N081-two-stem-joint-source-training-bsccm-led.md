# N081 · Two-stem joint-source training: BSCCM LED stem + S-BIAD452 BF/SSC stem, shared trunk + shared marker heads with per-dataset adapters; BSCCM GOAL/MM3 + IFC held-out-donor joint metrics

type: exp · date: 2026-11-23 · status: running
goal_effect: pending · scale: module

## Goal
Review N079 idea #2 (Codex 1, lit B): add NEW joint observations to BSCCM training -- S-BIAD452 cells with all 9 markers (5 overlap with
BSCCM single targets: CD14, CD19, CD3, CD45, CD56) through a second input stem on the shared trunk.

## Plan
Codex writes the harness extension (--ext-ifc/--ext-w/--ext-test-donor; IFC conv1 stem -> shared r50 trunk; shared reg rows + per-dataset
affine for the 5 overlapping markers; IFC-only head for CD15/Siglec8/CD4/CD8; one IFC batch per BSCCM step; BSCCM eval unchanged; extra
IFC held-out-donor eval). Claude red-team + flag-off equality suite before GPU. Screen: GOAL r50 recipe + --ext-ifc, ext-w .5, seed 0,
both directions vs control n051_v10_base seed 0. PRIMARY (pre-registered): GOAL delta > +.0054 OR CD19/CD56/CD3/CD14/CD45 mean delta >=
.02 with GOAL >= -.0027; MM3 reported. ~2 GPU-h.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N081 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N081-slug/ + external absolute paths -->

## Next
