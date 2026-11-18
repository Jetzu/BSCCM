# N062 · Panel-matrix latent head: shared molecular factors for single and pooled assays with assay-specific decoders and per-slide adapters for pooled training slides, GOAL r50 recipe, 2 seeds

type: exp · date: 2026-11-18 · status: open
goal_effect: pending · scale: paradigm

## Goal
N059 review (paradigm): the 8 single markers and 4 pooled groups are modelled as independent outputs; pooled cells (10.2k draw 0, 27.1k draw 1) never supervise the dim single markers, and slide-specific pooled readouts are averaged in.

## Plan
Writer Codex (results/N060-*/codex/brief.md, one brief for N060-N062), Claude red-team + integration into scripts/train_lf.py.
GOAL recipe otherwise unchanged (SSL r50, base view, v10 labels, 10 epochs). 2 seeds x 2 directions per arm (per-item seed noise up
to .124 on CD123, N059 AB s8). PRE-REGISTERED bar: arm single-model GOAL (mean of 2 seeds) > r50 base .6730 + .0054, OR the arm as a
3rd member of N048 (base+zoom2) > .6862 + .0054 (= .6916). Report CD123-excluded GOAL alongside (CD123 carried 59 % of N053's gain).
Placement GPU 1 x H100 (r50 fine-tune).
Arm: --panel-head (8 latent molecular factors; singles = monotone links; pooled = non-negative W z with W initialised from the panel
design, learnable; per-training-slide affine adapters on pooled outputs in the training loss only). No lineage/exclusivity priors.
Expected +.003-.01 if dim markers are label-starved (K033: +.035 on weak targets 50->100 % labels), ~0 otherwise. Kill: <= base.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N062 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N062-slug/ + external absolute paths -->

## Next
