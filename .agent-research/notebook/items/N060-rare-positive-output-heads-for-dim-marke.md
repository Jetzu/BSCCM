# N060 · Rare-positive output heads for dim markers: HL-Gauss distributional head and two-part gate x amplitude head on the GOAL r50 recipe, 2 seeds, CD123 seed-bagged

type: exp · date: 2026-11-18 · status: open
goal_effect: pending · scale: module

## Goal
N059 review: weak markers are few-shot rare-positive problems (CD123 top-1 % cells = .88-.92 of test variance; HLA-DR, CD19, CD56 top-2 % = .45-.65). An MSE point head shared by every member (errors r .94-.96) may be the common limit.

## Plan
Writer Codex (results/N060-*/codex/brief.md, one brief for N060-N062), Claude red-team + integration into scripts/train_lf.py.
GOAL recipe otherwise unchanged (SSL r50, base view, v10 labels, 10 epochs). 2 seeds x 2 directions per arm (per-item seed noise up
to .124 on CD123, N059 AB s8). PRE-REGISTERED bar: arm single-model GOAL (mean of 2 seeds) > r50 base .6730 + .0054, OR the arm as a
3rd member of N048 (base+zoom2) > .6862 + .0054 (= .6916). Report CD123-excluded GOAL alongside (CD123 carried 59 % of N053's gain).
Placement GPU 1 x H100 (r50 fine-tune).
Arms: --head hlgauss (HL-Gauss histogram head, all targets, expectation readout + .1 MSE); --head twopart (gate x amplitude for
CD123 / HLA-DR / CD19 / CD56, gate threshold = training-label q.98). Expected +.003-.008 (outsider, Codex). Kill: both arms <= base.

## Changes

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N060 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N060-slug/ + external absolute paths -->

## Next
