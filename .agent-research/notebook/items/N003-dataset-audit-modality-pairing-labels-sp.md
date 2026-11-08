# N003 · Dataset audit: modality pairing, labels, splits, Giemsa-LF match quality, batch structure

type: analysis · date: 2026-11-08 · status: open
goal_effect: pending · scale: ?

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Giemsa-LF pairing (scripts/audit_match.py, job 1274223, results/N003-*/giemsa_lf_match.csv, _summary.json, montage_giemsa_lf.png):
  Spearman(area_giemsa, area_dpc-phase) = 0.74 vs 0.00 for shuffled pairs; area_bf 0.26; DF_70 mean 0.54. Montage: shapes/lobes correspond.
- Mismatch-rate estimate (log-area residual, robust sd 0.157, n = 15,565): 2.70 % beyond 3 sd vs 32.6 % for shuffled pairs and ~0.3 % expected
  for Gaussian residuals -> crude mismatch fraction f from 0.003(1-f) + 0.326 f = 0.027 => f ~ 7 % (upper-ish; heavy tails also count).
  Paper (strand E report): pairs come from slide alignment + nearest-cell matching, not pixel registration -> treat G-L as noisy correspondence.
- 'dpc' array looks like a quantitative-phase reconstruction (smooth positive blob, range -1.6..4.1); confirmation pending (paper/code agent).
- Giemsa (Wright) images are soft/low-contrast relative to clinical CellaVision smears (montage) -> domain shift for transfer.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N003 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N003-slug/ + external absolute paths -->

## Next
