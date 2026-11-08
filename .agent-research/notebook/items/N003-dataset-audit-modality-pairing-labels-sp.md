# N003 · Dataset audit: modality pairing, labels, splits, Giemsa-LF match quality, batch structure

type: analysis · date: 2026-11-08 · status: open
goal_effect: pending · scale: ?

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Giemsa-LF pairing (scripts/audit_match.py, job 1274223, results/N003-dataset-audit-modality-pairing-labels-sp/giemsa_lf_match.csv, _summary.json, montage_giemsa_lf.png):
  Spearman(area_giemsa, area_dpc-phase) = 0.74 vs 0.00 for shuffled pairs; area_bf 0.26; DF_70 mean 0.54. Montage: shapes/lobes correspond.
- Mismatch-rate estimate (log-area residual, robust sd 0.157, n = 15,565): 2.70 % beyond 3 sd vs 32.6 % for shuffled pairs and ~0.3 % expected
  for Gaussian residuals -> crude mismatch fraction f from 0.003(1-f) + 0.326 f = 0.027 => f ~ 7 % (upper-ish; heavy tails also count).
  Paper (strand E report): pairs come from slide alignment + nearest-cell matching, not pixel registration -> treat G-L as noisy correspondence.
- 'dpc' array looks like a quantitative-phase reconstruction (smooth positive blob, range -1.6..4.1); confirmation pending (paper/code agent).
- Giemsa (Wright) images are soft/low-contrast relative to clinical CellaVision smears (montage) -> domain shift for transfer.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- Own read of the paper (arXiv 2402.06191v1 PDF, 34 pp, 2026-11-08), additions beyond the agent report:
  * Each batch = ONE 12 mL venipuncture split into antibody tubes -> all conditions within a batch share donor + isolation (within-batch population comparisons valid; batch 0 GS cells vs batch-0 'all' cells = same blood).
  * Slides stored up to 2 weeks before imaging, ~16 h per slide scan; each condition = own slide/date -> condition x storage-time confound.
  * Lymphocytes partly apoptotic ("dumbbell" shapes) and preferentially EXCLUDED by the QC CNN -> lymphocyte composition biased.
  * Histology autofocus weaker (focal stacks on a subset of positions, running-average interpolation) -> variable GS sharpness: add a GS focus score (gradient sum) as QC covariate/filter.
  * Two-spectrum NNMF has L1 shrinkage (lambda 0.7); four-spectrum model ignores autofluorescence; CD16/CD45/CD123 leak into CD3/CD19/CD56 group; background B per slide.
  * Download completeness: BSCCM full (50/50 chunks, sha256) + tiny. Not downloaded: BSCCMNIST (derived 28x28), BSCCM-coherent (4,304 cells, 566 single-LED, all-antibody/none only; separate subset, no histology), README (7 KB) -> nothing essential missing. User 2026-11-08: no additional data.

## Outcome
<!-- on close: bin/effect.sh N003 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N003-slug/ + external absolute paths -->

## Next
