# N007 · SSL pretraining of the LF encoder: cross-illumination views vs augmentation-only (SIGReg/JEPA-style), linear-probe eval

type: exp · date: 2026-11-08 · status: open
goal_effect: pending · scale: paradigm

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Codex writer (results/N007-ssl-pretraining-of-the-lf-encoder-cross/codex/): bsccm_ssl.py (views, SIGReg Epps-Pulley, ssl_loss), train_ssl_lf.py, probe_lf.py, tests. Claude review: loss/SIGReg/resume logic correct; data path CPU-bound (CPU LFNorm + per-view CPU warps, prefetch 1) -> integrated with GPU path (bsccm/gpu_norm.py RawSet+GPUNorm, bsccm/ssl.py make_views_batch). Tests: 8/8 Codex unit tests + make_views_batch test pass (bsccm.sif CPU).
- GPU loader: GPUNorm == LFNorm max |diff| 1.9e-6 (32 cells incl. FOV-edge); worker cost 3.6 -> 0.13 ms/cell; throughput ~1.4k -> ~2.7k cells/s (cold Lustre reads, job 1275478).
- Codex red-team of Claude code (results/N007-ssl-pretraining-of-the-lf-encoder-cross/codex_redteam/REDTEAM.md): 1 critical + 4 major + 2 minor, ALL reproduced first-hand and fixed:
  F1 retrieval ties counted as hits (collapsed embeddings R@1 = 1.0) -> pessimistic tie rank; F2 MK near-dup ignore used set INTERSECTION (pairs sharing only autofl ignored) -> identical 12-marker set + within tau; F3 train_lf resume without config/row check -> validated; F4 target calibration fitted on all rows incl. eval -> targets_v3 (fit on split-0 rows per batch; v2 vs v3 per-target corr >= 0.992); F5 screen SigLIP sampled with replacement (110 dup in 300) -> randperm; F6 bind standardisation on all rows -> training rows; F7 screen SigLIP maps crashed (grad) -> no_grad.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N007 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N007-slug/ + external absolute paths -->

## Next
