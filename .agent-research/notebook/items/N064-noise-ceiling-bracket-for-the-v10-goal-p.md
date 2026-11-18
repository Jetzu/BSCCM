# N064 · Noise-ceiling bracket for the v10 GOAL: pixel split-half reliability of each marker label re-derived from raw fluorescence (checkerboard halves through the label pipeline, Spearman-Brown), unstained-cell noise floor, FOV-level residual variance -> per-marker r_max

type: analysis · date: 2026-11-18 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal
N059 review: decide whether .70 is reachable by bracketing the label noise ceiling per marker (lit_noisy_ceiling idea 1, outsider M2).

## Plan
Writer Codex (codex/: n064_ceiling.py, n064_cpu.sbatch, report.md with pre-registration written before computing). Reads:
mean r_max <= .72 -> .70 near-ceiling; CD19/CD56 r_max >= .75 -> pixel-noise "label-limited" refuted. Upper-bound caveat: pixel
split-half sees photon/pixel noise only.

## Changes

## Evidence
- Smoke (2,000 cells; codex/smoke.log, validation.log): pixel aggregates reproduce the authors' disk/corner geometry (<= 5.7e-5
  counts), unmixing fidelity r >= .998, v10 curation replay error < 2e-7, ensemble rescore .686184 (= GOAL best). Pixel split-half
  r_max ~ .9999 for every target (CD19/CD56 .9994-.9999) -> PHOTON/PIXEL NOISE IS NEGLIGIBLE. But labels rebuilt with LOCAL corner
  backgrounds agree poorly with v10 for dim markers: CD19 r .36/.36, CD123 .52/.57, CD56 .44/.77, HLA-DR b01 .47 (CD16/CD45/CD3
  >= .96) -> the dim-marker labels are dominated by the BACKGROUND MODEL (authors' smoothed background_lowess vs local), not by
  photon noise; consistent with K028. FOV residual variance share .325 (null .175). Claude check: light (validation numbers +
  conclusion robust to small bugs: r_max cannot fall from .9999 to < .9 by pipeline details).
- Full run 1383477 (preempt_cpu, 48 cores, 69,576 cells) -> results/N064-*/full/ (reliability.csv, reconstruction_fidelity.csv,
  decision.json). Pixel split-half mean r_max .99987 (all 11 targets, both directions, eval + matched training cells): pre-registered
  read "CD19/CD56 r_max >= .75" -> pixel-noise version of "label-limited" REFUTED; "near ceiling (<= .72)" not supported for pixel
  noise. Background-model dependence (r of local-background labels vs v10): CD19 .24/.40 eval (.14/.20 train), CD56 .76/.67,
  CD123 .92/.87, HLA-DR .76 (b01 eval, draw 1) vs .98 (b10), grp:CD45 .68 (b01 eval) vs .95, CD16/CD3/CD45/CD14 >= .87-.999.
  The draw-specific deficits found in N059 (HLA-DR and grp:CD45 on draw 1) are exactly where the label depends most on the
  background model in that draw. pixel_chunks deleted after COMPLETE (563 files, /hnvme inode cap).

## Outcome
Measurement (n/a). Label error for dim markers is NOT photon/pixel noise (r_max ~ 1) but background-estimation error, which is
largest exactly on the weak and draw-shifted targets (CD19, CD56, HLA-DR/grp:CD45 in draw 1). Its size relative to the true
signal is not identifiable without a reference measurement, so no numeric ceiling for the v10 labels; the evidence points to the
label background model as the binding limit, out of reach for any LF model.

## Outputs
<!-- files in .agent-research/results/N064-slug/ + external absolute paths -->

## Next
