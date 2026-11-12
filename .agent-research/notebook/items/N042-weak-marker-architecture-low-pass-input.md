# N042 · Weak-marker architecture: low-pass input probe (image- vs label-limited), then separate BF/DPC, DF-low, DF-high stems with late fusion + private late blocks for CD19/CD56/CD123/HLA-DR as an extra ensemble member

type: exp · date: 2026-11-12 · status: done
goal_effect: flat · scale: module · value: void

## Goal
Module: is the weak-target signal image-limited? If yes, give it a representation that keeps darkfield angular detail
and private capacity (Codex ideas 1+2; literature: resolution is the lever with controlled evidence).

## Plan
Stage 1 probe (falsifier): N031 base recipe on low-pass-filtered inputs (Gaussian sigma 1.5 px, all LEDs), both directions.
Image-limited if >= 2 weak targets drop >= .05 while lineage markers drop < .02; else label/count-limited -> stop stage 2.
Stage 2: new member = separate stems for BF+DPC+phase / DF-low / DF-high (GroupNorm), 32x32 grid kept to late fusion, shared trunk +
private last blocks + heads for CD19/CD56/CD123/HLA-DR; other targets from the reference. Blend weight chosen on training-draw
split-1 FOVs only. Score frozen v5 GOAL (member alone and blended with N031). Expected +.004-.015 if stage 1 says image-limited.
Compute: stage 1 ~.6 GPU-h, stage 2 ~2.5 GPU-h.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Probe result (n042_score.py -> n042_lowpass_per_target.csv; plot .agent-research/plots/n042-lowpass-probe-r_2026-11-12_1836.png): GOAL common-11 (v5) low-pass
  .6411 (b10 .6478, b01 .6344) vs N031 base .6425 (.6488/.6363). Mean drop over both directions: CD19 .011, CD123 .006, CD56 .002,
  HLA-DR -.002; lineage CD14 .001, CD16 -.001, CD45 -.001. Pre-registered rule (>= 2 weak targets drop >= .05, lineage < .02):
  NOT image-limited at the >= 1.5 px scale.
- Stage 1 probe: train_lf.py --lowpass (new opt-in flag; depthwise separable Gaussian on the normalised input, train+eval; CPU unit test: mass 1.0000, peak .07076 vs analytic .07074); jobs 1313154 n042_lp15_b10, 1313155 n042_lp15_b01 (sigma 1.5 px, N031 inductive base recipe); baseline n031_base.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
KILLED AT THE FALSIFIER. Removing detail finer than ~1.5 px (sigma; ~.25 um at .166 um/px) costs the weak targets <= .011 and the GOAL
.0014 (within noise): the weak-marker signal the model uses is not in fine spatial detail, so a detail-preserving angular branch has
nothing to recover (consistent with the N038 diagnosis: CD19/CD56/CD3 are label-limited by FOV-level background error). Stage 2 not run.
Caveat: one blur strength, one seed; a coarser blur would show where the signal does live.
<!-- on close: bin/effect.sh N042 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N042-slug/ + external absolute paths -->

## Next
