# N035 · Frozen confirmation of N034-A in the untouched direction: CFM vs same-net regression trained on blood draw 0, tested on draw 1 (primary split 1, secondary split 2) with the sampler FROZEN at Heun 48 steps + CFG 2 (chosen on N034 in-batch FOVs), K 8, 400 cells/stain; pre-registered r_avgread, CRPS, conditional W1, outside-fraction

type: exp · date: 2026-11-11 · status: open
goal_effect: pending · scale: ?

## Goal
Untouched test of the N034-A conclusions (rereview N1: N034's sampler was selected while looking at its new-draw scores, so those are
development results). Opposite direction: train blood draw 0 (split 0), test blood draw 1 (see Plan: primary split 1 = never scored in N034 and not used for
model/sampler selection, but FLUORESCENCE-ASSISTED -- its fluorescence decided the 72/132 image-gain acceptances and it was N034's
val-loss set; split 2 was N034's in-batch set on which CFG 2 was chosen -> secondary only). Check3 F5: draw-1 split-1 CD14 labels
cluster unusually within FOVs (FOV-mean share .38 vs .22-.24 on splits 0/2) -> report per-marker values and CD14 with that caveat. Everything frozen before any result: fm_prep.npz (same sha as N034; a_c from draw-1 split-0 border noise =
camera noise constant, values identical to draw-0, disclosed), targets_v5c, CondUNet width 64 patch 2, 30 epochs both modes, seed 0,
sampler Heun 48 steps + CFG 2.0 (best N034 in-batch CRPS/condW1), K 8, 400 cells/stain, eval seed 0.

## Plan
Pre-registered predictions (from N034): flow r_avgread < regression r_reg by .03-.10 (subtle markers CD3/HLA-DR worse); flow CRPS and
conditional W1 lower than regression; outside-fraction above the exchangeable 2/9 = .22 (calibration NOT achieved). In-batch (draw 0
split 2) reported alongside. Any deviation is reported as is; no re-selection of the sampler.
Test cohort: PRIMARY = draw 1 split 1 (validation FOVs: never scored or inspected in N034; used only in aggregate for the gain
acceptance and as the N034 trainer's 2048-cell val loss); SECONDARY = draw 1 split 2 (it was N034's in-batch set, on which the sampler
was chosen with the N034 model) and draw 0 split 2 (N034's development new-draw set, here in-batch).
Frozen commands (after both trainings complete; R: = /hnvme/workspace/v181bb10-bsccm/runs/n035/, ckpt = ckpt.pt):
  primary  : fm_eval.py --cfm R:cfm_b0/ckpt --reg R:reg_b0/ckpt --out R:eval_split1 --per-stain 400 --K 8 --steps 48 --cfg 2.0 --seed 0 --newdraw-split 1
  secondary: same with --newdraw-split 2 --out R:eval_split2
Decision rule (primary, mean over the 12 labels with a valid readout): the N034 conclusion 'flow is distributionally better' is
CONFIRMED if flow CRPS < regression CRPS AND flow conditional W1 < regression conditional W1; 'flow is not better per cell' is
CONFIRMED if r_avgread <= r_reg + .01. Calibration counts as achieved only if outside-fraction <= 2/9 + .05. Each verdict reported
separately; no re-selection, no extra arms on these cohorts.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Training 1303666 n035_cfm (runs/n035/cfm_b0), 1303667 n035_reg (runs/n035/reg_b0): preempt, 1 H100 each, ~1 h each (102k cells x 30 ep).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N035 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N035-slug/ + external absolute paths -->

## Next
