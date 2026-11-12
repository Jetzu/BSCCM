# N035 · Frozen confirmation of N034-A in the untouched direction: CFM vs same-net regression trained on blood draw 0, tested on draw 1 (primary split 1, secondary split 2) with the sampler FROZEN at Heun 48 steps + CFG 2 (chosen on N034 in-batch FOVs), K 8, 400 cells/stain; pre-registered r_avgread, CRPS, conditional W1, outside-fraction

type: exp · date: 2026-11-11 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

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
- Training 1303666 n035_cfm (runs/n035/cfm_b0), 1303667 n035_reg (runs/n035/reg_b0): preempt, 1 H100 each, 58 min each, 30 epochs /
  23,940 steps, final EMA val loss cfm .3388, reg .0813 (draw-0 split 1).
- Frozen evals as pre-registered: 1305263 (primary, draw 1 split 1 -> runs/n035/eval_split1), 1305264 (secondary, split 2 -> eval_split2).
  Chains preempted (16/20 groups lost) -> fm_eval.py made resumable per (cohort, stain) group (each group seeds its own generator;
  no change to any computed value), restarted 1311490 -> 1312304 (primary) and 1311491 -> 1312303 (secondary), both COMPLETED 16:45.
  Consistency check of the resume: the shared in-batch cohort (draw 0 split 2, same seed) gives identical means in both evals
  (r_avgread .511, CRPS .362, condW1 .263) although different groups were resumed in each.
- Tables: results/N035-frozen-confirmation-of-n034-a-in-the-unt/n035_per_label.csv, n035_means.csv, n035_verdict.json
  (script n035_summary.py in the same folder). Plots: .agent-research/plots/n035-primary-crps_2026-11-12_1647.png,
  .agent-research/plots/n035-primary-condw1_2026-11-12_1647.png, .agent-research/plots/n035-primary-r_2026-11-12_1647.png (one dot per label;
  the 'noise' band there is a display convention, not a measured rerun sd -- single seed).
  Figure: /home/hpc/v181bb/v181bb10/bsccm_deliverables/figures/fig4b_marker_images_flow_confirmation.png.

PRIMARY (draw 1 split 1, 12 labels, mean)       flow    regression  shuffled-LF  real-image ref
  r with curated label                          .476    .576        .012         .617
  CRPS (lower better)                           .674    .815        -
  conditional W1                                .692    .727        .934
  marginal W1                                   .646    .624        .641
  outside-fraction (ideal .222)                 .526
  sd ratio samples/real readouts                1.25
SECONDARY (draw 1 split 2): r .486 vs .616; CRPS .692 vs .844; condW1 .718 vs .756; outside .515 -- same direction everywhere.
IN-BATCH (draw 0 split 2): r .511 vs .627; CRPS .362 vs .380; condW1 .263 vs .280; outside .376.
Per label (primary): flow CRPS lower on 10/12 (not grp:CD3/CD19/CD56, autofl); condW1 lower on 7/12 (worse on CD123, CD3,
grp:CD3/CD19/CD56, autofl, and 0 margin elsewhere); r below regression on all 12 (largest gaps CD123 .08 vs .37, CD3 .34 vs .56,
CD19 .10 vs .24, grp:CD45 .51 vs .68).
Caveats: HLA-DR real-image reference is -.11 on draw 1 (the readout fitted on real draw-0 images does not transfer to real draw-1
HLA-DR images, while both models give .42) -> HLA-DR stain/label relation differs between draws; CD14 FOV clustering on this split
(check3 F5) as disclosed. On the NEW draw the frozen sampler (CFG 2) OVER-disperses (sd ratio 1.25, in-batch .92): CFG 2 was chosen
in-batch and does not transfer -- consistent with outside .53.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
Pre-registered verdicts (primary, applied exactly as frozen):
- 'flow is distributionally better': CONFIRMED (CRPS .674 < .815 AND condW1 .692 < .727; condW1 margin small, 7/12 labels).
- 'flow is not better per cell': CONFIRMED (r_avgread .476 <= r_reg .576 + .01; gap .10, at the edge of the predicted .03-.10).
- calibration: NOT achieved (outside .526 > .272); new-draw samples over-dispersed (sd ratio 1.25).
So N034-A replicates in the untouched direction: flow samples give a better per-cell predictive DISTRIBUTION (mostly through
spread) but a worse per-cell point estimate than the same net trained by regression, and are not calibrated. No goal metric touched.

POST-HOC (after the verdicts; not part of the pre-registration): results/N035-frozen-confirmation-of-n034-a-in-the-unt/n035_posthoc_hladr.py
-> n035_posthoc_label_vs_integrals.json. Ridge of the curated label on the 6x4 log fluorescence integrals: HLA-DR within-draw FOV-CV r
.88 (draw 0) / .65 (draw 1) but the draw-0 map transfers to draw 1 at .11 (every other single marker transfers at .42-.96, CD16 .96,
CD45 .80, CD14 .73). So the draw-1 HLA-DR labels are coherent (also predicted from LF at .45 by the GOAL ensemble) but the spectral
image-to-label relation differs between draws (dimmer HLA-DR relative to autofluorescence in draw 1) -> the cross-draw image readout
yardstick is invalid for HLA-DR; HLA-DR image-readout numbers in N034/N035 should be read as uninformative. GOAL labels are per-draw
unmixed by the dataset authors and unaffected.
<!-- on close: bin/effect.sh N035 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N035-slug/ + external absolute paths -->

## Next
