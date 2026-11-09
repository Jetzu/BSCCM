# N014 · Teacher v4 portfolio: resnet50 x15 ep, SSL(aug)-init finetune, DANN batch-invariance; batch-held-out GOAL

type: exp · date: 2026-11-09 · status: done
goal_effect: improved · scale: module · value: 0.5742

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Batch-held-out mean r (12 markers), targets_v4, jobs 1282456-61 (.agent-research/results/N014-teacher-v4-portfolio-resnet50-x15-ep-ssl/n014_goal.csv, n014_final_evals.json; plot .agent-research/plots/batch-holdout-lf-mk-r-mean-v4_2026-11-09_1228.png): r18 baseline .5545 | r50 15 ep .568/.563 = .5654 | SSL(aug)-init r50 finetune 10 ep (lr 5e-4) .571/.577 = .5742 | DANN .1 r18 .559/.550 = .5541 (flat).
- Gain of SSL-init concentrated in CD123 (.40->.52 / .29->.46), CD19 (+.05), CD56 (+.05 on b0to1); CD16/CD14/CD45 unchanged (already ~.85-.90).
- => self-supervised pretraining helps once FINE-TUNED (its linear probe was weak, N007); adversarial batch invariance does not.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N014 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N014-slug/ + external absolute paths -->

## Next
