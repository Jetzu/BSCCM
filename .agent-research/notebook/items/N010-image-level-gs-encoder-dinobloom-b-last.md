# N010 · Image-level GS encoder (DinoBloom-B, last 4 blocks + heads) with stain/optics augmentation, SigLIP to frozen LF + distillation; cross-slide test

type: exp · date: 2026-11-08 · status: open
goal_effect: pending · scale: module

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Cross-slide (train b0 slide, test b1 slide 5,697), 2,500 steps, sup-LF teacher: aug only autofl-truth r .284 R@10 .041 medr 615; +Macenko .287 / .038 / 522; Macenko+DANN(.1) .304 / .032 / 587; DANN only .308 / .033 / 662 -> all within noise; stain normalisation and adversarial slide invariance do NOT fix the cross-slide gap. Macenko check: slide identity from colour stats .883 -> .707 (results/N011-*/macenko_check.png).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N010 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N010-slug/ + external absolute paths -->

## Next
