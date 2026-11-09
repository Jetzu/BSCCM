# N010 · Image-level GS encoder (DinoBloom-B, last 4 blocks + heads) with stain/optics augmentation, SigLIP to frozen LF + distillation; cross-slide test

type: exp · date: 2026-11-08 · status: done
goal_effect: flat · scale: module · value: 0.5545

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Cross-slide (train b0 slide, test b1 slide 5,697), 2,500 steps, sup-LF teacher: aug only autofl-truth r .284 R@10 .041 medr 615; +Macenko .287 / .038 / 522; Macenko+DANN(.1) .304 / .032 / 587; DANN only .308 / .033 / 662 -> all within noise; stain normalisation and adversarial slide invariance do NOT fix the cross-slide gap. Macenko check: slide identity from colour stats .883 -> .707 (results/N011-*/macenko_check.png).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- v4 LABELS + v4 teacher (LF sup v4 fov encoder, probe r .634; runs/n013_gsbind_v4_*): cross-slide b0->b1 (5,697 test GS cells): student vs DETRENDED autofl truth r .648 (aug) / .624 (Macenko) vs LF teacher .559 on the same cells; student-vs-teacher mean r (12 markers) .67-.69 (v3: .54); retrieval R@10 .042 medr ~450-500. Within-batch FOV run (pooled slides): autofl .26 / teacher .10 (pooled-slide evaluation of per-slide-detrended target; use per-slide numbers).
- The student beats its teacher on truth (.65 vs .56): regression of the teacher onto GS averages out teacher errors uncorrelated with morphology.
- Caveat (declared): the per-slide detrending trend of the GS slides' autofl was fitted on that slide's split-0 labelled cells (target-side cleaning; predictors never see labels).
- Replication (jobs 1282372-74): b0->b1 seed1 student .658 (teacher .559); b1->b0 seeds 0/1 student .323/.318 vs teacher .305 (test 10,015 GS cells); student-vs-teacher markers .53 (b1->b0, trained on 5.7k cells) vs .68 (b0->b1, 10k). Direct GS ridge (frozen DinoBloom) on detrended autofl cross-slide: .55 (b0->b1) / .47 (b1->b0) [N012].
- Plot: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/plots/gs-student-autofl-truth-r_2026-11-09_1122.png
- => the GS student matches or beats its LF teacher in BOTH directions; the transitive step is no longer the bottleneck -- the teacher is (LF->detrended autofl on the b0 GS slide only .31). Next lever: teacher quality (esp. batch 0).

## Outcome
<!-- on close: bin/effect.sh N010 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N010-slug/ + external absolute paths -->

## Next
