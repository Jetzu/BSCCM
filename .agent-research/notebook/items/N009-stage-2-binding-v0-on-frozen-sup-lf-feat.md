# N009 · Stage-2 binding v0 on frozen sup-LF features: GS (DinoBloom-B) + MK set-encoder SigLIP to LF; distillation vs head transfer; autofl truth on held-out GS slide; zero-training GS-LF screen

type: exp · date: 2026-11-08 · status: open
goal_effect: pending · scale: module

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- binding v0 (scripts/train_bind.py, frozen sup-LF n006v3 fov + frozen DinoBloom-B, 20k steps; runs/n009_bind_v0*): held-out stained slide (b1, 5,697): GS->LF R@1 .0007 R@10 .011 medr 1862 (chance R@1 .00018); autofl truth on GS: LF teacher .61, distilled GS .27, head transfer -.05, direct GS ceiling .37; MK->LF R@10 .158 ('all' cells, chance .005). FOV split (both slides, 1,623 test): GS->LF R@1 .04 R@10 .19 medr 81. Per-slide feature standardisation: worse (R@10 .004, distilled autofl .07).
- zero-training screen (scripts/screen_gs_lf.py, results/N009-stage-2-binding-v0-on-frozen-sup-lf-feat/screen_B/screen.{json,csv}): unaligned mutual-kNN .018 (cross-slide test) / .049 (within); within-slide with all 12.5k pairs: linear SigLIP R@1 .072 R@10 .349 medr 22 > ridge (.046/.267/32) ~ MLP > Procrustes (.033/.179/88); Procrustes from 100 pairs R@10 .037 (6x chance). Cross-slide (train b0 slide, test b1 slide): best linear SigLIP R@10 .033 medr 591; Procrustes .018 -> rotation does NOT transfer better OOD here (JEPA-PRH prediction not supported); slide shift is the core obstacle.
- => binding v0's extra losses/MLPs HURT GS-LF alignment (within-slide R@10 .19 vs .35 for a linear SigLIP head).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N009 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N009-slug/ + external absolute paths -->

## Next
