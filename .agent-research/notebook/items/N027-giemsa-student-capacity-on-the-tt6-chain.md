# N027 · Giemsa student capacity on the TT6 chain: DinoBloom-B with 8 trainable blocks + 6000 steps vs DinoBloom-L (8 trainable of 24) + 6000 steps; MLL23/PBC + within-slide autofl truth

type: exp · date: 2026-11-10 · status: done
goal_effect: worse · scale: module · value: void

## Goal
The recommended Giemsa chain (TT6, N025: v4 r18 markers + v6 r18 autofl) still has most MLL23 fine contrasts near chance and batch-0-slide autofl truth .58. Test student capacity/training length: is the student, not the teacher, the bottleneck?

## Plan
TT6 anchored students (PBC half A anchors, anchor-ce .5, fov holdout, aug), 6000 steps (was 3000), 2 seeds each: DinoBloom-B freeze 4 (8 trainable blocks, was 4) | DinoBloom-L freeze 16 (8 trainable of 24). Baseline TT6 (N025, 3 seeds): within-slide autofl .578 / .855, MLL23 fine .611. Jobs .agent-research/results/N027-giemsa-student-capacity-on-the-tt6-chain/jobs.txt (applies; students 1294877/79/81/83). ~3 GPU-h.

## Changes
- scripts/train_gs_bind.py, scripts/apply_gs_student.py: --backbone B|L (DinoBloom-L = ViT-L/14, 1024-d).

## Evidence
- (n027_metrics.csv, transfer_multiseed.csv; plots .agent-research/plots/n027-gs-autofl-within-b0_2026-11-10_0908.png, .agent-research/plots/n027-mll23-fine-mean_2026-11-10_0908.png): within-slide autofl b0 / b1 | MLL23 fine mean: TT6 baseline (3000 steps, 4 trainable blocks, n=3) .578 / .855 | .611 ; B 8 trainable + 6000 steps (n=2) .579 / .835 | .529 ; DinoBloom-L 8 of 24 trainable + 6000 steps (n=2) .581 / .860 | .420 (below chance).
- Jobs: students 1294877/79/81/83, applies 1294878/80/82/84.

## Outcome
WORSE for clinical transfer, flat on same-cell truth: the student is NOT capacity-bound; more trainable capacity/steps overfits the BSCCM Wright domain and destroys clinical transfer (L: .42). Next: fewer trainable blocks (N030). Not a GOAL attempt.
<!-- on close: bin/effect.sh N027 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N027-slug/ + external absolute paths -->

## Next
