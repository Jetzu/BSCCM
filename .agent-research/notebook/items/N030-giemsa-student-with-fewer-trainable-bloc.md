# N030 · Giemsa student with FEWER trainable blocks (TT6 chain): freeze 10 of 12 and fully frozen DinoBloom-B (heads + projections only), 3 seeds; does less BSCCM-domain capacity improve clinical transfer?

type: exp · date: 2026-11-10 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal
N027 showed more student capacity destroys clinical transfer. Test the other direction: fewer trainable DinoBloom-B blocks on the TT6 chain.

## Plan
TT6 anchored students (3000 steps, anchor-ce .5, PBC half A, fov holdout), freeze 10 (2 trainable) and freeze 12 (frozen backbone), 3 seeds each; baseline TT6 freeze 8 (N025). Jobs: .agent-research/results/N030-giemsa-student-with-fewer-trainable-bloc/jobs.txt (applies) + students 1295187-97.

## Changes

## Evidence
- (n030_metrics.csv, transfer_multiseed.csv; plot .agent-research/plots/n030-mll23-fine-mean_2026-11-10_0940.png): within-slide autofl b0 / b1 | MLL23 fine mean (n=3 each): 4 trainable (TT6 baseline) .578 / .855 | .611 ; 2 trainable .567 / .852 | .589 ; 0 trainable .532 / .853 | .482. With N027: 8 trainable .529, DinoBloom-L 8/24 .420.
- Per contrast (2 trainable vs 4): eos CD16 .79 vs .74, baso CD123 .50 vs .41, but myelocyte .68 vs .76, hairy HLA-DR .76 vs .85.

## Outcome
FLAT/WORSE: clinical transfer peaks at the current 4 trainable blocks (0: .48, 2: .59, 4: .61, 8: .53, L: .42); same-cell autofl flat. Student capacity is tuned; the clinical bottleneck is the domain gap / teacher, not student size. Not a GOAL attempt.
<!-- on close: bin/effect.sh N030 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N030-slug/ + external absolute paths -->

## Next
