# N032 · Teacher-run variance of clinical transfer: r18 v4c teachers seeds 1/2 (+ N029 seed 0) x TT6c students (2 seeds each) -> MLL23/PBC + autofl truth; is the v4 chain's MLL23 .62 a lucky teacher draw?

type: exp · date: 2026-11-10 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal
All clinical comparisons so far used ONE LF-teacher run per version (red-team N025 F2); retrained clean teachers lost ~.07 on MLL23 (N029). Measure teacher-run variance.

## Plan
r18 FOV teachers on targets_v4c with seeds 1 and 2 (seed 0 = N029), each -> probe -> 2 TT6c anchored students (v6c autofl aux, PBC half A anchors) -> MLL23/PBC-B; plus the old v4 teacher (N025 TT6) as a 4th draw. Jobs .agent-research/results/N032-teacher-run-variance-of-clinical-transfe/jobs.txt.

## Changes

## Evidence
- (n032_n033_metrics.csv, transfer_all.csv; plot .agent-research/plots/n032-mll23-fine-by-teacher_2026-11-10_1603.png) MLL23 mean over 12 fine contrasts per teacher run (student seeds): old v4 teacher .611 (.609/.622/.602), clean t0 .551 (.554/.571/.528), clean t1 .493 (.495/.491), clean t2 .477 (.512/.443) -> teacher-run spread .48-.61, mean over 4 teachers .53. Same-cell autofl within slide is teacher-INDEPENDENT: .564-.570 (b0) / .843-.848 (b1).
- Robust contrasts (all 4 teachers, worst >= .6): MLL23 segneut>lym CD16 .92 (min .87), mono CD14 .89 (.82), hairy>lym HLA-DR .72 (.63), myelocyte<segneut CD16 .70 (.61); PBC-B lineage contrasts .95-1.00, held-out erythroblast CD45 .87 (.76), IG CD16 .80 (.75), HLA-DR mono .85 (.72). NOT robust / failing: eos<segneut CD16 .47-.74, LGL CD56/CD16 ~.5, baso CD123 .29-.51, plasma/normoblast CD45 and hairy CD19/CD3 below .5 (inverted).

## Outcome
Clinical fine-contrast transfer is dominated by LF-teacher-run variance (.48-.61); the earlier .61-.62 was the luckiest teacher. Honest clinical claim: lineage markers + IG/myelocyte CD16-low + hairy-cell HLA-DR transfer robustly; eosinophil CD16 is teacher-dependent; NK/basophil/plasma/hairy CD19 fail. Future clinical numbers must average over >= 3 teacher runs. Not a GOAL attempt.
<!-- on close: bin/effect.sh N032 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N032-slug/ + external absolute paths -->

## Next
