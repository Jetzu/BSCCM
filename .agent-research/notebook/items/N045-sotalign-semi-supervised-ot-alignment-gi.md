# N045 · SOTAlign semi-supervised OT alignment Giemsa <-> label-free (15.7k pairs + unpaired clinical smears + LF cells)

type: exp · date: 2026-11-17 · status: done
goal_effect: flat · scale: paradigm · value: void

## Goal
User request 2026-11-17 ('why not SOTAlign again', github.com/ExplainableML/SOTAlign, arXiv 2602.23353, ICML 2026). SOTAlign = semi-
supervised alignment of frozen encoders: CCA teacher on the few pairs, alignment layers trained with SigLIP/SAIL on pairs + KLOT (KL
between entropic OT plans of teacher and student affinities) on UNPAIRED data. Here: X = Giemsa (frozen DinoBloom-B, 15,712 BSCCM cells),
Y = label-free (frozen r18 marker-teacher features); pairs = same cells (K001: no Giemsa cell has markers; Giemsa cells sit on the two
unstained slides; autofluorescence is the only same-cell truth); unpaired X = PBC half A, unpaired Y = 100k training-draw non-Giemsa LF.
N009 (pairs only, Procrustes / SigLIP adapter) gave cross-slide R@10 .018/.033 -> distillation was chosen instead.

## Plan
Arms (same pairs/budget/selection): CCA teacher, SigLIP pairs only, SOTAlign (pairs + KLOT with PBC + LF unpaired), SOTAlign LF-unpaired
only, controls raw_ridge, Procrustes, N009-style SigLIP. Protocols: cross-slide b0->1, b1->0 (test = all cells of the other Giemsa slide)
and FOV split within each slide; fit on paired split 0, select on split 1. Metrics: Giemsa->LF exact-cell R@1/R@10; transferred
autofluorescence r (ridge heads on aligned LF space, training-draw non-Giemsa cells) vs measured autofl of the test Giemsa cells
(compare: distillation student .56/.85 within slide, K022/K025); PBC-B 5-NN cell-type accuracy and lineage contrasts (N021/N032).
Screen 1 seed (seed 0), dim 128, 1,500 steps, batch 256+256; 3 seeds only if SOTAlign clearly beats pairs-only/CCA.
Not a GOAL attempt (GOAL = LF->markers) -> effect void on the GOAL, scale paradigm (new training signal: unpaired OT).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Code: Codex writer (codex/: scripts/n045_sotalign.py, merge_n045.py, tests/check_contracts.py 5/5 pass, README, report.md; executes
  upstream AlignmentLayer/MatchingModel/CCA unchanged via AST extraction; mapping audit: hist rows exact, PBC 17,092 x 768, halves A/B as
  N021). CPU smoke (1,664 cells, 6 steps; numbers meaningless). Claude red-team running (redteam_claude.md).
- v1 GPU (n045_gpu.sbatch, 1 GPU, lion-pytorch in env/pylib_n045): 1374892-5 failed (runner refuses output inside the data tree);
  rerun 1374934-7 -> ~/bsccm_runs/n045/ (2.5 min each). Test-split results, seed 0 (summary.csv per protocol):
  cross-slide R@10 (chance .0018/.0010): siglip_pairs .037/.022, sotalign .030/.018, sotalign_lf_only .034/.020, n009-style .025/.012,
  cca .021/.014, procrustes .020/.012, raw_ridge .017/.008; within-slide (chance .010/.016): siglip .30/.36, sotalign .30/.35.
  Transferred autofluorescence r (cross b0->1, b1->0 | FOV b0, b1): raw_ridge .65/.46 | .51/.78, procrustes .52/.27 | .38/.66,
  sotalign .38/.13 | .34/.36, siglip_pairs .21/.03 | .13/.14; reference distillation student TT6c (N029, same cells, clean truth)
  cross .715/.601, within .564/.848. PBC-B lineage contrasts mean AUROC: sotalign .58/.54, lf_only .62/.55, siglip .52/.57,
  raw_ridge .45/.38 (distilled students .95-1.00 on lineage, K025). (My first chat table averaged val+test R@10 -- corrected here.)
- Claude red-team (redteam_claude.md): LAND WITH FIXES. Leakage, upstream faithfulness, metric definitions and smoke reproduction OK.
  B1 output path (fixed); M1 stop exit 75 kills preempt chains -> 143 (fixed); M2 CPU suffices + cross/FOV protocols share training
  (PBC rows duplicated); M3 CCA teacher collapses on PBC (mean cosine .60 vs .012 BSCCM, near-uniform plans) -> confounds the PBC
  arm; m1-m5 undocumented deviations (CCA rank 128, grad clip, no warmup, 1,500 steps underfit, tie-prone selection, n009 arm not
  N009); m6 add N029 clean references (done above).
- v2 CPU (n045_cpu.sbatch, preempt_cpu 48 cores, 4,500 steps): PBC original vs PBC re-centred so half A (unlabelled) has the training
  draw's split-0 Giemsa mean (make_pbc_recentred.py; shift norm 36.6 vs PBC mean norm 21.7); jobs 1375024-31 -> ~/bsccm_runs/n045_v2/.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- v2 results (n045_v2_table.py -> v2_summary.csv; ~/bsccm_runs/n045_v2/, 11 min per job on 48 cores): test R@10 cross-slide
  siglip_pairs .038/.022, sotalign .036/.020 (re-centred .037/.019), lf_only .037/.022 (chance .0018/.0010); within-slide siglip
  .399/.384, sotalign .379/.374. Transferred autofluorescence r cross | within: raw_ridge .65/.46 | .51/.78, procrustes .52/.27 |
  .38/.66, sotalign .44/.21 | .38/.43 (re-centred .45/.21 | .39/.41), siglip_pairs .38/.18 | .34/.30; distillation student TT6c
  .715/.601 | .564/.848. PBC-B lineage contrasts mean AUROC (cross b0->1/b1->0): sotalign .58/.53 (re-centred .56/.51), siglip
  .53/.61, lf_only .46/.56, raw_ridge .45/.38; distilled students .95-1.00 (K025). PBC 5-NN .96-.97 for all learned aligners.

## Outcome
FLAT / NEGATIVE for this project (void on the GOAL). Semi-supervised OT alignment (SOTAlign) of frozen Giemsa and label-free encoders
never beats pairs-only SigLIP on exact-cell retrieval, improves marker (autofluorescence) transfer over pairs-only (+.06) but stays
below a plain ridge bridge, and its clinical lineage contrasts (AUROC .51-.58) are far below the distilled Giemsa student (.95-1.00).
Re-centring PBC (red-team M3) and 3x longer training change nothing material. Reason: an aligned shared space finds the same cell
(~20x chance cross-slide) but carries little marker information; distillation trains the Giemsa encoder on the marker targets
directly. SOTAlign's unpaired-OT idea assumes same-domain unpaired data; PBC is a different domain (K015/K016).
<!-- on close: bin/effect.sh N045 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N045-slug/ + external absolute paths -->

## Next
