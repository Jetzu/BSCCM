# N036 · Label position detrend v7: cross-fit the per-slide HGB position trend by FOV folds (no cell's trend fitted on its own FOV), keep a trend only if it reduces out-of-fold variance; targets_v4cx/v5cx; held-out diagnostics vs v5c and GOAL re-score (N034 check3 F6: v4 detrend overfits split-0 FOVs)

type: fix · date: 2026-11-12 · status: done
goal_effect: n/a · scale: ? · value: - · why: infra

## Goal
Fix N034 check3 F6: the v4 per-slide HGB position detrend is fitted on split-0 cells and subtracted from the same cells, so it absorbs
split-0 FOV-composition noise (FOV-mean variance share of v5c labels lower on split 0 than on splits 1/2, e.g. CD19 b1 .34 vs .54/.57;
v3c equal across splits). Labels only; inputs untouched.

## Plan
xfit: per slide x target, 5 deterministic FOV folds (hash of global FOV id); each cell's trend from an HGB (same hyper-parameters) fitted
on split-0 Giemsa-free cells of the OTHER folds; trend kept only if it lowers out-of-fold split-0 variance by > 1 %; centred on split 0.
Env BSCCM_DETREND=xfit in make_targets_v4.py / make_targets_v5.py -> targets_v4cx / targets_v5cx (default path unchanged).
Checks: (1) FOV-mean share equal across splits on v5cx; (2) held-out (splits 1+2) FOV-grouped CV position predictability of the label
not larger than v5c; (3) GOAL re-score of the N031 best predictions on v5 / v5c / v5cx. Kill: if (1) still differs, the trend model
itself (not the in-sample fit) carries FOV composition.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Build 1311495 (preempt_cpu, 48 cores; HGB fits, no GPU use): targets_v4cx/v5cx written in seconds (HGB early stopping); detrend
  summary results/N036-label-position-detrend-v7-cross-fit-the/targets_v4x_detrend_summary.csv: out-of-fold trend still explains most
  autofl (.824 vs in-sample .827) and CD19 (.805 vs .816) variance -> the slide-position artefact is real, not an overfit; correlations
  drop where the in-sample fit was absorbing noise (CD3 trend-target corr .28 vs .49, CD16 .11 vs .28).
  The diagnostics step crashed (forked Pool workers reading one lazy NpzFile: BadZipFile 'Overlapped entries'); arrays now materialised
  before the Pool, rerun as 1312652.
- GOAL re-score (n036_rescore.py -> n036_rescore_goal.csv, n036_rescore_per_target.csv): the batch-held-out test cells are ALL split 2 (Codex C2; my first
  check showed split 0 empty) -> GOAL test labels were never in-sample detrended; F6 affects training labels, split-0-fitted
  readouts and FOV-split protocols only. N031 inductive ensemble common-11: v5 .6527 (GOAL as defined), v5c .6532, v5cx .6547
  (b10 .6685, b01 .6408); base .6444, zoom2 .6462 on v5cx; v3c (no detrend, reference) .6080. Change v5c->v5cx +.0015 < 2 x noise.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->
- Red-team (Codex, results/N036-label-position-detrend-v7-cross-fit-the/codex/redteam_N036_N037.md; codex-run.sh printed a FALSE
  'CODEX-UNAVAILABLE model/quota refused' -- the gate matched text inside a command output; the report and last message are complete):
  C2 (GOAL test cells: all split 2), C3 (N037 args differ only in --targets/--out), C4 (scores reproduce with N031's own scorer),
  C5 (default hgb path unchanged) HOLD. F1 major: the trend VALUES are out-of-fold, but the keep/drop gate (one bit per slide x
  target) and the centring constant use all clean split-0 cells incl. the own FOV -> claim narrowed (verified by reading
  make_targets_v4.py; the constant cannot change within-slide Pearson; the bit is a slide-level selection) and documented in the
  code. F2 minor: a fold with no training rows would crash the Pool -> guard added (< 200 training cells or < 2 training FOVs ->
  zero trend for that slide x target); Codex scanned all current jobs (min 4,004 training cells), so v5cx outputs are unchanged.
- Diagnostics 1312652 (preempt_cpu; fov_share.csv, pos_r_heldout.csv): (1) split-0 minus held-out (splits 1/2) FOV-mean variance share,
  mean |gap| over target x draw: v3c .016, v5c .055, v5cx .017 -> the in-sample absorption is removed (largest v5c gaps CD19 b1 -.209,
  autofl b0 -.141, CD3 b0 -.131 become -.011/-.021/+.033). (2) held-out FOV-grouped CV position predictability (median over slides):
  v3c .452, v5c .046, v5cx .059 -- equal per target except grp:CD3/CD19/CD56 .056 vs .015, grp:CD16 .052 vs .034, CD56 .142 vs .125
  (weaker out-of-fold trend estimates / the keep-gate); both far below no-detrend. (3) GOAL re-score +.0015 (flat) and N037 retrain
  +.0010 (flat).

## Outcome
FIX WORKS, NO GOAL EFFECT. Cross-fitting removes the split-0 in-sample absorption (FOV-share gap .055 -> .017 = no-detrend level) with
about the same residual held-out position signal; the GOAL is unchanged (test cells were never in-sample detrended; N037 retrain flat).
targets_v5cx is available as the more correct training label; the N038 portfolio keeps v5 for comparability with N031, GOAL scored on
v5 as defined. Red-team F1 narrowed in the code comment, F2 guarded. Effect: n/a (label tool; its GOAL attempt is N037).
<!-- on close: bin/effect.sh N036 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N036-slug/ + external absolute paths -->

## Next
