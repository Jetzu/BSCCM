# N040 · Dense label completion (noisy student): label-count learning curve, then N031-ensemble pseudo-labels for all 11 targets on every training-draw cell and an augmented student on real + down-weighted pseudo labels

type: exp · date: 2026-11-12 · status: done
goal_effect: flat · scale: paradigm · value: 0.6564

## Goal
Paradigm bet: each training cell carries 1 of 11 labels (single-antibody slides) and unstained slides none; supervision for the
rare populations (B/NK/basophil/pDC) is ~10^2-10^3 positives per draw (DeepIFC saturates ~3,200 per type). Dense pseudo-labels from
the current ensemble on every training-draw cell + a noisy (augmented) student = semi-supervised, a new training signal.

## Plan
Stage 1 learning curve (falsifier): N031 recipe base model with --max-train 50 % and 25 % of the training draw, both directions,
seed 0; weak-target (CD19/CD56/CD123/HLA-DR) mean r vs 100 %. Kill the bet if 50 % -> 100 % gains < .01 on weak targets.
Stage 2 (only if stage 1 passes): teacher = N031 ensemble predictions on ALL training-draw non-Giemsa cells (never test-draw images);
student = same recipe, loss = real-label MSE + w * pseudo-label MSE (w .3, pseudo for unobserved targets only), train augmentation
(translation + flips where legal + gain/noise), 1 seed both directions; score frozen v5 GOAL vs N031 (single and ensemble).
Expected +.003-.012. Compute: stage 1 ~1 GPU-h, stage 2 ~2 GPU-h.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Stage 1 learning curve (results/N040-dense-label-completion-noisy-student-lab/jobs.txt): 1313150/1313151 b10 at 50 %/25 % (134,641/67,320 cells), 1313152/1313153 b01 at 50 %/25 % (63,973/31,986); N031 inductive init, v5; baseline n031_base (100 %). 1 H100 each, preempt.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->
- Stage 1 result (n040_lc_score.py -> n040_lc_per_target.csv; plot .agent-research/plots/n040-label-count-curve-r_2026-11-12_1847.png): GOAL common-11 (v5)
  25 % .6036 / 50 % .6272 / 100 % (n031_base) .6425; weak4 (CD19/CD56/CD123/HLA-DR) .347 / .396 / .430; other7 .750 / .760 / .764.
  Weak-target gain 50 -> 100 % = +.035 >= .01 -> PASS (stage 2 goes ahead). Largest: CD123 .34/.41/.50, CD19 .22/.31/.35.
  Caveat: --max-train removes cells (images + labels together), so this measures 'more labelled cells', not 'more labels per cell';
  pseudo labels add no new image-level information about rare positives -> expect a smaller gain than the curve suggests.
- Stage 2 code (Claude): train_lf.py --dump-train-preds (finished-checkpoint teacher pass over the training rows, no wandb, exits
  before result.json), --pseudo/--pseudo-w (teacher-mean targets only where the cell lacks a real label; asserts one training draw,
  coverage of tr, no Giemsa). Teacher dumps 1313822-1313825 (n031 base/zoom2 x b10/b01). Codex red-team running (codex/brief.md).
- Codex red-team (codex/redteam_N040.md): LAND WITH FIXES, all four applied after reading the code: F1 dump branch moved before the
  training loop and any run-owned write (in-flight dumps 1313822-5 use FINISHED checkpoints, so their loop was empty -- unaffected);
  F2 pseudo columns restricted to the common-11 targets with >= 1000 real source labels (excludes autofl and the unsupported
  batch-1 grp:CD123/HLA-DR/CD14); F3 resume fingerprint pins teacher content (sha256 of P+G), weight and columns; F4 explicit
  integrity checks (finite P of shape (N,K), unique in-range integer G, schema names, finite pseudo-w >= 0) and PY zeroed outside
  the mask (no NaN*0). Dumps written before the fix lack `names` (check skipped for them).

- Stage 2 (jobs 1374321/22/25/26, 1 H100 each, 12-22 min; runs/n040_student_{base,zoom2}_{b10,b01}; teacher = N031 base+zoom2
  predictions on the training draw, pseudo labels only where a cell lacks a real label, common-11 columns, w .3, phys aug .7):
  GOAL (results/N041-*/scores_goal.csv, scores_per_target.csv; plot .agent-research/plots/n040-noisy-student-goal-r_2026-11-17_1839.png):
  student ensemble .6564 (b10 .6650, b01 .6477) vs N031 ensemble .6527 (+.0037; better on 16/22 marker x direction items, Wilcoxon
  p .0013); student base alone .6529 vs N031 base .6425 (+.0103, 19/22, p 3e-5) = one model matches the old two-model ensemble.
  Phys aug alone (N041) gives base +.0025, so pseudo labels add ~+.008 to a single model.

## Outcome
FLAT by the GOAL rule (+.0037 < 2 x seed noise .0054), but a consistent paired gain in both directions; strongest single-model result
so far. Next: second seed + a second student round (teacher = student ensemble) to see whether the gain compounds past the noise.
<!-- on close: bin/effect.sh N040 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N040-slug/ + external absolute paths -->

## Next
