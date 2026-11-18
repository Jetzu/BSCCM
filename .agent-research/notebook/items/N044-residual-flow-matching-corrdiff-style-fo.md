# N044 · Residual flow matching (CorrDiff-style) for LF -> fluorescence images: frozen regressor gives the per-cell mean, flow learns the scaled residual conditioned on LF + mean

type: exp · date: 2026-11-17 · status: done
goal_effect: flat · scale: module · value: void

## Goal
Fix N034/N035's trade-off: flow samples are distributionally better than the same net as a regressor (CRPS .674 vs .815) but worse per
cell (r .476 vs .576) and over-dispersed on a new draw (sd ratio 1.25, outside .53 vs ideal .22). Residual flow (CorrDiff, Mardani et
al. 2023): y = mu(LF) + sigma_c * r, mu = frozen reg-mode CondUNet (EMA), flow learns r = (y - mu)/sigma_c conditioned on LF + mu
(31 condition channels, CFG drops both), sigma_c = per-channel residual rms on 4,096 training cells. Not a GOAL attempt (image model;
GOAL = scalar LF->MK marker r) -> effect void on the GOAL. User request 2026-11-17: 'go on further develop the model'.

## Plan
Development direction = N034 (train draw 1 = batch 1 split 0; mean = runs/n034/reg_v2), same recipe as cfm_v2 (width 64, patch 2,
30 ep, bs 128, lr 2e-4, cfg-drop .1, EMA .9995, seed 0). Sampler: Heun 48 steps, K 8; CFG chosen in {1, 2} on the IN-BATCH set
(draw 1 split 2) only, by CRPS then outside-fraction; newdraw (draw 0 split 2, N034's development set) reported with that choice.
Baselines on the same cells: reg_v2 (mean only) and cfm_v2 at s48 g2/g3 (N034 sweep). Pre-registered (newdraw, mean over labels):
SUCCESS if r_avgread >= r_reg - .01 AND CRPS < plain-flow CRPS (s48 g2) AND outside-fraction closer to 2/9 than plain flow.
KILL if CRPS >= plain-flow CRPS and r_avgread < r_reg - .02. If success: frozen confirmation in the N035 direction (train draw 0,
mean = runs/n035/reg_b0, untouched draw 1 split 1) as a separate pre-registered item.
Compute: smoke (2k cells, 1 ep) then full training ~3 H100-h (frozen mean adds a forward pass) + 2 evals ~1 h each; preempt, 1 H100.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Code (Claude): fm_common.make_cond(mu=), sample(mu=), Generator; fm_train --mode rcfm --mean-ckpt (sigma_c estimated once, stored
  in ckpt; mean-ckpt sha256 pinned in the resume config); fm_eval accepts rcfm in the --cfm slot (shuffled-LF control recomputes mu
  from the shuffled LF). CPU wiring check: cond (2,31,128,128), CFG-dropped cond all zero, finite samples, plain path unchanged.
  Codex red-team (codex/redteam_N044.md): LAND WITH FIXES. Residual target/decode, 31-channel condition + all-zero CFG null, frozen
  EMA mean, sigma selection (4,096 clean draw-1 split-0 rows; smoke scales .236-.267), sigma resume, mean-hash resume, shuffled-LF
  control (mu recomputed from shuffled LF) and bitwise-unchanged plain cfm/reg outputs all PASS; CUDA [unverified]. Fixed after
  reading the code: (1) mean-checkpoint gate now checks targets_sha256, lf_norm_sha256 and stain order (was the targets PATH only);
  (2) mean_ckpt omitted from plain cfm/reg configs so pre-N044 checkpoints resume. Noted (inherited, disclosed in N035): fm_prep's
  a_c comes from draw-1 split-0 border noise, so draw-0-trained targets share that camera-noise constant.
- Jobs (jobs.txt): 1374389 smoke -> 1374390 full rcfm_b1 (afterok), runs/n044/.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

- Training finished (epoch 30, step 50,610, ~560 cells/s). Evals: g1 1375827 done (runs/n044/eval_s48_g1.0); g2 1375828 preempted ->
  1376415 resumed per group; plain-flow baseline RE-EVALUATED in the current harness on the same cells (1376502; the N034 sweep used an
  older harness/labels: its r_reg .664 vs .695 here, so it is not a valid same-cell baseline).
  g1 provisional: in-batch r_avgread .692 vs r_reg .695, CRPS .254 vs reg .384, condW1 .116 vs .299, outside .298 (ideal .222), sd ratio
  .96; newdraw r .594 vs .599, CRPS .591 vs .778, condW1 .594 vs .739, outside .73, sd ratio .72 (under-dispersed on the new draw).
FINAL (n044_summary.py -> n044_summary.csv, n044_entries.csv; mean over every stain x target entry; plain flow re-evaluated in the
same harness on the same cells, 1376502):
| set | arm | r_avgread | r_reg | CRPS | outside (ideal .222) | sd ratio |
| in-batch | residual g1 | .692 | .695 | .254 | .298 | .96 |
| in-batch | residual g2 | .687 | .695 | .260 | .274 | 1.01 |
| in-batch | plain g2 | .612 | .695 | .307 | .352 | 1.02 |
| new draw | residual g1 | .594 | .599 | .591 | .730 | .72 |
| new draw | residual g2 | .590 | .599 | .572 | .696 | .75 |
| new draw | plain g2 | .550 | .599 | .569 | .699 | .79 |
Pre-registered rule: CFG chosen on in-batch by CRPS -> g1. New draw: r_avgread .594 >= r_reg - .01 PASS; CRPS .591 vs plain .569
FAIL; outside .730 vs plain .699 (farther from .222) FAIL -> NOT a success. Reading: the residual flow removes the per-cell accuracy
loss of the plain flow (in-batch r .692 vs .612, new draw .594 vs .550; regressor-level), but on a new blood draw it is not better
calibrated than the plain flow -- both under-disperse (sd ratio .72-.79, ~70 % of real readouts outside the 80 % band): the draw
shift, not the generator, dominates calibration. Plots: plots/n044-newdraw-r_2026-11-18_0913.png, n044-newdraw-crps_2026-11-18_0913.png.
Not a GOAL attempt (GOAL = LF->MK r) -> effect void.

## Outputs
<!-- files in .agent-research/results/N044-slug/ + external absolute paths -->

## Next
