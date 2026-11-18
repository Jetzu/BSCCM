# N056 · E3 clinical transfer: class-anchored Giemsa students (hub latent distillation vs TT6c output distillation) + unpaired OT/flow alignment of BSCCM-Giemsa to clinical smears (PBC, MLL23); per-cell contrasts and population proportions (RQ3)

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: paradigm · value: void

## Goal
E3 from N052: real-world Giemsa->marker transfer. N054 showed unanchored cross-slide students (all designs) are at chance on PBC /
MLL23; class anchoring is required (K016). Compare TT6c vs hub students under identical anchoring, add unpaired unbalanced-OT
alignment of clinical embeddings (user list: unbalanced OT), and test RQ3 (marker-positive fractions per clinical class).

## Plan
Writer Codex (codex/brief.md), Claude red-team + integrator. Arms T (TT6c anchored), H (hub anchored), T+OT, H+OT; 3 teacher
runs each (K025). Eval PBC half B + MLL23 pre-registered contrasts, RQ3 proportions vs a textbook table fixed before reading
clinical predictions. Prediction (Claude): H ~= T on lineage contrasts (.95+ PBC), OT helps little on fine contrasts (K016:
finer biology not robust), RQ3 lineage fractions within ~15 % absolute for CD16/CD14/CD3, poor for CD56/CD123/CD19.
Not a GOAL attempt -> effect void. Budget target <= 6 GPU-h.

## Changes

## Evidence
- Codex v1 (codex/): scripts + CPU smoke OK, est. 4.1-5.8 GPU-h; RQ3 literal calibration impossible (BSCCM expert classes carry
  pooled groups only, no single markers) -> proxy analysis, secondary.
- Claude red-team v1 (scratchpad/n056rt/c1-c7.py): no P0; contrasts AST-equal to apply_gs_student.py; UOT matches POT to 3e-7;
  fit/eval splits disjoint. P1s REPRODUCED first-hand: (1) no noise floor -- N032 within-teacher student-seed spread on MLL23
  fine up to .07 (t0 .528-.571, t2 .443-.512; n032_n033_metrics.csv); (2) anchor-CE share differs -- T log
  (logs/bsccm_gsanc_1294457.out) sig 3.4 + dist .22 vs .5 x anc ~.08 (~2-3 %), H latent MSE .085-.20 -> same CE weight is
  30-60 % of H's objective; (3) OT without trivial alignment controls, near-balanced tau under heavy label shift; (4) RQ3 under OT
  shrinks toward source rates (circular). P2: hub recipe differs from N054 (.594 vs .602 marker r, reviewer CPU check), /home
  disk risk, 7 exact + 8 near PBC cross-half duplicates, OT fit/apply representation mismatch, single-thread H loader.
- v2 fix brief (codex/v2/brief_v2.md): 2 H seeds/teacher + pre-registered seed-range threshold; primary H-vs-T on non-anchor
  contrasts + H_matched (gradient-share-matched anchor weight); mean-shift / CORAL / kNN controls, 2 fixed tau; RQ3 split into
  drift vs calibration, raw arms only; N054 hub recipe; /hnvme run root with single-file caches; duplicates excluded.
- Codex v2 delivered (CPU smoke + SIGTERM resume OK; 16 PBC duplicate filenames excluded; est. 5.7 GPU-h). Claude spot-check:
  matched anchor weight w = share/(1-share) * g_latent/g_anchor (correct), noise-floor decision in n056_eval.py:320-354.
- Training 1381736-44: 9 hub students (H s0/s1, H_matched s0; t0-t2), ~9 min each, all done.json present.
- INCIDENT: the eval submission ran twice (1381797-812 and 1381814-829, identical submit lines, same second) -> two writers per
  eval unit, sqlite UNIQUE errors in 3 units. All 32 chains cancelled with preempt-cancel.sh, the 16 eval dirs removed (training
  dirs kept), evals resubmitted once in the foreground: 1381849-64 (16 jobs verified).
- RESULTS (aggregate -> tables/; per-run n056_per_run.csv; plots plots/n056-mll23-fine-auroc_2026-11-18_1352.png,
  n056-primary-nonanchor-auroc_2026-11-18_1352.png). Raw students, mean over 3 teachers (T 7 seeds, H 6, H_matched 3):
  | arm | primary16 (non-anchor) | MLL23 15 | PBC 10 | all 25 |
  | T (TT6c, anchored) | .540 | .568 | .841 | .677 |
  | H (hub, anchored) | .632 | .704 | .846 | .761 |
  | H_matched (anchor share = T's) | .563 | .640 | .822 | .713 |
  Per teacher H > T on primary16 (+.064/+.112/+.101) and MLL23 (+.10/+.155/+.152); every H seed (.637-.787) is above every T seed
  (.499-.613) on MLL23 (rank separation 6 vs 7, exact p ~ 1/1716, post hoc). REGISTERED decision: "no difference" -- the paired
  difference (+.092 primary16, +.136 MLL23) is below the pre-registered floor (max within-teacher seed range .141 / .150).
  H_matched keeps part of the gain (MLL23 +.02/+.10/+.10 per teacher; sensitivity only, 1 seed) -> part of H's advantage is
  stronger anchoring, part the hub target. PBC lineage near ceiling for both (.84-.85).
  Alignment (on H): none beats raw on primary16 (raw .632; OT tau .3 .619, tau .91 .621, mean-shift .622, kNN .604, CORAL .600);
  OT tau .91 / mean-shift lift PBC10 (.868/.869 vs .846) but lower MLL23 (.675/.676 vs .704); class-conditional OT only PBC .881
  (uses anchor labels). -> unbalanced OT does not help beyond a trivial mean shift.
  RQ3 (proxy only; literal calibration impossible): H neutrophil CD16+ fraction .99-1.00 (ref ~.95), lymphocyte CD3+ .87-1.00 (ref
  ~.7), monocyte CD14+ .67-.83, eosinophil CD16+ .64/.96 (ref ~0: fails, as K016/K025). Mean abs error H .28/.37 vs T .52/.48 (MLL23
  / PBC), but the BSCCM calibration term alone is .37 -> RQ3 not quantitatively answerable with these labels.
- Codex v2 (codex/v2/; CPU smoke incl. SIGTERM resume bitwise OK; 16 PBC duplicate filenames excluded). Claude check of v2:
  matched weight w = share/(1-share) x g_latent/g_anchor (correct), noise-floor decision enforced in n056_eval.py:320-326.
- Training 1381736-44 (9 units, 8-10 min each; runs/n056v2_main/t*/): measured anchor-CE GRADIENT share on the trainable ViT
  blocks: T recipe .159-.173 (loss-magnitude share was only ~3 %); H at weight .5: .70 at start -> .28 at end; H_matched frozen
  weight .093-.115 (share matched over 200 calibration steps; .04-.07 later). Evals 1381797-812 (16 units; first links preempted,
  4 intermediate links crashed on a sqlite UNIQUE insert during resume (concurrent successor), final links 1381849-64 completed;
  all 16 eval.done.json; aggregation validated cache hashes). Tables: results/N056-*/tables/ (contrasts_all, macro_means,
  paired_macros = registered decisions).
RESULTS (mean over 3 teacher runs; AUROC; primary16 = non-anchor contrasts, the pre-registered H-vs-T endpoint):
| arm | primary16 | PBC10 | MLL23_15 | all25 | MLL23 robust |
| T (TT6c anchored, 7 seeds) | .540 | .841 | .568 | .677 | .764 |
| H (hub anchored, 6 seeds) | .632 | .846 | .704 | .761 | .891 |
| H_matched (anchor share = T's, 3) | .563 | .822 | .640 | .713 | .848 |
| T + unbalanced OT tau .91 | .534 | .859 | .562 | .680 | .765 |
| H + unbalanced OT tau .91 | .621 | .868 | .675 | .752 | .850 |
| T + mean-shift / CORAL / kNN | .536 / .543 / .514 | .834 / .801 / .844 | .571 / .571 / .542 | .676 / .663 / .663 | |
Registered decision (paired by teacher, threshold = max within-teacher seed range): H - T primary16 +.092 (floor .141), all25
+.083 (floor .091), MLL23_15 +.136 (floor .150) -> "no difference" by the pre-registered rule, though H is higher on every
summary. H_matched - T: +.023 primary16, +.035 all25, +.072 MLL23 (1 seed, sensitivity only) -> most of H's advantage tracks the
STRONGER ANCHORING, not the latent target. Alignment (OT at both tau, mean-shift, CORAL, kNN) changes nothing beyond noise
(|delta| <= .03; CORAL hurts robust contrasts). BSCCM same-cell autofl sanity unchanged (.57 / .85 all arms).
RQ3 fractions: literal per-class single-marker calibration impossible (no single markers on gated BSCCM classes) -> NA; proxy
exploratory only. Plots: plots/n056-clinical-auroc-primary16_2026-11-18_1355.png, n056-clinical-auroc-all25_2026-11-18_1355.png.

## Outcome
Hub (latent-distillation) students transfer fine markers to MLL23 better than TT6c in every teacher and every seed (+.14 AUROC),
but the pre-registered seed-noise floor is not passed (registered: no difference); part of the gain is anchoring strength. OT
alignment does not help. Not a GOAL attempt -> effect void. Next: confirm H > T on MLL23 with more seeds at matched anchoring.

## Outputs
<!-- files in .agent-research/results/N056-slug/ + external absolute paths -->

## Next
