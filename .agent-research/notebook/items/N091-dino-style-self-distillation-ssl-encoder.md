# N091 · DINO-style self-distillation SSL encoder (N088 lit A-P3) as a diverse ensemble member: ResNet-50 DINO on unlabelled cells of each training draw, fine-tuned both directions; pre-registered E8 + DINO member vs .7021

type: exp · date: 2026-11-27 · status: done
goal_effect: flat · scale: paradigm · value: 0.6951

## Goal

## Plan
Hypothesis (N088 lit A-P3; N085: ensembles gain from diverse members): a DINO self-distillation encoder (clustering-friendly, unlike
the isotropic SIGReg space) trained on the same unlabelled training-draw cells gives a different, complementary member.
Recipe: Codex adds --objective dino to scripts/train_ssl_lf.py (default sigreg byte-identical); ResNet-50, 40 epochs, bs 512, per draw
(--exclude-batch 0 / 1; ~2.5-3 H100-h each), teacher encoder exported; fine-tune with the n051 recipe (v10 labels, 10 ep, lr 5e-4,
seeds 0 and 1, both directions; ~2 H100-h). Total ~8 H100-h (preempt).
PRE-REGISTERED: PRIMARY = E9 = equal-weight mean of the 8 N085 E8 members + the DINO member (2-seed mean per direction) > .6967 + .0054
= .7021. Secondary: DINO single model (2-seed mean) vs the SIGReg control .6706; MM3; clustering of its frozen features (N087 harness).
Kill: DINO single < .6600 (member too weak to add) or collapse (teacher entropy ~0).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Codex writer (codex/: --objective dino in train_ssl_lf.py, 5 CPU tests; default sigreg byte-identical incl. old-ckpt resume; exact
  resume; teacher encoder export loads in train_lf --init-ssl). Claude red-team (redteam_claude.md): NOTHING BREAKS, 1 MAJOR: the
  prototype layer was not DINO-standard (only the weight-norm magnitude g frozen in epoch 1, then trained + decayed). Claude fixes
  (codex/train_ssl_lf.py; original kept as train_ssl_lf.codex_orig.py): g fixed at 1 always (norm_last_layer), whole prototype layer
  (directions) frozen in epoch 0, biases/norm params exempt from weight decay (MINOR 1); test updated to assert g == 1; 5/5 tests pass.
  Integrated into scripts/train_ssl_lf.py + tests/test_train_ssl_lf.py. Remaining MINORs (bf16 prototype logits, KL = logK - entropy,
  local crops share augmentation) accepted (local crops off).
- Jobs 1466822 / 1466823 n091_dino_r50_ex0 / _ex1 (1 x H100 preempt, 40 ep, bs 512, lr 1e-3 = DINO rule 5e-4 x bs/256): 1h32 / 35 min.
  No collapse: teacher entropy 6.4 -> 1.1-1.2 (log K = 8.3), marginal KL to uniform 1.63, embedding effective rank 107-130.
- Fine-tunes 1467297-300 (n051 recipe, seeds 0/1, both directions). score_goal_arms.py: DINO single .6519 (b10 .6692 / b01 .6346) vs
  SIGReg control .6706 -> -.0187 (kill line .6600 crossed). PRE-REGISTERED n091_score.py: E9 (E8 + DINO 2-seed member) .6951
  (b10 .7053 / b01 .6849, MM3 .7841) vs E8 .6967 and bar .7021 -> FLAT (adding DINO -.0016). Plot: .agent-research/plots/goal-direction-r-v10-n091_2026-11-27_1015.png.
- Secondary (cluster_with_dino.csv; frozen teacher features via n091_extract_teacher.py, which checks the export == teacher and that
  draw 0 was excluded): DINO NMI .340 / ARI .242 / kNN bal-acc .336 vs SIGReg .340 / .205 / .306, AE64 .390 / .260 / .346, image
  statistics .351 / .269 / .354 -> DINO clusters better than SIGReg (as lit A predicted) but not better than raw image statistics.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
FLAT: DINO single .6519 (-.019 vs SIGReg); E9 .6951 < E8 .6967 (bar .7021). DINO features cluster better than SIGReg, not better than image statistics. The DINO trainer (--objective dino) stays available.
<!-- on close: bin/effect.sh N091 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N091-slug/ + external absolute paths -->

## Next
