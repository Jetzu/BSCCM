# N076 · Multi-marker final fine-tune: continue trained GOAL models only on 'all'-slide cells (pooled multi-marker panel), user priority 2026-11-23; multi-marker metric + GOAL

type: exp · date: 2026-11-23 · status: done
goal_effect: flat · scale: module · value: 0.6681

## Goal
USER 2026-11-23: decoding ALL markers of a cell jointly is "really really important"; the last fine-tune should focus only on the
multi-marker data. In BSCCM the only cells with several markers measured together are the 'all'-antibody slides (4 pooled panels:
grp:CD3/CD19/CD56, grp:CD45, grp:CD16, grp:CD123/HLA-DR/CD14 [draw 0 only]); single-stain slides never co-measure markers.

## Plan
train_lf.py --init-full <trained run ckpt.pt> (strict full-state load: encoder + all heads) --train-panel all (training-draw cells of the
'all' slides only; evaluation rows unchanged). Verified: Codex N074 suite 7/7 flag-off byte-equal vs the pre-N074 harness; flag-on smoke
(verify/test_ft.py: panel restriction, strict load, finite losses, weights move).
Screen: base = control n051_v10_base (s0) / n051_v10_base_s1, both directions; fine-tune 5 epochs, lr 1e-4, bs 256, same SSL stem.
4 runs (~.1-.2 GPU-h each). Multi-marker metric MM3 = mean r of the 3 pooled panels in common-11 on held-out 'all' test cells, mean over
b10/b01; control MM3 .7577 (seed sd .0035 over 4 control seeds, verify/mm3_noise_controls.csv -> bar 2 x sd = .007).
PRIMARY (pre-registered, user priority): MM3(ft) - MM3(own base), paired by seed; success if > +.007. SECONDARY: GOAL (expected to
drop on singles; reported, not the decision). If success: apply the same final stage to every ensemble member and rebuild the ensemble.

## Changes

## Evidence
Jobs 1430118-121 (exit 0, ~4 min each). Scored with scripts/score_goal_arms.py (v10) + MM3 (mm3_scores.csv).
RESULT (paired by seed vs its own base): MM3 .7564 vs .7577 -> -.0012 (s0 -.0016, s1 -.0008; bar +.007) = FLAT; GOAL .6681 vs .6706
(-.0025; b10 +.0009, b01 -.0059). Per target all within +-.012 (HLA-DR .481 -> .469, CD19 .320 -> .310). A low-LR multi-marker-only
stage neither improves pooled-panel decoding nor (at this LR) erodes singles: the base models already fit the 'all'-slide cells as well
as this data allows; training focus is not the bottleneck for joint decoding.
Plots: plots/mm3-r-n076_2026-11-23_1536.png, plots/goal-r-n076_2026-11-23_1536.png

## Outcome
<!-- on close: bin/effect.sh N076 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N076-slug/ + external absolute paths -->

## Next
Keep the harness options (--init-full/--train-panel) for final stages; joint decoding needs NEW joint measurements (S-BIAD452, N077), not re-weighting of the existing ones.
