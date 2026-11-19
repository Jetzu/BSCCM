# N066 diagnosis: where the remaining gap (.70 - .6929 descriptive 6-way; - .6862 registered best) lives

Instruments (all on the SAME held-out GOAL cells, v10 labels; 6-way = N065 composite members + N053 members, GOAL .6929):
1. Direction split (n066_perslide.csv): b10 (train draw 1 -> test draw 0) .7036 is already above .70; b01 (train draw 0 -> test
   draw 1) .6822. b01 deficits: grp:CD45 .660 (b10 .779), grp:CD16 .784 (.868), HLA-DR .445 (.525). Draw 0 has ONE 'all' slide
   (10,210 pooled-label cells) vs 27,074 in draw 1.
2. Replicate-slide split inside the b01 test draw (same blood, same stain, same model): grp:CD45 r .881 on 'all' replicate 1 vs
   .578 on replicate 0 (label sd 0.73 vs 1.19; predicted sd .84 vs .84 -> the model sees the same population on both slides).
   grp:CD16: slide label means -.12 vs +.21 while predicted means are +.17 vs +.14 (slide offset not in the image).
3. FOV clustering of the ensemble residual (n066_fovresid.csv, permutation null within draw): excess FOV share CD19 +.46/+.43,
   CD56 +.29/+.30, CD14 b01 +.29, grp:CD45 +.14/+.21, HLA-DR b01 +.17; strong singles ~0 (matches N064 fov_error_share ANOVA).
4. Is the FOV-level residual predictable from LF-observable acquisition context (FOV cell count, within-FOV radius/x/y, focus z
   rel. to slide median)? Cross-draw residual model (fit on the other draw's held-out residuals, HGB): GOAL .6929 -> .6284, every
   target worse; |r(residual, context)| < .1 except draw-specific focus terms -> NOT predictable inductively.
5. Alternative-background labels (n066_altlabels.csv; N064 local-corner-background reconstruction, otherwise the identical v10
   pipeline): the ensemble scores .6652 against local-bg labels alone but .6996 against the CONSENSUS z(v10)+z(local)
   (b10 .7085, b01 .6908); gains on exactly the targets where the two background models disagree: CD56 +.04/+.03, HLA-DR b01
   +.023, CD14 b01 +.021, grp:CD3/19/56 b01 +.014, grp:CD45 +.005/+.011; CD19 is the exception (local-bg CD19 is near noise,
   v10-local r .24-.40, consensus .31/.29 < v10).
6. Re-used: ensemble asymptote ~.693 (N059 greedy test-selected .6922; N065 6-way .6929); pixel noise negligible (N064 r_max .9999);
   within-FOV r of the ensemble +.040 over pooled r (N059); label count 50 -> 100 % lifts weak targets +.035 (K033).

Bottleneck hypothesis: the remaining gap is LABEL error from the fluorescence background model -- FOV-clustered (LOWESS background)
and partly independent between background models -- on the weak/pooled targets, concentrated in b01 (draw 1 'all' replicate 0).
It acts twice: (a) on EVALUATION labels, which no inductive model can fix (instruments 2-4); (b) on TRAINING labels, which limits how
well the shared true signal is learned on weak targets (instrument 5 shows the predictions already track the background-model-
independent component; K033 shows weak targets are still label-starved). Lever (b) is reachable: train on denoised targets
(consensus over background models / reliability-weighted per FOV) while keeping v10 as the evaluation label.
Expected size: (a) caps GOAL near ~.70-.71 for any inductive model on v10; (b) is at most a few hundredths on CD56/HLA-DR/groups.

What would falsify it:
- Training on consensus (v10 + local-bg) targets and scoring on v10 gives no gain on CD56/HLA-DR/grp targets -> training-label
  noise is not limiting (the model already averages it away) and (b) is closed; the remaining gap is then purely (a) + finite data.
- A model whose b01 grp:CD45 r on 'all' replicate 0 rises toward replicate 1 (.88) without seeing draw-1 labels -> the replicate-0
  deficit is image-reachable, not label error.
- If the consensus gain (instrument 5) came from local-bg labels being smoother (lower variance of error on the bright tail)
  rather than independent error, a third background model (e.g. annulus background) would NOT add further consensus gain.

## Corrections after the Codex diagnosis red-team (codex/diag_rt/diag_redteam.md), reproduced first-hand (n066_rt_repro.py/.csv)
- Consensus gain is real paired information, not "averaging any correlated variable": a noise-matched second label with no extra
  information gives .6722 (reproduced; Codex .6722); FOV bootstrap CI of the gain [.0023, .0116] (Codex, not reproduced).
- The gain lives at the FOV level: adding half of only the FOV-mean of z(local)-z(v10) gives .7005, half of only the cell-centred
  difference .6916 (both reproduced; Codex .7005/.6916) -> consistent with FOV-level (LOWESS) background disagreement.
- NOT identified which label carries the error: a label distorted toward the model's predictions reproduces the consensus number
  exactly (Codex construction; logical point, accepted). Local-bg labels could carry an LF-morphology-aligned bias instead.
- Pooled-target FOV excess in b01 is partly between-slide offset (Codex: grp:CD16 .114 -> .040 after slide conditioning;
  [unverified], not reproduced, does not change the plan). Observable composition does not explain CD19/CD56/CD14/HLA-DR excess.
- The replicate gap (grp:CD45 .88 vs .58) survives label-SD matching (.81 vs .49, Codex) -> not a range artefact; its source unknown.
- WITHDRAWN: "inductive ceiling ~.70-.71" and "training-label noise limits learning" -- neither follows from evaluation-label
  diagnostics. The training-side lever is a hypothesis whose size AND sign need the matched run (N067).
Revised hypothesis: a real FOV-level disagreement between background models exists on weak/pooled targets; whether training on the
consensus helps the v10 objective is the open, directly testable question (N067). The data-quantity reading of b01 (draw 0 has
2.6-4.8x fewer labels for pooled/CD3/CD16) is weakened by C1: r(log label ratio, b10-b01) across targets (outsider check, below).
  C1 (outsider, computed here from attempts.md numbers): r(log(draw-1/draw-0 label count), b10 - b01) across the 11 targets = .18;
  CD45 and CD56 have 2.0x/1.6x more draw-1 labels yet b10 is WORSE (-.100/-.087); CD19 has fewer draw-1 labels yet b10 is better
  -> label count does not explain the direction gap per target; draw-specific test difficulty does at least as well.
- Not augmented so far (checked): training uses +-4 px translations only; rotations/flips were excluded because LED channels are
  directional (bsccm/data.py:42). n066_d4_aniso.py: the anti-diagonal reflection maps every Bottom half-ring/DPC channel onto its
  Right partner (DF_50/60/70 r .90/.90/.74, DPC B->R .91, T->L .93; DF_80 too isotropic to tell, .21); full rings are NOT the sum of
  halves (half/full brightness 1.02-1.13) -> only {identity, anti-diagonal reflection + channel swap} is exact; LED119 has no partner.
