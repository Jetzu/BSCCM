# Codex WRITER brief v2: N054 fixes after red-team (Claude reproduced every P1 below first-hand)

Your v1 delivery is in the parent folder (../n054_trimodal.py etc., ../brief.md = original spec, ../report.md). Write v2 ONLY into this
folder (codex/v2/); leave v1 files untouched. Same rules as v1 (repo + data read-only, scratch /tmp/codex_n054/, CPU smoke only).
Red-team report scripts (read them): /scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/n054rt/c1-c13.py

Reproduced findings and the REQUIRED changes:
P1-1 Modality gap: flow trained from P2(f) of LF cells but run from z_G = E_G(img). On TT6c batch1-holdout test cells cos(z_G, own z_L)
  = .318 +- .233, ||z_G - z_L|| 1.15 vs NN distance .37 (c3.py). Change A1 into two flows, both kept:
  A1-LF  = bridge trained on LF-cell pairs (P2(f), P1(f)) (literal "align the two spaces through LF");
  A1-GS  = bridge trained on training-slide GS pairs (E_G(img_i) -> P1(f_i)), initialised from A1-LF (split 0 train / split 1 select).
  Diagnostics rows (test slide): oracle = A1-LF flow started from P2(f_own); direct MLP z_G -> P1(f_own) regression on the same GS pairs
  (no flow; same capacity); cos(z_G, P2(f_own)) and cos(z_G, z_L) gap stats.
P1-2 A2 decodes from a region D_M never saw. Change: D_M is trained on P1(f) AND leave-one-out E_M rows AND full E_M rows. A2 primary =
  unpaired minibatch-OT flow from z_G of training-slide GS cells (source marginal) to P1(f) of DIFFERENT random non-GS LF cells
  (target marginal of space 1; no correspondences, same-cell matches forbidden). Keep the E_M-row-target variant as secondary A2-EM.
P1-3 A3 MSE averaged over 256 dims -> 0.5 SigLIP has 450-560x the gradient (c11.py). Use (zg - zl).square().sum(-1).mean()
  (= 2 - 2 cos on unit vectors). Keep w_sig in {0, 0.5}.
P1-4 Identical models retrained per config. Restructure: LF-side models (space 1: P1/E_M/D_M; hub: P_L/H/E_M; completion) do not depend
  on the GS holdout -> train ONCE. Per holdout train ONCE: space 2 (P2 + E_G SigLIP); hub GS encoders (w_sig 0, 0.5). Flows (A1-LF,
  A1-GS, A2, A2-EM; sigma 0 and .1) all reuse the same frozen space 1 / space 2 / E_G -> paired comparison.
P1-5 C2 has no labels-only baseline (other pooled groups alone predict the held-out group at r .84/.78/.75 on split 2, c10.py). Add for
  C1 and C2: (c) labels-only (other observed entries only; linear AND the same MLP), (d) ridge on [f, other observed entries], next to
  (a) LF only and (b) fused shared-space model. The hidden-information claim = (b) > max(a, c, d).
P1-6 Not preempt-safe on this cluster (PreemptMode=CANCEL, GraceTime 900 s; SIGTERM to the batch shell only). Use the house wrapper:
  read ~/.claude/skills/work-log/bin/preempt-run.sh and ~/.claude/skills/work-log/templates/preempt.sbatch and
  /home/hpc/v181bb/v181bb10/BSCCM/scripts/gpu_job.sbatch (exec preempt-run.sh -- apptainer exec ... python3 -I $CMD; no srun). Instead of
  77 dependent jobs: ONE driver per unit (python stage runner `--stage all` that runs stages in order and SKIPS any stage whose exported
  output + signature already exists; resumes inside a stage from ckpt.pt). Units: (1) LF unit (space 1, hub LF, completion), (2) holdout
  batch1, (3) holdout batch0 (each: space 2, hub GS x2, all flows, selection on source split 1, eval, clinical for A0 + SELECTED configs
  only). Units 2/3 wait for unit 1 (I will chain them; document it). Python must save on SIGTERM promptly (check STOP flag also inside
  ridge/stat loops or make them cheap to recompute).
Also fix (P2): completion early-stopping/validation loss only on entries masked out of the input; audit fields computed, not constants;
  near-duplicate negatives: rows whose only observed entry is autofl are near-duplicates when |autofl diff| < tau; selection must be
  CONSUMED: write n054_aggregate.py that builds the primary table from the frozen selection (A1-GS sigma, A1-LF sigma, A2 sigma, A3 w_sig
  chosen on source split-1 autofl r only) plus a secondary full-grid table; outputs under versioned dirs runs/n054v2_*; resume checks a
  code signature (sha256 of the script) as well as args; per-stage logs appended, not truncated.
C1 (new): use targets_v5cx_v10.npz for the completion unit (C1, C2, C3): it has autofl observed on all 312,590 single-marker cells (and
  current v10 marker labels; group-8 draw-2 QC entries already masked -- verify). Keep targets_tt6c.npz for the GS arms. For C1 report the
  gain (b vs a, c, d) PER observed marker, because dye crosstalk into the autofl channel could fake a gain (flag markers where (c)
  labels-only alone predicts autofl).
GPU estimate: give one with stated per-stage assumptions; target <= 8 GPU-h total.
Deliver: v2/n054_trimodal.py, v2/n054_clinical.py (+ contrasts), v2/n054_aggregate.py, v2/n054_unit.sbatch (house wrapper), v2/README.md
(exact sbatch commands, unit order), v2/report.md (what changed per finding, smoke results), CPU smoke of every unit on tiny subsets
(include a SIGTERM-mid-stage resume test via the driver). Handback <= 1.5k tokens.
