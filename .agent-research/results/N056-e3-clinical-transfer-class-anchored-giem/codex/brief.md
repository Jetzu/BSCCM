# Codex WRITER brief: N056 (E3) -- clinical transfer of Giemsa->marker students to real smears (PBC, MLL23)

Repo /home/hpc/v181bb/v181bb10/BSCCM READ-ONLY; deliver into this folder only. Data /hnvme/workspace/v181bb10-bsccm read-only.
Scratch /tmp/codex_n056/. No GPU/apptainer in your sandbox: deliver code + CPU smoke on tiny subsets; Claude runs GPU jobs.

## Context (read first)
- .agent-research/notebook/KNOWLEDGE.md: K016 (class-anchored alignment N021: anchors neutrophil/lymphocyte/monocyte on PBC half A
  transfer lineage markers to PBC half B, finer biology not robust, DANN harmful), K025 (clinical numbers are teacher-run dependent;
  report means over >= 3 teacher runs; robust vs failing contrasts listed), K022, K038 (N054: unanchored cross-slide students are at
  chance on PBC/MLL23; LF hub latent distillation ~= TT6c output distillation on same-cell autofl).
- Code: scripts/train_gs_bind.py (TT6c student; --target-zip PBC --anchor-ce --target-half A = the existing class-anchored recipe
  used in N029/N032 with holdout fov), scripts/apply_gs_student.py (pre-registered PBC + MLL23 contrasts and preprocessing),
  N054 v3 hub code: .agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v3/n054_trimodal.py (A3 hub:
  LF projection P_L + marker head H on frozen teacher features; GS encoder regresses the frozen hub latent, loss (zg-zl)^2 summed).
  Runs: runs/n029_tt6c_s0..s2, runs/n032_tt6c_t1_s0.. (anchored TT6c students, holdout fov) and their teachers runs/n029_r18_v4c_fov,
  runs/n029_r18_v6c_fov, runs/n032_r18_v4c_fov_t1/_t2 (+ matching v6c teachers -- find them).

## Question
Which Giemsa student transfers markers best to clinical smears from other labs/scanners, and does unpaired distribution alignment
(OT) to the clinical domain help beyond class anchoring? Also RQ3: can the student recover per-class marker-POSITIVE FRACTIONS?

## Arms (same teacher pair per comparison = paired; holdout fov as N032; PBC half A for anchoring only, half B + MLL23 for eval)
- T  : TT6c anchored student (existing recipe; re-use the N032 checkpoints where they exist, else retrain with identical args).
- H  : hub-anchored student = N054 A3 latent distillation (GS encoder -> frozen hub latent, markers via the hub head) + the SAME
       PBC class-anchor CE (same anchors, same weight .5, same images) -- implement in a new script, reuse train_gs_bind's anchor
       loader/augmentation verbatim.
- T+OT / H+OT : post-hoc unbalanced entropic OT (POT-free torch implementation; marginal relaxation tau) that maps clinical-smear
       student embeddings onto the BSCCM-Giemsa embedding distribution (source BSCCM training cells), barycentric projection,
       then the marker head. Fit on PBC half A / an MLL23 fit split (images only, no labels); eval on PBC half B / MLL23 eval split.
       Also a class-conditional variant that only uses the 3 anchor classes' labels on half A (as K016) if cheap.
Teacher runs: 3 teacher pairs (t0, t1, t2) x 1 student seed for T and H (K025: report means over teacher runs); OT arms are
post-hoc (no extra training).
Metrics: the 10 PBC + 15 MLL23 pre-registered contrast AUROCs (apply_gs_student.py definitions verbatim), mean over teacher runs,
robust/failing breakdown as K025; BSCCM same-cell autofl r (sanity, should stay ~.56/.85 within-slide as K025); RQ3 proportions:
per clinical class, fraction of cells predicted marker-positive (threshold = BSCCM training-cell quantile matching the marker's
positive rate in BSCCM stained cells of that class; define and pre-register the rule in report.md BEFORE looking at clinical
numbers) vs the textbook expectation table you write down first (e.g. neutrophil CD16+ ~ 95 %, lymphocyte CD3+ ~ 70 %, monocyte
CD14+ ~ 90 %, eosinophil CD16- ; cite sources) -> absolute error per class x marker.
Deliver: n056_hub_student.py, n056_ot_align.py, n056_eval.py (+ contrasts reused verbatim), n056_unit.sbatch using the house
preempt wrapper (read ~/.claude/skills/work-log/bin/preempt-run.sh; copy the pattern of
.agent-research/results/N054-tri-modal-latent-user-two-space-design-l/codex/v3/n054_unit.sbatch: no --mem, python3 -I -B, exit
143 on SIGTERM checkpoint), README.md with exact commands and a GPU-hour estimate (target <= 6 GPU-h), report.md, CPU smoke log.
Rules: test images (PBC half B, MLL23 eval split) never used for any fit or selection; no tuning on clinical metrics; the textbook
proportion table and thresholds are fixed before any clinical prediction is read. Handback <= 1.2k tokens.
