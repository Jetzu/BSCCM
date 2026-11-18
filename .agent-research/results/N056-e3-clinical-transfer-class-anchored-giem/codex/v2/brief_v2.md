# Codex WRITER brief v2: N056 fixes after Claude red-team (P1s reproduced first-hand)

v2 starts as a copy of your v1 scripts in THIS folder (codex/v2/); edit here; leave v1 untouched. Same rules as v1. Red-team
scratch (read its scripts/outputs): /scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/n056rt/ (c1-c7.py, hub_*.json).

P1-1 no noise floor (reproduced: N032 student-seed spread inside one teacher on MLL23 fine up to .07, t2 .443/.512). Train H with
  2 student seeds per teacher (t0, t1, t2). Pre-register in report.md BEFORE any clinical number: H-vs-T claim only if the
  teacher-paired mean difference exceeds the max within-teacher seed range of either arm; otherwise "no difference".
  T: use all existing seeds (n029_tt6c_s0..s2 for t0, n032 t1/t2 s0..s1).
P1-2 anchor strength differs (reproduced from logs/bsccm_gsanc_1294457.out: T loss sig 3.4 + dist .22 vs .5 x anc ~.08; H latent
  MSE .085-.20 -> the same .5 CE is ~30-60 % of H's objective). Fix: (a) PRIMARY H-vs-T comparison = contrasts that do NOT
  involve the anchor classes (PBC: eosinophil, ig, basophil, erythroblast contrasts; MLL23: all except segneut>lym CD16,
  mono>lym+neut CD14, lym>neut CD3 -- list them explicitly in report.md); (b) add arm H_matched: anchor weight chosen so that the
  anchor-CE gradient-norm share on the shared trainable ViT blocks over the first 200 steps equals T's share measured with
  T's recipe on the same batches (log per-term gradient norms for both; compute T's share with the train_gs_bind loss terms
  reimplemented or measured by a short instrumented run; 1 seed x 3 teachers). Log per-term gradient norms in all H runs.
P1-3 OT attribution: add trivial alignment controls on the same cached embeddings: mean-shift (per-dimension mean match), CORAL
  (mean + covariance whitening/colouring), kNN-to-source (replace by mean of k=10 source neighbours). Fix two unbalanced tau values a
  priori (e.g. .91 and .3; no selection on clinical numbers) and report both. Keep CCOT. Note the MLL23 fit frame shares the eval
  class composition (transductive; label as such).
P1-4 RQ3 circular under OT: report |fraction - pi_c| (transfer drift) and |pi_c - textbook| (BSCCM calibration) separately; RQ3
  interpreted for raw (non-aligned) arms only; OT-arm RQ3 rows labelled "not interpretable (shrinkage toward source)".
P2-1 hub recipe: use the N054 production recipe (lf_lr 1e-3, batch 256, 2500 steps, K019 group-8 draw-2 mask) for every teacher.
P2-3 storage: default run root /hnvme/workspace/v181bb10-bsccm/runs/n056v2_* (NOT /home: quota); /hnvme is inode-tight -> write
  each cache as ONE file (concatenate and delete per-batch .parts immediately; never keep both).
P2-4 exclude the PBC cross-half exact/near duplicates found by the red-team (EO_225902 = BNE_191112 etc.; reproduce the list with
  CRC32 + size and thumbnail cosine >= .995) from both halves; record the excluded list.
P2-6 OT fit and apply on the same representation (both per-view, or both on the 8-view mean).
P2-7 H data loading: DataLoader with 12 workers like T.
P2-2 note in report: H reads both teachers' features; do not compare across-teacher SDs between arms.
Update README (exact commands; GPU estimate, target <= 6 GPU-h), report.md (per finding: change + evidence), rerun the CPU smoke
(incl. SIGTERM resume). Handback <= 1k tokens.
