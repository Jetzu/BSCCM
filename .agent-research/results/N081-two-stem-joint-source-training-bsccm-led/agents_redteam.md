# N081 red-team (Claude) of Codex `--ext-ifc` patch — 2026-11-23

Scope: codex/train_lf.py (+diff, tests), scripts/pack_sbiad452.py. Repro scripts/logs: /tmp/n081_rt/bn_mix.py, /tmp/n081_rt/n074_rerun.log.
Verdict: flag-OFF claim holds; flag-ON claim "BSCCM evaluation unchanged / IFC test donor never enters training" does NOT hold.

## BLOCKER-1 Shared BN running stats mix two sources -> train/eval mismatch for BOTH (train_lf.py:408, 430, 617)
Both forwards run in train mode, so every trunk BN (bn1 + layers 1-4) normalises each source with its own batch stats during training,
but the EMA running stats alternate BSCCM/IFC (IFC last): steady state = 0.474 BSCCM + 0.526 IFC. evaluate() and evaluate_ifc() then use
these mixed stats -> a normalisation the network never trained with. No per-source BN, no stat save/restore.
Repro (real N031 SSL encoder, 96 real b1 train / b0 split-2 eval cells via GPUNorm, 96 real cells from the S-BIAD452 smoke pack, IFC stem
initialised exactly as the patch):
- bn1 batch stats: BSCCM mean|mu| 16.2, mean var 553 vs IFC 0.15 / 1.0 (3 orders of magnitude; train-mode BN gives no pressure to close it).
- BSCCM eval pooled features, mixed vs BSCCM-only BN stats: relL2 0.54, cos 0.88, median per-dim r 0.87 (p10 0.66).
  For scale: the known batch-0/1 domain shift (AdaBN recompute) is only relL2 0.36 / cos 0.94 / r 0.95.
- IFC eval features, mixed vs IFC-only stats: relL2 0.85, median per-dim r 0.13 -> IFC metrics effectively garbage.
Failure scenario: GOAL delta of the IFC arm vs control is dominated by BN-stat corruption (likely negative), not by joint-source learning; IFC
held-out r is meaningless. Fix: domain-specific BN (separate running stats per source; e.g. snapshot/restore BSCCM running buffers around
the IFC forward and keep an IFC buffer set swapped in for evaluate_ifc), or recompute BSCCM stats on BSCCM training data before evaluate().

## MAJOR-1 Real pack donor names break the defaults; the obvious fix leaks the test donor (train_lf.py:266; pack_sbiad452.py:81)
Packer sample = run directory: {sample0, sample1_early, sample1_late, sample2_early, sample2_late} (verified on disk; N080 brief says donor =
sample0/1/2 and N080 code maps with a regex). Harness compares raw strings and tests used synthetic 'sample0/1/2'. README commands
(--ext-test-donor sample2) raise 'IFC test donor missing' on the real pack. A user who then sets --ext-test-donor sample2_late and lists
sample2_early in train passes every assertion while donor 2 is in training. Fix: map run -> donor (regex as N080) and exclude by donor.

## MAJOR-2 Shared reg rows: IFC label scale ~1 vs BSCCM 0.2-0.9, affine starts at 1 (train_lf.py:344, 604)
v10 train-row label SD for single:CD14/CD19/CD3/CD45/CD56 = 0.26/0.20/0.49/0.92/0.61 (b1 sample); IFC labels are per-run z-scores (SD 1).
With scale init 1 and Adam moving it ~lr/step (needs ~8k steps at 5e-4 to reach 5x), the shared rows are pulled toward 2-5x larger outputs
for much of training, conflicting with BSCCM MSE on exactly the 5 overlapping targets; AdamW wd 0.05 also decays `scale` toward 0.
Fix: init scale/offset from training-row SD/mean ratios (BSCCM train rows, IFC train donors), exclude affine from weight decay.

## MINOR
- m1 (e) Test-donor statistics enter training preprocessing: chan_scale (inputs, pack:98) and marker cofactors c (training labels, pack:101)
  are fitted on all cells incl. the test donor. Disclosed in README, but contradicts "never enters training".
- m2 Global clip 2.0 over both sources (train_lf.py:611). At init: BSCCM 1.39, 0.5*IFC 0.31, joint 1.46 -> inactive at init; if it activates
  later the IFC gradient shrinks BSCCM's step (N074 clipped dense separately for this reason). Clip per source for a clean control.
- m3 RNG confound: ifc_stem/ifc_only init and prep_ifc torch.randint draw from the global RNG -> BSCCM shuffle order and GPUNorm shifts differ
  from the matched control; 1-seed deltas include seed noise.
- m4 --enc-lr-mult != 1 puts ifc_stem in the HEAD lr group (train_lf.py:369). --adabn not rejected with --ext-ifc.
- m5 (g) Two graphs alive per step -> ~2x activation memory and ~2x step time (+ per-cell Python D4 loop, train_lf.py:396-401: ~3 kernels x
  256 cells); 4 persistent loaders x --workers 15 = 60 workers. GPU-h estimate for the approval rule must be doubled.
- (d) D4 + circular roll on IFC is fine (zero-padded background); BSCCM prep/fwd/evaluate unchanged.
- (c) New params are in the optimizer (created before AdamW); total_steps/sched.step unchanged.

## Verified OK
- (f) Default-off: diff applies to scripts/train_lf.py (sha256 4cf89b09...) and reproduces codex/train_lf.py byte-for-byte; N074 regression
  suite re-run first-hand (ORIGINAL = live repo file): 7/7 OK, exit 0, incl. full-loop weights/eval_preds equality.
- Cell-level donor exclusion is correct for exact string names (test-donor rows never read in training; poisoned-input test passes).
