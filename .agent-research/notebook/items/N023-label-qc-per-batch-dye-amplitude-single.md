# N023 · Label QC per batch: dye amplitude (single-antibody spectra) + fluorescence-independent LF-class pattern per (slide,target); mask unrecoverable labels -> targets_v5; GOAL re-run

type: exp · date: 2026-11-09 · status: done
goal_effect: improved · scale: module · value: 0.6404

## Goal
Find out why the batch-held-out GOAL is stuck near .57 and why grp:CD123/HLA-DR/CD14 and autofl come out NEGATIVE on the held-out batch (N022: -.08 / -.08 vs FOV +.28 / +.35). Hypothesis: label-side, not model-side.

## Plan
1. Fluorescence-independent QC: lym/gran/mono from LF features only (logistic on frozen LF-teacher features, batch-0 gated cells, held-out-FOV acc .945); per (slide, target) class means + Cohen's d of a canonical contrast fixed in scripts/label_qc.py BEFORE reading per-slide numbers.
2. Physics: per-batch dye spectra from single-antibody slides (raw 6-channel shading-corrected).
3. Fix labels only (inputs keep artefacts): targets_v5. Metric-only control = re-score v4-trained models on v5 labels; training effect = retrain on v5 (same config as N014 best). Kill: v5-trained <= v4-trained re-scored on v5 + 2x noise -> label fix is evaluation-only.

## Changes
- scripts/label_qc.py, scripts/dye_spectra.py, scripts/label_subset_check.py (QC); scripts/make_targets_v5.py (targets_v5.npz); scripts/rescore_targets.py (metric-only control); train_lf.py: r_mean_markers_valid / n_markers_valid (mean over targets with >= 1000 training labels).
- targets_v5 = v4 + (a) grp:CD123/HLA-DR/CD14 MASKED on batch-1 'all' (27,074 cells); (b) per-slide robust z (median / IQR on split-0 rows, then batch z) where a batch has several slides for one target: grp:* on batch-1 'all' rep0/rep1, autofl on all 19 slides. Single-antibody targets unchanged (one slide per batch).

## Evidence
- QC table (label_qc.csv; classes from LF image features): grp:CD123/HLA-DR/CD14 canonical mono>lym d = +1.21 (batch 0) vs -0.08 / -0.28 (batch-1 rep1 / rep0). SSL-only control features (n007, never fluorescence-supervised): +1.43 / -0.08 / -0.26 (reproduced; red-team F3, Codex also +1.49/-.14/-.30 and, with non-A fluorescence gates as classes, +1.21/-.44/-.61). Provenance caveat: the default classes come from a teacher trained on v4 fluorescence labels; batch-1 classes have no independent annotation; N023 was motivated by observed negative held-out r (outcome-informed revision).
- Crosstalk control (dye_spectra.txt, reproduced red-team F2): released pooled A value on slides WITHOUT the A antibodies, q90: batch 1 CD16-only 87.8 > batch-1 'all' 41.9 (CD45-only 24.4, unstained 14.9); batch 0 CD16-only 32.7 < 'all' 75.1. => in batch 1, CD16 crosstalk alone exceeds the whole pooled-A signal; in batch 0 the real signal dominates. Masking = conservative exclusion of an assay that fails QC, NOT proof that the cells carry no marker information.
- Dye estimates (gate-based, clipped; population-dependent, not calibrated brightness): b1/b0 norm ratio HLA-DR .29, CD14 .41, CD123 .68, CD45 .23, CD16 .59, CD19 .86, CD56 .71, CD3 .82. CORRECTION: the first estimator (top-10 % by unmixed value) gave .19/.36/.28 and claimed CD56 nearly absent in batch 0 and a "4 %" share -- both withdrawn (estimator artefacts / never computed; red-team F2).
- Slide offsets within a batch: grp:CD16 gran mean 1.08 (rep1) vs .09 (rep0) -> after v5 per-slide z .66/.72. autofl: per-slide positive-affine z (within-slide order and Pearson unchanged); estimand = within-slide standardised autofl. Class relations of autofl differ between co-stained slides (e.g. batch-1 CD45 slide lym>gran, CD14 slide lym<gran; red-team F4, visible in label_qc.csv) -> stained-slide autofl is contaminated by the co-stain; marginal normalisation is not biological validation. GS truth (unstained slides, within-slide Pearson) is unaffected.
- Weak but real single labels (subset_check.txt): CD3 .72/.80, CD19 .16/.17, CD56 .11/.11 (dye SNR 5.1/4.3: present in both batches), CD14 .87/.72, HLA-DR .46/.16 (batch-1 HLA-DR weak; gate-based dye SNR 14.1 vs 1.7), CD45 .72/.98 (null .05) -> kept, flagged.
- METRIC-ONLY control (rescore_v4_models.json, protocol/training rows from result.json): N014 SSL-init r50: legacy 12-target v4 .5742; common-11 on v4 labels .6287; common-11 on v5 labels .6343 (b1to0 .6290 / b0to1 .6397). => 90 % of the apparent .060 jump is dropping the failed target; .0056 is label normalisation (b0to1 only). Re-based yardstick, NOT progress. Other models on common-11 v5: N014 r50 .6252, N022 base_b10 .6343 (b1to0 only), tail05 .6306, tail10 .6320.
- GOAL re-registered (goal.sh init --force --since N023): common-11 on v5, best .6343 (= N014 re-scored), noise .002 (N013 seed pair on v5 common-11: .6119 vs .6136, rescore_v5_noise_n013.json), target .67 (proposal: half the FOV-vs-batch gap). Legacy GOAL file kept in results (GOAL_legacy_v4_12.md); legacy metric unresolved.
- Red-team: Codex (codex/redteam_N023.md): target construction reproduced EXACTLY (max err 0; only intended entries change). F1 blocker (metric framing) -> fixed by the common-11 re-base above; F2/F3/F4 reproduced and wording corrected; F5 (GS calibration overlap: v5 per-slide stats use split-0 rows incl. 12.5k GS eval cells; affine -> GS Pearson unaffected; train_gs_bind still uses v4) -> documented, to address before migrating the GS chain; F6 (rescorer guessed protocol from run name) -> fixed (reads result.json).
- Training effect (goal_runs_v5.json, goal_common11_v5.csv; plot .agent-research/plots/goal-common11-v5_2026-11-09_2107.png): SSL(aug)-init r50, 10 ep, lr 5e-4; all scored on common-11 v5 labels. Trained on v4: .6343 (N014 b1to0 + b0to1) / .6348 (N022 base_b10 + n023_v4_b01_s1, job 1288894); trained on v5: .6423 (jobs 1288695/96) / .6384 (seed 1, jobs 1288892/93) -> mean .6346 vs .6404 (+.006; both v5 seeds above both v4 seeds). Per marker (v5-v4 means): b1to0 grp:CD3/CD19/CD56 +.097, grp:CD16 +.051, others within +-.01 except single:CD19 -.034; b0to1 all within +-.012 (batch-0 training labels unchanged except autofl) -> the gain sits exactly where the training labels were repaired (batch-1 'all' slides). LF autofl on the held-out batch: b1to0 -.08 (v4) -> +.28 (v5 seed 0).

## Outcome
IMPROVED (re-based GOAL common-11 on v5): .6404 (2-seed mean) vs .6346 seed-matched v4-trained control and .6343 baseline. Two parts, reported separately: (1) yardstick re-base (legacy 12-target .5742 -> common-11 .6343 with the SAME model; 90 % from dropping the failed target) -- not progress; (2) training on repaired labels +.006 on the new yardstick -- real, mechanistically localised. Legacy 12-target v4 GOAL unresolved. grp:CD123/HLA-DR/CD14 cannot be learned from batch 1 (b1to0 A r -.21/-.13 with an untrained head).
<!-- on close: bin/effect.sh N023 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N023-slug/ + external absolute paths -->

## Next
v5 r18 teacher for the Giemsa chain (N024, running); autofl from unstained slides only; reliability-weighted labels (HLA-DR batch 1); GS chain calibration disjoint from GS eval rows before any scale-dependent GS claim (red-team F5).
