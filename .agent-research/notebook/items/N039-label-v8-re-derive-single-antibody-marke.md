# N039 · Label v8: re-derive single-antibody marker labels with the authors' PER-CELL background (not background_lowess) + draw-1 own-dye QC; pre-registered label QC; GOAL recipe retrain scored on frozen v5 and on v8; proper in-distribution ceiling (r50 FOV on v5, data-matched)

type: exp · date: 2026-11-12 · status: done
goal_effect: flat · scale: module · value: void

## Goal
GOAL attempt + eval repair (N038 diagnosis rev. 2). The authors' unmixed values use a smoothed background (background_lowess);
in the dim dye channel shared by CD3/CD19/CD56 its error gives 34-49 % FOV-level label variance no LF model can predict (within-FOV
GOAL .7211 vs pooled .6527). v8 = single-antibody labels re-derived from total_raw - per-cell `background` (/shading), unmixed with
source-split-0-fitted spectra (dye spectrum from the authors' positive-gate examples, autofluorescence basis from same-draw
unstained cells), then the existing v3->v5 calibration and the N036 cross-fitted position detrend; grp/autofl targets unchanged.

## Plan
Stage 1 (CPU, 0 GPU-h) label QC, PRE-REGISTERED, checked before any model is scored on v8: per single target x draw (a) FOV-level
excess share <= .05; (b) CD19 positive mixture component 1-6 % with d' >= 3 in both draws; (c) r(v8, v5) >= .95 for CD16/CD45/CD14;
(d) draw-1 lymphocyte stratum: r(label, own dye) >= 0 for HLA-DR/CD19/CD14 (else flag/mask that stratum, as N023). Fail -> stop, record.
Stage 2 (GPU) (i) clean in-distribution ceiling: r50 SSL-init FOV split on v5 (and --max-train matched to b01's 128k), scored on the
GOAL cells; (ii) GOAL recipe (N031 inductive base + zoom2, both directions) trained on v8, scored on the FROZEN v5 GOAL and on v8.
Expected: v5 GOAL +0 to +.005 (eval-label cap unchanged); v8-scored ~.70-.72 = re-base, NOT progress; user decides the re-base.
Kill: QC fails -> v8 not used. Compute: ceiling 2 x ~.5 GPU-h + retrain ~1.3 GPU-h, 1 H100 each, preempt.
From review N038 (results/N038-review-lf-mk-common-11-batch-held-out-go/ideas.md).
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Stage 2(i) ceiling jobs (results/N039-label-v8-re-derive-single-antibody-marke/jobs.txt): 1313148 n039_ceil_fov_v5_mt0, 1313149 n039_ceil_fov_v5_mt128000 (r50, n007 SSL init = transductive, acceptable for a ceiling measurement only; FOV split, v5 labels); 1 H100 each, preempt. Stage 1 label v8: Codex writer (codex/brief.md) running.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->
- Stage 1 (Codex writer, results/N039-label-v8-re-derive-single-antibody-marke/codex/report.md, qc_v8.md/csv, unmixing_fidelity.csv):
  the authors' unmixing is recovered as a clipped 6-channel affine map of (total_raw - background_lowess)/shading per marker x draw
  (fidelity r >= .998 on all clean split-0 cells; split-2 validation r >= .9989); v8 applies the same maps to the PER-CELL background
  and runs the existing v3 -> v4 xfit -> v5 code (BSCCM_RAW=v8; candidate targets_v8cx.npz kept in /tmp only, SHA-256 in
  build_manifest.json). PRE-REGISTERED QC = FAIL: (a) FOV excess <= .05: 15/16 pass, CD14 draw 0 .060 (v5 itself .059 -- the
  criterion did not allow for inherited excess); (b) CD19 positive mode: draw 0 4.3 %, d' 17.3 PASS, draw 1 2.8 %, d' .02 FAIL;
  (c) r(v8, v5) >= .95 for CD16/CD45/CD14: CD14 draw 1 .925 FAIL; (d) draw-1 lymphocyte own-dye sign: unassessable (no same-cell
  gate on single-antibody slides; proxy CD19 -.14 FAIL).
  Claude first-hand check (verify_qc_a_c.py on the /tmp candidate) reproduces (a) and (c) exactly: CD19 FOV excess v8 -.002/.016
  vs v5 .509/.333, CD56 .038/.009 vs .458/.557, CD3 .005/.002 vs .342/.295; r(v8,v5) CD19 .53/.69, CD56 .56/.48, CD3 .70/.68,
  CD14 .985/.925, HLA-DR .988/.885.
  -> Per the pre-registration v8 is NOT used; stage 2(ii) (GOAL retrain on v8) is cancelled. The mechanism is confirmed (per-cell
  background removes the FOV-level offsets of the dim-dye targets), but v8 changes CD19/CD56/CD3 labels so much (r .48-.70 with v5)
  that it is a different label, and draw-1 CD19 has no positive population in either version. Codex code/diffs not landed.
- Stage 2(i) clean ceiling (n039_ceiling.py -> n039_ceiling_per_target.csv; jobs 1313148 37 min, 1313149 15 min): r50 SSL(n007) FOV
  split trained on v5, scored on the GOAL split-2 cells, common-11 v5: full 318k .6776 (draw 0 .6628 / draw 1 .6924), data-matched
  128k .6585 (.6465 / .6705); held-out N031 base .6425 (.6488 / .6363), ensemble .6527. Within-FOV: full .7462, ensemble .7205.
  -> draw 0 as test (trained on draw 1, 269k): no draw-shift cost (.6488 held-out vs .6465 in-distribution at 128k); draw 1 as test
  (trained on draw 0, 128k): .034 shift cost, concentrated in HLA-DR (.61 vs .51 overall), grp:CD45/CD16/CD3-19-56 (+.05-.08) --
  the draw-1 label-definition differences (autofluorescence-driven single HLA-DR strata; batch-1 'all' slides with CD19 at 35 %;
  two grp:CD45 slides 5x apart). Data quantity: 128k -> 318k +.019 in-distribution.

## Outcome
STAGE 1 QC FAILED -> v8 not used, GOAL retrain cancelled (pre-registered). Mechanism confirmed: per-cell background removes the
FOV-level offsets of the dim-dye targets (CD19 FOV excess .51 -> .00), but v8 is a different label for CD19/CD56/CD3 (r .48-.70 with
v5) and draw-1 CD19 has no positive mode in any version. Clean ceiling: a single model that SEES both draws reaches .678 on the GOAL
cells (data-matched .659); the held-out gap lives in the draw-0 -> draw-1 direction and in draw-1 label-definition differences.
A post-hoc v8b (per-cell background only for CD3/CD19/CD56) needs the user's go (IDEAS.md).
<!-- on close: bin/effect.sh N039 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N039-slug/ + external absolute paths -->

## Next
