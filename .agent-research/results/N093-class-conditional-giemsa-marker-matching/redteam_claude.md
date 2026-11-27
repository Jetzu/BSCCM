# N093 red-team (Claude, 2026-11-27)

Scripts: /tmp/n093_rt/diag.py, diag2.py. Outputs: diag_h1.json, diag2.json (CPU, t0, full folds).

**Code holds up. Run will not break.** Tests (e) pass 8/8.
- (b) Metrics are the harness's own. N058 saved predictions through `harness.score_contrasts` + `macro` (n093_class_match.py:41) reproduce n058_per_run.csv exactly (3 rows checked).
- (a) No leakage:
  - anchors = harness PBC `fit` half (:99);
  - eval IDs equal N058 T, disjoint from anchors (:74-79);
  - held-out cells used only for scoring (:120, :139);
  - t1/t2 replay excludes GS slides; parity error .0021/.0013;
  - LF `c3` head vs gated cls3: acc .955;
  - Sinkhorn converges at full size.
- (d) Mapping matches make_targets_v3.py:50 and harness ANCH/CLASS (n056_eval.py:29). T/H_matched also use PBC-A anchor CE (n056_hub_student.py:119).

**MAJOR 1: GS classifier collapses on BSCCM (N008 again).**
- Training-slide routes (lym/neu/mono): 39/7949/19 (h1), 128/4352/15 (h0).
- Held-out routes: 160/5518/19 vs same-cell LF routes 1113/3545/1039.
- Route agreement .596 vs .622 majority base rate (h0 .579 vs .582). Only 10/1113 LF lymphocytes recovered.
- F1 vs teacher, mean marker r: -.044 (h1), .012 (h0). Autofl r .07 vs teacher .81.
- MLL23 (n=72): 54 route to neutrophil, including 5/6 lymphocytes and 6/6 LGL.
- So held-out agreement/autofl and most MLL23 contrasts measure this collapse (n093_match.py:76, :94).

**MAJOR 2: F2 coordinate encodes domain, not biology.**
- GS support pools BSCCM cells with PBC-A anchors (n093_match.py:77-84). Lym/mono supports: 607/646 and 708/727 anchors.
- Own-centroid rank separates the domains with AUROC 1.00 in every class.
- Mean ranks: anchors .47/.08/.49, BSCCM .97/.59/.99, MLL23 lym/mono .96/.99.
- GS ranks nearly collinear (r .91-.94); LF ranks not (.09/.90/.30). A 1-D "un-PBC-ness" axis is matched to an unrelated LF axis.
- Effect (h1, mean r vs teacher): F2 .004 vs F1-hard -.043. Within-route residual r .075 vs shuffled null -.016. Autofl F2 .045 < F1 .07.
- Expect F2 = F1-hard + domain offset. An F2-F1 gap on MLL23 is an artifact.

**MINOR**
- 3: PBC10 reads out a PBC classifier (PBC-B accuracy .996), not GS→marker transfer.
- 4: Autofl r is re-implemented (:34-38, :146) on all FOVs of the held-out slide (N054 M1). Not comparable to N058 `sanity()` (n056_eval.py:289, split 2).
- 5: Two clinical rows per teacher (one per source slide). MLL23 single frozen view vs T/H 8-view TTA.
