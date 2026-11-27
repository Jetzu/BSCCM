# N093 red-team round 2 (Claude, 2026-11-27)

Scratch: /tmp/n093_rt2/{diag,diag2}.py, diag.json, diag2.json (CPU, full t0 folds).

**NOTHING BREAKS. The claim holds.**

- **(e)** 11/11 fixture tests pass (/tmp/n093_rt2/fixture.log).
- **(a) No F1-bsccm leakage.**
  - The router fits only `train_gs`/`paired_class` (n093_class_match.py:167-171). Held-out hist-row overlap is 0, and the bank never touches the test slide.
  - The full-size router converges in 123/190 of 1000 iterations, so the guard (n093_match.py:128) will not abort the job.
  - Held-out route agreement with LF is .903 (slide 19) and .921 (slide 2): 1023/3563/1111 vs LF 1113/3545/1039.
- **(b) The oracle cannot reach clinical metrics.** `gs_predictions` has no oracle (:92-97), `add_oracles` is called only on the BSCCM path (:207), and `per_run` uses fixed clinical arms (:191, :202-204).
- **(c) Class means are correct.** The bank is split-0 LF-only cells of the training draw (n093_data.py:41). F1, F1-bsccm and both oracles share `student.mu`.
- **F1/F2 unchanged.** The patch only renames `predict` to `predict_with_probabilities` (old route = argmax).
- **(d) Denominators verify** (verify.log). Round-1 counts reproduce: 160/5518/19 and 56/9936/23.

**MINOR 1: "UPPER_BOUND" is not a bound.**
- Full t0 slide 19: F1-bsccm autofl r .287 vs oracle-hard .243 and oracle-soft .269. Marker r is .692 for both.
- README:17 concedes this, but the arm names (:88-89) invite a false "exceeds bound means leak" reading.

**MINOR 2: The BSCCM fraction reference is circular.**
- The only reference is the LF route, which is also F1-bsccm's training target.
- Measured cls3 exists only on draw 0 (3026/4905/1501, not reported). It is absent from paired cells, from slide-2/19 LF-only cells and from all of draw 1.
- So c3 accuracy on draw 1 (holdout-0 bank, holdout-1 test) is unmeasurable.

**Note (pre-existing):** `include_gs=0` drops only paired rows (scripts/train_lf.py:101). LF encoders saw the held-out slides' unpaired LF cells, as in N058. This is not new.
