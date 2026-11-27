# N087 red-team (Claude), 2026-11-24

**Verdict: NOTHING BREAKS. There is no leakage, no scoring difference, and `none` stays byte-identical. Seven MINOR findings.** Scratch: /tmp/n087_rt.

## Checked
- **(a) Leakage.** Decoders fit only on `is_eval==0` rows, which are train_lf's `tr`: one draw, `hist_row<0`. The scaler, the FOV-grouped lambda split, the kNN bank and the networks all see source rows only. AE rows equal the n031 SSL ckpt rows exactly (274,980 / 137,961), with 0 eval rows. targets_v3 matches v10 on batch, slide, fov and split. Running the wrong-fold SSL raises a ValueError.
- **(b) Scoring.** The factored scorer is identical: the diff reproduces the copy. The real reference gives .68674590 / .65918953 / .67296771.
- **(c) Marker prior.** All 49 tests pass in the SIF. The prior is fitted on `y[va]`, which is per-run z with revealed-only stats, the same scale as `y[tr]`. It stays frozen. `none` leaves cfg unchanged.
- **(d) Image AE.** The input is train_lf's eval GPUNorm. Loss covers withheld and finite pixels only. Two real CPU steps ran.
- **(e) Cluster eval.** cls10 exists in draw 0 only. GroupKFold groups on (slide, fov). It ran on the real dump.

## MINOR
1. **decoders.py:86, loss weighting.** The loss is a per-entry masked mean; train_lf uses a per-target mean. autofl is 46% / 44% of entries. Measured on 60k rows, 1 seed, at k=2: GOAL .407 vs .507 with per-target weighting. k=4, k=8, full and MLP stay within .012. The low-k end of the curve is biased down.
2. **decoders.py:21,58, lambda grid.** The grid steps by 100× and the chosen lambda is not logged. CD19 picks the edge value, 100. This can only handicap ridge.
3. **decoders.py:22,210-218, hard-coded reference.** The `RECORDED` assert fires on any other dump that has `P`, such as the D2 bottleneck runs, and aborts the sweep. It fails closed.
4. **decoders.py:71, k ≥ 13.** k ≥ 13 equals `full` because there are only 13 outputs, so the capacity curve is informative only for k ≤ 8.
5. **cluster_eval.py:27, cohort size.** It scores 1,181 of the 10,137 draw-0 cls10 cells; classes 6 and 7 have 9 and 8 cells. All 10,137 are valid, because no b10 encoder saw draw 0. That needs a re-dump. This costs power; it is not leakage.
6. **image_ae.py:126,89,218 vs extract.py:87, masking.** The AE trains on inputs with 25% of patches zeroed, but extraction feeds full images. The brief's plain AE is `--mask-fraction 0`.
7. **sandbox.py:637-640 + validation_loss, validation donor.** It both fits the prior and selects the checkpoint. It also adds joint labels that the N080 `single` control lacks, which confounds the comparison. The test donor is untouched.
