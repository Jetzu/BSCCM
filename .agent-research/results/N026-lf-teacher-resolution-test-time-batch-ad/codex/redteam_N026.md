Verdict: **C1's reported scores reproduce, but the evidence for a specifically multi-scale benefit is incomplete; C2's dual-zoom collapse is confounded by shared BatchNorm running statistics.** No blocker was found in row alignment or common-11 scoring. Do not treat this as a verified architectural win or a clean rejection of dual scale.

This was a read-only review of the live tree and external run artifacts. All audit outputs are inside this `codex/` directory. No training, SLURM, GPU, dependency installation, notebook changes, or commits were performed. The requested Apptainer command failed with `fuse: device not found`; the explicitly permitted `triad/bin/python` fallback was used. CPU calculations use one PyTorch/BLAS thread; the forward diagnostic is capped at 540 seconds. No scratch directory was needed.

Reproduction environment, from this directory:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
"$PY" audit_n026.py > audit_n026.txt 2>&1
"$PY" audit_controls.py > audit_controls.txt
"$PY" audit_scope.py > ssl_scope.txt
timeout 540 "$PY" audit_bn.py > audit_bn.txt 2>&1
```

The scripts only read source/data/checkpoints; they write small local evidence files. `audit_controls.txt` records SHA256 hashes of the reviewed production scripts. Model state hashes and all run configurations are in `audit_metrics.json`; measured scores are also in `audit_metrics.csv`.

1. **MAJOR — The omitted zoom+zoom control substantially weakens “beyond plain ensembling.”**

   `../ensemble_scoring.py:8–12` includes base+base and base+zoom, but omits zoom+zoom despite having both zoom seeds. The omitted equal-cost two-model ensemble scores **0.6491005993**, versus **0.6512268983 / 0.6517534288** for the reported mixed ensembles. The additional gain is consequently **0.002126299 / 0.002652829**, rather than the **0.007406650 / 0.007933181** gain against base+base. The mixed average exceeds zoom+zoom by only **0.002389564**. This does not erase the measured record, but substantially changes the mechanism claim.

   Evidence command: `"$PY" audit_n026.py`; output `audit_n026.txt:41–46`:

   ```text
   multiscale_s0 b10=0.6545958293 b01=0.6478579673 GOAL=0.6512268983
   multiscale_s1 b10=0.6573654758 b01=0.6461413819 GOAL=0.6517534288
   same_base    b10=0.6462679024 b01=0.6413725940 GOAL=0.6438202482
   same_zoom    b10=0.6566955959 b01=0.6415056028 GOAL=0.6491005993
   ```

   Evidence command: `"$PY" audit_controls.py`; `audit_controls.txt:1–2`:

   ```text
   SINGLE goal means base 0.6403523142352877 zoom 0.6421812505614444
   MULTISCALE advantage over same_zoom [0.002126298978427088, 0.0026528294997242297]
   ```

   Concrete fix: add zoom+zoom to the reported controls and describe the scale-specific increment separately from the benefit of using two models. Compare predeclared mixed and same-scale ensembles on additional seeds with matched training/inference budgets. The current result supports an observed benchmark improvement, not a settled explanation that two scales are necessary.

2. **MAJOR — Dual zoom mixes two view distributions in every BatchNorm buffer; the collapse cannot be called a clean architectural failure.**

   `scripts/train_lf.py:76–82` calls the same encoder sequentially on full and zoomed images. During training, BatchNorm normalizes each view using that view's batch moments, while both calls update the same running moments. At evaluation (`:87–91`), both branches use those shared moments. These cannot generally represent both view distributions. With momentum 0.1 and stationary moments, the running mean after each pair weights the full view by 0.9/1.9 and the second view by 1/1.9. Averaged within-view variances also omit the between-view mean contribution of a true mixture. Simply reversing the calls does not fix the train/evaluation mismatch.

   Evidence command: `timeout 540 "$PY" audit_bn.py`. The diagnostic executes the actual production `fwd` AST on a tiny encoder, disabling only CUDA autocast. `audit_bn.txt:1–4`:

   ```text
   full mean=1 zoom mean=4 BN running mean=[0.4900000095,...] BN count=2
   head_shapes [(4,13),(4,3),(4,10),(4,96)]
   SSL ones channel running mean [0.1899999976] (expected .19 after two updates)
   STRICT torchvision checkpoint load b10 <All keys matched successfully>
   BN update counts b10 actual unique [85500] SSL unique [64480]
   expected SSL+2*steps 85500
   ```

   The actual b01 checkpoint similarly has 74460 updates = 64480 inherited SSL updates + 2×4990 fine-tuning steps. This is the behavior of the saved runs, not just a hypothetical path. The 4096-feature production regression head is dimensionally consistent; cropping/interpolating the 24 appended all-ones SSL mask channels preserves ones. No head-width or SSL-mask crop error was found.

   A completed CPU diagnostic strictly loads the encoder state into torchvision's matching ResNet-50 and checks against saved GPU predictions on fixed, slide-stratified samples. It then recalibrates **separate** full/zoom BN buffers using only source-batch split-0 images, without changing weights or using label values. Evidence: `audit_bn.txt:5–11`, `audit_bn_metrics.json`:

   | Direction | Sample eval cells | Source calibration cells | Original CPU r | Separate-view source-BN r | Change |
   |---|---:|---:|---:|---:|---:|
   | b1→0 | 480 | 384 | .408542 | .521614 | +.113072 |
   | b0→1 | 576 | 320 | .517437 | .596753 | +.079316 |

   The CPU predictions correlate .99999764 / .99999824 with saved GPU predictions on these same cells; their sample scores differ by only .00001805 / .00000871. The CPU LF normalization reference agrees with production normalization to max absolute error 9.54e-7 on two checked cells. Computation completed in 74.3 + 80.1 seconds, with no checkpoint writes.

   This is a small-sample diagnosis, **not a replacement GOAL score**. Samples have 48 labelled cells per single-marker target and 48/96 per pooled target; full-population scores are not estimated by these tables. Recalibration changes both the moment estimates and their separation by view, so this does not isolate which accounts for how much of the rescue. It demonstrates a material normalization confound and does not prove that all of the collapse is a BN bug or that corrected dual scale would beat the baseline.

   Concrete fix: maintain view-specific BN buffers while sharing convolutional weights; alternatively train with a normalization scheme whose inference rule matches training. Jointly batching views would require training with joint moments too. Recheck full held-out evaluation after fixing/calibrating the normalization path. Until then report “this shared-BN implementation scored .5074,” not “dual scale is intrinsically worse.”

3. **MAJOR — Two seeds do not establish the small advantage over the stronger control; the selection benchmark has also been reused.**

   `audit_n026.txt:47–48` estimates single-run sample SD **0.002711548** for baseline and **0.002597770** for zoom, pooled **0.002655269**. Each arm has only two seeds, so this variance estimate is unstable. The paired zoom-minus-baseline effects change sign: **−0.001925319 / +0.005583191**; their sample SD is **0.005309318** (`audit_controls.txt:3`). Thus “zoom alone is flat” describes the observed screen; it is not evidence of equivalence.

   The additional mixed-over-zoom-ensemble gain is about the estimated single-run SD. That is a warning about power, **not** a valid significance test for an ensemble contrast. Only one two-seed same-scale ensemble exists per scale, and both reported mixed contrasts reuse it. Their small observed mixed-score SD of **0.000372313** (`audit_controls.txt:5`) cannot establish a population variance from two observations.

   Diagnostic cross-seed mixed pairings produce **0.654749160 / 0.649333800** (`audit_n026.txt:42,44`). These reuse existing models and are **not extra independent replicates**; the range shows sensitivity to model pairing. Do not promote the best diagnostic pairing as another headline.

   The N026 item plan names multiple resolution/AdaBN arms, and the same split-2 metric is computed every epoch (`scripts/train_lf.py:158`). Final-epoch predictions are used consistently, so no best-epoch cherry-picking was found. However, architecture/ensemble selection and replication occur on the repeatedly reused GOAL evaluation batches. Two fresh fine-tuning seeds do not supply a new isolation-level test. The notebook's `ORCHESTRATION.md` also calls for at least three seeds before a new headline is verified.

   Concrete fix: predeclare the comparison including zoom+zoom, add independent seed pairs, and reserve a fresh isolation or untouched outer evaluation for a generalization claim. Retain the current .6512/.6518 as development-benchmark measurements with an uncertainty caveat. Do not bootstrap individual cells as a substitute for training-seed or isolation replication.

4. **MAJOR, inherited — “NEW blood isolation” needs a transductive-SSL qualification.**

   This is not new label leakage from zoom, and it affects baseline and zoom equally. Nevertheless, the common initializer has already trained on every evaluated image, including target split-2 rows. Evidence command: `"$PY" audit_scope.py`; `ssl_scope.txt:1–4`:

   ```text
   SSL config ... 'rows': 'all', 'exclude_batch': [], 'seed': 0
   SSL rows 412941 epoch 40
   b10 eval rows 13113 in SSL training 13113
   b01 eval rows 26322 in SSL training 26322
   ```

   `scripts/train_ssl_lf.py:159–170` constructs those rows, and `:238–249` forwards them during SSL training. Both fine-tuning seed replicates also share this exact seed-0 pretrained encoder; they are distinct **fine-tuning** replicates, not independent SSL pretraining or isolation replicates.

   Concrete fix: label the result “batch-held-out supervised regression with transductive SSL initialization.” For strictly unseen-isolation performance, pretrain without the destination batch, and keep evaluation images out of any adaptation cohort. This limitation does not invalidate the measured mixed-versus-baseline difference under the current frozen benchmark.

5. **MINOR — AdaBN's “labels never read” comment is literally false, although no label-value leakage was found.**

   `scripts/train_lf.py:23–24,164` uses mask/class availability to define `has`. `:169` calls `mk(tgt,False)`, whose dataset receives `LAB` (`:43–45`). `bsccm/gpu_norm.py:14` eagerly returns the labels, and the AdaBN loop discards them. The forward call receives only images/phase/global row, runs under `no_grad`, and does not optimize against labels. Adaptation rows are explicitly target-batch split 0, not split 2.

   Evidence commands: `nl -ba scripts/train_lf.py` and `"$PY" audit_n026.py`; `audit_n026.txt:4,7,10,13` reproduces the four baseline AdaBN changes:

   ```text
   -0.009070054  -0.005144465  -0.010832225  -0.005472172
   rows=20000, ev_overlap=0 for each
   ```

   `ssl_scope.txt:5–6` shows that dropping the label-availability filter would change **zero** eligible non-Giemsa split-0 cells in either batch (102097 / 215838 eligible). Thus the filter is not an actual selection leak for these artifacts. The image data path reads LED and phase, not fluorescence.

   Concrete fix: use `RawSet(tgt, labels=None)` and image/metadata-only eligibility, or say “labels are loaded but ignored.” The observed AdaBN degradation is reproduced. The item's further causal statement that flat-field normalization has already removed batch shift is not established by this negative adaptation experiment alone.

6. **MINOR, inherited — Resume flag compatibility is preserved, but input contents are not authenticated.**

   I executed the actual `_cfg` AST (`scripts/train_lf.py:126–127`) and checked every reviewed checkpoint against its reconstructed config and rows. Removing new options only when they equal their old behavior defaults does **not** make zoom/stem/dual checkpoints interchangeable. `audit_n026.txt:49–54`:

   ```text
   RESUME change zoom 2 accepted False
   RESUME change stem nomaxpool accepted False
   RESUME change dual_zoom 2 accepted False
   RESUME change adabn 20000 accepted True
   RESUME change seed 9 accepted False
   RESUME change targets targets_v6.npz accepted False
   ```

   Omitting `adabn` is consistent with it being post-training-only: the saved checkpoint precedes adaptation (`:160–170`), and adapted predictions/results have separate filenames. No incompatible-checkpoint acceptance caused by the new defaults was reproduced.

   The guard still compares argument strings and row indices only (`:132–134`). Changing `Y` or target ordering at the same targets pathname, while preserving eligibility/rows, is undetectable. `audit_controls.txt` executes the actual resume guard with unchanged config/rows and changed label values:

   ```text
   Y=[0.0,1.0] ACCEPTED
   Y=[99.0,-99.0] ACCEPTED
   ```

   Concrete fix: record immutable target-content/name-order digests and a preprocessing/schema version in checkpoint provenance; verify them on resume. This is an existing integrity limitation, not evidence that the reviewed artifacts were overwritten. Exact reproducibility after interruption is also limited because the LF checkpoint does not save/restore RNG state (`:160–161`); no interrupted-run lineage was checked.

The requested scoring and fairness controls otherwise passed:

- `audit_n026.py` asserts identical ordered `G` arrays, no duplicate evaluation rows, the exact expected target-batch split-2 rows (13113 / 26322), checkpoint `tr`/`ev` agreement, disjoint supervised train/evaluation rows, finite masked predictions/labels, and prediction shape `(len(G),13)`. Every ensemble combination passes. Its common-11 set is the first 12 marker targets excluding only `grp:CD123/HLA-DR/CD14`; autofluorescence is excluded. It verifies that the >=1000-training-label criterion in `train_lf.py` selects exactly these 11 targets in these runs. Train-style float32 Pearson and independent float64 `corrcoef` agree with saved metrics to far below 1e-6. Means are equal-weight over targets and then the two directions, as GOAL requires.
- The actual `rescore_targets.py` was also run, writing only `rescore_control.json` here. Reproduce with `PYTHONPATH=/home/hpc/v181bb/v181bb10/BSCCM "$PY" /home/hpc/v181bb/v181bb10/BSCCM/scripts/rescore_targets.py --runs n023_v5_b10 n023_v5_b01 n023_v5_b10_s1 n023_v5_b01_s1 n026_zoom2_b10 n026_zoom2_b01 n026_zoom2_b10_s1 n026_zoom2_b01_s1 n026_dual2_b10 n026_dual2_b01 --targets targets_v5.npz --out rescore_control.json`. `rescore_control.txt:1–10` agrees with the independent audit. The original ensemble script's AST was executed through line 19 (omitting only its external-file write); `ensemble_control.txt:1–8` reproduces its full reported table.
- Configurations match for 10 epochs, lr .0005, batch size 256, ResNet-50, channels/phase, SSL initializer, targets_v5, class weights, no DANN/tail/focal weighting, and training/evaluation rows. The only behavior differences between baseline and zoom are zoom and the explicitly paired fine-tuning seed. Epoch-10 checkpoints and full ten-epoch histories were checked. All 14 reviewed model-state hashes are distinct (`audit_n026.txt:40`, `audit_metrics.json`). Rep1 uses seed-1 baseline **and** seed-1 zoom for each direction; it reuses no rep0 fine-tuned model (`../ensemble_scoring.py:12`). Hashes/configs establish distinct saved states and declared seeds; they do not independently reconstruct launch history.
- Zoom is applied after image normalization by `prep` (`scripts/train_lf.py:48–52`), which is called in training (`:139`), evaluation (`:91`), and AdaBN (`:169`). Hence zoom is applied at evaluation too; it is not train-only augmentation. Its crop is `[32:96,32:96]`, bilinear upsampled to 128×128. No new label-value or target-split-2 training use was found in zoom/AdaBN themselves. The SSL exception is finding 4.
- The other C2 screen numbers reproduce: zoom-only GOAL .640344350 / .644018152; dual .507400623; nomaxpool alone .631829144; zoom+nomaxpool .632452933. The baseline mean is .640352314. These are final-epoch observations, with one seed for stem/dual, not general impossibility results.

Plan for the integrator: first add the missing same-scale control and qualify the claims; separately correct dual-view normalization and reproduce its full evaluation. Then perform predeclared seed/isolation replication. Any code fix would touch `scripts/train_lf.py`; control/reporting changes touch the N026 scoring script, item, and GOAL/knowledge wording only as warranted. I changed none of those live files. The integration agent must update/commit the notebook record; the read-only brief and writable scope prohibit doing that here.

Not checked: full CPU or GPU re-evaluation, retraining, an unseen third isolation, new seeds, FOV-cluster confidence intervals, the complete upstream target-construction pipeline, image near-duplicates, launch/preemption logs, or wandb/cron lineage. The review uses the current targets_v5 artifact and declared run provenance; no claim is made that source-only SSL or a corrected dual model achieves the headline score.
