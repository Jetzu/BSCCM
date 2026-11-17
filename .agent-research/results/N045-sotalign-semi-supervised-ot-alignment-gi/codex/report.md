# N045 design and CPU smoke report

The delivery implements SOTAlign on frozen embeddings and completes a bounded real-data CPU smoke. It does **not** establish an alignment improvement: the trained experiment remains for the H100. Every write made by this writer is under `/tmp/codex_n045/`; the BSCCM repository, data and MIT upstream checkout were read only.

## Evidence read before implementation

Read KNOWLEDGE.md (including K001, K005, K006, K011, K015, K016, K021–K025), notebook RULES, N009/N021/N024/N029/N032 item text, N009 `screen_B/screen.json`, N021 `pbc_anchor_auroc.csv`, N032 `transfer_all.csv`, and the implementation/provenance of feature extraction, clean targets and autofluorescence detrending. Inspected upstream `scripts/sotalign.sh`, `matching.py`, `similarity.py`, `klot.py`, `utils.py`, `sail_model.py`, `linear.py`, `loss.py` and MIT license. `README.md` gives executable commands and output schemas.

K001 governs interpretation: no BSCCM Giemsa cell has measured surface markers. This task evaluates exact-cell G–L matching, measured same-cell autofluorescence, and clinical lineage plausibility. G–L prediction agreement is not G–marker ground truth. Autofluorescence cannot rank teachers for surface-marker transfer (K011).

## Mapping audit

| Item | Verified result |
|---|---|
| Global LF / targets rows | 412,941 |
| Giemsa features | 15,712 × 768; pack/embed code writes in hist-row order |
| `targets.hist_row[pack.hist_cells]` | Exactly `arange(15712)` |
| LF hist sidecars | Exact agreement with global target hist-row indices; complete unique Giemsa coverage |
| Draw 0 Giemsa | 10,015 cells, slide id 2, `antibodies_unstained/batch_0/slide_replicate_0` |
| Draw 1 Giemsa | 5,697 cells, slide id 19, `antibodies_unstained/batch_1/slide_replicate_1` |
| Draw 0 paired split 0/1/2 | 8,007 / 994 / 1,014 |
| Draw 1 paired split 0/1/2 | 4,495 / 593 / 609 |
| Distinct paired FOVs, draw 0 split 0/1/2 | 2,219 / 279 / 276 |
| Distinct paired FOVs, draw 1 split 0/1/2 | 1,550 / 195 / 217 |
| PBC | 17,092 × 768; all labels match classes reconstructed from sorted archive paths |
| PBC halves | A even rows / B odd rows, 8,546 each; same split as previous clinical studies |
| Giemsa pair suspects | 567; retained in every test, never filtered by truth/pair quality |

`lf_feats_hist.npy` and the f16 global LF array are **separate extraction passes**. Their max absolute difference is 0.0078125, mean absolute difference 8.59e-8, RMSE 1.35e-5. This is tiny but not exact f16 round-trip identity. The implementation checks the disagreement and consistently indexes **only `lf_feats_all.npy`** for pair training, unpaired LF, marker heads and LF evaluation. Hist sidecars establish the mapping; the extra feature pass is an audit only.

Scale metadata reports PBC median cell fraction .104568, BSCCM .108318, crop side 398 px, scale reference 3922. The scalematched array is exposed as a predeclared alternative, not selected using clinical scores. The full-view Giemsa array is the default for N009 continuity.

## Fitting, isolation and controls

All four protocols use paired split 0 for fitting and source-slide split 1 for checkpoint selection. Cross-slide tests use **all** cells of the opposite stained slide; the source slide's split 2 is unused. Within-slide tests use only split 2 of that slide. FOV consistency is asserted globally, with explicit train/validation/test separation. The default cross-slide training pair counts are therefore smaller than historical N009's 10,015 full-source-slide pairs: 8,007 and 4,495. This is the cost of retaining honest source-slide validation.

Unpaired X is PBC half A only; unpaired Y is a fixed seeded sample of up to 100k non-Giemsa cells from the training draw, split 0. Both draws have at least 100k eligible LF cells (102,097 / 215,838). Test-draw non-Giemsa LF is also excluded from the unpaired pool, even though only exclusion of test Giemsa was explicitly required. No BSCCM held-out-slide Giemsa features enter either unpaired arm. PBC class labels never enter an alignment loss, hyperparameter choice, checkpoint selection or marker-head fit. Half-B features are untouched until evaluation. The fixed-k kNN sanity classifier uses half-A labels **only inside evaluation**, as any supervised kNN evaluation must.

The four requested arms are implemented, plus three controls. Regularized upstream CCA is fitted only on paired training rows. The learned arms use the original `AlignmentLayer` with raw input LayerNorm and linear maps, random initialization, a common seeded paired minibatch schedule, and the same optimizer/update budget. There is no CCA initialization of students. The upstream `MatchingModel` supplies SigLIP/SAIL and KLOT. KLOT's forward value and surrogate gradient are preserved; no nearest-neighbour substitute, guessed pseudo-pairing or differentiated truncated Sinkhorn is introduced.

The upstream module imports heavyweight text/image encoders, although `AlignmentLayer` does not use them. The runner extracts the original `L2Norm` and `AlignmentLayer` class ASTs and executes them unchanged with their real dependencies, loading `linear.py` unchanged. It directly imports `MatchingModel` and CCA. This avoids encoder downloads and dummy modules. Source hashes are recorded. The CCA orientation/centering/normalization is checked against upstream `AlignmentLayer.fit_affinity(method='cca')` to avoid silently swapping Giemsa and LF. Internally X=Giemsa occupies upstream's text slot, Y=LF its image slot; these are names only, not text/image encoder execution.

The LF-only arm needs an X support to define a cross-modal OT plan. It uses randomly sampled **training-pair Giemsa anchors** against extra LF Y, with no additional unpaired Giemsa or PBC. For upstream expanded SAIL, only extra LF supplies negatives: reused X anchors are not appended as duplicate negative Giemsa examples. This is a defined asymmetric ablation, not a claim that a two-marginal OT objective can be computed from Y alone.

`raw_ridge` keeps the raw LF space and learns a standardized raw Giemsa→LF ridge bridge on the same pairs. `raw_giemsa` supplies the PBC raw-feature kNN baseline. A direct cosine between the original 768/512 dimensional spaces is undefined. `procrustes` and `n009_siglip` use separate PCA-whitened modalities fitted **only on paired training features**, unlike N009's older all-row transductive PCA. N009-style refers to the architecture/control, not an exact reproduction of the old optimizer, teacher or leakage-prone preprocessing. The controls have the same available pairs; learned controls have the same update count and common optimizer settings. Closed-form controls naturally have no optimizer steps.

## Hyperparameters and selection

Inherited defaults: Lion, lr 1e-4, betas .9/.99, weight decay 1e-5, cosine schedule, scale 20, bias −10, CCA ridge .1/.1, eig epsilon 1e-6, KLOT weight .001, student/teacher epsilon .05/.01, 100 fixed Sinkhorn iterations, noise zero. Dimension is reduced from 1,024 to 128 and each paired/unpaired batch from the web-scale source sizes to 256. CCA dimension is capped by training-pair rank. Training is 1,500 updates, validation every 100, with the earliest maximal source-FOV R@10 checkpoint. The prescribed budget and defaults were not chosen by looking at test or PBC scores.

Upstream default SAIL includes unpaired negatives. Consequently the main SOTAlign contrast changes both its SAIL negatives and KLOT, and its pair-positive fraction differs from pairs-only under the source's mean-over-logits loss normalization. `--sail-negatives pairs` is a separately documented, smoke-tested control in which SOTAlign adds only KLOT to the identical paired loss. This is the clean contrast for attributing a gain specifically to KLOT. No assertion about which variant is better is made from the smoke.

Marker heads use normalized aligned LF embeddings, a training-draw-only `StandardScaler`, per-target masked Ridge, and alphas 1/10/100/1000 selected by that draw's non-Giemsa split-1 MSE. The pool is at most 100k training / 10k validation cells; missing labels never become zeros. All targets, including unavailable ones, retain a status. The scaler/head never sees Giemsa truth. Giemsa autofluorescence is used only after the complete head/alignment selection.

Primary truth is `targets_v5c.Y[global_index, autofl]` with its mask: position-detrended clean calibration, followed by the v5 per-slide scaling/QC. Clean v3/v4/v5 target generation excluded every Giemsa row from calibration, and there are 5,088 / 6,604 non-Giemsa split-0 autofluorescence calibration cells on the respective histology slides. Test-slide truth preprocessing is an existing target definition, not an N045 fitting step. The separately reported N012 residual is the old within-slide region-blocked cross-fitted spatial residual and predates clean calibration; it is evaluation-only, never a replacement silently mixed with v5c truth.

## CPU smoke results

One alignment seed (0), six steps per learned arm, dimension 32, 32 paired / 32 unpaired per batch. Model fitting/evaluation used 768 unique Giemsa cells, 640 other BSCCM cells, and 256 PBC cells: **1,664 unique cells total**. Each training/validation paired pool has 128 cells. Cross-slide galleries contain 384 cells (R@10 chance .02604); within-slide galleries contain 128 (chance .078125). The separate exhaustive mapping audit reads all mapping indices and compares the two LF extraction passes; it does not expand the model sample budget.

Giemsa→LF exact-cell R@10 (smoke only):

| Method | draw 0→1 | draw 1→0 | FOV draw 0 | FOV draw 1 |
|---|---:|---:|---:|---:|
| raw_ridge | 0.1146 | 0.1224 | 0.3984 | 0.3047 |
| procrustes | 0.0990 | 0.1172 | 0.3828 | 0.3359 |
| cca | 0.1484 | 0.1562 | 0.4688 | 0.4297 |
| siglip_pairs | 0.0391 | 0.0521 | 0.1484 | 0.1172 |
| sotalign | 0.0365 | 0.0521 | 0.1484 | 0.1172 |
| sotalign_lf_only | 0.0391 | 0.0521 | 0.1484 | 0.1250 |
| n009_siglip | 0.0234 | 0.0104 | 0.1016 | 0.0625 |

Transferred autofluorescence Pearson r against clean v5c truth (smoke only):

| Method | draw 0→1 | draw 1→0 | FOV draw 0 | FOV draw 1 |
|---|---:|---:|---:|---:|
| raw_ridge | 0.3596 | 0.2871 | 0.1514 | 0.3946 |
| procrustes | 0.1648 | 0.2868 | -0.0225 | 0.3727 |
| cca | 0.0881 | 0.3092 | 0.0455 | 0.2016 |
| siglip_pairs | 0.2429 | 0.2976 | 0.2865 | 0.1403 |
| sotalign | 0.2420 | 0.2978 | 0.2860 | 0.1401 |
| sotalign_lf_only | 0.2426 | 0.2973 | 0.2863 | 0.1400 |
| n009_siglip | -0.0807 | -0.1173 | 0.1144 | -0.0508 |

PBC-B sanity/selected requested contrasts using draw-0-trained maps (128 PBC-B cells, smoke only):

| Method | 5-NN accuracy | CD16 neut>lym | CD14 mono>all other | CD3 lym>neut |
|---|---:|---:|---:|---:|
| raw_ridge | 0.5000 | 0.7538 | 0.7054 | 0.4811 |
| procrustes | 0.6562 | 0.1288 | 0.6373 | 0.6439 |
| cca | 0.6953 | 0.1894 | 0.4894 | 0.6856 |
| siglip_pairs | 0.6797 | 0.4735 | 0.3901 | 0.2576 |
| sotalign | 0.6797 | 0.4735 | 0.3895 | 0.2576 |
| sotalign_lf_only | 0.6797 | 0.4735 | 0.3901 | 0.2576 |
| n009_siglip | 0.4922 | 0.6818 | 0.5240 | 0.5379 |

Raw Giemsa PBC 5-NN accuracy is .8672. These tiny random subsets, sparse heads and six-step students cannot establish clinical utility or rank the arms. All R@1, chance, median-rank, both truth definitions, all 12 PBC contrasts, both draw directions, within-slide tests, target agreement and missing metrics are retained in `smoke/metrics.csv` (1,037 rows). `comparison.csv` additionally retains 175 historical context rows. PBC metrics for cross-slide and within-slide protocols of the same source draw use the same fitted map; they are not independent replication.

There are 28 unavailable target-head instances: draw-0 `single:CD19` lacks enough validation observations in the tiny label-blind smoke sample (14 arm/protocol instances), and draw-1 `grp:CD123/HLA-DR/CD14` has no usable labels because v5 QC masks that group (14 instances). The latter remains unavailable on a full run too. Missing clinical metrics stay null with reasons and counts. No sign flipping, class balancing by adaptation labels, test-cell filtering or silent target omission was used.

KLOT and gradients are finite. The largest smoke teacher mean-row-marginal error relative to the target row mass is ~3.46% after the fixed 100 iterations. This is recorded, not tuned away: small teacher epsilon and differing class compositions may require attention on source-only validation. The source's fixed-iteration approximate OT plans are not guaranteed converged. `training.json` retains absolute and relative residuals for both teacher and student.

## Verification and operations

Five scientific contracts passed (`contracts.log`): identity/permuted/collapsed retrieval, agreement of CCA affinities with the original upstream injected layer, KLOT finite-difference gradient and zero self-KL, actual source/FOV/PBC/label-pool exclusions, and exact CPU model-weight continuation after a checkpointed interruption. The synthetic alignment trainer in the resume test has no labels/targets attribute. A full completed-run `--resume` passed (`resume.log`). The alternative paired-negative objective passed its real-data smoke (`smoke_pairs_only/`). The H100 launcher passed shell syntax validation, a CPU invocation (`launcher_smoke.log`), failure/success exit-code propagation and actual SIGUSR1 forwarding (`launcher_contracts.log`); real scheduler requeue, H100 numerics, CUDA package setup and GPU wall time cannot be validated in this sandbox.

`last.pt` saves model, optimizer, step, best validation state and history atomically. Pair and unpaired samplers derive independent RNG streams from (seed, step), so unpaired draws never perturb pair scheduling and resumed training replays exactly from its last saved step. There is no stochastic layer in the learned linear architecture. Best checkpoint, metrics and manifests use atomic replacement; an interrupted marker-head stage is recomputed. A signal requests exit at a safe boundary, exit code 75; the Slurm shell waits for the child before requesting requeue. Hard termination can lose progress since the previous checkpoint but not a completed checkpoint. Persistent storage and an unchanged Python/source/input environment are required.

## Interpretation and remaining risks

1. **Encoder holdout scope:** the supplied clean supervised r18 LF encoder excludes all Giemsa cells (`include_gs=0`), and its LF inputs are 23 LEDs plus phase. It was nevertheless supervised on non-Giemsa split-0 cells from **both draws**. The cross-slide result is an alignment/readout holdout conditional on a FOV-trained encoder. A fully new-draw pipeline requires fresh draw-held-out LF features; the code can consume those without relaxing partitions. Existing target calibration also uses per-slide non-Giemsa split-0 labels from both draws.
2. **Single teacher:** alignment seeds measure optimizer variability, not LF-teacher-run variability. K025/N032 calls for at least three LF teachers for a robust clinical claim. The GPU script runs three alignment seeds of the supplied teacher. The merger preserves teacher identity and flags fewer than three teachers; it averages seeds inside teachers before averaging teacher means. No clinical robustness claim follows from the present single-teacher smoke or its future single-teacher GPU run alone.
3. **Historical context is not a paired comparison:** N009's Procrustes .01755 versus linear SigLIP .03318 R@10 used another LF teacher, full source-slide pairs and transductive PCA. K011's cross-slide distilled autofluorescence .648/.658 and .32 used older truth/teachers; K022/N029's clean TT6c .564/.848 are within-slide students with different training, including PBC class anchors. N021/N032 clinical students used class-labelled adaptation, whereas N045 uses none. Every historical row is marked non-comparable; none enters current means. A contemporaneous image-distillation rerun is outside this embedding-only delivery.
4. **Distribution mismatch:** balanced entropic OT enforces approximately uniform marginal mass across batches. PBC and BSCCM LF have different lineage composition, and PBC morphology is far from BSCCM Wright (K005). A CCA teacher learned from a few BSCCM pairs may impose incorrect confident matches on PBC. Report lineage inversions and teacher residuals rather than selecting favourable signs/contrasts. `sotalign_lf_only` helps diagnose whether PBC adds harm.
5. **What the truth measures:** autofocus/position artefacts may remain in input embeddings by design. Primary targets are clean detrended v5c, with known FOV-level label issues from later knowledge entries. Same-cell autofluorescence transfer is not evidence of unmeasured CD19/CD56/CD123 accuracy. The v5c autofluorescence head pools training-draw non-Giemsa slides under their curated labels, unlike a separate unstained-only v6c autofluorescence teacher.
6. **Clinical split scope:** PBC-A/B follows the established alternating filename split, not a patient/scanner split. It cannot exclude near-duplicate patients/images across halves. All classes can appear as unlabelled PBC-A features, so IG/erythroblast contrasts are label-held-out rather than image-class-held-out. PBC 5-NN is a morphology preservation sanity metric, not cross-modal marker validation.
7. **Budget and numerical scope:** main defaults were prespecified rather than optimized. Low-rate Lion over 1,500 steps may underfit; head dimension/regularization and the original SAIL mean normalization matter. Full GPU/runtime/accuracy are unmeasured. No 30-minute completion guarantee is made; the delivery relies on checkpoint/requeue support and 29-minute allocations. The launcher requires the allocated CUDA device to be an H100.
8. **Skipped scope:** MLL23 lacks frozen embeddings and is optional in the brief; generating them would introduce image extraction/GPU work. No MLL23 features/results are fabricated, and no external images or fluorescence arrays were loaded as LF input.

The reviewable outcome is a complete runner, source/data provenance checks, all-arm CPU outputs, a preemption-aware single-GPU launcher, numerical/split/resume checks and documented experimental limits. Run the frozen H100 experiment before drawing conclusions about SOTAlign on BSCCM.
