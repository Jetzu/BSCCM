# N054 v3 second-round fixes

All edits and exported evidence are in `codex/v3/`. V2, repository code, production outputs and input data were left untouched. Checkpoints and numerical arrays are under `/tmp/codex_n054/v3_final_smoke/`; reviewable JSON/CSV exports are in `smoke_artifacts/`. CPU smoke only; no GPU/Slurm/container execution or submission.

## P1-A: out-of-sample maps and direct pretraining

Primary `A1-GS_oos_s0` and `direct` train exclusively on source-slide split-1 cached E_G latents. E_G optimizer training remains source split 0. A deterministic seed-5401 partition reserves 20% of source split-1 slide/FOV groups, rounded up, from both maps. Sigma is fixed at zero, budget at 600 updates, and the final iterate is exported. Neither holdout loss nor autofl r selects a map checkpoint, sigma, or primary path. Other families retain v2 source-validation selection. E_G itself retains v2's split-1 checkpoint validation, so “out-of-sample” means absent from E_G optimizer training; split 1 is not an untouched encoder-selection set.

A1-GS initializes from `A1-LF_s0`. Direct first fits `P2(f) -> P1(f)` on non-GS LF split-0 same-cell pairs for 2,500 updates, then receives exactly the same GS fine-tune rows, normalization, learning rate, budget and holdout as A1-GS. Both use the same 535,808-parameter network. Direct fixes time to zero and regresses normalized endpoints. In-sample split-0 versions remain explicitly labelled secondary, using the same 600-update fine-tune budget and LF initializers (A1-GS retains both sigmas).

Evidence: full-data metadata census in `production_partition_audit.json` gives:

| GS holdout | Source slide | Map training cells/FOVs | Informational holdout cells/FOVs |
|---|---:|---:|---:|
| batch1 | 2 | 795 / 223 | 199 / 56 |
| batch0 | 19 | 469 / 156 | 124 / 39 |

Each map exports optimizer source/validation row IDs and `finetune_curve.json`: pre-fine-tune, steps 250/500/600, full-holdout loss, mapped cosine and autofl r. CPU smoke uses steps 0/10/20. The smoke checks split-1 isolation from E_G training, disjoint FOV groups, identical partition/initialization contracts, and bitwise equality of each fine-tuned export with its last optimizer state. `pre_post.json` per holdout and aggregate `pre_post.csv` report both maps before and after fine tuning, plus the in-sample secondary. Metrics include measured autofl r, native retrieval, mapped retrieval and mapped CKA.

## P1-B: frame-invariant A2

Primary A2 now uses minibatch entropic GW on intra-space cosine-distance matrices of source GS and independently sampled different non-GS LF P1 endpoints. A torch implementation follows squared-loss fixed-point/Sinkhorn iteration from [Peyré, Cuturi & Solomon (2016)](https://proceedings.mlr.press/v48/peyre16.pdf): epsilon .005, at most 40 outer iterations, 100 log-domain Sinkhorn iterations, uniform marginals, and numerical balancing. Rotation/permutation-invariant sorted-distance profiles initialize the nonconvex solver; no cross-frame coordinate cost or cell identity enters initialization or optimization. Flow matching samples joint pairs from the soft coupling. Original Hungarian A2 remains as `A2_euclidean_s*`, labelled frame-dependent. A2-EM remains secondary.

POT is absent from the available CPU environment; apptainer is unavailable to this writer. The implementation adds no dependency. A 256×256 random coupling took approximately 0.21 seconds on two CPU threads; production GPU cost is unmeasured.

Evidence: `n054_gw_test.py` checks exact recovery of random rotation+permutation, rotation invariance of the full coupling, nonnegative finite entries, and balanced marginals. Each A2 GW run separately evaluates five batches of the SAME real non-GS LF cells encoded through P2 and P1, recording argmax identity, diagonal coupling mass and chance in `gw_identity_sanity.json`. Aggregation emits explicit sanity rows and `gw_identity_sanity.csv/json`. This empirical check can be imperfect: independently trained spaces need not preserve the same geometry, and GW is a nonconvex structural alignment, not evidence of true biological matches across different cells.

## P1-C: completion controls

Added (e) linear stacking with intercept on `[pred_LF, pred_labels_MLP]`, fitted only to source split-1 task-eligible rows; C1 has separate coefficients for each observed marker and the aggregate. Fit row IDs, masks and coefficients are saved in `*_stack.npz`. Added (f) a plain MLP on `[f, y*mask, mask]`, with the fused head's two 512-wide hidden layers, no marker encoder and no shared space. It trains on the same rows and hidden-entry masking regime as the fused head. Neural validation still scores only observed entries removed from input.

Every completion metric row reports gains against a, c_mlp, c_linear, d, e and f. `hidden_information_supported` requires fused r strictly greater than the maximum of all six controls, with every score available. Test predictions/truth never fit stacking coefficients. C1's per-marker and aggregate stacked predictions are saved separately for reproduction. Source-unobserved/QC-failed targets remain explicitly unavailable.

Evidence: the smoke checks source split-1 stacking rows, source-draw isolation, hidden-target exclusion, finite neural outputs, every per-control difference, and recomputes the flag from all six scores. It retains 39 completion rows, including 24 per-observed-marker C1 rows, three independently fitted protocols, and C3 descriptive coherence.

## P2: execution and LF overlap disclosure

All Python launchers, subprocesses, README commands and sbatch execution use `-I -B`; no `__pycache__` is created here. Default output prefix/path is `n054v3`. README contains the exact three sbatch commands and waits for the complete LF preemption chain before starting either holdout. Existing house wrapper behavior and eight-hour per-unit request remain.

LF-side training deliberately retains held-out-draw non-GS unstained rows because LF is a shared unit: **6,604 slide-19 rows (2.08%) / 5,088 slide-2 rows (1.60%)**, out of 317,935 non-GS LF split-0 training rows. These are absent from GS training, but they limit whole-slide-exclusion claims for LF. Every `data_audit.json` now writes both per-holdout full-data counts, actual allowed-row counts and slide IDs in `lf_heldout_non_gs_unstained_by_holdout`. The smoke asserts the two full-data counts. Supplied frozen LF feature teachers also retain their prior across-draw training exposure.

## GPU estimate and remaining limits

Revised planning estimate **8.60 GPU-hours total**: shared LF 0.25; completion 0.40; six GS fits 5.10; maps/caching/GW 0.75; internal inference 0.40; twelve clinical runs 1.70. Assumes a fast A100/H100-class GPU, warm data, approximately one second per GS update plus validation/I/O, and 8.5 minutes per clinical run. GW double-precision throughput and convergence may change this materially. Production counts, two-sigma grids and clinical TTA are retained; maps are shared with six total GS fits. No GPU throughput was measured.

CPU scores verify integration, not scientific superiority. Full CUDA/container/Slurm preemption behavior and production metrics await execution. No cell has both GS and surface-marker truth; M2 products remain assumption-dependent, and C3 remains descriptive. Clinical labels never tune fitting or selection.

## Final CPU verification

**PASS: 173.75 seconds**, using the actual three unit drivers with real data, 20 updates per fit, at most 2,000 non-GS LF rows and 300 GS rows. SIGTERM interrupted the batch1 GS driver at step 10/20; it exited 143 in 0.741 seconds and resumed to **bitwise-identical weights** versus uninterrupted training. Completed-unit replay preserved all model mtimes; changed-code replay was rejected as expected. The expected rejection traceback in smoke.log is part of the passing suite.

Six GS fits, two shared LF-space fits, three completion protocols and twelve clinical evaluations passed. Aggregation produced **12 primary / 34 grid / 14 diagnostic rows**. Completion retained 39 rows, including 24 per-observed-marker C1 rows. Clinical smoke used 16 PBC and 24 MLL23 images per primary/baseline run, all 25 contrasts; new arms used eight-view TTA. All direct/flow maps have 535,808 parameters. `verification.json` additionally confirms identical primary train/holdout rows, no stack/test overlap, parsed Python and no bytecode directories; shell syntax also passed.

`gw_test_results.json`: **100% recovery in all four rotation+permutation cases**, including 256 dimensions; largest marginal error below 2e-17 and largest rotation error below 2e-18. Empirical same-LF cross-model GW argmax identity in the smoke (five batches, n=16) was:

| GS holdout | Sigma | Identity rate | Chance |
|---|---:|---:|---:|
| batch1 | 0 | 0.2750 | 0.0625 |
| batch1 | 0.1 | 0.2750 | 0.0625 |
| batch0 | 0 | 0.2250 | 0.0625 |
| batch0 | 0.1 | 0.2250 | 0.0625 |

CPU smoke measured test autofl r, before and after fine tuning (integration checks only):

| Holdout | Map | LF-pretrained | OOS final | In-sample secondary |
|---|---|---:|---:|---:|
| batch1 | A1-GS | 0.0991 | -0.2609 | -0.2660 |
| batch1 | direct | -0.0388 | -0.0228 | -0.0289 |
| batch0 | A1-GS | 0.2283 | -0.1864 | -0.0687 |
| batch0 | direct | -0.3599 | -0.4071 | -0.2396 |

Informational OOS holdout autofl-r curves (full holdout; no model selection):

| Holdout | Map | Step 0 | Step 10 | Step 20 |
|---|---|---:|---:|---:|
| batch1 | A1-GS | 0.4081 | -0.3912 | -0.7900 |
| batch1 | direct | -0.5511 | -0.5351 | -0.5607 |
| batch0 | A1-GS | 0.1861 | 0.2711 | -0.0109 |
| batch0 | direct | -0.2353 | -0.2487 | -0.2462 |

Final main-script SHA256: `811ffe1c23e7d78d9788b89c70d36dee3c3740ead25000c42c1d9bace14ec511`. Commands and detailed evidence: `smoke.log`, `smoke_results.json`, `gw_test_results.json`, `verification.json`, and `smoke_artifacts/index.json`.
