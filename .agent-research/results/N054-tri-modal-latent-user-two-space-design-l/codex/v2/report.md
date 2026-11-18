# N054 v2 delivery and verification

V2 is entirely in this directory. V1, repository code and input data were left untouched. Scratch checkpoints/NPZs remain under `/tmp/codex_n054/v2_final_smoke/`. Reviewable CSV/JSON exports are in `smoke_artifacts/`. No GPU job, real container or Slurm submission was run.

Read before implementation: v1/original brief/report, all supplied red-team scripts c1.py–c13.py, repository knowledge/N052, and the requested house preemption wrapper/template and repository GPU job template.

## Changes per finding

| Finding | Fix and evidence |
|---|---|
| P1-1 modality gap | A1-LF retained at both sigmas; A1-GS initializes from its same-sigma LF bridge and fits source split-0 GS→own-P1 pairs, validating on source split 1. Direct regression uses the same pairs and exactly the same parameter count, without integration. Each LF bridge exports own-P2-start oracle predictions. Evaluation exports native GS/LF cosine mean/SD, native distance, within-P2 nearest-other distance, P1 cosine, and hub GS/LF gap statistics. Native z_L means P2(f_own) for space 2 and P_L(f_own) for the hub. |
| P1-2 decoder support / A2 | D_M trains on P1, exact leave-one-out E_M and full E_M. A2 uses source-slide GS latents and independently sampled non-GS LF P1 targets. A2-EM is secondary. Hungarian OT forbids equal global IDs; saved audits measure actual source/target GS counts and same-cell assignments. |
| P1-3 loss scale | A3 uses squared error summed over dimensions, averaged over rows: 2−2cos on unit vectors. Both SigLIP weights remain. Numerical smoke confirms the gradient is exactly 256× the old dimension-mean gradient. |
| P1-4 repeated training | One LF unit fits space 1, the LF hub and three completion protocols. Each holdout fits space 2 once and hub GS once per weight. All eight bridges and direct regression reference the same frozen space-1/space-2 paths. Six GS fits total. |
| P1-5 controls | C1/C2 export LF, fused, labels-only MLP, labels-only linear ridge and ridge on [f, observed values×mask, mask]. Labels-only has its own marker encoder and identical output MLP architecture to fused. Target-specific ridges use only source split-0 rows and task-appropriate other entries. Hidden-information flag requires fused > every control, with all correlations available. |
| P1-6 preemption | Three --stage all drivers replace the dependent-job graph. Sbatch executes the house wrapper without srun. Python checkpoint exits are 143, accepted by the wrapper. Training checkpoints include current/best weights, optimizer/scheduler/RNG and pending validation. Source-cache/internal/clinical inference have batch cursors. Stats/ridge/geometry check STOP between bounded chunks. README documents waiting for the entire LF resubmission chain before either holdout. |
| P2 validation | All three completion neural prediction losses score only observed entries removed from input. Validation contains those errors only, with no contrastive component or visible-entry reconstruction. Saved validation audits count zero scored-visible entries. |
| P2 audit | LF/GS exclusion, split and overlap counts are computed from actual row arrays. Flow audits record actual assignments. Completion masks/scored counts are saved and checked. |
| P2 negatives | Autofl-only rows are near-duplicates when absolute autofl difference < tau; diagonal positives remain. Smoke exercises nearby and distant AF-only rows. |
| P2 selection consumed | Freeze A1-GS sigma, A1-LF sigma, A2 sigma and A3 weight using only source split-1 autofl truth r; ties prefer lower sigma/weight. Aggregator reads those exact paths and validates source artifacts. No test/clinical score selects. |
| P2 signatures/logs | SHA256 of the script plus arguments and input/output fingerprints govern stage reuse. Training checkpoints also check code/arguments. Changed code is rejected. Stage logs append. Defaults use runs/n054v2_*; changed configurations require a fresh versioned prefix. |
| New completion file / C1 | Completion uses targets_v5cx_v10.npz; GS models retain TT6c. Verified all 312,590 single-marker cells have observed autofl, and zero draw-2 group-8 observations survive in v10. Names and split/batch/hist_row/slide/FOV arrays match TT6c. C1 reports each observed marker separately, gains versus each control and the strongest control, and flags abs(labels-only r) >= .2 for descriptive crosstalk review. |

The v10 check corrects v1's unavailable-C1 conclusion. TT6c retains 27,074 failed pooled entries; these are masked for new models/reference scoring. Original baseline teachers retain historical masks for compatibility. All clinical contrast dictionaries remain byte-for-byte identical to v1.

## Final CPU smoke

**PASS: 156.77 seconds**, real data, 20 updates per fit. Final code SHA256:

`05183097e503337572daab9b36e79516252050f050c0ed0428f321b3845d7411`

The suite executes all three actual unit drivers, both directions, all bridge sigmas/hub weights, A0, direct regression, completion, clinical inference and aggregation. Limits: 2,000 non-GS LF cells (680/660/660 train/validation/test), 300 GS cells (50 per split/draw). Trainable GS uses the declared random two-block ViT-tiny at 32 pixels. A0 loads real DinoBloom/TT6c checkpoints.

- A real SIGTERM interrupts the batch1 driver mid-GS-stage at **step 10/20**. It saves ckpt.pt, exits **143** within **0.752 seconds**, and resumes at step 10. Exported GS weights are **bitwise equal** to uninterrupted training with identical seed/data/architecture.
- Reissuing completed LF and batch1 units skips every stage and preserves every .pt modification time. Shared LF outputs remain unchanged across both holdouts. A one-comment code change is rejected; its expected traceback is near the end of smoke.log.
- Exactly two shared LF-space fits, three completion protocol fits and six GS fits occur. Every direct/bridge map has **535,808 parameters** and uses the expected frozen paths.
- OT audits confirm GS source/non-GS target populations and **zero same-cell assignments**. LF/GS training exclusions and source/test isolation pass. Training rows are split 0; source validation is split 1.
- Completion exports **39 metric rows**, including **24 per-observed-marker C1 rows** (eight markers × three protocols). Saved hidden-target input masks are zero; neural predictions are finite. The reverse-draw failed group has no source truth and is explicitly unavailable. Validation audits contain zero scored-visible entries.
- Exactly **10 clinical evaluations**: A0 plus four selected families per direction. Each has **16 PBC + 24 MLL23 images** and all **25 contrasts**. Unselected configurations have no clinical outputs. New arms exercise eight-view TTA; A0 uses one CPU view.
- Aggregation yields **10 primary rows**, **24 full-grid rows**, and **6 diagnostic rows** (two oracle sigmas plus direct per holdout), using frozen selections.
- All Python files parse and shell syntax passes. A mocked-container invocation through the actual house wrapper's non-Slurm fallback verifies all three unit commands, python3 -I, stage all and shared LF path. This does not verify Slurm resubmission/container signal propagation.

Commands/results are in smoke.log, smoke_results.json, smoke_artifacts/index.json and wrapper_check.json. n054_smoke.py reproduces them with a fresh scratch directory. Smoke scores are integration checks and cannot rank the scientific arms.

## Budget and open limits

Target **7.60 GPU-hours**: shared LF 0.25; completion 0.30; six GS fits 5.10; maps/source caching 0.25; internal evaluation 0.30; ten clinical runs 1.40. README states the per-stage assumptions: about 1 second per GS update plus validation/I/O, and 8.4 minutes per eight-view clinical run on a fast A100/H100 with warm storage. This is unmeasured; slow memmap access, clinical throughput, GPU class or repeated preemption can exceed eight hours. Production step counts, grid and TTA are retained.

Full CUDA training, container dependencies, actual scheduler signal forwarding/resubmission, full-slide metrics and full clinical inference await GPU execution. A mock cannot prove cluster behaviour.

No cell has both GS and surface-marker truth. M2 products require the transitive/CIA assumption; retrieval/geometry cannot validate marker fidelity. C3 remains descriptive. Labels-only autofl predictability can reflect crosstalk or biology and is not causal evidence. Completion's new models are source-only in draw tests, while supplied LF feature teachers were trained across both draws. Clinical conclusions remain conditional on this teacher pair/seed; source autofl selection need not optimize fine-marker transfer. Clinical labels never tune models or selection.
