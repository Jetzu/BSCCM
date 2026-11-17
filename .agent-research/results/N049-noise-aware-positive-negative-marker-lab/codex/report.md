N049 writer delivery, 2026-11-17. Implementation and verification are complete in this directory. The repository and original data were read only; no jobs were submitted and no GPU was used. This is a code/fit delivery, with no claim of GOAL improvement or full-data AUROC efficacy.

Read the N049 Plan, GOAL.md, KNOWLEDGE.md K001/K007/K019/K028, the current trainer, RawSet/GPUNorm and the v5 target builder. The frozen GOAL remains batch-held-out common-11 Pearson on v5, from the intensity head. Spatial label error and failed pooled-label QC are inherited constraints; this work neither repairs the FOV offsets nor restores the masked pooled assay.

The reviewable files are [the patched trainer](scripts/train_lf.py), [the fitter](scripts/n049_noise_mixture.py), [archive/loss helpers](bsccm/n049_pospos.py), [CPU tests](tests/test_n049.py), [the real-data smoke](tests/smoke_real_cpu.py), and [optional fit plotting](scripts/n049_plot_diagnostics.py). [implementation.patch](implementation.patch) contains those changes with repository-relative paths, for later application in a writable checkout. No existing repository file was patched in place. Baseline trainer SHA256: `7aeed6fd07f587d51f68d1aa4115cf2744cb42d76dc636f161ed0484409f4eb0`.

The model uses latent true positivity Z and latent observed-positive component S:

```text
Z ~ Bernoulli(pi)
P(S=1 | Z=1) = eta; P(S=1 | Z=0) = phi
Y | S=s ~ Normal(mu_s, sigma²), mu_0 <= mu_1
q = P(S=1) = pi*eta + (1-pi)*phi
P(Z=1 | y) = P(S=1|y)*pi*eta/q
             + P(S=0|y)*pi*(1-eta)/(1-q)
```

Gaussian emissions are a simple working model for the curated asinh/z-scored labels, rather than raw photon counts. The variance is shared: this forces an intensity location split and makes P(true positive | y) monotone in y. Unequal variances can create variance-only populations and send extreme high intensities back toward the negative component. Shared variance has a cost: skew and outliers can still look like populations, and narrow negative peaks are imperfectly described. The [fit plot](diagnostics/mixture_fits.png) exposes these failures; BIC is no biological validation. No threshold, lineage prior, diagnosis metric, or fluorescence image is used.

The priors are eta ~ Beta(18,2), truncated to [0.60,0.99]; phi ~ Beta(2,48), truncated to [0.001,0.15]; pi ~ Uniform[0,1]. Their modes are 0.94444 and 0.020833. These are explicit working assumptions, not independently measured efficiencies. Joint MAP EM sums the four (Z,S) responsibilities, updates pi/eta/phi and the two means/shared variance, then maximises the prior along the fixed-q ridge. Four deterministic percentile initialisations compete on the MAP objective. The shared variance floor is 0.01 times total training variance; convergence is relative objective change <=1e-8, with at most 600 iterations per start. Inputs with fewer than 32 fitting labels, nonfinite observed source labels, or constant fitting labels fail explicitly. The JSON reports convergence, iterations, variance floor and minimum objective step as well as the requested d′/eta/phi/BIC.

**Identifiability is the central limitation.** The two Gaussian emissions identify q; they cannot identify pi, eta and phi individually. The ridge step makes this dependence explicit instead of allowing arbitrary EM initialisation to determine the error rates. In most fits eta/phi equal their prior modes. When q is below the phi mode, the MAP can set pi=0 and phi=q. These are prior-conditioned soft labels, not calibrated measurements of biological positivity. [prior_sensitivity.json](diagnostics/prior_sensitivity.json) holds q and emissions fixed while reducing phi to 0.001: for draw-0 CD123 and CD19, P(Z=1|S=1) changes from 0 to about 0.796 and 0.947. The same marginal intensity likelihood supports both answers.

Each draw is fitted separately. A marker uses only `batch == source_draw`, `split == 0`, `hist_row < 0`, and that single-marker mask. Outputs cover every originally observed single-marker entry in that same clean draw, including its other FOV splits, using frozen parameters. Missing entries stay masked and zero in storage. No held-out-draw value enters fitting, initialisation, priors, or source hashes. Giemsa cells are absent from both fitting and soft-target output. The trainer's existing batch protocols still train on all source-draw splits; the stricter split-0 restriction applies to mixture estimation.

The `.npz` schema is `G` (sorted global clean source rows), `P` (float32 N_source x 8), `mask`, `fit_mask` (bool N_source x 8), `names` (single-marker order), and scalar JSON `metadata`. Sidecar JSON contains the parameters and elapsed time. The loader requires exact row coverage, schema, original single-marker mask and split-0 fit mask; finite [0,1] targets; zero storage on missing entries; matching source-draw provenance and a hash of source labels/metadata. It rejects the wrong direction, FOV protocol and `--include-gs 1`. This also catches archives prepared against incompatible labels. The source hash covers all source target splits for provenance; fitting itself only uses split 0.

`--pospos FILE --pos-w W` opts into one new linear head with eight logits. Both defaults are off (`''`, `0.0`). The head is created after the old modules, only with a supplied archive. The existing forward result and intensity loss are unchanged. BCE selects observed entries before doing any arithmetic, averages within each present marker, then averages across present markers. An all-missing batch contributes exactly zero with zero gradients. Neither pseudo-label masks nor tail weights affect this BCE. Soft labels are real-valued BCE targets; there is no hard-threshold conversion. Joint loss is the existing loss plus W times this BCE. With a file and W=0 the head exists but receives no BCE supervision; supply a positive W to train it.

Enabled evaluation retains `P`, `G`, `C3` and adds `P_pos` (sigmoid probabilities) and `pos_names`. GOAL metrics continue to use `P`. Head sharing means enabled training may change intensity predictions, as intended; the prediction definition remains intensity. Disabling the feature creates no head and consumes no extra RNG. `_NEW_DEFAULTS` contains pospos/pos_w and the new `--device` option (default cuda); default values are omitted from `_cfg`. Active configs pin the archive byte SHA256, source digest, draw, names, path and nondefault loss weight. Changed archive contents or weight reject resume. Existing pseudo/phys-aug/lowpass defaults and fingerprint logic are retained. CPU mode uses float32 rather than CUDA autocast, and DataLoader now supports workers=0.

The real-data fits below use `/hnvme/workspace/v181bb10-bsccm/targets_v5.npz`. d′ is `(mu1-mu0)/sigma`. Delta BIC is BIC(1 Gaussian) minus BIC(2 tied-variance Gaussians); positive favors two. Parameter counts are 2 versus 4 identifiable marginal parameters, not 6 nominal latent parameters. Mixture BIC is a descriptive MAP plug-in diagnostic with the usual mixture/boundary caveats. Pi is included to reveal degenerate noise decompositions.

| Draw | Marker | Fitting cells | d′ | eta | phi | pi | Delta BIC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | CD45 | 8,885 | 3.694 | 0.9444 | 0.0208 | 0.2124 | +2702.0 |
| 0 | CD123 | 14,483 | 21.678 | 0.9444 | 0.0049 | 0.0000 | +16200.0 |
| 0 | CD19 | 10,305 | 4.413 | 0.9444 | 0.0186 | 0.0000 | +1417.9 |
| 0 | CD56 | 11,077 | 2.699 | 0.9444 | 0.0208 | 0.9983 | +521.3 |
| 0 | CD14 | 14,788 | 6.405 | 0.9444 | 0.0208 | 0.1336 | +14798.8 |
| 0 | CD16 | 7,431 | 5.308 | 0.9444 | 0.0208 | 0.4773 | +5418.5 |
| 0 | HLA-DR | 16,694 | 4.675 | 0.9444 | 0.0208 | 0.0235 | +5307.0 |
| 0 | CD3 | 5,292 | 2.945 | 0.9444 | 0.0208 | 0.0564 | +411.4 |
| 1 | CD45 | 17,753 | 3.771 | 0.9444 | 0.0208 | 0.2352 | +6140.9 |
| 1 | CD123 | 25,050 | 0.055 | 0.9444 | 0.0208 | 0.4643 | -20.2 |
| 1 | CD19 | 6,150 | 0.057 | 0.9444 | 0.0208 | 0.6855 | -17.5 |
| 1 | CD56 | 18,786 | 2.992 | 0.9444 | 0.0208 | 0.9274 | +2071.5 |
| 1 | CD14 | 20,598 | 6.111 | 0.9444 | 0.0208 | 0.0960 | +17684.7 |
| 1 | CD16 | 22,108 | 5.612 | 0.9444 | 0.0208 | 0.5997 | +18156.4 |
| 1 | HLA-DR | 25,980 | 4.993 | 0.9444 | 0.0122 | 0.0000 | +3653.4 |
| 1 | CD3 | 25,208 | 2.908 | 0.9444 | 0.0208 | 0.4533 | +3256.5 |

All 16 selected fits converged, with nondecreasing recorded MAP objectives and no active variance floors. Draw 0 used 88,955 fitting entries and produced 111,413 soft targets over 127,946 clean rows in 4.57 seconds. Draw 1 used 161,633 fitting entries and produced 201,177 soft targets over 269,283 clean rows in 7.09 seconds (one CPU thread; wall time includes reading/writing labels). Each run used its own draw only. Deliverables: [draw-0 targets](diagnostics/pospos_draw0.npz), [draw-0 parameters](diagnostics/pospos_draw0.json), [draw-1 targets](diagnostics/pospos_draw1.npz), [draw-1 parameters](diagnostics/pospos_draw1.json).

The result does not support a blanket interpretation as recovered positive populations. Draw-1 CD123/CD19 prefer one component and have almost coincident means; their inferred probabilities are largely prior/prevalence driven. Draw-0 CD123/CD19 and draw-1 HLA-DR have exactly zero MAP true prevalence under these priors, hence all-zero observed soft targets. Draw-0 CD56 has pi=0.9983, giving nearly constant high targets. The very large draw-0 CD123 d′ comes from a tiny high-intensity tail, not evidence of a well-supported biological population. These cases are flagged in JSON. They are retained under the original mask as the requested model output; there is no hidden filtering or relabelling of missing markers as negative. A future training run should treat the full eight-marker head as an experiment, given these failure cases.

[The test log](diagnostics/tests.log) records seven passing tests, including the full synthetic trainer smokes, in 6.82 seconds. Tests cover the latent-posterior formula, separation and one-Gaussian null, bounded priors, monotonicity, constant-label rejection, invariance to changes in the other draw/nonzero splits/Giemsa/missing values, archive integrity, soft-BCE values and gradients, unequal marker counts, all-missing batches and NaNs outside the mask. A fixed-seed ResNet18 run with actual RawSet/GPUNorm and a tiny synthetic BF pack compares the patched default against the original trainer: every model/optimizer/scheduler tensor, prediction array and training loss is bit-identical on CPU. The original's CUDA-only API calls are mapped to CPU by the test harness; the training functions and loss are otherwise the original ones. This is CPU evidence, not a GPU bitwise-equivalence claim.

The same test compares the legacy config with the patched GPU-default config exactly, then exercises strict old-checkpoint loading on CPU with only the device field adapted. Enabled runs verify head gradients and probability shapes, successful unchanged resume, rejection of changed archive bytes/weight, and coexistence of pseudo-label tuple entries plus lowpass. [cpu_smoke.json](diagnostics/cpu_smoke.json) records the synthetic result. Tiny runs have fewer than 1,000 training observations per marker, so the existing GOAL-valid support metric is empty and emits NumPy mean warnings; the inherited result-file writer also emits ResourceWarnings. Neither is a failed test or an efficacy measurement.

[The additional real-data CPU smoke](diagnostics/real_cpu_smoke.json) ran two AdamW steps on eight original draw-0 cells, six observed single-marker entries, then predicted 24 original draw-1 CD45 cells. It used the delivered source archive without refitting, actual LED/phase memmaps, original BF calibration, and ResNet18. Only BF entered the network. Loss was 1.633455; saved P_pos was 24 x 8 and finite (range 0.47948–0.51422). Temporary checkpoint/data copies were deleted. The observed tiny-sample intensity correlation is recorded only for debugging; it is not a GOAL score. No full-data head training, GOAL comparison, AUROC measurement, expected-mixture-intensity alternative, or GPU experiment was run.

To reproduce from this delivery directory (baseline checkout location can be overridden with `BSCCM_REPO` in tests):

```bash
mkdir -p /tmp/codex_n049
UV_CACHE_DIR=/tmp/codex_n049/uv-cache uv venv /tmp/codex_n049/venv
UV_CACHE_DIR=/tmp/codex_n049/uv-cache uv pip install --python /tmp/codex_n049/venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
UV_CACHE_DIR=/tmp/codex_n049/uv-cache uv pip install --python /tmp/codex_n049/venv/bin/python numpy scipy timm scikit-learn matplotlib
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/codex_n049/matplotlib
/tmp/codex_n049/venv/bin/python scripts/n049_noise_mixture.py --source-draw 0 --out diagnostics/pospos_draw0.npz
/tmp/codex_n049/venv/bin/python scripts/n049_noise_mixture.py --source-draw 1 --out diagnostics/pospos_draw1.npz
/tmp/codex_n049/venv/bin/python -m unittest discover -s tests -v
/tmp/codex_n049/venv/bin/python tests/smoke_real_cpu.py
/tmp/codex_n049/venv/bin/python scripts/n049_plot_diagnostics.py
```

The tested package versions are in [environment.txt](diagnostics/environment.txt): Python 3.13.13, torch 2.15.0+cpu, torchvision 0.29.2+cpu and timm 1.0.31. The trainer file is a replacement for the repository's scripts/train_lf.py and relies on that repository's existing bsccm package; tests overlay the helper while importing that read-only package. The fitter runs directly from this delivery. For later use in a writable checkout, apply implementation.patch, retain a fixed archive per run, and add `--pospos /absolute/path/pospos_draw0.npz --pos-w 0.25` for b0to1, or the draw-1 archive for b1to0. CUDA remains the production default; use `--device cpu --workers 0` for CPU checks. These production runs have not been submitted.
