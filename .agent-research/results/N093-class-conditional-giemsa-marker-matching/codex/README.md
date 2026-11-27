# N093 · Class-conditional Giemsa to marker matching

F1 predicts a soft mixture of three LF-teacher class means. F2 assigns a GS image to one of the three classes and predicts a weighted average of LF-teacher profiles from that class. Each profile has the twelve TT6c marker outputs and the shared auxiliary autofluorescence output. F1, F2 and the new F1-bsccm student each take a frozen 768-dimensional DinoBloom-B image embedding as their only inference input.

## Fix round

Round-one t0 routed 5,518/5,697 held-out slide-19 cells and 9,936/10,015 slide-2 cells to granulocyte. The same-cell LF reference on slide 19 routed 3,545/5,697 there. Claude's red-team also found that the F2 within-class rank separated PBC from BSCCM with AUROC 1.00. F1 and F2 remain the round-one comparison arms. The two additions reuse their fold- and teacher-specific matrix of three mean profiles.

| Arm | Routing and use |
|---|---|
| F1-bsccm | Fit a multinomial logistic regression on both split-0 paired GS cells of the single source slide and N058's three duplicate-filtered PBC-A anchor classes. The BSCCM target is the argmax of **that row's own LF encoder and trained cls3 head**, indexed by the global cell ID. PBC uses its morphology label. Predict the probability-weighted mean profile on GS alone. |
| F1-oracle-hard_UPPER_BOUND | On each held-out BSCCM row, route with the argmax of its own LF teacher's cls3 head and look up that class's mean profile. |
| F1-oracle-soft_UPPER_BOUND | On the same rows, apply softmax to the same LF head logits and mix the three profiles with those probabilities. |

F1-bsccm assigns **50% total weight to each domain**, uniformly per cell within a domain, for both standardization and the logistic loss. Sample weights sum to the pooled sample size. The solver is L-BFGS, with L2 regularization, `C=0.1`, and seed 0. The fold audit records the source slide, global row hashes, source weight shares and class counts. Every teacher/fold fits a separate GS router before clinical scoring.

The oracle is an **upper-bound control for access to the teacher's route** in a class-mean decoder. Its inputs include an LF view of the scored cell. Its marker outputs still come entirely from training-bank means. A particular Pearson coefficient can be higher for a learned router. The oracle has no clinical endpoint because clinical images have no paired LF view.

## Classes and training cells

| PBC half-A anchor | BSCCM LF head output | Contributing BSCCM cls10 gates |
|---|---|---|
| lymphocyte | 0 · lymphocyte | 0 |
| neutrophil | 1 · granulocyte | 1, 2, 3 |
| monocyte | 2 · monocyte | 7, 8, 9 |

Neutrophil is the harness's morphology anchor for the broader granulocyte gate. The remaining five PBC classes have no training role. Clinical cells outside the three anchored classes still receive predictions. This is coarse-lineage matching, with class assignment error possible on both sides. Gates and their grouping come from the repository's `scripts/make_targets_v3.py`. N008 previously observed a large PBC-to-BSCCM shift. The fold audit therefore records the number of BSCCM GS cells assigned to each route.

The original F1/F2 GS classifier is multinomial logistic regression (`C=0.1`) on standardized PBC-A anchor embeddings. Both scaler and classifier fit only the registered, duplicate-filtered anchors. The PBC feature row is its position in the original sorted archive listing from `scripts/embed_external.py`. The N058 harness performs the alternating split before removing duplicates. The loader checks labels against archive names for the rows it takes.

Each fold uses split-0 GS cells and split-0 non-GS LF cells from the source draw. The LF bank includes single-marker, pooled and unstained acquisitions. Every entry carries the teacher's whole predicted profile. F1 uses **every** eligible bank entry. Each registered marker encoder supplies its own trained linear `c3` head on raw 512-dimensional LF features. The implementation also handles an absent head by fitting a classifier on draw-0, split-0, pooled, singly gated `cls3` rows from slides outside both GS slides. The registered t0–t2 runs all have the head.

## Within-class OT

GS and marker vectors have different dimensions and unrelated axes. F2 describes each cell by its distances to the three corresponding class centroids in its own modality. On the GS side the centroids use labeled PBC-A anchors in standardized embedding space. On the LF side they use the routed LF bank in standardized teacher-output space.

Within each class the GS support combines source-slide GS cells routed by the classifier and PBC-A cells carrying that anchor label. The LF support contains only bank cells routed to that class by the teacher. With seed 0 by default, F2 samples up to 1,024 rows uniformly without replacement on each side of each class. It converts each of the three distances to its empirical midrank within that support. The cost is mean squared distance between the two rank vectors. POT Sinkhorn fits uniform marginals with entropy regularization 0.035. The solver checks both marginals. For a new GS image, the frozen GS rank functions and fitted LF column potential give the conditional transport weights. Their barycentre uses the original LF marker values. No evaluation-batch statistics enter the fit. The round-one domain separation limits biological interpretation of F2.

## Teacher and endpoint provenance

- The seed-0 entries of N058's `code/teachers.json` specify the three marker feature sources and the common v6c auxiliary source. Each run checks the TT6c configuration hash, LF encoder configuration, global row maps and registered target order. The GS student's regression head is never used to generate a marker bank.
- The original TT6c marker teacher is a per-output ridge on v4c LF features. For t0 the program reads the N054 A0 frozen ridge from the seekable archive `runs/archive_n054v3_n057_runs.tar`, member `n054v3_batch1/A0/baseline_teacher.npz`, and verifies its SHA-256. This member also supplies the shared v6c autofluorescence ridge for all three teachers.
- For t1 and t2 the program reconstructs the registered marker ridge with the corresponding LF features, original marker masks, split-0 stained normalization, per-marker seed, 60,000-row cap, unpenalized intercept and ridge penalty 10. It checks that these label rows exclude both GS slides. All teachers undergo a fixed numerical parity check against sixteen original TT6c saved teacher predictions (absolute tolerance 0.02 in the saved output units). CPU and original GPU arithmetic can differ. The smoke audit records t0's maximum difference of 0.00257.
- Induction applies to the new students. The historical LF teachers are frozen references from FOV-split training on both draws. Their training excluded all GS-paired cells, and the shared auxiliary teacher used other non-GS cells on both unstained slides. The provided frozen GS features also retain their original intensity preprocessing. A completely new-draw LF-teacher experiment would require different teachers.
- All GS students are fitted before loading clinical evaluation features. Scoring uses N058's own `clinical_manifest`, `score_contrasts` and primary flags. It checks the ordered evaluation IDs against the saved N058 T run (8,538 PBC-B and 7,265 MLL23) and disjointness from anchor IDs. MLL23 evaluation images use the N008 frozen DinoBloom-B single-view preprocessing, computed once on CPU across teachers/folds. Historical T/H used eight-view inference. Endpoint and case membership match.
- N054 M1 is within-slide Pearson correlation against observed same-cell detrended v6c autofluorescence, with all FOV splits on the held-out slide. A second table reproduces the N058 sanity definition on the held-out slide, restricted to observed paired cells with split 2, using its Pearson support and variance rule. The twelve marker correlations compare predictions to the paired cell's own LF-teacher output in global-row order. They measure teacher agreement. Missing or constant Pearson inputs produce a blank coefficient with its finite-pair count.

## Commands

Run from the N093 `codex/` directory. For a different installation layout, supply the full N058 `code/` directory with `--harness`. The input root defaults to `/hnvme/workspace/v181bb10-bsccm`.

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex
data=/hnvme/workspace/v181bb10-bsccm
work=/tmp/n093_codex2
mkdir -p "$work"
py=(apptainer exec -B /hnvme "$data/env/bsccm.sif" python3 -I -B)
```

On the writer host, the existing unpacked copy of this SIF supports CPU fixture and smoke runs without a FUSE mount.

```bash
if [[ -d /tmp/n093_codex/image/usr/local/lib/python3.11/site-packages/ot ]]; then
  py=(apptainer exec --userns -B /hnvme /tmp/n093_codex/image python3 -I -B)
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR="$work"
"${py[@]}" "$PWD/tests/test_class_match.py"
smoke_parent=$(mktemp -d "$work/smoke.XXXXXX")
"${py[@]}" "$PWD/n093_class_match.py" --smoke --seed 0 --teachers t0 t1 t2 \
  --holdouts 1 0 --threads 4 --out "$smoke_parent/run"
"${py[@]}" "$PWD/tests/test_class_match.py" --verify-real "$smoke_parent/run"
```

**Full evaluation for integration.** Request one 48-core CPU allocation. Ridge solves, logistic regression, Sinkhorn matrix products and batched MLL23 image encoding use the CPU thread pools. A single process shares the expensive clinical embedding across all six fits. GPU allocation would require evidence of a twofold speedup over the threaded CPU job. This command uses a fresh round-two result directory and publishes only on success.

```bash
sbatch --partition=preempt_cpu --cpus-per-task=48 --mem=160G --time=06:00:00 \
  --job-name=n093_router_fix --output="$PWD/../n093_router_fix_%j.out" <<'SBATCH'
#!/bin/bash
set -eu
cd /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex
data=/hnvme/workspace/v181bb10-bsccm
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
exec apptainer exec -B /hnvme "$data/env/bsccm.sif" python3 -I -B "$PWD/n093_class_match.py" \
  --seed 0 --teachers t0 t1 t2 --holdouts 1 0 --threads 48 --out "$data/runs/n093_class_match_fix_s0"
SBATCH
```

## Reading the output

Each holdout has its own result directory. Holdout 1 is slide 19 (5,697 paired cells) and holdout 0 is slide 2 (10,015). Both use the same clinical evaluation IDs. Compare each fold to the seed-0 T and H_matched rows in N058 separately. The historical students used training FOVs from both slides.

| File | Contents |
|---|---|
| `n093_per_run.csv` | N058 macro columns for F1, F2 and F1-bsccm. A missing contrast invalidates the macro. Smoke outputs use the `smoke_` prefix. |
| `contrasts.csv` | All 25 registered clinical AUROCs for each GS student and teacher. |
| `sanity.csv`, `sanity_n058.csv` | Observed same-cell v6c autofluorescence r on all held-out FOVs (N054 M1) and on held-out split 2 (N058), respectively, for all five arms. |
| `agreement.csv` | Twelve per-marker same-cell LF-teacher correlations and finite-pair counts per arm and teacher. |
| `router_fractions.csv` | Hard counts and fractions and mean routing weights for every applicable arm. The same row carries the reference fractions and support. Includes overall MLL23 and PBC-B and each morphology label, overall paired BSCCM, both autofluorescence subsets and strata by LF class. |
| `t*_paired.npz` | Paired global and histology IDs, split, profiles, round-one route, three router probability matrices, class means, measured cls3, LF marker reference and observed autofluorescence. Classes use the order in the mapping table above. |
| Run-level `audit.json` | Teacher identity and parity, fit/eval row hashes, support and source counts, training weight shares and clinical denominators. |

In the fractions table, `routed_fraction` uses every cell in its population. For clinical reference comparisons, compare `routed_fraction_on_gated` and `gated_fraction`, both on exactly `n_gated` cells covered by N058's registered morphology-to-cls3 mapping (`n056_eval.CLASS`). Other morphologies still count in the overall router fractions. A clinical reference is a morphology label. On paired BSCCM cells v4c supplies zero measured cls3 gates, so those fields are blank. The paired LF columns give both the reference argmax fractions and mean posterior on exactly the scored rows. A blank fraction or correlation indicates unavailable support. PBC10 also reflects the supervised PBC-A anchors.

## Validation and integration

The CPU fixture log and capped real-data outputs are in `tests/fix_smoke/`. The real run uses all six seed-0 teacher/fold combinations, with 32 paired source cells, 96 LF bank rows, 48 anchors, 24 held-out cells, 16 PBC-B images and 24 MLL23 images per fit. The t1/t2 LF marker ridge replay uses its registered training rows even in smoke mode. The verifier independently refits the weighted classifier from raw IDs and checkpoint logits, checks both oracle readouts and recomputes correlations, routing denominators and macro scores. It also checks the original t0 slide-19 predictions against the stored round-one smoke fixture in `tests/real_smoke/`. The capped slide-2 sample contains zero observed split-2 rows and exercises the blank-coefficient case. Smoke scores establish execution and pairing only.

Land `n093_data.py`, `n093_match.py`, `n093_class_match.py`, the updated test, its evidence and this README together under N093. The integrator owns the 48-core production evaluation, independent review and notebook commit.
