# N056 preregistration — 2026-11-18

Status: protocol and code only. No new clinical predictions have been read or
scored. N032/K025 historical findings were supplied in the brief. CPU smoke
results are engineering checks, not clinical evidence. GPU execution is pending.

## Locked design

Three paired comparisons, student seed 0: t0 uses n029_r18_v4c_fov/probe,
t1/t2 use n032_r18_v4c_fov_t1/t2/probe. ALL THREE use the same
n029_r18_v6c_fov/probe autofluorescence teacher, as their saved result.json
arguments specify. There are three marker-teacher runs, not three independently
retrained autofluorescence teachers. Reuse n029_tt6c_s0, n032_tt6c_t1_s0,
n032_tt6c_t2_s0, final step 3000. Do not reuse old clinical predictions.

H uses the N054 A3 supervised LF hub (masked marker MSE + masked-marker
SigLIP, 1024-D concatenated frozen teacher features -> normalized 256-D latent;
512-wide, depth-2 MLPs), then freezes the entire hub. Marker labels are v4c;
autofl is v6c, matching T. Preserve v4c masks for pairing, including the known
K019 unreliable pooled group: report this caveat rather than silently changing
the teacher task. H student loss = mean sum_d (z_G-z_L)^2 + 0.5 times the
EXACT two CE terms of train_gs_bind (source N008 pseudo classes and PBC
lymphocyte/neutrophil/monocyte half A). No student SigLIP or output loss in H.
Final iterate, 3000 hub steps and 3000 student steps; no clinical selection.
ViT-B/14 DinoBloom, freeze 8 blocks, batch 128, AdamW lr 1e-4, wd .05,
OneCycle pct_start .05, clip 1; hub batch 512, lr 1e-4. Augmentation/anchor
loader copied verbatim from train_gs_bind. Same legacy image scale for pairing.
Source GS train=split 0, sanity eval=split 2, both slides; split 1 unused.

PBC sorted JPEG names, A=even/B=odd, exactly legacy. MLL23: within each zip,
sorted valid filenames, RNG(0) permutation truncated to 1500 exactly as legacy,
then alternating selected names into fit/eval. No labels used by unconditional
OT; archive class membership only balances this fixed sampling frame. No test
image enters normalization, hub/student fitting, OT fitting, thresholds or
selection. MLL23 remains a historically reused development benchmark (K022).
Evaluation preprocessing and all 10+15 contrast definitions are copied verbatim
from apply_gs_student. TTA fixed to 8 rotations/flips. OT averages decoded
per-view mapped embeddings, matching the baseline TTA decoding convention.

OT: clinical fit embeddings -> source GS split-0 embeddings, uniform marginals,
KL-unbalanced entropic OT, epsilon .05, marginal KL strength rho=.5, generalized
Sinkhorn exponent tau=rho/(rho+epsilon), 200 log-domain iterations, no tuning.
Use at most 2048 cells from each distribution, RNG(560), costs squared Euclidean
/ median of fit-to-source squared distances. Frozen column potential extends
the barycentric map to unseen images; L2-normalize projected latent before
frozen marker head. This is an inductive extension, not OT refitted on evaluation.
Optional PBC conditional OT fits three maps on A anchor labels and N008 source
pseudo labels; evaluation routing uses the frozen student class head, never true
evaluation classes. Do not run conditional OT on MLL23 (no anchors authorized).

## RQ3 locked before clinical predictions

Working textbook expectations, fractions in [0,1]:

| Clinical class | Marker | Fixed expectation | Interpretation |
|---|---|---:|---|
| neutrophil / neutrophil_segmented | single:CD16 | .95 | mature positive phenotype; .95 is a declared working target, not a universal measured percentage |
| lymphocyte | single:CD3 | .70 | typical adult T-cell share; reference interval 51.3–83.5% |
| monocyte | single:CD14 | .95 | high/positive majority; includes intermediate cells; not the classical subset frequency |
| eosinophil | single:CD16 | .00 | resting peripheral eosinophils negative |

Sources: [healthy adult lymphocyte reference study](https://pubmed.ncbi.nlm.nih.gov/25450652/),
[monocyte subset primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5406034/),
[JSLH differential SOP: CD16 neutrophils, CD14 monocytes](https://jslh.kenkyuukai.jp/images/sys/information/20230410155539-287B6537AF7256D51C818052377BA23D7B3FABEF54065EA18F1156A17D614D30.pdf),
[flow cytometry guidelines: eosinophils lack CD16](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/38563/Nourshargh%20Guidelines%20for%20the%20use%202017%20Accepted.pdf?isAllowed=y&sequence=1).
The near-unity targets are explicit approximations to qualitative phenotypes.
MLL23 disease classes do not have universal normal proportions: comparisons
are descriptive compatibility with these references, not measured diagnostic error.

Positive-rate rule: for each single marker and batch, fit a deterministic
2-component Gaussian mixture to clean source training stained-cell T values
(targets_v3c.npz, undetrended asinh readouts); positive = posterior(high-mean
component) >= .5. Require >=100 observations, both weights >=.02 and mean
separation >=1 pooled component SD; otherwise unavailable. This estimates an
assay-specific positive mode, not a validated flow gate. No clinical labels or
textbook rates enter mixture fitting.

For each source class c and marker j, rate pi_cj = mean of those binary source
readouts among measured stained split-0 cells of class c. Threshold q_cj =
quantile_(1-pi_cj) of that student's GS split-0 marker predictions in class c
(linear quantile); positive iff prediction > q_cj. pi=0/1 gives +inf/-inf.
Use the SAME frozen threshold for baseline, OT and conditional OT; never
recalibrate after alignment. Require >=100 measured stained and >=30 GS
training cells for a class-marker threshold. Quantile tie error is reported.

Identifiability limitation discovered from source metadata: expert cls3 gates
exist on pooled-marker cells only; all eight individual-marker masks are false
on those cells. Histology cells also lack expert class labels. Consequently the
literal measured-class/single-marker RQ3 is unavailable, not a zero fraction.
Primary output records this explicitly. A separately labeled exploratory proxy
analysis is implemented: fit a fixed alpha=10 one-hot ridge classifier on
split-0 expert cls3 LF features; apply to single-stained cells, and use existing
N008 LF pseudo classes for GS training cells. Granulocyte proxies neutrophil;
eosinophil uses the granulocyte threshold as a transfer diagnostic, with NO
claim that source eosinophils were identified. Other clinical classes have no
class-specific threshold and remain unavailable. All eight marker fractions
are exported where support exists; absolute errors are reported only for the
four fixed reference pairs above. Missing mixtures/support remain NA.

Threshold files, source training predictions, and this report's SHA256 are
sealed before the first clinical prediction. Evaluation refuses a changed
preregistration or missing seal. Thresholds are teacher/student specific;
all fit/calibration cell IDs are saved for auditing.

## Locked summaries and decision rule

Export every contrast per teacher/arm, then mean and sample SD over exactly
three teachers (no pooling of cells/predictions across runs), paired H−T,
T+OT−T, H+OT−H, and conditional counterparts. Report macro mean across 25,
PBC 10, MLL23 15, MLL23 fine 12. No hyperparameter search. Missing contrast
classes return NA and invalidate the corresponding complete macro score.
K025 strata: PBC robust = six anchor lineage contrasts + IG CD16 +
erythroblast CD45, failing=basophil CD123, teacher-dependent=eosinophil CD16.
MLL23 robust=three lineage references + hairy HLA-DR + myelocyte CD16;
failing=LGL CD56/CD16, basophil CD123, plasma/normoblast CD45, hairy CD19/CD3;
teacher-dependent=eosinophil CD16; remaining hairy CD123/band CD16 separate.
Mean within-slide autofl Pearson r uses v6c masked GS split-2 truth, separately
by slide and teacher (historical expectation ~.56/.85). No pooled-slide r.
Declare best only after all 3 paired runs, using 25-contrast macro mean;
report fine/robust strata even if they disagree. Positive-fraction recovery
cannot be established from these labels alone, irrespective of proxy MAE.

## Execution results

CPU smoke passed: exact copied ASTs; real archive filename split audit (PBC
8546 fit / 8546 eval, MLL23 7268 fit / 7265 eval); UOT identity/serialization,
fixed unseen-point extension and conditional routing; synthetic 25-contrast and
three-teacher paired aggregation; RQ3 missing support and source-only mixture
positivity; tiny actual source/anchor-A training; OS SIGTERM exit 143 followed by
bitwise identical resumed parameters; completed-run idempotence; full t0 T model
loading and eight-view CPU inference on two source training cells. See
cpu_smoke.log. The synthetic summary printed there is a test fixture, not a
clinical result. Source LF hist/all feature equality was checked on ten cells.

Pending GPU execution. No clinical performance conclusion is asserted. Exact
launch commands and a conditional 4.1–5.8 H100 GPU-hour estimate are in README.md.
The primary RQ3 requested measurement remains non-identifiable with available
expert class gates; proxy analyses cannot resolve this limitation.
