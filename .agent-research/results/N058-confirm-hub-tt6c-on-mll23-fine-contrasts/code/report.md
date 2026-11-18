# N056 v2 preregistration — 2026-11-18

Protocol registered before any new clinical predictions. GPU execution belongs to
Claude, as in the original writer brief; this handoff supplies code and CPU checks.
Historical red-team findings are inputs, not new v2 clinical outcomes.

## P1-1: locked claim rule and seed design

H student seeds 0 and 1 for each teacher t0/t1/t2; H_matched seed 0 for each.
T reuses ALL saved students: n029_tt6c_s0/s1/s2 (t0),
n032_tt6c_t1_s0/s1 and n032_tt6c_t2_s0/s1.
For each contrast or macro, average seeds within teacher first; average the three
paired teacher differences second, giving each teacher equal weight. An H-vs-T
claim requires abs(teacher-paired mean difference) to EXCEED the maximum
within-teacher student-seed range of EITHER arm. Otherwise say "no difference".
A negative difference exceeding the floor supports T, not H. H_matched has only
one seed per teacher: its own noise floor is unmeasured, so comparisons remain
sensitivity analyses and never establish superiority.
H reads BOTH teachers' features (marker teacher + common v6c autofl teacher).
Do not compare across-teacher SDs between arms; they do not measure comparable
independent teacher variability. Export seed observations and ranges, no SD ranking.

## P1-2: primary contrasts and anchor gradient matching

Primary PBC contrasts (held-out biology relative to the three anchor classes):
- CD16 neutrophil>eosinophil
- CD123 basophil>rest
- CD16 neutrophil>ig (held-out)
- CD45 lymphocyte>erythroblast (held-out)
These contrast held-out classes against reference anchors; they necessarily retain
reference-anchor cells. This is the brief's eos/ig/baso/erythroblast definition.
Primary MLL23 contrasts:
- LGL>lym CD56
- LGL>lym CD16
- hairy>lym CD123
- hairy>lym CD19
- hairy>lym HLA-DR
- hairy<lym CD3
- plasma<lym CD45
- eos<segneut CD16
- band<segneut CD16
- myelocyte<segneut CD16
- baso>rest CD123
- normoblast<lym CD45
Exclude segneut>lym CD16, mono>lym+neut CD14, lym>neut CD3 from primary.
Report PBC-primary4, MLL23-primary12 and combined primary16; all25 is secondary.

H uses latent squared-distance sum + .5 CE (source pseudo-class CE + PBC A CE).
H_matched instruments an independent T-recipe student from DinoBloom initialization
for its first 200 updates on EXACTLY the same augmented source/anchor batches as H.
T terms are original train_gs_bind SigLIP, per-output ridge distillation and .5 CE;
original masks and teacher-specific normalization are retained for T ridge targets.
The shared parameter set for gradient norms is the trainable ViT blocks (8–11).
Define share = w*||g_CE|| / (sum of non-anchor term norms + w*||g_CE||).
This is a norm-budget share, not a norm of the summed gradient (cancellation).
For steps 1–200 choose H's weight each step to equal that batch's measured T
share exactly. Freeze the arithmetic mean of these 200 weights thereafter.
This preregistered adaptive calibration avoids a clinical or loss-magnitude search;
export both shares, every term norm and chosen weights. Vanishing norms fail closed.
Log H term norms in every run for first 200 steps and every 100 steps afterward.
Matching is a sensitivity control; it does not remove all recipe differences.

## P2-1 / P2-7: production hub and loading

N054 A3 architecture: 1024 concatenated LF features -> 256 normalized latent,
masked target MSE + marker SigLIP; 512-wide depth-2 MLPs. Every teacher uses
lf_lr=1e-3, batch=256, 2500 steps, K019 batch==1 target/group index 8 masked.
Final iterate is fixed in advance (no clinical selection). Freeze hub for student.
Student: DinoBloom ViT-B/14, freeze first 8 blocks, 3000 steps, batch128,
AdamW lr1e-4 wd.05, OneCycle pct_start .05, gradient clip1.
Augmentation and legacy anchor loader remain verbatim; wrapper filters duplicates.
DataLoader uses 12 workers; deterministic step-addressed batches and private loader
RNG permit optimizer/RNG resume. H hub seed0, independent student seeds0/1.
GS source split0 only; split2 used for same-cell autofl sanity only.

## P1-3 / P2-6: alignment attribution and representation

All alignment fits AND applications use normalized EIGHT-VIEW MEAN embeddings;
raw arm keeps legacy mean of eight decoded predictions. No clinical tuning.
Controls fit the same cached fit/source vectors: per-dimension mean shift;
CORAL mean match + regularized covariance whitening/colouring (ridge1e-4);
kNN-to-source mean of k=10 nearest source embeddings. All mapped vectors are
L2-normalized before the frozen head. Controls use the same RNG560 support cap2048
as OT, and are reported beside OT, not selected by performance.
Unbalanced OT epsilon=.05, tau=.91 AND .30 fixed a priori;
rho=epsilon*tau/(1-tau), 200 log-domain iterations, squared distance / fit median.
Keep PBC anchor-conditional CCOT at tau=.91; frozen C routes mean embeddings.
PBC split: globally sorted JPEG alternating A/B BEFORE duplicate exclusion.
MLL23: sorted per archive, RNG0 first1500, alternate fit/eval. Its sampling frame
shares evaluation class composition: TRANSDUCTIVE composition information,
although fit and eval image IDs are disjoint. This is a reused development benchmark.
No eval pixel enters fits, thresholds, hyperparameter selection or duplicate-derived
model decisions; duplicate audit necessarily reads thumbnails for hygiene only.

## P2-3 / P2-4: storage and contamination

Default GPU run root /hnvme/workspace/v181bb10-bsccm/runs/n056v2_main.
Each inference cache is ONE transactional SQLite file; completed batches resume
inside that file, no .parts files or second concatenated copy. Code never writes
to /home except the requested v2 handoff. CPU scratch is /tmp/codex_n056v2.
PBC cross-half CRC32+size exact groups and ALL cross-half thumbnail pairs with
centered RGB24x24 cosine >=.995 are excluded from BOTH original halves and H
anchors. n056_duplicates.py writes the complete excluded filenames and pair evidence
to pbc_excluded.json; clinical eval refuses a missing/mismatched archive audit.
Existing T training cannot be retroactively decontaminated: evaluation removes both
members; disclose historical T exposure to removed anchor images as a limitation.

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


## P1-4: RQ3 interpretation

Export transfer_drift=|fraction-pi_c| and bsccm_calibration=|pi_c-textbook|
separately, as well as descriptive fraction-vs-expectation absolute_error.
Interpret RQ3 only for raw arms. All aligned rows (including trivial controls) say
"not interpretable (shrinkage toward source)"; reducing drift through alignment
cannot validate positivity recovery. Literal measured-class calibration remains
unavailable; proxy results retain their explicit exploratory labels.

## Evidence and execution status

Red-team scripts c1–c7 and hub_lr1e-3.json/hub_lr1e-4.json were read directly.
P1-1: c1/brief establish seed variability; aggregation now enforces the floor.
P1-2: original train_gs_bind terms were inspected; instrumented replay and
same-batch gradient accounting replace the unjustified equal-weight argument.
P1-3: c6/c7 establish UOT numerical behavior and shrinkage; controls and fixed taus
are now compulsory. P2-6 mean-fit/per-view-apply mismatch removed.
P2-1: c4 and hub JSON compare production vs older recipe; production is locked.
P2-4: c2/c3 motivate CRC+thumbnail exclusion, audited independently here.
P2-2/P2-3/P2-7: explicit teacher caveat, single-file caches and 12-worker loading.
CPU verification evidence is cpu_smoke.log and validation.md; they are separate
from this immutable preregistration so completed-run hashes remain stable.
No new clinical numerical result or GPU timing is asserted in this report.

## N058 addendum (Claude, 2026-11-18)
Confirmation run of H_matched with seeds 0-2 per teacher; the same registered seed-floor rule now applies to H_matched vs T
(sensitivity flag off). Primary endpoint MLL23_15 (pre-registered in N058 item), secondary primary16, all25, PBC10.
