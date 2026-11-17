**N047 diagnosis red team — BREAKS: the measurements reproduce; saturation and causal attribution remain unproven.**

A simpler explanation fits equally well: these models share a representation, initialization, training recipe, supervision and errors, and average away little additional variance. Finite supervision and ordinary cross-draw generalization error can coexist with substantial label artefacts. The evidence does not separate their contributions sufficiently to conclude that the remaining gap to .67 is mostly evaluation-label noise, or that this predictor family cannot close it. Conversely, this audit does not establish that .67 is achievable inductively.

Read: GOAL.md; KNOWLEDGE.md, including K001/K019/K023/K028; N047 attempts.md, declined.md and diagnosis.md. The declined directions remain binding. No training, jobs, label changes, source edits or fixes were performed. Dependencies, scripts and intermediate results are under `/tmp/codex_n047/`; this requested report is the sole deliverable written in the repository tree. Computation used the uv venv, one BLAS/OpenMP thread and saved predictions, not apptainer. The independent audit reported 0.199 CPU seconds; the remaining checks also completed in seconds.

**What survives reproduction.** Equal weighting of the 11 targets and two directions gives:

| Predictions | Frozen pooled GOAL | Within FOV, all FOVs | Within FOV, ≥3 labelled cells |
|---|---:|---:|---:|
| N031 base + zoom2 | .652741 | .720545 | .721100 |
| N037 base + zoom2 | .653675 | .721282 | .722138 |
| N040 students | .656371 | .724592 | .724916 |
| N046 students | .654589 | .722881 | .723314 |
| All 10 | .657061 | .725879 | .726266 |

All-10 direction scores are b10 .664762 and b01 .649360. Its gain over N031 is .004320; its remaining target gap is .012939. The project's fixed .0054 flat threshold classifies that gain as flat. That is a stopping convention, not an estimate of a performance ceiling or a confidence interval for this ensemble. N038's seed estimate comes from two single-model seed pairs; transferring it to this correlated ten-model comparison does not supply either interval.

The three original scripts regenerate N047 D1's 66 rows, N038 D2's 22 rows and N039's 88 rows to tolerance 1e-12. The ten prediction files in each direction have exactly identical, unique `G` arrays: 13,113 b10 and 26,322 b01 rows, all split 2, correct draw, no Giemsa. The N039 references use exactly these same test rows after draw selection. Thus neither row misalignment nor a different evaluation cohort explains these results. N047 includes all FOVs when centering; N038 restricts to FOVs with ≥3 labelled cells. This explains .720545 versus .721100 and is a minor metric-definition difference, not the main objection.

**The ten-model average is a valid score, but a weak saturation test.** For each target/direction I computed all 45 pairwise prediction correlations and all 45 residual correlations, using residual `P - Y_v5` on the same masked cells. The table averages those correlations across the eleven targets, rather than pooling incomparable marker scales.

| Test direction | Mean prediction correlation | Mean residual correlation | Residual correlation after individual oracle affine calibration |
|---|---:|---:|---:|
| b10 | .962684 | .953145 | .967235 |
| b01 | .950443 | .972722 | .975337 |

The targets central to the diagnosis are particularly redundant:

| Target | b10 prediction / residual correlation | b01 prediction / residual correlation |
|---|---:|---:|
| single:CD19 | .946820 / .984645 | .875103 / .988231 |
| single:CD56 | .947258 / .990366 | .927500 / .982763 |
| single:CD3 | .989571 / .993310 | .964952 / .982843 |

Saved `result.json` arguments verify all ten members use ResNet-50, 10 epochs, batch size 256, learning rate .0005 and seed 0, with the same N031 inductive encoder within each direction. N037 changes the training labels to v5cx; the other listed models use v5. Crops, curation, augmentation and student losses vary, but these are closely related trained predictors, not ten independent challenges to the family. N040's teacher is N031 and N046's teacher is N040, as recorded in their plans/jobs.

For intuition only, the equal-variance, equal-correlation approximation `n_eff = 10 / (1 + 9*rho_residual)` averages to 1.045 in b10 and 1.025 in b01. This is a variance-reduction diagnostic, not an effective count of experimental replicates or a literal learning-theoretic bound. Error variances differ. Shared noisy labels themselves induce correlated residuals, so high correlation does **not** prove model bias rather than noise. It establishes precisely why a small averaging gain cannot distinguish the two explanations. Failure of equal-weight averaging also cannot bound all functions in the predictor family.

**Within-FOV correlation is a conditional diagnostic, not an achievable pooled score or a noise ceiling.** Both predictions and held-out labels are centered using the empirical mean of the target's labelled cells in each FOV. It estimates association after removing all between-FOV means. It uses held-out label information to define the evaluation; it is legitimate as a diagnostic, but is not a deployed predictor's pooled performance.

For the implicated single targets, the actual fraction of label variance removed by this operation on all FOVs is:

| Target | b10 | b01 |
|---|---:|---:|
| CD19 | .660297 | .569136 |
| CD56 | .622199 | .637055 |
| CD3 | .548582 | .452861 |

These are raw between-FOV shares, **not** N038's permutation-corrected excesses. N038's .342–.488 excesses also reproduce. A permutation null establishes more grouping than shuffled labels would produce; it does not identify all that grouping as measurement error. Centering removes finite-sample differences in the cells' true marker composition as well as offsets. Actual excess attributable to real composition in these data is **[unverified]**. The point is identifiability, not a claim that slide coordinates carry biology or should be used as features; the binding prohibition on position features remains intact.

A minimal noise-free construction reproduces the logical ambiguity: let an exactly measured label equal within-group biology plus a between-group biological component with equal variance, and let predictions capture only the within-group component. Pooled r is .707107; within-group r is 1.000000. There is no label error. This construction does not establish the biological explanation in BSCCM; it disproves the inference that a within/pooled gap by itself demonstrates label error.

There is also considerable unbounded within-FOV prediction error: all-10 within r is .725879, not 1. As an explicitly nondeployable algebraic diagnostic, setting prediction to `Y - FOVmean(Y)` gives mean pooled r .812657 on these cells, despite removing *all* between-FOV variation. This does not prove LF can learn that signal. It shows that attributing the between-FOV component to unpredictable offsets alone does not mathematically force a ceiling near .657.

N039's per-cell-background intervention is additional mechanistic evidence for a label problem, beyond the within-FOV comparison. Its failed QC does not establish a validated alternative truth or identify the fraction of the residual GOAL gap caused by error. **[unverified in this audit]** The raw fluorescence reconstruction, candidate v8 labels and reported causal attribution to `background_lowess` were not rerun; this audit reproduced v5 prediction scores and FOV statistics, not that unmixing experiment. Calling 34–49% of variance proven *measurement error*, rather than excess FOV-associated variation, therefore goes beyond the reproduced statistic.

**N039 is a useful reference, not a ceiling or a clean label-definition intervention.** Its v5-trained scores reproduce:

| Arm | Test draw 0 | Test draw 1 | Mean |
|---|---:|---:|---:|
| Inductive N031 base | .648803 | .636296 | .642549 |
| FOV model, 128k | .646452 | .670469 | .658460 |
| FOV model, full | .662806 | .692364 | .677585 |

The .034173 improvement on test draw 1 and −.002351 on draw 0 are the **128k** contrasts. With the full reference they are +.056068 and +.014003. It is inaccurate to read “no shift cost in the other direction” without specifying the smaller reference: b10's inductive model trained on 269,283 cells, not 128,000. b01 used 127,946. N039 uses 317,935 or 128,000 mixed-draw training cells and the n007 transductive SSL initialization, rather than N031's inductive initialization. The 128k arm matches b01's total count approximately, not its draw composition or pretraining regime; it does not match b10's count. Full versus 128k additionally changes optimizer updates at fixed epochs.

Access to target-draw supervision can resolve real distribution shift or missing coverage as well as label calibration. This comparison changes those factors together. Hence “draw-specific label definitions cause the entire shift cost” is **[unverified] as a causal allocation**, even if separate QC makes label differences plausible. “An inductive model cannot learn a definition it never sees” needs the additional premise that the necessary correction has no transferable relationship to allowed LF inputs; these scores do not establish that premise. The .677585 result is not a valid inductive GOAL counterexample, but does show that the frozen labels do not impose a universal .657 scoring cap.

N040's learning curve is also batch-held-out, not an in-distribution result: weak-target means at 25/50/100% training cells are .3474/.3955/.4303, and the 50→100% gain is .0348. It changes images, labels and optimization exposure together. It cannot project a future gain, but it does not support describing all supervision gains as merely “in-distribution-like.” Real additional labelled examples and teacher-generated missing labels are different interventions. A second student round's failure tests that recursion, not the value of independent supervision.

**Scale audit of attempts.md.** These are judgments of the executed intervention; the table's labels are not measurements.

| Item | Recorded scale | Audit |
|---|---|---|
| N039 | module | Reasonable for the planned label pipeline. QC failed, so the GOAL retrain was cancelled; this is not a failed trained predictor module. The measured reference is diagnostic. |
| N042 | module | Planned architecture module was never run. Executed change was one low-pass input knob plus a diagnostic/gating decision. Counting it as a tested architecture module overstates exploration. |
| N041 | module | Reasonable: new instrument augmentation and consistency-loss component. Still the same scalar regression family. |
| N040 | paradigm | Defensible for a change to semi-supervised training signal, consistent with the notebook's usage. It is not a new predictor family: same ResNet, initialization, targets and regression task, with teacher-generated supervision. Treating “paradigm” as evidence of independent family coverage overstates it. Under a predictor-family taxonomy this would be a training module. |
| N046 | knob | Appropriate: change teacher/round while retaining the student recipe. |

Thus five entries do not represent five independently trained module/paradigm challenges to the GOAL. In particular, two entries stopped at diagnostics or QC. N042's reproduced blur result (.6425→.6411) only shows that this recipe is insensitive to that blur strength. It cannot establish that the model extracts all useful LF information, including information already present at coarser spatial scales or across illumination channels. The intended architecture was not tested. “All five assumed the frozen target stays reachable” is also **[unverified] as an attribution of intent**: QC and gating probes can be performed to test that premise.

**The stated falsifier is unnecessarily restrictive.** Under the diagnosis's own additive-offset model, `Y = S + U`, with U independent of predictions and signal, `corr(P,Y) = corr(P,S) * sqrt(var(S)/var(Y))`. A genuine improvement in predicting S therefore naturally improves within-FOV correlation more than pooled correlation. Requiring a successful model change to improve within no more than pooled excludes ordinary progress under the hypothesis being tested. The existing all-10 change illustrates the direction: pooled +.004320, within +.005335. Crossing .67 on the frozen inductive metric would refute a ceiling below .67 regardless of which correlation increased more. Likewise, FOV offsets could be genuine label artefacts while other target errors still improve enough to reach .67.

The defensible diagnosis is: **the tested related recipes have diminishing ensemble returns; frozen v5 labels contain substantial FOV-associated variation and documented QC concerns; the achievable inductive score and the causal share of the remaining gap are unresolved.** This supports neither declaring the family saturated nor treating a metric rebase as a demonstrated necessity. GOAL labels stay frozen, the .67 proposal stays unconfirmed, and no declined direction is reopened here.

**Reproduction commands and actual output.** The original scripts are run with output redirected to scratch. Python 3.13.13, numpy 2.5.5, pandas 3.1.1 were used. Save the two independent script listings below as `/tmp/codex_n047/audit.py` and `/tmp/codex_n047/provenance.py` before running this block. The commands only read repository/data files and write scratch outputs. Pandas printed future API warnings about positional `mean(1)` in the original scripts; all runs completed successfully.

```bash
mkdir -p /tmp/codex_n047
UV_CACHE_DIR=/tmp/codex_n047/uv-cache uv venv /tmp/codex_n047/venv
UV_CACHE_DIR=/tmp/codex_n047/uv-cache uv pip install --python /tmp/codex_n047/venv/bin/python numpy==2.5.5 pandas==3.1.1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/tmp/codex_n047/venv/bin/python
R=/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results
"$PY" "$R/N047-review-2-lf-mk-goal-6527-target-67-after/d1_big_ensemble.py" /tmp/codex_n047
"$PY" "$R/N038-review-lf-mk-common-11-batch-held-out-go/d2_within_fov.py" /tmp/codex_n047
"$PY" "$R/N039-label-v8-re-derive-single-antibody-marke/n039_ceiling.py" /tmp/codex_n047
"$PY" /tmp/codex_n047/audit.py
"$PY" /tmp/codex_n047/provenance.py
"$PY" "$R/N040-dense-label-completion-noisy-student-lab/n040_lc_score.py" /tmp/codex_n047
"$PY" "$R/N042-weak-marker-architecture-low-pass-input/n042_score.py" /tmp/codex_n047
"$PY" "$R/N041-physics-based-source-only-nuisance-rando/n041_n040_score.py" /tmp/codex_n047 /tmp/codex_n047
```

The independent audit computes ensemble means in float64; original scripts average float32 predictions. Scores agree at the shown precision. FOV IDs were checked not to combine different slides within a target/draw. Affine calibration is an oracle sensitivity diagnostic only and is never used to report the GOAL.

**Independent audit: actual output.**

```text
ALIGNMENT b10 models 10 identical_unique_G 13113
N039 reference test rows exactly match b10
ALIGNMENT b01 models 10 identical_unique_G 26322
N039 reference test rows exactly match b01

RECOMPUTED SCORE MEANS (equal target and draw weighting)
         pooled  within_all  pooled_ge3  within_ge3
arm                                                
N031   0.652741    0.720545    0.653220    0.721100
N037   0.653675    0.721282    0.654752    0.722138
N040   0.656371    0.724592    0.656301    0.724916
N046   0.654589    0.722881    0.654714    0.723314
all10  0.657061    0.725879    0.657244    0.726266

ALL10 BY DIRECTION
       pooled  within_all
dir                      
b01  0.649360    0.722495
b10  0.664762    0.729264

PAIRWISE CORRELATIONS (45 member pairs per target/draw)
     pred_mean  pred_min  residual_mean  residual_min  affine_residual_mean  equalvar_neff
dir                                                                                       
b01   0.950443  0.904802       0.972722      0.937553              0.975337       1.025378
b10   0.962684  0.927577       0.953145      0.909652              0.967235       1.045474
Across target/draw pairs: prediction mean range 0.807601 0.98994 residual mean range 0.856265 0.99331

WEAK TARGET DIVERSITY
dir      target  pred_mean  residual_mean  equalvar_neff
b10 single:CD19   0.946820       0.984645       1.014013
b10 single:CD56   0.947258       0.990366       1.008747
b10  single:CD3   0.989571       0.993310       1.006058
b01 single:CD19   0.875103       0.988231       1.010706
b01 single:CD56   0.927500       0.982763       1.015758
b01  single:CD3   0.964952       0.982843       1.015684

WITHIN-FOV LABEL VARIANCE REMOVED (all FOVs, no null subtraction)
dir      target    n  nfov  kept  label_share
b10 single:CD19 1299   301  1181     0.660297
b10 single:CD56 1500   299  1392     0.622199
b10  single:CD3  629   110   601     0.548582
b01 single:CD19  722    82   715     0.569136
b01 single:CD56 2112   317  2024     0.637055
b01  single:CD3 2968   341  2908     0.452861

PERFECT WITHIN-FOV ORACLE, ZERO BETWEEN-FOV COMPONENT
mean pooled r if p = y - FOVmean(y): 0.812657

N039 SAVED TABLE SUMMARY (raw predictions independently reproduced by n039_ceiling.py)
                         r  r_within
arm         draw                    
fov_v5_128k 0     0.646452  0.700773
            1     0.670469  0.745605
fov_v5_full 0     0.662806  0.721328
            1     0.692364  0.771167
goal_base   0     0.648803  0.716282
            1     0.636296  0.707108
goal_ens    0     0.661222  0.725274
            1     0.644260  0.715816
arm draw-balanced means {'fov_v5_128k': 0.65846, 'fov_v5_full': 0.677585, 'goal_base': 0.642549, 'goal_ens': 0.652741}

NO-LABEL-ERROR CONSTRUCTION: y = within-cell biology + between-group biology; p = within-cell biology
pooled_r 0.707107 within_r 1.0
Audit CPU seconds 0.199
```

**CSV equality and model configuration: actual output.**

```text
d1_big_ensemble.csv MATCH 66 rows
d2_within_fov.csv MATCH 22 rows
n039_ceiling_per_target.csv MATCH 88 rows
b10 arch ['resnet50']
b10 epochs ['10']
b10 bs ['256']
b10 lr ['0.0005']
b10 seed ['0']
b10 init_ssl ['/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/encoder_final.pt']
b10 training label versions ['targets_v5.npz', 'targets_v5cx.npz']
b01 arch ['resnet50']
b01 epochs ['10']
b01 bs ['256']
b01 lr ['0.0005']
b01 seed ['0']
b01 init_ssl ['/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex1/encoder_final.pt']
b01 training label versions ['targets_v5.npz', 'targets_v5cx.npz']
n031_base_b10 n_train 269283 init_ssl /hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/encoder_final.pt
n031_base_b01 n_train 127946 init_ssl /hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex1/encoder_final.pt
n039_ceil_fov_v5_mt0 n_train 317935 init_ssl /hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt
n039_ceil_fov_v5_mt128000 n_train 128000 init_ssl /hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt
```

**Original learning-curve scorer: actual output.**

```text
frac       25      50      100
group                         
other7  0.7500  0.7596  0.7639
weak4   0.3474  0.3955  0.4303
dir      b01     b10    goal
frac                        
25    0.5846  0.6227  0.6036
50    0.6233  0.6310  0.6272
100   0.6363  0.6488  0.6425
frac                 25     50     100
target                                
grp:CD16           0.792  0.789  0.795
grp:CD3/CD19/CD56  0.765  0.774  0.778
grp:CD45           0.679  0.687  0.686
single:CD123       0.337  0.410  0.502
single:CD14        0.817  0.832  0.842
single:CD16        0.883  0.892  0.900
single:CD19        0.218  0.311  0.345
single:CD3         0.560  0.579  0.570
single:CD45        0.755  0.763  0.776
single:CD56        0.342  0.368  0.363
single:HLA-DR      0.493  0.493  0.511
weak4 gain 50->100 %: 0.0348 -> PASS (continue)
```

**Original low-pass scorer: actual output.**

```text
dir                  b01               b10                      b01
arm                 base lowpass1.5   base lowpass1.5   drop   drop
target                                                             
grp:CD16           0.799      0.799  0.791      0.805 -0.014 -0.000
grp:CD3/CD19/CD56  0.790      0.788  0.766      0.762  0.003  0.002
grp:CD45           0.650      0.648  0.722      0.717  0.005  0.002
single:CD123       0.419      0.413  0.586      0.580  0.005  0.007
single:CD14        0.824      0.823  0.860      0.858  0.002  0.001
single:CD16        0.898      0.898  0.901      0.903 -0.002  0.000
single:CD19        0.385      0.381  0.305      0.286  0.019  0.004
single:CD3         0.588      0.585  0.552      0.556 -0.004  0.003
single:CD45        0.845      0.845  0.707      0.709 -0.002  0.001
single:CD56        0.358      0.353  0.368      0.369 -0.001  0.005
single:HLA-DR      0.443      0.446  0.579      0.580 -0.002 -0.003
dir            b01     b10    goal
arm                               
base        0.6363  0.6488  0.6425
lowpass1.5  0.6344  0.6478  0.6411
{'single:CD19': np.float64(0.011), 'single:CD56': np.float64(0.002), 'single:CD123': np.float64(0.006), 'single:HLA-DR': np.float64(-0.002), 'single:CD14': np.float64(0.001), 'single:CD16': np.float64(-0.001), 'single:CD45': np.float64(-0.001)}
VERDICT image-limited: False
```

**Original N040/N041 scorer: actual output.**

```text
dir                             b01     b10    goal
arm                                                
N031 base                    0.6363  0.6488  0.6425
N031 zoom2                   0.6333  0.6552  0.6442
N041 phys+tail               0.6374  0.6521  0.6447
N041 phys                    0.6360  0.6540  0.6450
N040 student zoom2           0.6342  0.6586  0.6464
N041 phys+tail + N031 zoom2  0.6441  0.6609  0.6525
N031 ensemble                0.6443  0.6612  0.6527
N040 student base            0.6459  0.6598  0.6529
N041 phys + N031 zoom2       0.6437  0.6626  0.6531
N040 student ensemble        0.6477  0.6650  0.6564
target                       single:CD123  single:CD19  single:CD56  single:HLA-DR
arm                                                                               
N031 base                           0.502        0.345        0.363          0.511
N031 ensemble                       0.537        0.347        0.373          0.511
N031 zoom2                          0.511        0.320        0.365          0.495
N040 student base                   0.542        0.360        0.369          0.512
N040 student ensemble               0.541        0.357        0.378          0.511
N040 student zoom2                  0.489        0.338        0.374          0.498
N041 phys                           0.505        0.328        0.376          0.516
N041 phys + N031 zoom2              0.535        0.337        0.379          0.513
N041 phys+tail                      0.533        0.312        0.364          0.504
N041 phys+tail + N031 zoom2         0.547        0.328        0.373          0.509
```

**Independent audit script.**

```python
import time
from pathlib import Path
import numpy as np
import pandas as pd
start=time.process_time()
R=Path('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results')
W=Path('/hnvme/workspace/v181bb10-bsccm')
T=dict(np.load(W/'targets_v5.npz',allow_pickle=True))
names=list(map(str,T['names'])); cols=[j for j in range(12) if 'CD123/HLA' not in names[j]]
fam=['n031_base','n031_zoom2','n037_base','n037_zoom2','n040_student_base','n040_student_zoom2','n041_phys','n041_phystail','n046_student2_base','n046_student2_zoom2']
def load(run):
 with np.load(W/'runs'/run/'eval_preds.npz') as e: return e['P'],e['G']
def corr(x,y): return np.corrcoef(x,y)[0,1]
def center(x,inv): return x-(np.bincount(inv,x)/np.bincount(inv))[inv]
def offdiag(c): return c[np.triu_indices(len(c),1)]
scores=[]; diversity=[]; fovrows=[]
for d,b in [('b10',0),('b01',1)]:
 models=[load(f'{m}_{d}') for m in fam]; G=models[0][1]
 assert all(np.array_equal(g,G) for _,g in models)
 assert len(np.unique(G))==len(G)
 assert np.all(T['batch'][G]==b) and np.all(T['split'][G]==2) and np.all(T['hist_row'][G]<0)
 print('ALIGNMENT',d,'models',len(models),'identical_unique_G',len(G))
 P=np.stack([p for p,_ in models]).astype(float)
 ensembles={'N031':P[:2].mean(0),'N037':P[2:4].mean(0),'N040':P[4:6].mean(0),'N046':P[8:10].mean(0),'all10':P.mean(0)}
 for j in cols:
  sel=T['mask'][G,j]; y=T['Y'][G[sel],j].astype(float); x=P[:,sel,j]
  cp=offdiag(np.corrcoef(x)); ce=offdiag(np.corrcoef(x-y[None,:]))
  xc=x-x.mean(1,keepdims=True); yc=y-y.mean(); slopes=(xc@yc)/(xc*xc).sum(1)
  cca=offdiag(np.corrcoef(slopes[:,None]*xc-yc))
  diversity.append(dict(dir=d,target=names[j],pred_mean=cp.mean(),pred_min=cp.min(),residual_mean=ce.mean(),residual_min=ce.min(),affine_residual_mean=cca.mean(),equalvar_neff=10/(1+9*ce.mean())))
  _,inv,count=np.unique(T['fov'][G[sel]],return_inverse=True,return_counts=True)
  assert all(len(np.unique(T['slide'][G[sel]][inv==k]))==1 for k in range(len(count)))
  keep=count[inv]>=3
  fovrows.append(dict(dir=d,target=names[j],n=len(y),nfov=len(count),kept=int(keep.sum()),label_share=np.var(y-center(y,inv))/np.var(y)))
  for arm,pred in ensembles.items():
   z=pred[sel,j]; _,invk=np.unique(T['fov'][G[sel]][keep],return_inverse=True)
   scores.append(dict(arm=arm,dir=d,target=names[j],pooled=corr(z,y),within_all=corr(center(z,inv),center(y,inv)),pooled_ge3=corr(z[keep],y[keep]),within_ge3=corr(center(z[keep],invk),center(y[keep],invk))))
 for run in ['n039_ceil_fov_v5_mt0','n039_ceil_fov_v5_mt128000']:
  p,g=load(run); assert np.array_equal(g[T['batch'][g]==b],G)
 print('N039 reference test rows exactly match',d)
S,D,F=map(pd.DataFrame,[scores,diversity,fovrows])
S.to_csv('/tmp/codex_n047/scores.csv',index=False); D.to_csv('/tmp/codex_n047/diversity.csv',index=False); F.to_csv('/tmp/codex_n047/fov.csv',index=False)
print('\nRECOMPUTED SCORE MEANS (equal target and draw weighting)')
print(S.groupby('arm')[['pooled','within_all','pooled_ge3','within_ge3']].mean().round(6).to_string())
print('\nALL10 BY DIRECTION')
print(S[S.arm=='all10'].groupby('dir')[['pooled','within_all']].mean().round(6).to_string())
print('\nPAIRWISE CORRELATIONS (45 member pairs per target/draw)')
print(D.groupby('dir')[['pred_mean','pred_min','residual_mean','residual_min','affine_residual_mean','equalvar_neff']].mean().round(6).to_string())
print('Across target/draw pairs: prediction mean range',D.pred_mean.min().round(6),D.pred_mean.max().round(6),'residual mean range',D.residual_mean.min().round(6),D.residual_mean.max().round(6))
weak=['single:CD19','single:CD56','single:CD3']
print('\nWEAK TARGET DIVERSITY')
print(D[D.target.isin(weak)][['dir','target','pred_mean','residual_mean','equalvar_neff']].round(6).to_string(index=False))
print('\nWITHIN-FOV LABEL VARIANCE REMOVED (all FOVs, no null subtraction)')
print(F[F.target.isin(weak)].round(6).to_string(index=False))
print('\nPERFECT WITHIN-FOV ORACLE, ZERO BETWEEN-FOV COMPONENT')
print('mean pooled r if p = y - FOVmean(y):',np.sqrt(1-F.label_share).mean().round(6))
print('\nN039 SAVED TABLE SUMMARY (raw predictions independently reproduced by n039_ceiling.py)')
C=pd.read_csv(R/'N039-label-v8-re-derive-single-antibody-marke/n039_ceiling_per_target.csv')
print(C.groupby(['arm','draw'])[['r','r_within']].mean().round(6).to_string())
print('arm draw-balanced means',C.groupby('arm').r.mean().round(6).to_dict())
print('\nNO-LABEL-ERROR CONSTRUCTION: y = within-cell biology + between-group biology; p = within-cell biology')
within=np.tile([-1.,1.,-1.,1.],2); group=np.repeat([-1.,1.],4); y=within+group; inv=(group>0).astype(int)
print('pooled_r',round(corr(within,y),6),'within_r',round(corr(center(within,inv),center(y,inv)),6))
print('Audit CPU seconds',round(time.process_time()-start,3))
```

**CSV equality and model configuration script.**

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
R=Path('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results'); W=Path('/hnvme/workspace/v181bb10-bsccm')
pairs=[('N047-review-2-lf-mk-goal-6527-target-67-after','d1_big_ensemble.csv'),('N038-review-lf-mk-common-11-batch-held-out-go','d2_within_fov.csv'),('N039-label-v8-re-derive-single-antibody-marke','n039_ceiling_per_target.csv')]
for folder,name in pairs:
 a=pd.read_csv(R/folder/name); b=pd.read_csv(Path('/tmp/codex_n047')/name)
 pd.testing.assert_frame_equal(a,b,atol=1e-12,rtol=1e-12)
 print(name,'MATCH',len(a),'rows')
fam=['n031_base','n031_zoom2','n037_base','n037_zoom2','n040_student_base','n040_student_zoom2','n041_phys','n041_phystail','n046_student2_base','n046_student2_zoom2']
for d in ['b10','b01']:
 configs=[json.load(open(W/'runs'/f'{m}_{d}'/'result.json')) for m in fam]
 for k in ['arch','epochs','bs','lr','seed','init_ssl']:
  print(d,k,sorted(set(str(c['args'].get(k)) for c in configs)))
 print(d,'training label versions',sorted(set(c['args']['targets'] for c in configs)))
for run in ['n031_base_b10','n031_base_b01','n039_ceil_fov_v5_mt0','n039_ceil_fov_v5_mt128000']:
 c=json.load(open(W/'runs'/run/'result.json'))
 print(run,'n_train',c['n_train'],'init_ssl',c['args']['init_ssl'])
```
