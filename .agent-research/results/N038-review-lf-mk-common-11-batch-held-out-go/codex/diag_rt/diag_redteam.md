# N038 diagnosis red team

**BREAKS: the scores reproduce, but the ceiling and causal interpretation do not.** A simpler explanation is that the GOAL recipe has scarcely been challenged by the five listed attempts, while D1b compares different supervised labels, training cohorts, and evaluation regimes. Existing predictions already provide an in-distribution counterexample: **N022 tail05 scores .679976**, using the same split-2 cells, v5 labels, common-11 targets, and equal weighting of the two draws. This does not establish .67 on a new isolation; it rejects .658 as an upper bound for this predictor family.

All numerical findings below were recomputed from saved predictions unless marked otherwise. No training, jobs, input cleaning, position features, fluorescence inputs to LF models, or repository changes were performed. Only this report was written under results; all scripts, dependencies, logs, and temporary outputs are in `/tmp/codex_n038/`.

## What simpler explanation fits the attempts?

The recent sequence is mostly work on other objectives, followed by one narrow test of training-label curation. N027/N030 change the Giemsa student, N033 adds synthetic Giemsa training data, and N034 evaluates fluorescence images/distributions and set matching. Their void GOAL entries cannot show saturation of LF-to-marker regression. N037 is the only direct attempt in that table; its reproduced ensemble change is .652741 → .653675, +.000934. That supports a narrow conclusion about this cross-fitted-label intervention under this recipe, not a general conclusion that labels, resolution, representation, or training have reached their limits.

Thus **limited exploration of the actual objective, plus a confounded reference comparison**, fits the record without invoking an information ceiling. This is an interpretation of the recorded interventions, not proof that a particular alternative would improve the GOAL. The sentence saying all five attempts “assumed” a ceiling is **[unverified] as an attribution of intent**. N034's written plan explicitly investigates distributional/set-level behavior; it need not assume that scalar regression is optimal.

## D1b: valid matched-cell scores, invalid ceiling test

Both original scripts were rerun with their output directory redirected to scratch. Both regenerated CSVs match the originals numerically. D1b genuinely compares identical evaluation cells. This removes test-cohort mismatch within that comparison; it does not match the interventions that produced the models.

| Saved run | Supervised labels | Training cells | Training cohort | SSL initialization |
|---|---|---:|---|---|
| n015_sslinit_fov | v4 | 317,935 | split 0, both draws | n007, both draws |
| n023_v5_b10 | v5 | 269,283 | draw 1, all splits | n007, both draws |
| n023_v5_b01 | v5 | 127,946 | draw 0, all splits | n007, both draws |

All three use ResNet-50, 10 epochs, batch size 256, learning rate .0005, seed 0. Consequently:

1. **Different training-label definitions matter numerically.** All eight single-antibody labels and draw-0 common groups are identical between v4 and v5. On draw-1 split-2 cells, however, v4/v5 label correlations are .731836 for grp:CD3/CD19/CD56, .806196 for grp:CD45, and .777898 for grp:CD16. Per-slide normalization changes pooled correlations across slides; it is not merely one affine transformation that Pearson r would ignore. N015's draw-1 common-11 is .709325 against v4, versus .661177 against v5, with its predictions and cells unchanged. Its draw-balanced score correspondingly changes from .682262 to .658188. This sensitivity does **not** make v4 the right GOAL labels; it makes the v4-trained model an unsuitable v5 ceiling reference.
2. **“Same 10 epochs” does not mean equal optimization exposure.** The training counts differ substantially and asymmetrically, and FOV training uses source split 0 while batch training uses all source splits. “Half the data” is inaccurate. The contrast combines access to target-draw labels, source/target composition, sample count, loss targets, and optimizer updates. It is not an intervention that removes only draw shift.
3. **Both D1b arms are transductive in SSL.** n007's saved configuration says `rows='all'`, `exclude_batch=[]`, `n_rows=412941`. Even the “held-out” D1b model has seen target-draw images during SSL, including evaluation images. That is compatible with a transductive diagnostic control, but fails the inductive GOAL constraint. N031 instead uses encoders excluding the held-out draw. Reproduced N031 scores confirm the GOAL .652741; they do not make D1b an inductive causal comparison. Nor does this evidence establish that transduction explains away the scores: the recorded N031 aggregate control was essentially flat.
4. **The evaluation is split 2 only.** N015 has 39,435 unique non-Giemsa evaluation cells: 13,113 from draw 0 and 26,322 from draw 1. These are precisely the batch-model evaluation cohorts, so there is no hidden cell mismatch here. But two fixed draws, one FOV partition, and one trained instance per D1b arm cannot bound performance on a future isolation, another split, or other recipes. The .002 seed-noise estimate is not a population-level confidence bound on an information ceiling.

The D1b summary is a **net score difference**, with substantial cancellation:

| Target subset | Draw 0 FOV − batch | Draw 1 FOV − batch | Equal-draw mean |
|---|---:|---:|---:|
| common-11 | +.011541 | +.020296 | +.015918 |
| eight singles, unchanged v4/v5 labels | −.005976 | +.052162 | +.023093 |
| three common groups | +.058253 | −.064680 | −.003214 |

Across the 22 target/draw entries, positive contributions to the common-11 mean sum to +.032313 and negative contributions to −.016394. In draw 1 alone, grp:CD3/CD19/CD56 and grp:CD16 have gaps −.106418 and −.087848. This cancellation cannot justify “draw shift costs only .016.” Conversely, the unchanged-label single-target subset does not cleanly identify shift either, because training size and the multi-output training objective still differ.

Adding a batch-held-out ensemble gain of roughly .010 to the FOV single-model score is also **[unverified]**: correlation is nonlinear and gains depend on prediction-error covariance. No matched FOV base-plus-zoom ensemble was tested in D1b. The extrapolated .668 is not a measured ceiling.

## Existing counterexample to .658

Using the same cells and the actual v5 common-11 metric:

| FOV model | Draw 0 | Draw 1 | Equal-draw mean |
|---|---:|---:|---:|
| N015 SSL-init R50 | .655198 | .661177 | .658188 |
| N022 base R50 | .657514 | .660203 | .658859 |
| **N022 tail05 R50** | **.686617** | **.673334** | **.679976** |
| N024 supervised R18, v5-trained | .645697 | .674081 | .659889 |

N022 tail05 retains the SSL-init R50, 128-pixel input recipe, and v4 training labels; it changes tail weighting and the focal class head. Its evaluation-row array is exactly equal to N015's. It beats N015 by .021788 under the diagnosis's own evaluation conditions, without new image information. This is stronger evidence against a family-wide ceiling than an architecture mismatch. Its previously reported failure to improve held-out b1to0 remains compatible with this finding: **an in-distribution capacity/training result and generalization to a new draw are different questions.**

For uncertainty, a paired bootstrap resampled slide/FOV clusters within each draw, with equal weighting of draw means (1,000 replicates, seed 20261112). The 95% percentile interval for tail05 minus N015 is [.011940, .028988]. N015's mean interval is [.646425, .670094]; tail05's is [.663999, .693722]. These are conditional intervals for these trained models and these draws, with no adjustment for historical model selection. They reject treating .658 as an exact limit; they do not prove a population score above .67. The paired overall D1b gap interval is [.005498, .028719], describing that confounded contrast only.

The old “FOV ~.71 versus batch .634” numbers should indeed not be interpreted as a controlled shift experiment. Replacing them with a differently confounded .016 comparison does not solve that problem.

## Are the four dim-target conclusions robust?

Only the descriptive statement that several targets are weaker than CD14/CD16 is robust. The four do not exhibit one common failure mechanism.

| Single target | N015 FOV, draws 0 / 1 | N023 batch singles, draws 0 / 1 | N022 tail05 FOV, draws 0 / 1 | What the saved predictions support |
|---|---|---|---|---|
| CD19 | .279 / .467 | .280 / .365 | .325 / .437 | Weak and draw-dependent; draw-0 shift effect near zero, draw-1 positive |
| CD56 | .411 / .386 | .362 / .372 | .421 / .376 | Consistently weak across these recipes; no measured label-reliability ceiling |
| CD123 | .408 / .528 | .555 / .461 | **.646 / .676** | Strong sensitivity to training objective; low N015 score is not an information bound |
| HLA-DR | .614 / .599 | .560 / .454 | .640 / .605 | A persistent generalization gap, particularly draw 1; not evidence that draw shift is irrelevant |

The currently compliant N031 ensemble already reaches CD123 .614830 on draw 0, above N015's purported in-distribution reference .407966. N024's CD123 reference is weaker still (.304377/.399814). Calling those references a ceiling is empirically backwards for that target.

HLA-DR's D1b FOV-minus-batch gaps are +.054284/+.145156. The paired FOV bootstrap intervals are [.024866, .083515] and [.113254, .177785]. Both are positive in this conditional comparison; the cause remains confounded. CD19 and CD56 gaps vary by draw; CD123 gap intervals span zero in both draws. A uniform “four dim targets are capped” conclusion therefore exceeds the evidence.

HLA-DR is not even the fourth-lowest N015 target by draw-averaged correlation: single CD3 is .584988 versus HLA-DR .606430. Statements that high-scoring lineage markers are “at their ceiling” are also **[unverified]** without an independent upper-bound estimate. Adding .10 to any four targets would increase the 11-target mean by .03636; that arithmetic does not identify a mechanism or achievable intervention.

The fluorescence-integral evidence is especially easy to overread. Reading N035's stored JSON reproduces its reported within-draw correlations:

| Target | Draw 0 | Draw 1 |
|---|---:|---:|
| CD19 | .322119 | .558051 |
| CD56 | .536775 | .481121 |
| CD123 | .828996 | .708230 |
| HLA-DR | .879508 | .648520 |
| CD16 | .968576 | .965159 |

**[Unverified: the underlying N035 ridge fits were not rerun.]** Source inspection establishes that these are FOV-grouped cross-validation scores of RidgeCV on 24 sign-log fluorescence-integral features, targeting **v5c**, across that study's cohort. They are not technical replicate reliability estimates, nor the same split-2 v5 test. A low score for a restricted linear readout can reflect missing spatial/position-correction information, nonlinear relationships, or cross-FOV transfer; it does not show that the curated label mostly lacks the cell's photons. CD123/HLA-DR's own much higher stored within-draw values also resist lumping all four together.

**[Unverified]** Image-information limits, independent measurement-noise ceilings, and irreducibility of any particular target remain hypotheses. Neither dimness nor a low trained-model correlation identifies which applies. The attenuation bound `r <= sqrt(reliability)` requires a suitable measurement-error model and an independently valid reliability estimate. Bimodality alone is not such an estimate; spectral halves require justified independence and comparability. K004's pooled-group within-class results and downstream clinical failures also do not independently identify reliability of the four single-antibody labels.

## Scale audit of attempts.md

The following classifications are judgments using “knob” for settings of an existing method, “module” for a substantial component/pipeline change, and “paradigm” for a changed learning formulation. The underlying interventions were checked against the corresponding notebook items; their clinical outcomes were not rerun here.

| Attempt | Recorded | Assessment |
|---|---|---|
| N027 | module | Reasonable for B→L backbone replacement; trainable-block count and step count within that portfolio are knobs. Entirely downstream of GOAL. |
| N030 | module | **Knob**: changing how many existing DinoBloom-B blocks are frozen, holding the student formulation fixed. Calling this a module overstates the exploration breadth. |
| N033 | module | Reasonable: new virtual-stain rendering/supervision pipeline and measured-label synthetic examples. It changes student training, not the LF GOAL predictor. |
| N034 | paradigm | Defensible for set-level matching with hidden pairs. Paired CFM alone changes model/loss/output and can be called a module; a bundle-level paradigm label cannot make a void GOAL measurement a failed scalar-GOAL paradigm. |
| N037 | module | Reasonable for replacing the target-generation estimator with cross-fitting and clean calibration; predictor settings remain fixed. Only this row directly measures GOAL. |

The consequential error is counting four void objective measurements alongside one flat objective measurement as evidence for an exhausted GOAL predictor. N030's scale inflation adds to that impression. No new fixes or experiments are proposed or implemented in this review.

## Reproduction record

The prescribed container failed before Python started:

```sh
TMPDIR=/tmp/codex_n038/tmp XDG_CACHE_HOME=/tmp/codex_n038/cache \
APPTAINER_CACHEDIR=/tmp/codex_n038/cache APPTAINER_TMPDIR=/tmp/codex_n038/tmp \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm --bind /tmp/codex_n038 \
/hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I /tmp/codex_n038/audit.py
```

```text
FATAL: container creation failed: image driver mount failure:
squashfuse_ll exited: fuse: device not found, try 'modprobe fuse' first
```

To avoid unpacking the 4.3-GB image, the read-only numerical audit used host Python with dependencies installed only in scratch:

```sh
mkdir -p /tmp/codex_n038/tmp /tmp/codex_n038/cache
TMPDIR=/tmp/codex_n038/tmp XDG_CACHE_HOME=/tmp/codex_n038/cache \
PIP_CACHE_DIR=/tmp/codex_n038/cache \
python3 -I -m pip install --target /tmp/codex_n038/site --only-binary=:all: numpy pandas
```

Environment: Python 3.13.13, NumPy 2.5.4, pandas 3.1.1. This fallback is disclosed because it differs from the requested runtime. Original script CSVs matched; numerical audit CPU time was .318 seconds and bootstrap .544 seconds (dependency installation was also lightweight). No model inference or fitting was performed.

The next two sections contain the complete scratch scripts, execution commands, and actual outputs, so the numerical claims are independently repeatable. Run metadata and saved SSL provenance are separate from a proof that historical jobs followed their configuration exactly; training was not rerun.

### Prediction audit

Save the following as `/tmp/codex_n038/audit.py`:

```python
import json,pathlib,runpy,sys,time
import numpy as np
import pandas as pd
start=time.process_time()
R=pathlib.Path('/home/hpc/v181bb/v181bb10/BSCCM')
D=R/'.agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go'
W=pathlib.Path('/hnvme/workspace/v181bb10-bsccm')
print('=== ORIGINAL SCRIPTS ===')
for script in ['d1_fov_vs_batch.py','d1b_same_arch.py']:
 sys.argv=[str(D/script),'/tmp/codex_n038']
 runpy.run_path(str(D/script),run_name='__main__')
 old=pd.read_csv(D/script.replace('.py','.csv')); new=pd.read_csv(pathlib.Path('/tmp/codex_n038')/script.replace('.py','.csv'))
 print('CSV match',script,old.shape==new.shape and np.allclose(old.select_dtypes('number'),new.select_dtypes('number'),atol=1e-12))
T={v:dict(np.load(W/f'targets_{v}.npz',allow_pickle=True)) for v in ['v4','v5']}
t=T['v5']; names=list(map(str,t['names'])); common=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]; groups=[j for j in common if j>=8]
def load(run):
 z=np.load(W/'runs'/run/'eval_preds.npz'); return z['P'],z['G']
def scores(p,g,v='v5'):
 y=T[v]['Y'][g]; m=T[v]['mask'][g]
 return np.array([np.corrcoef(p[m[:,j],j],y[m[:,j],j])[0,1] if m[:,j].sum()>20 else np.nan for j in range(12)])
def fmt(x): return np.array2string(np.asarray(x),precision=6,floatmode='fixed')
print('=== RUN METADATA ===')
for run in ['n015_sslinit_fov','n023_v5_b10','n023_v5_b01','n024_r18_v5_fov','n022_base_fov','n022_tail05_fov','n031_base_b10','n031_base_b01','n031_zoom2_b10','n031_zoom2_b01']:
 z=json.load(open(W/'runs'/run/'result.json')); a=z['args']
 print(run,json.dumps({k:a.get(k) for k in ['arch','targets','protocol','epochs','bs','lr','seed','init_ssl','tail_weight','focal_c10','zoom']},sort_keys=True),'n_train',z.get('n_train'),'n_eval',z.get('n_eval'))
print('=== MATCHED D1B SUBSETS ===')
df=pd.read_csv('/tmp/codex_n038/d1b_same_arch.csv')
for label,sel in [('common11',np.ones(len(df),bool)),('single8',df.target.str.startswith('single:')),('group3',df.target.str.startswith('grp:'))]:
 print(label,df[sel].groupby('test_draw')[['r_fov','r_batch','gap']].mean().round(6).to_dict(),'overall',df[sel][['r_fov','r_batch','gap']].mean().round(6).to_dict())
print('positive/negative contributions',df.gap.clip(lower=0).sum()/22,df.gap.clip(upper=0).sum()/22)
print('=== IDENTICAL-CELL FOV CONTROL / LABEL-VERSION SENSITIVITY ===')
for run in ['n015_sslinit_fov','n022_base_fov','n022_tail05_fov','n024_r18_v5_fov']:
 p,g=load(run); print(run,'cells',len(g),'unique',len(np.unique(g)),'split',np.unique(t['split'][g]),'hist',int((t['hist_row'][g]>=0).sum())); vals=[]
 for b in [0,1]:
  keep=t['batch'][g]==b; gg=g[keep]; pp=p[keep]; v4=scores(pp,gg,'v4'); v5=scores(pp,gg); vals.append(v5)
  print(' draw',b,'fovs',len(np.unique(t['fov'][gg])),'v4 common11',fmt(v4[common].mean()),'v5 common11',fmt(v5[common].mean()),'single8',fmt(v5[:8].mean()),'group3',fmt(v5[groups].mean()))
  print(' single8 v5',fmt(v5[:8]),'group3 v5',fmt(v5[groups]))
 print(' draw-balanced v5 common11',fmt(np.mean([v[common].mean() for v in vals])))
print('=== LABEL CHANGE ON SAME SPLIT2 CELLS ===')
for b in [0,1]:
 for j in common:
  m=(t['batch']==b)&(t['split']==2)&t['mask'][:,j]; y4=T['v4']['Y'][m,j]; y5=t['Y'][m,j]
  print(b,names[j],'n',int(m.sum()),'label r',fmt(np.corrcoef(y4,y5)[0,1]),'maxdelta',fmt(np.max(np.abs(y4-y5))),'fovs',len(np.unique(t['fov'][m])))
print('=== GOAL RECOMPUTED ===')
for prefix in ['n031','n037']:
 means=[]
 for d in ['b10','b01']:
  p,g=load(f'{prefix}_base_{d}'); q,h=load(f'{prefix}_zoom2_{d}'); assert np.array_equal(g,h)
  ss=scores((p+q)/2,g); means.append(ss[common].mean()); print(prefix,d,'common11',fmt(ss[common].mean()),'single8',fmt(ss[:8]),'group3',fmt(ss[groups]))
 print(prefix,'mean',fmt(np.mean(means)))
print('CPU seconds',time.process_time()-start)
```

Run:

```sh
TMPDIR=/tmp/codex_n038/tmp XDG_CACHE_HOME=/tmp/codex_n038/cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -I -c 'import sys,runpy; sys.path.insert(0,"/tmp/codex_n038/site"); runpy.run_path("/tmp/codex_n038/audit.py",run_name="__main__")'
```

Actual output:

```text
=== ORIGINAL SCRIPTS ===
fov eval cells 39435 splits [2] batches (array([0, 1]), array([13113, 26322]))
 test_draw            target    n  r_fov_r18  r_batch_ens    gap
         0       single:CD45  987      0.716        0.711  0.005
         0      single:CD123 1820      0.304        0.615 -0.310
         0       single:CD19 1299      0.247        0.299 -0.052
         0       single:CD56 1500      0.407        0.383  0.024
         0       single:CD14 1771      0.868        0.861  0.007
         0       single:CD16 1006      0.886        0.904 -0.018
         0     single:HLA-DR 2274      0.623        0.575  0.048
         0        single:CD3  629      0.556        0.560 -0.005
         0 grp:CD3/CD19/CD56 1191      0.802        0.787  0.015
         0          grp:CD45 1191      0.830        0.752  0.078
         0          grp:CD16 1191      0.864        0.825  0.039
         1       single:CD45 2232      0.867        0.845  0.021
         1      single:CD123 2973      0.400        0.459 -0.059
         1       single:CD19  722      0.483        0.395  0.088
         1       single:CD56 2112      0.360        0.362 -0.003
         1       single:CD14 2168      0.830        0.826  0.004
         1       single:CD16 2911      0.899        0.898  0.001
         1     single:HLA-DR 3418      0.594        0.446  0.148
         1        single:CD3 2968      0.621        0.602  0.020
         1 grp:CD3/CD19/CD56 2807      0.833        0.794  0.039
         1          grp:CD45 2807      0.711        0.655  0.055
         1          grp:CD16 2807      0.818        0.803  0.014
           r_fov_r18  r_batch_ens     gap
test_draw                                
0             0.6457       0.6612 -0.0155
1             0.6741       0.6443  0.0298
CSV match d1_fov_vs_batch.py True
 test_draw            target    n  r_fov  r_batch    gap
         0       single:CD45  987  0.712    0.711  0.001
         0      single:CD123 1820  0.408    0.555 -0.147
         0       single:CD19 1299  0.279    0.280 -0.001
         0       single:CD56 1500  0.411    0.362  0.049
         0       single:CD14 1771  0.876    0.861  0.014
         0       single:CD16 1006  0.892    0.900 -0.008
         0     single:HLA-DR 2274  0.614    0.560  0.054
         0        single:CD3  629  0.550    0.560 -0.011
         0 grp:CD3/CD19/CD56 1191  0.773    0.767  0.006
         0          grp:CD45 1191  0.840    0.714  0.125
         0          grp:CD16 1191  0.853    0.810  0.044
         1       single:CD45 2232  0.879    0.837  0.042
         1      single:CD123 2973  0.528    0.461  0.067
         1       single:CD19  722  0.467    0.365  0.102
         1       single:CD56 2112  0.386    0.372  0.014
         1       single:CD14 2168  0.848    0.837  0.011
         1       single:CD16 2911  0.908    0.900  0.008
         1     single:HLA-DR 3418  0.599    0.454  0.145
         1        single:CD3 2968  0.620    0.593  0.028
         1 grp:CD3/CD19/CD56 2807  0.679    0.785 -0.106
         1          grp:CD45 2807  0.648    0.648  0.000
         1          grp:CD16 2807  0.711    0.799 -0.088
            r_fov  r_batch     gap
test_draw                         
0          0.6552   0.6437  0.0115
1          0.6612   0.6409  0.0203
overall {'r_fov': 0.6582, 'r_batch': 0.6423, 'gap': 0.0159}
CSV match d1b_same_arch.py True
=== RUN METADATA ===
n015_sslinit_fov {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": null, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt", "lr": 0.0005, "protocol": "fov", "seed": 0, "tail_weight": null, "targets": "targets_v4.npz", "zoom": null} n_train 317935 n_eval 39435
n023_v5_b10 {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt", "lr": 0.0005, "protocol": "b1to0", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": 1} n_train 269283 n_eval 13113
n023_v5_b01 {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt", "lr": 0.0005, "protocol": "b0to1", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": 1} n_train 127946 n_eval 26322
n024_r18_v5_fov {"arch": "resnet18", "bs": 256, "epochs": 8, "focal_c10": 0.0, "init_ssl": "", "lr": 0.001, "protocol": "fov", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": null} n_train 317935 n_eval 39435
n022_base_fov {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt", "lr": 0.0005, "protocol": "fov", "seed": 0, "tail_weight": 0.0, "targets": "targets_v4.npz", "zoom": null} n_train 317935 n_eval 39435
n022_tail05_fov {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 2.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/encoder_final.pt", "lr": 0.0005, "protocol": "fov", "seed": 0, "tail_weight": 0.5, "targets": "targets_v4.npz", "zoom": null} n_train 317935 n_eval 39435
n031_base_b10 {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/encoder_final.pt", "lr": 0.0005, "protocol": "b1to0", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": 1} n_train 269283 n_eval 13113
n031_base_b01 {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex1/encoder_final.pt", "lr": 0.0005, "protocol": "b0to1", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": 1} n_train 127946 n_eval 26322
n031_zoom2_b10 {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/encoder_final.pt", "lr": 0.0005, "protocol": "b1to0", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": 2} n_train 269283 n_eval 13113
n031_zoom2_b01 {"arch": "resnet50", "bs": 256, "epochs": 10, "focal_c10": 0.0, "init_ssl": "/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex1/encoder_final.pt", "lr": 0.0005, "protocol": "b0to1", "seed": 0, "tail_weight": 0.0, "targets": "targets_v5.npz", "zoom": 2} n_train 127946 n_eval 26322
=== MATCHED D1B SUBSETS ===
common11 {'r_fov': {0: 0.655198, 1: 0.661177}, 'r_batch': {0: 0.643658, 1: 0.640882}, 'gap': {0: 0.011541, 1: 0.020296}} overall {'r_fov': 0.658188, 'r_batch': 0.64227, 'gap': 0.015918}
single8 {'r_fov': {0: 0.592661, 1: 0.654449}, 'r_batch': {0: 0.598638, 1: 0.602288}, 'gap': {0: -0.005976, 1: 0.052162}} overall {'r_fov': 0.623555, 'r_batch': 0.600463, 'gap': 0.023093}
group3 {'r_fov': {0: 0.821963, 1: 0.679119}, 'r_batch': {0: 0.763711, 1: 0.743799}, 'gap': {0: 0.058253, 1: -0.06468}} overall {'r_fov': 0.750541, 'r_batch': 0.753755, 'gap': -0.003214}
positive/negative contributions 0.03231254873774455 -0.01639434184825041
=== IDENTICAL-CELL FOV CONTROL / LABEL-VERSION SENSITIVITY ===
n015_sslinit_fov cells 39435 unique 39435 split [2] hist 0
 draw 0 fovs 2592 v4 common11 0.655198 v5 common11 0.655198 single8 0.592661 group3 0.821963
 single8 v5 [0.711715 0.407966 0.278894 0.410953 0.875904 0.891889 0.614113 0.549856] group3 v5 [0.772883 0.839891 0.853117]
 draw 1 fovs 3774 v4 common11 0.709325 v5 common11 0.661177 single8 0.654449 group3 0.679119
 single8 v5 [0.878834 0.528337 0.466887 0.386366 0.847940 0.908363 0.598747 0.620121] group3 v5 [0.678706 0.647956 0.710696]
 draw-balanced v5 common11 0.658188
n022_base_fov cells 39435 unique 39435 split [2] hist 0
 draw 0 fovs 2592 v4 common11 0.657514 v5 common11 0.657514 single8 0.596091 group3 0.821309
 single8 v5 [0.713788 0.417235 0.286454 0.420426 0.881537 0.893463 0.612281 0.543544] group3 v5 [0.770245 0.840277 0.853405]
 draw 1 fovs 3774 v4 common11 0.708486 v5 common11 0.660203 single8 0.653036 group3 0.679316
 single8 v5 [0.878719 0.530568 0.456945 0.382386 0.847643 0.908669 0.599509 0.619845] group3 v5 [0.677624 0.649516 0.710809]
 draw-balanced v5 common11 0.658859
n022_tail05_fov cells 39435 unique 39435 split [2] hist 0
 draw 0 fovs 2592 v4 common11 0.686617 v5 common11 0.686617 single8 0.635964 group3 0.821692
 single8 v5 [0.717625 0.646037 0.324622 0.420549 0.878554 0.895805 0.640047 0.564472] group3 v5 [0.781397 0.837887 0.845792]
 draw 1 fovs 3774 v4 common11 0.724437 v5 common11 0.673334 single8 0.671547 group3 0.678099
 single8 v5 [0.886881 0.675963 0.437264 0.376207 0.850600 0.910276 0.605376 0.629810] group3 v5 [0.683655 0.657127 0.693515]
 draw-balanced v5 common11 0.679976
n024_r18_v5_fov cells 39435 unique 39435 split [2] hist 0
 draw 0 fovs 2592 v4 common11 0.645697 v5 common11 0.645697 single8 0.575852 group3 0.831951
 single8 v5 [0.716337 0.304377 0.246507 0.407178 0.867523 0.886157 0.623138 0.555600] group3 v5 [0.801816 0.829979 0.864058]
 draw 1 fovs 3774 v4 common11 0.655590 v5 common11 0.674081 single8 0.631675 group3 0.787165
 single8 v5 [0.866549 0.399814 0.483180 0.359607 0.829879 0.899206 0.593707 0.621455] group3 v5 [0.832994 0.710828 0.817672]
 draw-balanced v5 common11 0.659889
=== LABEL CHANGE ON SAME SPLIT2 CELLS ===
0 single:CD45 n 987 label r 1.000000 maxdelta 0.000000 fovs 238
0 single:CD123 n 1820 label r 1.000000 maxdelta 0.000000 fovs 299
0 single:CD19 n 1299 label r 1.000000 maxdelta 0.000000 fovs 301
0 single:CD56 n 1500 label r 1.000000 maxdelta 0.000000 fovs 299
0 single:CD14 n 1771 label r 1.000000 maxdelta 0.000000 fovs 268
0 single:CD16 n 1006 label r 1.000000 maxdelta 0.000000 fovs 206
0 single:HLA-DR n 2274 label r 1.000000 maxdelta 0.000000 fovs 363
0 single:CD3 n 629 label r 1.000000 maxdelta 0.000000 fovs 110
0 grp:CD3/CD19/CD56 n 1191 label r 1.000000 maxdelta 0.000000 fovs 252
0 grp:CD45 n 1191 label r 1.000000 maxdelta 0.000000 fovs 252
0 grp:CD16 n 1191 label r 1.000000 maxdelta 0.000000 fovs 252
1 single:CD45 n 2232 label r 1.000000 maxdelta 0.000000 fovs 282
1 single:CD123 n 2973 label r 1.000000 maxdelta 0.000000 fovs 388
1 single:CD19 n 722 label r 1.000000 maxdelta 0.000000 fovs 82
1 single:CD56 n 2112 label r 1.000000 maxdelta 0.000000 fovs 317
1 single:CD14 n 2168 label r 1.000000 maxdelta 0.000000 fovs 333
1 single:CD16 n 2911 label r 1.000000 maxdelta 0.000000 fovs 371
1 single:HLA-DR n 3418 label r 1.000000 maxdelta 0.000000 fovs 396
1 single:CD3 n 2968 label r 1.000000 maxdelta 0.000000 fovs 341
1 grp:CD3/CD19/CD56 n 2807 label r 0.731836 maxdelta 1.755117 fovs 632
1 grp:CD45 n 2807 label r 0.806196 maxdelta 2.918489 fovs 632
1 grp:CD16 n 2807 label r 0.777898 maxdelta 1.460042 fovs 632
=== GOAL RECOMPUTED ===
n031 b10 common11 0.661222 single8 [0.711466 0.614830 0.298978 0.383358 0.861004 0.904282 0.575386 0.560323] group3 [0.787067 0.752170 0.824580]
n031 b01 common11 0.644260 single8 [0.845433 0.458765 0.394989 0.362472 0.826295 0.898354 0.446185 0.601841] group3 [0.793626 0.655438 0.803460]
n031 mean 0.652741
n037 b10 common11 0.660158 single8 [0.711756 0.611922 0.297092 0.380040 0.860610 0.903337 0.570577 0.561972] group3 [0.790073 0.752064 0.822291]
n037 b01 common11 0.647192 single8 [0.847769 0.453223 0.417205 0.365996 0.829359 0.899480 0.449396 0.604104] group3 [0.792929 0.657927 0.801724]
n037 mean 0.653675
CPU seconds 0.318100475
```

### Paired FOV bootstrap

Save as `/tmp/codex_n038/bootstrap.py`:

```python
import sys,pathlib,time
sys.path.insert(0,'/tmp/codex_n038/site')
import numpy as np
start=time.process_time()
W=pathlib.Path('/hnvme/workspace/v181bb10-bsccm'); T=dict(np.load(W/'targets_v5.npz',allow_pickle=True)); names=list(map(str,T['names'])); jj=[j for j in range(12) if 'CD123/HLA' not in names[j]]
runs=['n015_sslinit_fov','n022_tail05_fov','n023_v5_b10','n023_v5_b01']
E={r:dict(np.load(W/'runs'/r/'eval_preds.npz')) for r in runs}
g=E[runs[0]]['G']
assert np.array_equal(g,E[runs[1]]['G'])
rng=np.random.default_rng(20261112); B=1000
result={}
for b,d in [(0,'b10'),(1,'b01')]:
 keep=T['batch'][g]==b; c=g[keep]; p0=E[runs[0]]['P'][keep]; p1=E[runs[1]]['P'][keep]
 eb=E['n023_v5_'+d]; loc=np.searchsorted(eb['G'],c); assert np.array_equal(eb['G'][loc],c)
 ps=[p0,p1,eb['P'][loc]]
 # Cluster on slide AND fov to avoid accidental sharing of integer FOV IDs across slides.
 key=np.stack([T['slide'][c],T['fov'][c]],axis=1); _,cluster=np.unique(key,axis=0,return_inverse=True); K=cluster.max()+1
 weights=rng.multinomial(K,np.ones(K)/K,size=B).astype(float)
 out=np.empty((B,3,len(jj)))
 for q,j in enumerate(jj):
  sel=T['mask'][c,j]; yy=T['Y'][c[sel],j].astype(float); cc=cluster[sel]
  for a,pp in enumerate(ps):
   x=pp[sel,j].astype(float)
   s=np.array([np.bincount(cc,weights=z,minlength=K) for z in [np.ones(len(x)),x,yy,x*x,yy*yy,x*yy]])
   n,sx,sy,sxx,syy,sxy=(weights@s.T).T
   out[:,a,q]=(sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
 result[b]=out
 print('draw',b,'clusters',K,'D1b gap CI',np.quantile(out[:,0,:].mean(1)-out[:,2,:].mean(1),[.025,.975]))
 for target in ['single:CD19','single:CD56','single:CD123','single:HLA-DR']:
  q=jj.index(names.index(target)); print(target,'D1b gap CI',np.quantile(out[:,0,q]-out[:,2,q],[.025,.975]),'tail minus base CI',np.quantile(out[:,1,q]-out[:,0,q],[.025,.975]))
combo=(result[0]+result[1])/2
for a,label in enumerate(['n015 FOV','n022 tail FOV','n023 heldout']): print(label,'common11 CI',np.quantile(combo[:,a,:].mean(1),[.025,.975]))
print('D1b overall gap CI',np.quantile((combo[:,0,:]-combo[:,2,:]).mean(1),[.025,.975]))
print('tail minus base overall CI',np.quantile((combo[:,1,:]-combo[:,0,:]).mean(1),[.025,.975]))
print('CPU seconds',time.process_time()-start)
```

Command:

```sh
TMPDIR=/tmp/codex_n038/tmp XDG_CACHE_HOME=/tmp/codex_n038/cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 -I /tmp/codex_n038/bootstrap.py
```

Actual output:

```text
draw 0 clusters 2592 D1b gap CI [-0.00407901  0.03056538]
single:CD19 D1b gap CI [-0.04280985  0.04056115] tail minus base CI [0.01071419 0.07668103]
single:CD56 D1b gap CI [0.02475839 0.07495339] tail minus base CI [-0.00813349  0.02603261]
single:CD123 D1b gap CI [-0.28229658  0.05449083] tail minus base CI [0.0476144  0.34517574]
single:HLA-DR D1b gap CI [0.02486629 0.08351498] tail minus base CI [0.00692529 0.04402668]
draw 1 clusters 3774 D1b gap CI [0.00537938 0.03804436]
single:CD19 D1b gap CI [0.01341339 0.18284401] tail minus base CI [-0.07418474  0.01863398]
single:CD56 D1b gap CI [-0.00505474  0.03342452] tail minus base CI [-0.02511817  0.00377219]
single:CD123 D1b gap CI [-0.05734742  0.2323037 ] tail minus base CI [0.05707945 0.22306928]
single:HLA-DR D1b gap CI [0.11325363 0.17778452] tail minus base CI [-0.00587574  0.01907727]
n015 FOV common11 CI [0.64642502 0.67009417]
n022 tail FOV common11 CI [0.66399938 0.69372215]
n023 heldout common11 CI [0.62550123 0.65621453]
D1b overall gap CI [0.00549803 0.02871894]
tail minus base overall CI [0.01194019 0.02898844]
CPU seconds 0.544475691
```

### Source provenance checks

Commands (read-only):

```sh
rg -n "^config:" /hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/wandb/offline-run-20261108_221340-n007_ssl_aug/logs/debug.log /hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/wandb/offline-run-20261110_095540-n031_ssl_aug_ex0/logs/debug.log
```

Output:

```text
/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0/wandb/offline-run-20261110_095540-n031_ssl_aug_ex0/logs/debug.log:8:config: {'arm': 'aug', 'out': '/hnvme/workspace/v181bb10-bsccm/runs/n031_ssl_aug_ex0', 'arch': 'resnet50', 'epochs': 40, 'bs': 512, 'views': 2, 'lr': 0.002, 'lam': 0.05, 'proj_dim': 256, 'workers': 15, 'rows': 'all', 'exclude_batch': [0], 'max_steps': 0, 'resume': 'auto', 'seed': 0, 'slurm_job_id': '1295538', 'n_rows': 274980, 'transductive': True, '_wandb': {}}
/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug/wandb/offline-run-20261108_221340-n007_ssl_aug/logs/debug.log:8:config: {'arm': 'aug', 'out': '/hnvme/workspace/v181bb10-bsccm/runs/n007_ssl_aug', 'arch': 'resnet50', 'epochs': 40, 'bs': 512, 'views': 2, 'lr': 0.002, 'lam': 0.05, 'proj_dim': 256, 'workers': 15, 'rows': 'all', 'exclude_batch': [], 'max_steps': 0, 'resume': 'auto', 'seed': 0, 'slurm_job_id': '1277302', 'n_rows': 412941, 'transductive': True, '_wandb': {}}
```

For attempt scopes, run from the repository root:

```sh
rg -n '^# N0|^TT6 anchored|^Generator =|^Recipe frozen|^User idea|^A \(paired\)|^B \(user' .agent-research/notebook/items/N027-*.md .agent-research/notebook/items/N030-*.md .agent-research/notebook/items/N033-*.md .agent-research/notebook/items/N034-*.md .agent-research/notebook/items/N037-*.md
```

Output:

```text
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N037-goal-retrain-on-cross-fitted-labels-n031.md:1:# N037 · GOAL retrain on cross-fitted labels: N031 inductive base + zoom2 fine-tunes, both directions, trained on targets_v5cx (Giemsa-free fits + FOV-fold cross-fitted position detrend, N036) instead of v5; scored by GOAL (v5) and on v5cx
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N037-goal-retrain-on-cross-fitted-labels-n031.md:13:Recipe frozen = N031 (GOAL best .6527): inductive SSL(aug) encoders n031_ssl_aug_ex0/ex1, resnet50, 10 ep, bs 256, lr 5e-4, w_cls .5,
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N034-lf-fluorescence-marker-images-by-conditi.md:1:# N034 · LF -> fluorescence marker IMAGES by conditional flow matching (A: paired per-cell CFM + same-net regression baseline, stain-conditioned, batch-held-out) as the bench for B: set-level distribution matching with pairs hidden (user idea 2026-11-11)
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N034-lf-fluorescence-marker-images-by-conditi.md:7:User idea 2026-11-11: first predict the marker IMAGES (6-ch fluorescence) from the label-free images, with a distribution-matching
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N034-lf-fluorescence-marker-images-by-conditi.md:12:A (paired): conditional rectified flow matching LF(24 ch) + stain token -> 6-ch fluor image (128 px), CFG-dropout 0.1, EMA; the SAME
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N034-lf-fluorescence-marker-images-by-conditi.md:21:B (user's set-level idea), after A: pairs hidden, loss only between SETS of cells (subset of one slide): B1 same cells shuffled,
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N033-virtual-stain-data-engine-giemsa-student.md:1:# N033 · Virtual-stain data engine: Giemsa student trained additionally on virtual Wright images of 318k marker-labelled cells with MEASURED labels (TT6c + synthetic, w 1.0), paired vs N029 TT6c (same teachers, seeds)
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N033-virtual-stain-data-engine-giemsa-student.md:10:Generator = N028 FOV model (trained on split-0 GS pairs only). Render all 317,935 non-GS split-0 cells (scripts/vs_generate.py -> /hnvme/workspace/v181bb10-bsccm/runs/n033_synth_fov/images.npy, 15.6 GB uint8). Labels = targets_tt6c.npz (v4c markers + v6c unstained-only autofl; scripts/make_targets_tt6c.py). Student: TT6c (N029 teachers t0) + masked MSE on measured labels of a synthetic batch per step (w 1.0), virtual 128 px image embedded at the registration field (.67 x 398 px) in a border-median canvas, student units x rgb_scale/scale. 3 seeds, paired vs N029 tt6c_s0-2 (same teachers, seeds, steps). Eval: within-slide autofl truth (v4c clean), MLL23 fine mean, PBC-B. Caveat: the generator only sees LF, so synthetic images cannot carry information beyond LF; the hope is better DOMAIN coverage (318k cells) for Giemsa->marker mapping. Jobs .agent-research/results/N033-virtual-stain-data-engine-giemsa-student/jobs.txt. ~2.5 GPU-h.
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N027-giemsa-student-capacity-on-the-tt6-chain.md:1:# N027 · Giemsa student capacity on the TT6 chain: DinoBloom-B with 8 trainable blocks + 6000 steps vs DinoBloom-L (8 trainable of 24) + 6000 steps; MLL23/PBC + within-slide autofl truth
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N027-giemsa-student-capacity-on-the-tt6-chain.md:10:TT6 anchored students (PBC half A anchors, anchor-ce .5, fov holdout, aug), 6000 steps (was 3000), 2 seeds each: DinoBloom-B freeze 4 (8 trainable blocks, was 4) | DinoBloom-L freeze 16 (8 trainable of 24). Baseline TT6 (N025, 3 seeds): within-slide autofl .578 / .855, MLL23 fine .611. Jobs .agent-research/results/N027-giemsa-student-capacity-on-the-tt6-chain/jobs.txt (applies; students 1294877/79/81/83). ~3 GPU-h.
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N030-giemsa-student-with-fewer-trainable-bloc.md:1:# N030 · Giemsa student with FEWER trainable blocks (TT6 chain): freeze 10 of 12 and fully frozen DinoBloom-B (heads + projections only), 3 seeds; does less BSCCM-domain capacity improve clinical transfer?
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/notebook/items/N030-giemsa-student-with-fewer-trainable-bloc.md:10:TT6 anchored students (3000 steps, anchor-ce .5, PBC half A, fov holdout), freeze 10 (2 trainable) and freeze 12 (frozen backbone), 3 seeds each; baseline TT6 freeze 8 (N025). Jobs: .agent-research/results/N030-giemsa-student-with-fewer-trainable-bloc/jobs.txt (applies) + students 1295187-97.
```

To check the stored N035 numbers (this reads results, not rerunning the experiment):

```sh
python3 -I - <<'PY2'
import json
p="/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N035-frozen-confirmation-of-n034-a-in-the-unt/n035_posthoc_label_vs_integrals.json"
d=json.load(open(p))
for n in ["single:CD19","single:CD56","single:CD123","single:HLA-DR","single:CD16"]:
 print(n,*(round(d[n]["within_b"+str(b)],6) for b in [0,1]))
PY2
```

Output:

```text
single:CD19 0.322119 0.558051
single:CD56 0.536775 0.481121
single:CD123 0.828996 0.70823
single:HLA-DR 0.879508 0.64852
single:CD16 0.968576 0.965159
```
