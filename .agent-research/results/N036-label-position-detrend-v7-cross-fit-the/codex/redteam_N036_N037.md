# Red-team review: N036 / N037

Reviewed branch `bsccm-data-model`, HEAD `7cb92661d08df6ae87f03872b54389ad30e7c1b2`, including working-tree changes and the saved labels/predictions. Two reproduced defects; no blocker. No source, notebook, label, or run files were changed. The report is the only review artifact written inside the repository; all executable probes and scratch are in `/tmp/codex_n036_rt/`.

## Environment and reproduction

The prescribed command was attempted:

```sh
apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -c 'import sys,numpy,pandas,sklearn; print(sys.version)'
```

Observed:

```text
FATAL:   container creation failed: image driver mount failure: image driver squashfuse_ll instance exited with error: squashfuse_ll exited: fuse: device not found, try 'modprobe fuse' first
```

Execution in that container is therefore **[unverified]**. For the checks below I used a temporary Python 3.11.15 environment, with the scientific-library versions matched to the SIF's `*.dist-info` directories: NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, PyArrow 26.0.1. The real-slide cross-fit rebuild below reproduced the saved float32 labels exactly. No training or complete diagnostic rebuild was run; fits were limited to small probes.

Set these variables for the reproduction commands below:

```sh
export TMPDIR=/tmp/codex_n036_rt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/tmp/codex_n036_rt/venv/bin/python
R=/home/hpc/v181bb/v181bb10/BSCCM
```

The environment was created with `uv venv --python 3.11 /tmp/codex_n036_rt/venv` and the five versions above installed with `uv pip install --python /tmp/codex_n036_rt/venv/bin/python ...`; `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` were also set under the scratch directory.

## Finding 1 — major: the completed correction is not independent of its own FOV

**Claim affected:** C1, when applied to the final trend subtracted from a cell. The narrower claim that each individual HGB fit excludes its prediction FOV **holds**.

**Location:** `scripts/make_targets_v4.py:28–32`.

The raw prediction is correctly cross-fitted: `f = FOLD[m]`, `te = f == k`, the training indices exclude `f == k`, and `P[m][te]` selects the corresponding feature rows. The defect is after prediction. The keep/drop decision at line 31 uses every clean split-0 label on the slide, including the FOV whose trend is being decided. Consequently that FOV can determine whether its own correction is applied. This is label-based selection on the same OOF observations being transformed; the selected transform as a whole is not held out from that FOV.

The centering at line 32 also pools predictions for all split-0 folds. Predictions in other folds come from models that *did* train on the current FOV, so even a fixed gate does not give the final correction strict FOV independence. This second dependency is a slide-wide constant; the gate dependency can remove the entire spatial correction.

**Concrete real-data reproduction:** extract and run the unchanged `_fit` function, perturb only the source labels of FOV 23268 in memory, and record each fold's raw predictions. This FOV contains 15 cells on `antibodies_CD3/batch_0/slide_replicate_0`. The large perturbation is a dependency test, not an estimate of natural error magnitude.

```sh
"$PY" -I -B /tmp/codex_n036_rt/repro_c1.py
```

Observed:

```text
changed_FOV=23268 cells= 15
raw_own_fold_max_change= 0.0
nonzero_final_trend_before_after= 6538 0
own_FOV_final_trend_max_change= 0.6566229979983477
```

A separate real-slide probe also reported:

```text
V4CX_ARTIFACT_MAX_REBUILD_DIFFERENCE 0.0
```

Thus the unaffected held-out HGB prediction becomes a different final correction solely because the cell's own FOV labels changed. A synthetic five-FOV probe independently reproduced the same transition. This does **not** demonstrate test-split labels entering HGB fits or invalidate the reproduced GOAL numbers; C2 and C4 below hold. The effect on ordinary-data training bias or GOAL performance is **[unverified]**.

**Suggested fix:** if C1 is meant to cover the complete correction, use an outer FOV fold for prediction and compute both the gate and centering reference entirely inside the other outer folds. Gate selection needs inner FOV CV there, or a separately frozen calibration rule. Alternatively, explicitly narrow C1 to raw HGB fits and disclose that selection/centering are fitted on all clean split-0 cells; that documentation change alone would not establish strict independence.

For a durable reproduction, this is the full `repro_c1.py` source; it never writes data:

```python
import ast
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
R=Path('/home/hpc/v181bb/v181bb10/BSCCM'); W=Path('/hnvme/workspace/v181bb10-bsccm')
T=dict(np.load(W/'targets_v3c.npz',allow_pickle=True))
cols=['data_path','fov_center_x','fov_center_y','fov_center_z','position_in_fov_x_pix','position_in_fov_y_pix']
idx=pd.read_parquet(W/'bsccm_index.parquet',columns=cols)
records=[]
class Record:
 def __init__(self,**kw): self.g=HGB(**kw)
 def fit(self,p,y): self.g.fit(p,y); return self
 def predict(self,p):
  z=self.g.predict(p); records.append(z.copy()); return z
ns=dict(np=np,HGB=Record,slide=idx.data_path.values,P=idx[cols[1:]].values.astype(float),
 Y=T['Y'].copy(),Mk=T['mask'],sp=T['split'],OKC=T['hist_row']<0,
 FOLD=((T['fov'].astype(np.int64)*2654435761)%2**32)%5,DET='xfit')
fn=next(n for n in ast.parse((R/'scripts/make_targets_v4.py').read_text()).body
 if isinstance(n,ast.FunctionDef) and n.name=='_fit')
exec(compile(ast.Module(body=[fn],type_ignores=[]),'v4','exec'),ns)
s='antibodies_CD3/batch_0/slide_replicate_0'; j=list(T['names']).index('single:CD3')
own=T['fov']==23268; k=int(ns['FOLD'][own][0])
_,_,m,a=ns['_fit']((s,j)); raw_a=records[k].copy(); records.clear()
ns['Y'][own,j]+=10000
_,_,_,b=ns['_fit']((s,j)); raw_b=records[k]
print('changed_FOV=23268 cells=',int(own.sum()))
print('raw_own_fold_max_change=',float(np.max(np.abs(raw_a-raw_b))))
print('nonzero_final_trend_before_after=',int(np.count_nonzero(a)),int(np.count_nonzero(b)))
print('own_FOV_final_trend_max_change=',float(np.max(np.abs(a[own[m]]-b[own[m]]))))
```

## Finding 2 — minor: total-cell guard does not prevent empty cross-fit training sets

**Claim affected:** C1 implementation robustness / small-slide fold degeneracy; no current-data overlap violation.

**Location:** `scripts/make_targets_v4.py:20,28–29`.

The only size guard tests all masked cells (`len(m) >= 1000`). It does not require any clean split-0 training cells outside a prediction fold. A slide with 1,200 masked cells concentrated in one FOV/fold passes the guard, then calls HGB with zero training rows. Because `_fit` runs inside `Pool.map`, that exception aborts the complete label build. The same problem occurs if all eligible split-0 clean cells lie in one fold while other cells inflate the total count. The previous non-cross-fitted HGB path could fit a slide containing one training FOV.

**Concrete reproduction:** the actual `_fit` function on a synthetic 1,200-cell, one-FOV slide:

```sh
"$PY" -I -B /tmp/codex_n036_rt/repro_small.py
```

Observed:

```text
ValueError: Found array with 0 sample(s) (shape=(0, 5)) while a minimum of 1 is required by HistGradientBoostingRegressor.
```

Full reproduction source:

```python
import ast
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
src=Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/make_targets_v4.py').read_text()
fn=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_fit')
n=1200
ns=dict(np=np,HGB=HGB,DET='xfit',slide=np.zeros(n,int),Mk=np.ones((n,1),bool),
 sp=np.zeros(n,int),OKC=np.ones(n,bool),FOLD=np.zeros(n,int),P=np.zeros((n,5)),Y=np.arange(n,dtype=float)[:,None])
exec(compile(ast.Module(body=[fn],type_ignores=[]),'v4','exec'),ns)
try: ns['_fit']((0,0))
except Exception as exc: print(type(exc).__name__+': '+str(exc))
```

This is a reproduced boundary defect, **not a failure observed on the current BSCCM labels**. Scanning every present eligible slide/target/fold found 47 jobs, no empty training folds, and a minimum of 4,004 training cells for a nonempty prediction fold.

**Suggested fix:** check eligible training counts and distinct training FOVs before fitting each fold. If the required exclusion cannot be satisfied, return a zero trend for the slide/target and record the reason, or use a supported smaller number of grouped folds. Do not fall back to fitting on the prediction FOV. Also explicitly handle absent clean calibration rows and nonfinite/zero baseline variance.

## Checks that did not produce defects

### C2: evaluation rows exclude split 0

Command:

```sh
"$PY" -I -B /tmp/codex_n036_rt/audit_data.py
```

Observed for **both** base and zoom2, **both** N031 and N037:

```text
b10: n=13113, split 2=13113, batch=[0], giemsa=0, unique=13113
b01: n=26322, split 2=26322, batch=[1], giemsa=0, unique=26322
N031_N037_G_IDENTICAL True
```

All cells are actually split 2, a stronger statement than “splits 1+2.” Across 412,941 cells and 65,096 FOVs, each global FOV belongs to exactly one slide and one split. Every actual training/prediction fold pair had disjoint FOV IDs. Metadata (`names`, `split`, `slide`, `batch`, `fov`, `hist_row`) agrees across v3c/v5/v5c/v5cx. There are no nonfinite masked labels or nonzero unmasked v5/v5c/v5cx labels.

An actual-function synthetic perturbation of all split-1/2 labels and dirty split-0 labels produced:

```text
TEST_AND_DIRTY_LABEL_PERTURBATION_MAX_TREND_CHANGE 0.0
```

The variance gate and centering use clean split 0 only. I found no split-1/2 label entry into the position fits, gate, or normalization. Held-out-batch split-0 calibration is the pre-existing, documented label protocol and is disjoint from these evaluation cells.

### C3: saved training arguments match

Command:

```sh
"$PY" -I -B /tmp/codex_n036_rt/probe_args.py
```

Observed for each of `base b10`, `base b01`, `zoom2 b10`, `zoom2 b01`:

```text
changed_args ['out', 'targets']
non_output_changes {'targets': ('targets_v5.npz', 'targets_v5cx.npz')}
same_train_eval_counts True
```

The initialization checkpoints, protocols, seed, optimizer-related options, epochs, zoom, dual-zoom and remaining recorded arguments match. `out` changes as expected for separate runs. Counts match: b10 train/eval 269,283/13,113; b01 127,946/26,322. This verifies the stated `result.json` comparison; historical equality of unrecorded software/runtime state is not established by those files.

### C4: scores reproduce with the original N031 function

`probe_sc.py` executes the unchanged definitions preceding N031's run loop, then calls its actual `load` and `sc` functions on both sets of ensemble predictions:

```sh
"$PY" -I -B /tmp/codex_n036_rt/probe_sc.py
```

Observed:

```text
n031 b10 0.6612222379
n031 b01 0.6442598721
n031 GOAL 0.6527410550
n037 b10 0.6601575310
n037 b01 0.6471919457
n037 GOAL 0.6536747384
```

That is a gain of 0.0009336834, rounding to the reported .6537 versus .6527. Both use mean base/zoom2 predictions, masked Pearson per common-11 target on v5 labels, equal target averaging and equal direction averaging. All 11 correlations are finite; the smallest target sample sizes are 629 and 722 in b10 and b01, respectively. Thus pandas' NaN-skipping mean does not create a difference from N031's `np.mean` here.

The complete N037 scorer was also rerun with output directed to scratch:

```sh
"$PY" -I -B "$R/.agent-research/results/N037-goal-retrain-on-cross-fitted-labels-n031/n037_score.py" /tmp/codex_n036_rt
diff -u "$R/.agent-research/results/N037-goal-retrain-on-cross-fitted-labels-n031/n037_goal.csv" /tmp/codex_n036_rt/n037_goal.csv
diff -u "$R/.agent-research/results/N037-goal-retrain-on-cross-fitted-labels-n031/n037_per_target.csv" /tmp/codex_n036_rt/n037_per_target.csv
```

Both diffs returned exit 0 with no output. Independent calculations also matched v5c and v5cx ensemble means:

| Labels | N031 | N037 |
|---|---:|---:|
| v5 | .6527410550 | .6536747384 |
| v5c | .6531973770 | .6541091474 |
| v5cx | .6546600976 | .6554673507 |

### C5: no default-path regression found

Commands:

```sh
"$PY" -I -B /tmp/codex_n036_rt/probe_defaults.py
"$PY" -I -B /tmp/codex_n036_rt/probe_real.py
```

The v5 probe executes HEAD and working-tree source with `BSCCM_DETREND` unset, intercepting `np.savez` in memory so it cannot write the workspace. It checks both `BSCCM_CLEAN=0` and `1`, using the actual v4/v4c files.

Observed:

```text
V5_DEFAULT clean 0 paths_equal True keys_equal True arrays_checked 20 mismatches []
V5_DEFAULT clean 1 paths_equal True keys_equal True arrays_checked 20 mismatches []
REAL_DEFAULT_HGB_OLD_NEW_EQUAL True
```

V4's original `_fit` and the default branch use identical HGB parameters, training rows, prediction rows and centering; their returned trends were array-identical on the real 6,538-cell CD3 slide and on a synthetic clean/split-mask probe. With the environment unset, both v4/v5 keep their prior input/output filenames. The new unconditional FOLD computation reads an existing valid `fov` key and does not affect the HGB calculation on these inputs. A full v4 rebuild was not performed.

### Multiprocessing and remaining diagnostic evidence

The previous `BadZipFile: Overlapped entries` is present in `logs/n036_build_1311495.out`, but the reviewed diagnostic code now materializes `TT[*]['Y']` and `TT[*]['mask']` before forking. It also materializes the T0 metadata before worker calls. The v4 builder uses `dict(np.load(...))` before its Pool. I found no remaining shared lazy-array read in these worker paths.

Command:

```sh
"$PY" -I -B /tmp/codex_n036_rt/probe_fork.py
```

A two-worker Pool executing the actual current `_pos` on v5c/v5cx for slide 2 completed successfully:

```text
TT Y/mask types: ndarray for all three versions
v5c:  target=autofl, slide=2, n=1235, pos_r_held=0.07373851615697012
v5cx: target=autofl, slide=2, n=1235, pos_r_held=0.08511068646849801
```

This checks the repaired worker path; one slide does not establish the planned aggregate position-predictability criterion. **[unverified]**: the full N036 held-out position diagnostic passes. At review time `pos_r_heldout.csv` and the claimed rerun's log `logs/n036_diag_1312652.out` were absent. Do not promote v5cx on the assumption that this criterion passed.

The existing FOV-share CSV gives these unweighted means over target/draw rows:

| Version | split 0 | split 1 | split 2 |
|---|---:|---:|---:|
| v3c | .4468 | .4517 | .4454 |
| v5c | .2666 | .3283 | .3146 |
| v5cx | .3230 | .3306 | .3157 |

The aggregate train/held-out gap is smaller under xfit, but this alone is neither a per-target equality test nor evidence that Finding 1 has no practical effect.

## Claim verdicts

C1: **BROKEN** — final applied trend depends on own-FOV labels via gate/centering; raw HGB fold exclusion and `P[m][te]` indexing hold.

C2: **HOLDS** — every inspected GOAL evaluation row is split 2.

C3: **HOLDS** — recorded args differ only in targets and the required output directory.

C4: **HOLDS** — the original N031 scorer reproduces both reported ensemble scores.

C5: **HOLDS** — no default HGB-path output change found in the diff and the regression checks described above.
