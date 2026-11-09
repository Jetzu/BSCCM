"""Targets v4 (N013) = targets_v3 with the slide-POSITION artefact removed from every fluorescence-derived target (labels only;
inputs are never touched). Per slide x target: a smooth, strongly regularised model of the target on position (FOV centre x/y/z +
position in FOV; HistGradientBoosting, 31 leaves max, min 200 cells/leaf) is fitted on that slide's TRAINING-split cells (split 0) only,
then its prediction minus its training-cell mean is subtracted from all cells of the slide (slide mean preserved; spatial trend removed).
User ruling 2026-11-09: position on the slide carries no biology. Also flags GS-LF pair outliers (log-area residual > 3 robust sd, N003)
as gs_pair_suspect (for label hygiene; never used to filter test inputs). Writes targets_v4.npz (+ v3 keys unchanged otherwise)."""
import numpy as np, pandas as pd, json, sys
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W = '/hnvme/workspace/v181bb10-bsccm'; T = dict(np.load(W + '/targets_v3.npz', allow_pickle=True)); T['slide_names'] = T['slide_names'].astype(str);   # own file; object->str so v4 loads without pickle
idx = pd.read_parquet(W + '/bsccm_index.parquet')
P = idx[['fov_center_x', 'fov_center_y', 'fov_center_z', 'position_in_fov_x_pix', 'position_in_fov_y_pix']].values.astype(float)
slide = idx.data_path.values; Y, Mk, sp = T['Y'].copy(), T['mask'], T['split']; names = list(T['names']); info = {}
def _fit(args):
    s, j = args
    m = np.where((slide == s) & Mk[:, j])[0]
    if len(m) < 1000: return s, j, None, None
    tr = m[sp[m] == 0]
    g = HGB(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=0).fit(P[tr], Y[tr, j])
    trend = g.predict(P[m]); trend -= trend[sp[m] == 0].mean(); return s, j, m, trend
import os; os.environ.setdefault('OMP_NUM_THREADS', '3')
from multiprocessing import Pool
jobs = [(s, j) for s in sorted(set(slide)) for j in range(len(names))]
with Pool(16) as pool: outs = pool.map(_fit, jobs, chunksize=1)
for s, j, m, trend in outs:
    if m is None: continue
    n = names[j]; before = float(np.var(Y[m, j])); orig = Y[m, j].copy(); Y[m, j] = orig - trend
    info[f'{n}|{s}'] = dict(n=int(len(m)), var_explained_by_trend=float(np.var(trend) / before), corr_trend_target=float(np.corrcoef(trend, orig)[0, 1]))
T['Y'] = Y.astype(np.float32); T['Y_v3'] = np.load(W + '/targets_v3.npz', allow_pickle=True)['Y']
# GS-LF pair hygiene flag from the N003 area audit
d = pd.read_csv('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N003-dataset-audit-modality-pairing-labels-sp/giemsa_lf_match.csv')
ok = (d.area_giemsa > 5) & (d.area_dpc > 5); x, yv = np.log(d.area_giemsa[ok]), np.log(d.area_dpc[ok]); b = np.polyfit(yv, x, 1); r = x - np.polyval(b, yv)
mad = np.median(np.abs(r - np.median(r))) * 1.4826; sus = np.zeros(len(Y), bool); sus[d.global_index[ok][np.abs(r) > 3 * mad].values] = True
sus[d.global_index[~ok].values] = True; T['gs_pair_suspect'] = sus
np.savez(W + '/targets_v4.npz', **T)
v = pd.DataFrame(info).T; v['target'] = [k.split('|')[0] for k in v.index]
summ = v.groupby('target')[['var_explained_by_trend', 'corr_trend_target']].median().round(3)
print(summ.to_string()); print('GS pair suspects:', int(sus.sum()))
json.dump(info, open(sys.argv[1] + '/targets_v4_detrend.json', 'w'), indent=0); summ.to_csv(sys.argv[1] + '/targets_v4_detrend_summary.csv')
